"""Differentiable single-strand gene-structure HMM/CRF layer.

All coordinates are in model (transcript) orientation, exactly as the windows
are fed to ``SpliceMamba``. A path assigns one of seven states to every base:

    0 IR         intergenic, UTR or any non-coding base
    1..3 E0..E2  coding base at codon position 0/1/2
    4..6 I0..I2  intron base; ``I_k`` re-enters the CDS at codon position k

Transition potentials are position dependent and come from the existing
candidate heads; state potentials come from a five-class state head
(non_CDS/frame0/frame1/frame2/intron). Transitions use the same motif
convention as the site labels:

    IR -> E0  at t if ATG starts at t                  start head at t
    E2 -> IR  at t if a stop codon occupies t-3..t-1   stop head at t-3
    E  -> I   at t if GT starts at t (donor)           donor head at t
    I  -> E   at t if AG occupies t-2..t-1 (acceptor)  acceptor head at t-2

A completed in-frame stop codon may only be followed by IR. Stop codons split
by an intron are not modelled; a codon interrupted by an intron never looks
like a contiguous stop because the intron-side bases are A/G of ``AG``.

The loss is the CRF negative log-likelihood of the label-constrained path set,
``logZ - logZ_constrained``: a sum over paths, not a max. Unconstrained bases
(label -1) allow every state, so the constrained term marginalises over them.
The partition function uses a parallel tree reduction in the log semiring.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import splice_candidate_masks, start_stop_candidate_masks

IR, E0, E1, E2, I0, I1, I2 = range(7)
N_STATES = 7
STATE_NAMES = ("IR", "E0", "E1", "E2", "I0", "I1", "I2")
# Five-class state head order; the first four match the joint-frame head.
STATE_CLASSES = ("non_CDS", "frame0", "frame1", "frame2", "intron")
STATE_TO_CLASS = (0, 1, 2, 3, 4, 4, 4)
NEG = -1.0e9
INFEASIBLE = 0.5 * NEG
BOUNDARIES = ("start", "stop", "donor", "acceptor")
# Longest motif look-behind (stop, t-3) / look-ahead (start, t+2).
EDGE_RELAX = 3


def sequence_masks(sequence):
    """Boolean motif masks indexed by motif start, as in ``common``."""
    gt, ag = splice_candidate_masks(sequence)
    atg, stop = start_stop_candidate_masks(sequence)
    return {"start": atg, "stop": stop, "donor": gt, "acceptor": ag}


def _shift_right(tensor, shift, fill):
    """Return ``out[..., t] = tensor[..., t - shift]`` with ``fill`` on the left."""
    if shift == 0:
        return tensor
    out = torch.full_like(tensor, fill)
    out[..., shift:] = tensor[..., :-shift]
    return out


def transition_potentials(sequence, boundary):
    """Build ``[B, L, S, S]`` log transition potentials into position t.

    ``boundary`` maps each name in ``BOUNDARIES`` to a ``[B, L]`` tensor
    indexed by motif start. Entry ``[:, 0]`` is unused (no transition into 0).
    """
    masks = sequence_masks(sequence)
    batch, length = sequence.shape
    dtype = boundary["start"].dtype
    neg = torch.tensor(NEG, dtype=dtype, device=sequence.device)
    zero = torch.zeros((), dtype=dtype, device=sequence.device)
    start_ok = masks["start"]
    donor_ok = masks["donor"]
    stop_end = _shift_right(masks["stop"], 3, False)
    acceptor_end = _shift_right(masks["acceptor"], 2, False)
    stop_score = _shift_right(boundary["stop"], 3, NEG)
    acceptor_score = _shift_right(boundary["acceptor"], 2, NEG)

    T = torch.full(
        (batch, length, N_STATES, N_STATES), NEG, dtype=dtype,
        device=sequence.device,
    )
    T[..., IR, IR] = 0.0
    T[..., IR, E0] = torch.where(start_ok, boundary["start"], neg)
    T[..., E0, E1] = 0.0
    T[..., E1, E2] = 0.0
    T[..., E2, E0] = torch.where(stop_end, neg, zero)
    T[..., E2, IR] = torch.where(stop_end, stop_score, neg)
    donor = torch.where(donor_ok, boundary["donor"], neg)
    T[..., E0, I1] = donor
    T[..., E1, I2] = donor
    T[..., E2, I0] = torch.where(stop_end, neg, donor)
    for exon, intron in ((E0, I0), (E1, I1), (E2, I2)):
        T[..., intron, intron] = 0.0
        T[..., intron, exon] = torch.where(acceptor_end, acceptor_score, neg)
    return T


def _log_matmul(a, b):
    return (a.unsqueeze(-1) + b.unsqueeze(-3)).logsumexp(-2)


def log_partition(emission, transition):
    """Free-start/free-end log partition by parallel tree reduction.

    ``emission`` is ``[B, L, S]`` and ``transition`` is ``[B, L, S, S]``.
    """
    steps = transition[:, 1:] + emission[:, 1:].unsqueeze(-2)
    while steps.shape[1] > 1:
        if steps.shape[1] % 2:
            identity = torch.full_like(steps[:, :1], NEG)
            identity.diagonal(dim1=-2, dim2=-1).zero_()
            steps = torch.cat((steps, identity), dim=1)
        steps = _log_matmul(steps[:, 0::2], steps[:, 1::2])
    if steps.shape[1] == 0:
        return emission[:, 0].logsumexp(-1)
    total = emission[:, 0].unsqueeze(-1) + steps[:, 0]
    return total.flatten(1).logsumexp(-1)


def label_mask(path_labels, dtype):
    """Additive ``[B, L, S]`` mask: 0 for allowed states, NEG otherwise."""
    allowed = F.one_hot(path_labels.clamp(min=0).long(), N_STATES).bool()
    allowed |= (path_labels < 0).unsqueeze(-1)
    return torch.where(
        allowed, torch.zeros((), dtype=dtype, device=path_labels.device),
        torch.tensor(NEG, dtype=dtype, device=path_labels.device),
    )


def viterbi(emission, transition):
    """Sequential max-product decoding; returns ``[B, L]`` state indices."""
    batch, length, _ = emission.shape
    score = emission[:, 0]
    pointers = []
    for t in range(1, length):
        candidates = score.unsqueeze(-1) + transition[:, t]
        score, pointer = candidates.max(-2)
        score = score + emission[:, t]
        pointers.append(pointer)
    path = torch.empty((batch, length), dtype=torch.long, device=emission.device)
    path[:, -1] = score.argmax(-1)
    for t in range(length - 1, 0, -1):
        path[:, t - 1] = pointers[t - 1].gather(1, path[:, t:t + 1]).squeeze(1)
    return path


class GeneHMM(nn.Module):
    """Learned scalar calibration of NN log-probabilities into CRF potentials."""

    def __init__(self):
        super().__init__()
        self.state_weight = nn.Parameter(torch.ones(()))
        self.boundary_weight = nn.Parameter(torch.ones(len(BOUNDARIES)))
        self.boundary_bias = nn.Parameter(torch.zeros(len(BOUNDARIES)))

    def potentials(self, sequence, splice_logits, start_stop_logits, state_logits):
        """Return fp32 ``(emission, boundary)`` potentials."""
        splice = splice_logits.float().log_softmax(-1)
        start_stop = start_stop_logits.float().log_softmax(-1)
        state = state_logits.float().log_softmax(-1)
        index = torch.tensor(STATE_TO_CLASS, device=state.device)
        emission = self.state_weight * state.index_select(-1, index)
        raw = {
            "start": start_stop[..., 1], "stop": start_stop[..., 2],
            "donor": splice[..., 1], "acceptor": splice[..., 2],
        }
        boundary = {
            name: self.boundary_weight[i] * raw[name] + self.boundary_bias[i]
            for i, name in enumerate(BOUNDARIES)
        }
        return emission, boundary

    def forward(
        self, sequence, splice_logits, start_stop_logits, state_logits,
        path_labels,
    ):
        """Per-window NLL / length and a feasibility mask.

        Labels on the ``EDGE_RELAX`` bases at each window end are ignored:
        motifs there are cut by the window, so their transitions cannot be
        checked. Windows whose labels still admit no path are returned with
        ``feasible=False`` and a zero loss so they never produce NaN/huge
        gradients; the caller decides how to count them.
        """
        with torch.autocast("cuda", enabled=False):
            emission, boundary = self.potentials(
                sequence, splice_logits, start_stop_logits, state_logits
            )
            transition = transition_potentials(sequence, boundary)
            log_z = log_partition(emission, transition)
            path_labels = path_labels.clone()
            path_labels[:, :EDGE_RELAX] = -1
            path_labels[:, -EDGE_RELAX:] = -1
            constrained = emission + label_mask(path_labels, emission.dtype)
            log_z_labels = log_partition(constrained, transition)
            feasible = log_z_labels > INFEASIBLE
            nll = (log_z - log_z_labels) / sequence.shape[1]
            nll = torch.where(feasible, nll, torch.zeros_like(nll))
        return nll, feasible

    def posteriors(self, sequence, splice_logits, start_stop_logits, state_logits):
        """Posterior state marginals and boundary-event marginals.

        Boundary marginals are indexed by motif start, i.e. directly comparable
        with the raw candidate probabilities of the same position.
        """
        with torch.enable_grad(), torch.autocast("cuda", enabled=False):
            emission, boundary = self.potentials(
                sequence, splice_logits.detach(), start_stop_logits.detach(),
                state_logits.detach(),
            )
            emission = emission.detach().requires_grad_(True)
            boundary = {
                name: value.detach().requires_grad_(True)
                for name, value in boundary.items()
            }
            transition = transition_potentials(sequence, boundary)
            log_z = log_partition(emission, transition)
            grads = torch.autograd.grad(
                log_z.sum(), [emission] + [boundary[name] for name in BOUNDARIES]
            )
        states = grads[0]
        events = {name: grad for name, grad in zip(BOUNDARIES, grads[1:])}
        return states, events

    @torch.no_grad()
    def decode(self, sequence, splice_logits, start_stop_logits, state_logits):
        with torch.autocast("cuda", enabled=False):
            emission, boundary = self.potentials(
                sequence, splice_logits, start_stop_logits, state_logits
            )
            return viterbi(emission, transition_potentials(sequence, boundary))


def path_is_grammatical(sequence, path):
    """True when every consecutive transition of ``path`` is allowed.

    ``sequence`` and ``path`` are 1-D integer tensors in model orientation.
    Used by label construction so that the training grammar is the single
    source of truth for which annotated transcripts are representable.
    """
    sequence = sequence.unsqueeze(0)
    zeros = torch.zeros(sequence.shape, dtype=torch.float32)
    boundary = {name: zeros for name in BOUNDARIES}
    T = transition_potentials(sequence, boundary)[0]
    if path.numel() < 2:
        return True
    t = torch.arange(1, path.numel())
    return bool((T[t, path[:-1], path[1:]] > INFEASIBLE).all())
