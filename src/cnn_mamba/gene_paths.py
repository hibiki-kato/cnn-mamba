"""Gene-structure supervision tracks for end-to-end HMM training.

For every EviAnn gene the transcript with the longest total CDS is converted
into a per-base state path (see ``gene_hmm``). Tracks are stored in genomic
coordinates; ``track[strand][p]`` is the state of base ``p`` when the strand is
read in transcript orientation, so a minus-strand window is the reversed slice,
exactly like the phase tracks.

Track values: -1 unconstrained, otherwise a ``gene_hmm`` state index.
Transcripts that the training grammar cannot represent (non-canonical sites,
non-ATG start, missing/split stop, internal stop, CDS length not divisible by
three) and clusters of overlapping same-strand genes are left unconstrained
over their CDS span instead of being forced to a wrong path.

``build_isoform_mask_tracks`` is the all-isoform alternative: instead of one
path it stores allowed-set masks (see ``gene_hmm``) whose path set is the
splice graph of every CDS isoform of the gene.
"""

from collections import defaultdict

import numpy as np
import torch

from .gene_hmm import (
    ALL_EVENTS, BOUNDARIES, BOUNDARY_EDGES, E0, EVENT_SHIFT, I0, IR, N_STATES,
    UNCONSTRAINED, path_is_grammatical,
)

ENCODE = np.full(256, 4, dtype=np.uint8)
for _base, _value in (("A", 0), ("C", 1), ("G", 2), ("T", 3)):
    ENCODE[ord(_base)] = _value
COMPLEMENT = str.maketrans("ACGTN", "TGCAN")


def _attrs(text):
    return dict(item.split("=", 1) for item in text.strip().split(";") if "=" in item)


def parse_gff_transcripts(path, accessions=None):
    """Return ``{transcript: {gene, chrom, strand, cds}}`` from a GFF3."""
    genes, cds, location = {}, defaultdict(list), {}
    with open(path) as handle:
        for line in handle:
            if line.startswith("#") or not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9 or (accessions is not None and fields[0] not in accessions):
                continue
            attrs = _attrs(fields[8])
            if fields[2] == "mRNA" and attrs.get("ID"):
                genes[attrs["ID"]] = attrs.get("Parent", attrs["ID"])
                location[attrs["ID"]] = (fields[0], fields[6])
            elif fields[2] == "CDS":
                for tid in attrs.get("Parent", "").split(","):
                    if tid:
                        cds[tid].append((int(fields[3]) - 1, int(fields[4])))
    transcripts = {}
    for tid, segments in cds.items():
        if tid not in location or location[tid][1] not in {"+", "-"}:
            continue
        chrom, strand = location[tid]
        transcripts[tid] = {
            "gene": genes[tid], "chrom": chrom, "strand": strand,
            "cds": sorted(set(segments)),
        }
    return transcripts


def longest_cds_transcripts(transcripts):
    """One transcript per gene: longest total CDS, ties broken by ID."""
    best = {}
    for tid in sorted(transcripts):
        record = transcripts[tid]
        length = sum(end - start for start, end in record["cds"])
        key = (record["chrom"], record["strand"], record["gene"])
        if key not in best or length > best[key][0]:
            best[key] = (length, tid)
    return sorted(tid for _length, tid in best.values())


def transcript_path(segments, lo, hi, strand):
    """State path over ``[lo, hi)`` in transcript orientation."""
    if strand == "-":
        segments = [(hi - end, hi - start) for start, end in segments]
        lo, hi = 0, hi - lo
    else:
        segments = [(start - lo, end - lo) for start, end in segments]
        hi, lo = hi - lo, 0
    segments = sorted(segments)
    path = np.full(hi, -1, dtype=np.int8)
    coding = 0
    for index, (start, end) in enumerate(segments):
        path[start:end] = E0 + (coding + np.arange(end - start)) % 3
        coding += end - start
        if index + 1 < len(segments):
            path[end:segments[index + 1][0]] = I0 + coding % 3
    return path, coding


def _validated_path(fetch, record, lo, hi, chrom, chrom_len):
    """Return ``(path, padded, reason)`` for one transcript over ``[lo, hi)``.

    ``path`` is in transcript orientation with IR outside the transcript's own
    CDS; ``padded`` adds one IR base of context on each side (edges into the
    span start at ``padded[1]``). ``reason`` names why the training grammar
    cannot represent the transcript, else ``None``.
    """
    strand = record["strand"]
    path, coding = transcript_path(record["cds"], lo, hi, strand)
    path[path < 0] = IR
    if coding % 3:
        return path, None, "cds_length_not_multiple_of_3"
    # One IR base of context on each side checks start and stop.
    left, right = max(0, lo - 1), min(chrom_len, hi + 1)
    text = fetch(chrom, left, right)
    if strand == "-":
        text = text.translate(COMPLEMENT)[::-1]
        pad_before, pad_after = right - hi, lo - left
    else:
        pad_before, pad_after = lo - left, right - hi
    if not (pad_before and pad_after):
        return path, None, "touches_chromosome_end"
    padded = np.concatenate((
        np.full(1, IR, dtype=np.int8), path, np.full(1, IR, dtype=np.int8),
    ))
    sequence = torch.from_numpy(
        ENCODE[np.frombuffer(text.encode("ascii"), dtype=np.uint8)].astype(np.int64)
    )
    if not path_is_grammatical(sequence, torch.from_numpy(padded.astype(np.int64))):
        return path, None, "ungrammatical"
    return path, padded, None


def _clusters(spans):
    clusters = []
    for lo, hi, key in spans:
        # Abutting genes (stop codon directly followed by ATG) have no IR
        # base between them and are not representable either.
        if clusters and lo <= clusters[-1][1]:
            clusters[-1][1] = max(clusters[-1][1], hi)
            clusters[-1][2].append(key)
        else:
            clusters.append([lo, hi, [key]])
    return clusters


def _background(spans, chrom_len, ir_margin, ir_value, unconstrained):
    if ir_margin is None:
        return np.full(chrom_len, ir_value, dtype=np.asarray(ir_value).dtype)
    track = np.full(chrom_len, unconstrained, dtype=np.asarray(ir_value).dtype)
    for lo, hi, _key in spans:
        track[max(0, lo - ir_margin):min(chrom_len, hi + ir_margin)] = ir_value
    return track


def build_path_tracks(fetch, transcripts, chosen, chrom, chrom_len, ir_margin=None):
    """Return ``({"+": track, "-": track}, stats)`` for one chromosome.

    ``fetch(chrom, start, end)`` returns upper-case genomic sequence. With
    ``ir_margin=None`` every base outside genes is supervised as IR; otherwise
    only bases within ``ir_margin`` bp of a chosen CDS span are.
    """
    stats = defaultdict(int)
    tracks = {}
    for strand in ("+", "-"):
        local = [
            tid for tid in chosen
            if transcripts[tid]["chrom"] == chrom and transcripts[tid]["strand"] == strand
        ]
        spans = sorted(
            (transcripts[tid]["cds"][0][0], transcripts[tid]["cds"][-1][1], tid)
            for tid in local
        )
        track = _background(spans, chrom_len, ir_margin, np.int8(IR), -1)
        for lo, hi, members in _clusters(spans):
            if len(members) > 1:
                track[lo:hi] = -1
                stats["overlap_masked_transcripts"] += len(members)
                continue
            path, _padded, reason = _validated_path(
                fetch, transcripts[members[0]], lo, hi, chrom, chrom_len
            )
            if reason is not None:
                track[lo:hi] = -1
                stats[f"invalid_{reason}"] += 1
                continue
            track[lo:hi] = path if strand == "+" else path[::-1]
            stats["supervised_transcripts"] += 1
        tracks[strand] = track
    return tracks, dict(stats)


# Boundary-event bit of each transition, for allowed-set masks.
EDGE_EVENT_BITS = np.zeros((N_STATES, N_STATES), dtype=np.int16)
for _index, _name in enumerate(BOUNDARIES):
    for _source, _target in BOUNDARY_EDGES[_name]:
        EDGE_EVENT_BITS[_source, _target] = 1 << (EVENT_SHIFT + _index)


def build_isoform_mask_tracks(fetch, transcripts, chrom, chrom_len, ir_margin=None):
    """Return ``({"+": masks, "-": masks}, stats)`` allowing any CDS isoform.

    Masks use the ``gene_hmm`` allowed-set encoding (int16). Within a gene the
    allowed states of a base are the union over its isoforms and a boundary
    event is allowed only where some isoform uses it, so the constrained path
    set is the splice graph of the annotated isoforms; frames are still checked
    by the grammar. Genes with any unrepresentable isoform and clusters of
    overlapping same-strand genes are unconstrained over their CDS span.
    """
    stats = defaultdict(int)
    tracks = {}
    ir_mask = np.int16(1 << IR | ALL_EVENTS << EVENT_SHIFT)
    for strand in ("+", "-"):
        genes = defaultdict(list)
        for tid in sorted(transcripts):
            record = transcripts[tid]
            if record["chrom"] == chrom and record["strand"] == strand:
                genes[record["gene"]].append(tid)
        spans = sorted(
            (
                min(transcripts[tid]["cds"][0][0] for tid in members),
                max(transcripts[tid]["cds"][-1][1] for tid in members),
                gene,
            )
            for gene, members in genes.items()
        )
        track = _background(spans, chrom_len, ir_margin, ir_mask, UNCONSTRAINED)
        for lo, hi, members in _clusters(spans):
            if len(members) > 1:
                track[lo:hi] = UNCONSTRAINED
                stats["overlap_masked_genes"] += len(members)
                continue
            gene = members[0]
            union = np.zeros(hi - lo, dtype=np.int16)
            paths = set()
            reason = None
            for tid in genes[gene]:
                path, padded, reason = _validated_path(
                    fetch, transcripts[tid], lo, hi, chrom, chrom_len
                )
                if reason is not None:
                    break
                union |= np.left_shift(1, path).astype(np.int16)
                union |= EDGE_EVENT_BITS[padded[:-2], padded[1:-1]]
                paths.add(path.tobytes())
            if reason is not None:
                track[lo:hi] = UNCONSTRAINED
                stats[f"invalid_genes_{reason}"] += 1
                continue
            track[lo:hi] = union if strand == "+" else union[::-1]
            stats["supervised_genes"] += 1
            stats["supervised_transcripts"] += len(genes[gene])
            stats["multi_path_genes"] += len(paths) > 1
        tracks[strand] = track
    return tracks, dict(stats)
