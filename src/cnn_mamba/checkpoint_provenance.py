"""Training progress metadata shared by training and published checkpoints."""


def checkpoint_progress(saved):
    """Preserve recorded progress; leave absent checkpoint fields unknown."""
    return {
        "checkpoint_step": saved.get("optimizer_step"),
        "checkpoint_epoch": saved.get("epoch"),
    }
