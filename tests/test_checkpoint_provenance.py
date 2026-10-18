import unittest

from cnn_mamba.checkpoint_provenance import checkpoint_progress


class CheckpointProvenanceTests(unittest.TestCase):
    def test_published_checkpoint_without_optimizer_step(self):
        saved = {"epoch": 4, "config": {}, "model_state_dict": {}}
        self.assertEqual(
            checkpoint_progress(saved),
            {"checkpoint_step": None, "checkpoint_epoch": 4},
        )

    def test_training_checkpoint_preserves_step_and_epoch(self):
        self.assertEqual(
            checkpoint_progress({"optimizer_step": 1200, "epoch": 3}),
            {"checkpoint_step": 1200, "checkpoint_epoch": 3},
        )

    def test_missing_progress_remains_unknown(self):
        self.assertEqual(
            checkpoint_progress({"config": {}, "model_state_dict": {}}),
            {"checkpoint_step": None, "checkpoint_epoch": None},
        )


if __name__ == "__main__":
    unittest.main()
