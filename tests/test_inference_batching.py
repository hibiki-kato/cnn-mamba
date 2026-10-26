import argparse
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from cnn_mamba.inference import add_batch_arguments, configure_batch_size, configure_threads, hardware_memory, completed_output
from cnn_mamba.model import SpliceMamba


class BatchTests(unittest.TestCase):
    def test_completed_legacy_outputs_are_fp32_and_precision_changes_need_new_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.assertFalse(completed_output(root, 'bf16'))
            (root/'.complete.json').write_text('{}')
            (root/'metadata.json').write_text(json.dumps({'backend': 'torch'}))
            self.assertTrue(completed_output(root, 'fp32'))
            with self.assertRaisesRegex(ValueError, 'different output directory'):
                completed_output(root, 'bf16')
            with self.assertRaisesRegex(ValueError, 'compile-scan'):
                completed_output(root, 'fp32', compile_scan=True)
            (root/'metadata.json').write_text(json.dumps({'precision': 'fp32', 'compile_scan': True}))
            self.assertTrue(completed_output(root, 'fp32', compile_scan=True))
            with self.assertRaisesRegex(ValueError, 'compile-scan'):
                completed_output(root, 'fp32')

    def setUp(self):
        self.parser = argparse.ArgumentParser()
        add_batch_arguments(self.parser)
        self.model = SpliceMamba(d_model=16, d_state=8, n_layers=1,
                                 dropout=0.0, joint_frame_auxiliary=True).eval()

    def test_auto_scales_with_window_and_respects_cap_and_manual_size(self):
        args = self.parser.parse_args(["--batch-size", "auto", "--max-batch-size", "8"])
        with patch("cnn_mamba.inference.hardware_memory", return_value=64 * 1024**2):
            small = configure_batch_size(args, self.model, 64, torch.device("cpu"))
            large = configure_batch_size(args, self.model, 10000, torch.device("cpu"))
            self.assertEqual(small, 8)
            self.assertLess(large, small)
            self.assertGreaterEqual(large, 1)
            args.batch_size = 3
            self.assertEqual(configure_batch_size(args, self.model, 10000, torch.device("cpu")), 3)

    def test_cpu_default_remains_one_and_unknown_capacity_uses_one(self):
        args = self.parser.parse_args([])
        self.assertEqual(configure_batch_size(args, self.model, 64, torch.device("cpu")), 1)
        args.batch_size = "auto"
        with patch("cnn_mamba.inference.hardware_memory", return_value=0):
            self.assertEqual(configure_batch_size(args, self.model, 64, torch.device("cpu")), 1)

    def test_invalid_memory_options_are_rejected(self):
        for argv in (["--memory-fraction", "0"], ["--memory-fraction", "1.1"],
                     ["--max-batch-size", "0"], ["--memory-gb", "0"]):
            with self.subTest(argv=argv), self.assertRaises(ValueError):
                configure_batch_size(self.parser.parse_args(argv), self.model, 64, torch.device("cpu"))

    def test_capacity_uses_hardware_specs_without_live_memory_queries(self):
        with patch("torch.cuda.get_device_properties", return_value=SimpleNamespace(total_memory=12 * 1024**3)), patch(
                "torch.cuda.mem_get_info", side_effect=AssertionError("must not query live VRAM")):
            self.assertEqual(hardware_memory(torch.device("cuda")), 12 * 1024**3)
        def sysconf(name):
            return {"SC_PHYS_PAGES": 4 * 1024**2, "SC_PAGE_SIZE": 4096}[name]
        with patch("cnn_mamba.inference.sys.platform", "linux"), patch(
                "os.sysconf", side_effect=sysconf), patch(
                "torch.mps.recommended_max_memory", return_value=12 * 1024**3), patch(
                "torch.mps.driver_allocated_memory", side_effect=AssertionError("must not query live MPS memory")):
            self.assertEqual(hardware_memory(torch.device("cpu")), 16 * 1024**3)
            self.assertEqual(hardware_memory(torch.device("mps")), 12 * 1024**3)

    def test_hpc_allocation_override_needs_no_hardware_probe_or_forward(self):
        args = self.parser.parse_args(["--batch-size", "auto", "--memory-gb", "0.5"])
        with patch("cnn_mamba.inference.hardware_memory", side_effect=AssertionError("unnecessary hardware probe")), patch.object(
                self.model, "forward", side_effect=AssertionError("must not run a trial forward")):
            size = configure_batch_size(args, self.model, 10000, torch.device("cpu"))
        self.assertGreaterEqual(size, 1)
        self.assertLessEqual(size, args.max_batch_size)

    def test_batched_forward_matches_individual_windows_for_all_heads(self):
        sequence = torch.randint(0, 5, (4, 65))
        with torch.inference_mode():
            batch = self.model(sequence, return_joint_frame=True)
            single = [self.model(row, return_joint_frame=True) for row in sequence.split(1)]
            for actual, parts in zip(batch, zip(*single)):
                torch.testing.assert_close(actual, torch.cat(parts), rtol=2e-5, atol=2e-6)

    def test_thread_defaults_respect_hpc_allocation_and_environment(self):
        with patch.dict("os.environ", {}, clear=True), patch(
                "cnn_mamba.inference.sys.platform", "linux"), patch(
                "torch.get_num_threads", return_value=16), patch(
                "os.sched_getaffinity", return_value=set(range(16)), create=True), patch("torch.set_num_threads") as setter:
            configure_threads(None, torch.device("cpu"))
            setter.assert_called_once_with(8)
        with patch.dict("os.environ", {"SLURM_CPUS_PER_TASK": "2"}, clear=True), patch(
                "torch.get_num_threads", return_value=16), patch(
                "os.sched_getaffinity", return_value=set(range(16)), create=True), patch("torch.set_num_threads") as setter:
            configure_threads(None, torch.device("cpu"))
            setter.assert_called_once_with(2)
        with patch.dict("os.environ", {"OMP_NUM_THREADS": "12"}, clear=True), patch(
                "torch.get_num_threads", return_value=12), patch("torch.set_num_threads") as setter:
            self.assertEqual(configure_threads(None, torch.device("cpu")), 12)
            setter.assert_not_called()
            configure_threads(3, torch.device("cpu"))
            setter.assert_called_once_with(3)


if __name__ == "__main__":
    unittest.main()
