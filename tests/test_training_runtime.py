import argparse
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
import numpy as np

from cnn_mamba.training_runtime import (
    add_training_runtime_arguments, amp_dtype, configure_training_runtime,
)


class TrainingRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.parser = argparse.ArgumentParser()
        add_training_runtime_arguments(self.parser)
        self.device = torch.device('cuda')

    def test_manual_batch_preserves_existing_defaults_without_capacity_queries(self):
        args = self.parser.parse_args([])
        with patch('cnn_mamba.training_runtime.hardware_memory', side_effect=AssertionError('no probe')):
            info = configure_training_runtime(args, self.device, 10000)
        self.assertEqual((args.batch_size, args.grad_accum), (4, 2))
        self.assertEqual(info['effective_batch_size'], 8)
        self.assertIsNone(info['memory_capacity_bytes'])
        self.assertIsNone(info['amp_dtype'])

    def test_auto_uses_fixed_capacity_and_preserves_effective_batch(self):
        for gb in (4, 12, 80):
            args = self.parser.parse_args(['--batch-size', 'auto', '--amp', '--amp-dtype', 'bfloat16'])
            with patch('cnn_mamba.training_runtime.hardware_memory', return_value=gb*1024**3) as capacity, patch(
                    'torch.cuda.mem_get_info', side_effect=AssertionError('no free-memory queries')):
                info = configure_training_runtime(args, self.device, 10000)
            capacity.assert_called_once_with(self.device)
            self.assertEqual(args.batch_size * args.grad_accum, 8)
            self.assertLessEqual(args.batch_size, 8)
            self.assertEqual(info['amp_dtype'], 'bfloat16')

    def test_amp_reduces_estimate_and_longer_windows_reduce_microbatch(self):
        sizes = []
        for flags, length in (([], 10000), (['--amp'], 10000), (['--amp'], 40000)):
            args = self.parser.parse_args(['--batch-size', 'auto', '--memory-gb', '12',
                                          '--effective-batch-size', '32', *flags])
            with patch('cnn_mamba.training_runtime.hardware_memory', side_effect=AssertionError('override wins')):
                configure_training_runtime(args, self.device, length)
            sizes.append(args.batch_size)
            self.assertEqual(args.batch_size * args.grad_accum, 32)
        self.assertGreater(sizes[1], sizes[0])
        self.assertLess(sizes[2], sizes[1])

    def test_non_power_of_two_effective_batch_is_preserved_with_cap(self):
        args = self.parser.parse_args(['--batch-size', 'auto', '--memory-gb', '80',
                                      '--effective-batch-size', '12', '--max-batch-size', '8'])
        configure_training_runtime(args, self.device, 10000)
        self.assertEqual((args.batch_size, args.grad_accum), (4, 3))
        args = self.parser.parse_args(['--batch-size', 'auto', '--grad-accum', '4', '--memory-gb', '80'])
        configure_training_runtime(args, self.device, 10000)
        self.assertEqual(args.batch_size * args.grad_accum, 16)

    def test_manual_effective_batch_and_incompatible_values(self):
        args = self.parser.parse_args(['--batch-size', '4', '--effective-batch-size', '20'])
        configure_training_runtime(args, self.device, 10000)
        self.assertEqual(args.grad_accum, 5)
        for flags in (['--batch-size', '3', '--effective-batch-size', '8'],
                      ['--grad-accum', '0'], ['--memory-gb', 'nan'],
                      ['--memory-gb', 'inf'], ['--memory-fraction', '0']):
            with self.subTest(flags=flags), self.assertRaises(ValueError):
                configure_training_runtime(self.parser.parse_args(flags), self.device, 10000)

    def test_unknown_capacity_defaults_to_one_and_precision_names_map(self):
        args = self.parser.parse_args(['--batch-size', 'auto'])
        with patch('cnn_mamba.training_runtime.hardware_memory', return_value=0):
            configure_training_runtime(args, self.device, 10000)
        self.assertEqual((args.batch_size, args.grad_accum), (1, 8))
        self.assertEqual(amp_dtype('float16'), torch.float16)
        self.assertEqual(amp_dtype('bfloat16'), torch.bfloat16)

    @unittest.skipUnless(torch.cuda.is_available() and importlib.util.find_spec('mamba_ssm'),
                         'native CUDA training environment unavailable')
    def test_training_cli_bf16_auto_and_resume_with_empty_scaler_state(self):
        from cnn_mamba import train
        from cnn_mamba.model import SpliceMamba
        if torch.version.hip is not None or not torch.cuda.is_bf16_supported():
            self.skipTest('native BF16 CUDA training required')
        def small_model(**kwargs):
            return SpliceMamba(**(kwargs | dict(d_model=64, d_state=16, n_layers=1)))
        with tempfile.TemporaryDirectory(prefix='cnn-mamba-training-test-') as temporary:
            root = Path(temporary)
            data, output = root/'data', root/'output'
            rng = np.random.default_rng(71)
            for split in ('train', 'val', 'test'):
                (data/split).mkdir(parents=True)
                for index in range(2):
                    arrays = dict(sequence=rng.integers(0, 5, 128),
                                  labels=rng.integers(0, 3, 128),
                                  start_stop_labels=rng.integers(0, 3, 128),
                                  splice_ignore=np.zeros(128), start_stop_ignore=np.zeros(128),
                                  chess_labels=np.zeros(128), chess_start_stop_labels=np.zeros(128),
                                  phase_labels=rng.integers(0, 3, 128), phase_mask=np.ones(128),
                                  reference_phase_labels=np.zeros(128), reference_phase_mask=np.ones(128),
                                  cds_labels=rng.integers(0, 2, 128), reference_cds_labels=np.zeros(128))
                    np.savez(data/split/f'{index}.npz', **arrays)
            argv = ['train', '--data-root', str(data), '--output-dir', str(output),
                    '--mode', 'baseline', '--all-splits-as-train', '--epochs', '1',
                    '--window-size', '128', '--stride', '64', '--workers', '0',
                    '--batch-size', 'auto', '--effective-batch-size', '4', '--memory-gb', '1',
                    '--amp', '--amp-dtype', 'bfloat16', '--joint-frame-aux-weight', '.1',
                    '--frame-dilation', '3', '--fuse-bidirectional']
            with patch.object(train, 'SpliceMamba', side_effect=small_model), patch.object(sys, 'argv', argv):
                train.main()
            checkpoint = torch.load(output/'last_model.pt', map_location='cpu', weights_only=True)
            policy = checkpoint['config']['batch_policy']
            self.assertEqual(policy['batch_size'] * policy['grad_accum'], 4)
            self.assertEqual(checkpoint['scaler_state_dict'], {})
            self.assertEqual(checkpoint['optimizer_step'], 1)
            self.assertTrue(all(torch.isfinite(v).all() for v in checkpoint['model_state_dict'].values()))
            resumed = argv.copy()
            resumed[resumed.index('--epochs')+1] = '2'
            resumed[resumed.index('--amp-dtype')+1] = 'float16'
            resumed.append('--resume')
            with patch.object(train, 'SpliceMamba', side_effect=small_model), patch.object(sys, 'argv', resumed):
                train.main()
            checkpoint = torch.load(output/'last_model.pt', map_location='cpu', weights_only=True)
            self.assertEqual(checkpoint['epoch'], 2)
            self.assertTrue(checkpoint['scaler_state_dict'])
            for handler in train.logging.getLogger('eviann_pu_pilot').handlers:
                handler.close()


if __name__ == '__main__':
    unittest.main()
