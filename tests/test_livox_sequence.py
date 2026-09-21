"""Sequence contracts that run without launching Isaac Sim."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / 'LidarSensor/LidarSensor/example/isaaclab/isaaclab/sensors/ray_caster/patterns/livox_sequence.py'
spec = importlib.util.spec_from_file_location('livox_sequence', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class SequenceTests(unittest.TestCase):
    def test_wrap_and_downsample_preserve_raw_scan_time(self):
        rays = torch.arange(30).reshape(10, 3)
        seq = module.LivoxSequence(rays, 2, 6, downsample=2, start=8)
        result = seq.next(torch.tensor([1]))
        torch.testing.assert_close(result[0], rays[[8, 0, 2]])
        self.assertEqual(seq.cursors.tolist(), [8, 4])

    def test_partial_reset_does_not_change_other_environment(self):
        seq = module.LivoxSequence(torch.zeros(10, 3), 2, 3)
        seq.next(torch.tensor([0, 1]))
        seq.reset(torch.tensor([0]))
        self.assertEqual(seq.cursors.tolist(), [0, 3])

    def test_multiple_wraps_keep_fixed_size(self):
        seq = module.LivoxSequence(torch.eye(3), 1, 8)
        self.assertEqual(seq.next(torch.tensor([0])).shape, (1, 8, 3))

    def test_mid360_angles_are_loaded_from_checkout(self):
        cfg = SimpleNamespace(pattern_path=None, sensor_type='mid360')
        directions = module.load_directions(cfg, 'cpu')
        self.assertEqual(directions.shape, (800000, 3))
        torch.testing.assert_close(directions.norm(dim=-1), torch.ones(800000))

    def test_missing_file_does_not_fall_back_to_random_rays(self):
        cfg = SimpleNamespace(pattern_path='missing-livox-file.npy', sensor_type='mid360')
        with self.assertRaises(FileNotFoundError):
            module.load_directions(cfg, 'cpu')

    def test_invalid_sampling_is_rejected(self):
        with self.assertRaises(ValueError):
            module.LivoxSequence(torch.eye(3), 1, 0)


if __name__ == '__main__':
    unittest.main()
