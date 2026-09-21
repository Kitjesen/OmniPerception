"""Livox angle sequence loading and batched, per-environment scan cursors."""
from pathlib import Path

import numpy as np
import torch


def load_directions(cfg, device):
    root = Path(__file__).resolve().parents[6] / "sensor_pattern/sensor_lidar/scan_mode"
    path = Path(cfg.pattern_path) if cfg.pattern_path else root / f"{cfg.sensor_type}.npy"
    angles = np.load(path, allow_pickle=False)
    if angles.ndim != 2 or angles.shape[1] != 2 or not len(angles) or not np.isfinite(angles).all():
        raise ValueError(f"Expected finite (N, 2) azimuth/elevation angles: {path}")
    angles = torch.as_tensor(angles, device=device, dtype=torch.float32)
    theta, phi = angles.unbind(-1)
    return torch.stack((theta.cos() * phi.cos(), theta.sin() * phi.cos(), phi.sin()), -1)


class LivoxSequence:
    def __init__(self, directions, num_envs, samples, downsample=1, start=0):
        if samples <= 0 or downsample <= 0:
            raise ValueError("samples and downsample must be positive")
        self.directions = directions
        self.samples = samples
        self.start = start % len(directions)
        self.cursors = torch.full((num_envs,), self.start, device=directions.device, dtype=torch.long)
        self.offsets = torch.arange(0, samples, downsample, device=directions.device)

    def next(self, env_ids):
        indices = (self.cursors[env_ids, None] + self.offsets) % len(self.directions)
        result = self.directions[indices]
        self.cursors[env_ids] = (self.cursors[env_ids] + self.samples) % len(self.directions)
        return result

    def reset(self, env_ids):
        self.cursors[env_ids] = self.start
