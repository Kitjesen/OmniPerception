# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""GPU LiDAR observations with per-environment Livox scan state."""
from __future__ import annotations

import torch
from isaaclab.utils.math import quat_apply, quat_apply_inverse

from .ray_caster import RayCaster
from .lidar_sensor_data import LidarSensorData
from .ray_caster.patterns.livox_sequence import LivoxSequence, load_directions


class LidarSensor(RayCaster):
    def __init__(self, cfg):
        if cfg.ray_alignment != "base" or cfg.attach_yaw_only:
            raise ValueError("LiDAR requires ray_alignment='base' to follow roll, pitch and yaw")
        if not 0 <= cfg.min_range < cfg.max_distance:
            raise ValueError("Require 0 <= min_range < max_distance")
        if cfg.update_period <= 0:
            if cfg.update_frequency <= 0:
                raise ValueError("update_frequency must be positive")
            cfg.update_period = 1.0 / cfg.update_frequency
        if cfg.normalize_range or cfg.random_angle_noise != 0:
            raise ValueError("Use get_observation() for normalization; angular noise is not implemented")
        super().__init__(cfg)
        self._data = LidarSensorData()
        self._sequence = None

    def _initialize_rays_impl(self):
        super()._initialize_rays_impl()
        self._offset_quat = torch.tensor(self.cfg.offset.rot, device=self.device).float()
        self._sensor_directions = quat_apply_inverse(
            self._offset_quat.expand(self._view.count, self.num_rays, 4), self.ray_directions
        ).clone()
        cfg = self.cfg.pattern_cfg
        if hasattr(cfg, "sensor_type") and not cfg.use_simple_grid:
            self._sequence = LivoxSequence(load_directions(cfg, self.device), self._view.count,
                                           cfg.samples, cfg.downsample, cfg.rolling_window_start)
        self._data.distances = torch.full((self._view.count, self.num_rays), self.cfg.max_distance, device=self.device)
        self._data.valid_mask = torch.zeros_like(self._data.distances, dtype=torch.bool)
        self._data.pointcloud = torch.zeros_like(self._data.ray_hits_w) if self.cfg.return_pointcloud else None

    def reset(self, env_ids=None):
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        if self._sequence is not None:
            self._sequence.reset(ids)
        self._data.distances[ids] = self.cfg.max_distance
        self._data.valid_mask[ids] = False
        if self._data.pointcloud is not None:
            self._data.pointcloud[ids] = 0

    def _update_buffers_impl(self, env_ids):
        ids = torch.arange(self._view.count, device=self.device)[env_ids]
        if self._sequence is not None:
            rays = self._sequence.next(ids)
            self._sensor_directions[ids] = rays
            self.ray_directions[ids] = quat_apply(self._offset_quat.expand(*rays.shape[:-1], 4), rays)
        super()._update_buffers_impl(ids)
        parent_quat = self._data.quat_w[ids]
        offset = torch.tensor(self.cfg.offset.pos, device=self.device).float().expand(len(ids), 3)
        origin = self._data.pos_w[ids] + quat_apply(parent_quat, offset)
        hits = self._data.ray_hits_w[ids]
        distances = torch.linalg.vector_norm(hits - origin[:, None], dim=-1)
        valid = torch.isfinite(hits).all(-1) & (distances >= self.cfg.min_range) & (distances <= self.cfg.max_distance)
        if self.cfg.enable_sensor_noise:
            distances = distances + torch.randn_like(distances) * self.cfg.random_distance_noise
            valid &= torch.rand_like(distances) >= self.cfg.pixel_dropout_prob
        distances = torch.where(valid, distances.clamp(self.cfg.min_range, self.cfg.max_distance), self.cfg.max_distance)
        self._data.distances[ids] = distances
        self._data.valid_mask[ids] = valid
        if self.cfg.return_pointcloud:
            points = self._sensor_directions[ids] * distances[..., None]
            if self.cfg.pointcloud_in_world_frame:
                points = quat_apply(self._offset_quat.expand(*points.shape[:-1], 4), points)
                points = quat_apply(parent_quat[:, None].expand(-1, self.num_rays, -1), points) + origin[:, None]
            self._data.pointcloud[ids] = torch.where(valid[..., None], points, 0.0)

    def get_distances(self, env_ids=None):
        data = self.data.distances
        return data if env_ids is None else data[env_ids]

    def get_pointcloud(self, env_ids=None):
        if not self.cfg.return_pointcloud:
            raise ValueError("Point cloud generation is disabled")
        data = self.data.pointcloud
        return data if env_ids is None else data[env_ids]

    def get_observation(self, env_ids=None):
        """Finite (E, N, 4): sensor-frame xyz / max_range and a valid-return mask.

        Directions change with the Livox sequence. Feed these points to a point
        encoder, or explicitly bin them before using an angular-grid encoder.
        """
        data = self.data
        ids = slice(None) if env_ids is None else env_ids
        points = self._sensor_directions[ids] * (data.distances[ids] / self.cfg.max_distance)[..., None]
        mask = data.valid_mask[ids]
        points = torch.where(mask[..., None], points, 0.0)
        return torch.cat((points, mask[..., None].float()), dim=-1)
