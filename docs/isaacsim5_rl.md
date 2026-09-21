# Isaac Sim 5.0 / Isaac Lab 2.2.1 LiDAR input

This fork adds a source-checkout integration that does **not** overwrite Isaac Lab.
Use the `LidarSensor` directory on `PYTHONPATH`, and import after `AppLauncher` starts.
No new dependency installation is needed in the tested training environment.

## Why the upstream version fails

[Upstream issue 29](https://github.com/aCodeDog/OmniPerception/issues/29) and
[issue 17](https://github.com/aCodeDog/OmniPerception/issues/17) report stale XForm poses.
On the tested Sim 5.0 host, both USD and Fabric queries for visual mesh children
remained at the initial pose while PhysX rigid-body tensors moved.
The fix resolves each mesh's rigid-body ancestor once, caches its relative
transform, and reads current PhysX poses at every sensor update. Articulation
link descendants use the same rigid-body ancestry mechanism; robot-specific
articulation validation remains to be performed. Non-physical prims still use USD.
Mesh topology and mesh-to-body transforms must remain fixed during a run.

## Scene configuration

```python
from LidarSensor.example.isaaclab.isaaclab.sensors import LidarSensorCfg, LivoxPatternCfg

# Assign to an InteractiveSceneCfg field named lidar.
lidar = LidarSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base",
    ray_alignment="base",
    offset=LidarSensorCfg.OffsetCfg(pos=(0.0, 0.0, 0.15)),  # Replace with calibrated mount pose.
    mesh_prim_paths=["/World/ground"],
    dynamic_env_mesh_prim_paths=["{ENV_REGEX_NS}/Obstacle/geometry/mesh"],
    max_distance=10.0,
    min_range=0.2,
    update_period=0.02,
    pattern_cfg=LivoxPatternCfg(sensor_type="mid360", samples=4000, downsample=8),
)
```

With a nominal 200,000 raw rays/s, a 20 ms scan advances 4,000 pattern rows;
downsampling by eight produces 500 rays per environment without slowing the
sequence clock. Select `samples` to match your chosen acquisition interval.
The file is a finite precomputed sequence and wraps; this is not a physical
optical model or a guarantee of equivalence to the real device.

For a manager-based observation term:

```python
def lidar_points(env, sensor_name="lidar"):
    return env.scene[sensor_name].get_observation()
```

Output is GPU `float32 [num_envs, num_points, 4]`: sensor-frame XYZ divided by
`max_distance`, followed by a valid-return mask. Misses, too-near returns and
dropped points are zero with mask zero. Distances remain meters in `.data.distances`.
Point directions change over time: use a point encoder or explicitly bin points
into angular cells before using a grid encoder. A fixed beam-index MLP assumption
does not match the moving Livox sequence. Keep actor observations free of ground
truth map/obstacle poses. The optional world-frame point cloud is for diagnostics.

The positive `update_period` is the scheduling source of truth. The legacy
`update_frequency` is only used when `update_period` was not set. Sample regularly;
lazy reads spanning multiple acquisition periods are not interpolated. An entire
scan uses one physical pose, so within-scan motion distortion is not modeled.

## Training boundaries

- No random-ray fallback when pattern data is missing; startup fails explicitly.
- Sequence cursors and reset are per environment, not shared global scan time.
- Mesh-update errors stop the run instead of silently returning stale geometry.
- The inherited mesh backend combines environments. **Dense environment layouts
  are not isolated.** Keep all foreign geometry outside every sensor's max range,
  including possible robot/obstacle motion. The smoke test uses 40 m spacing and
  10 m range. This is a small-scale input prototype, not a validated 4,096-env backend.
- Angular noise and legacy range normalization are rejected rather than ignored;
  `get_observation()` supplies explicit normalization. Range noise and dropout are supported.
- No changes are made to an existing `robot_lab` policy, rewards, checkpoints or
  training process. Full PPO training, articulated self-occlusion and large-scale
  throughput still require task-specific validation.

## Reproduce checks

From this fork's root, using the Python environment that already runs Isaac Lab:

```bash
export PYTHONPATH="$PWD/LidarSensor:${PYTHONPATH:-}"
python tests/test_livox_sequence.py
python tests/isaaclab_smoke.py --headless --device cuda:0
```

The smoke test moves two physical boxes without rendering and compares live
PhysX reads with the old USD reads. It also checks mount offset/rotation, finite
MID-360 observations, partial reset, dropout and three point-encoder optimizer
steps. This backward-pass check is not a learned locomotion policy or PPO result.

Test platform: Isaac Sim 5.0.0.0, Isaac Lab checkout VERSION 2.2.1
(Python package metadata 0.46.2), Python 3.11.14, PyTorch 2.7.0+cu128,
Warp 1.9.0, RTX 3090, Ubuntu 22.04. Recorded on 2026-09-22.
