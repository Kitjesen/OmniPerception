"""Run with Isaac Lab's Python, not pytest. No existing Lab files are modified."""
import argparse
import json
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch
import isaaclab.sim as sim_utils
from isaacsim.core.utils.prims import create_prim
from isaacsim.core.simulation_manager import SimulationManager
from LidarSensor.example.isaaclab.isaaclab.sensors import LidarSensor, LidarSensorCfg, LivoxPatternCfg

sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01, device=args.device, use_fabric=True))
create_prim('/World/envs', 'Xform')
ground_cfg = sim_utils.GroundPlaneCfg()
ground_cfg.func('/World/ground', ground_cfg, translation=(0.0, 0.0, -5.0))
for i in range(2):
    root = f'/World/envs/env_{i}'
    create_prim(root, 'Xform')
    create_prim(root + '/Sensor', 'Xform', translation=(0.0, i * 40.0, 0.0))
    create_prim(root + '/RotatedSensor', 'Xform', translation=(0.0, i * 40.0, 0.0),
                orientation=(math.sqrt(0.5), 0.0, 0.0, -math.sqrt(0.5)))
    cfg = sim_utils.CuboidCfg(size=(1.0, 1.0, 1.0),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0), collision_props=sim_utils.CollisionPropertiesCfg())
    cfg.func(root + '/Box', cfg, translation=(3.0, i * 40.0, 0.0))

paths = [f'/World/envs/env_{i}/Box' for i in range(2)]
cfg = LidarSensorCfg(prim_path='/World/envs/env_.*/Sensor', mesh_prim_paths=paths,
    max_distance=10.0, update_period=0.01,
    pattern_cfg=LivoxPatternCfg(sensor_type='mid360', use_simple_grid=True,
        horizontal_line_num=1, vertical_line_num=1, horizontal_fov_deg_min=0,
        horizontal_fov_deg_max=0, vertical_fov_deg_min=0, vertical_fov_deg_max=0))
sensor = LidarSensor(cfg)
rotated_cfg = cfg.replace(prim_path='/World/envs/env_.*/RotatedSensor',
    offset=LidarSensorCfg.OffsetCfg(pos=(0.0, 0.2, 0.0), rot=(math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5))))
rotated = LidarSensor(rotated_cfg)
mid = LidarSensor(LidarSensorCfg(prim_path='/World/envs/env_.*/Sensor', mesh_prim_paths=['/World/ground'],
    dynamic_env_mesh_prim_paths=['{ENV_REGEX_NS}/Box/geometry/mesh'],
    max_distance=10.0, update_period=0.01,
    pattern_cfg=LivoxPatternCfg(sensor_type='mid360', samples=2000, downsample=4)))
sim.reset()
assert len(mid._geometry.meshes) >= 3, 'Configured ground and dynamic meshes must be initialized'
view = SimulationManager.get_physics_sim_view().create_rigid_body_view('/World/envs/env_*/Box')
indices = torch.arange(2, device=args.device, dtype=torch.int32)
velocity = torch.zeros((2, 6), device=args.device)
velocity[:, 0] = 1.0
view.set_velocities(velocity, indices)

def scan():
    sim.step(render=False)
    sensor.update(0.01, force_recompute=True)
    mid.update(0.01, force_recompute=True)
    rotated.update(0.01, force_recompute=True)
    return sensor.data.distances.clone()

before = scan()
for _ in range(20):
    after = scan()
movement = (after - before).flatten()
assert torch.allclose(movement, torch.full_like(movement, 0.2), atol=0.025), movement
assert torch.allclose(after[0], after[1], atol=1e-4), after
assert torch.allclose(rotated.data.distances, after - 0.2, atol=1e-4)
assert torch.allclose(rotated.data.pointcloud[..., 0], after - 0.2, atol=1e-4)
assert torch.max(torch.abs(rotated.data.pointcloud[..., 1:])) < 1e-4

# Compare with the upstream USD read path on the same running scene.
live_reader = sensor._get_live_poses
sensor._get_live_poses = lambda view, ids: view.get_world_poses(indices=None if ids is None else ids.to(torch.int32), usd=True)
stale_start = scan()
for _ in range(10):
    stale_end = scan()
sensor._get_live_poses = live_reader
restored = scan()
assert torch.max(torch.abs(stale_end - stale_start)) < 1e-4
assert torch.min(restored - stale_end) > 0.2

obs = mid.get_observation()
assert obs.shape == (2, 500, 4), obs.shape
assert torch.isfinite(obs).all()
assert torch.all(torch.abs(obs[..., :3]) <= 1.00001)
cursor = mid._sequence.cursors.clone()
mid.reset(torch.tensor([0], device=args.device))
assert mid._sequence.cursors[0] == 0
assert mid._sequence.cursors[1] == cursor[1]
assert not mid._data.valid_mask[0].any()
scan()
assert mid._sequence.cursors[0] == 2000
assert mid._sequence.cursors[1] == (cursor[1] + 2000) % len(mid._sequence.directions)
mid.cfg.enable_sensor_noise = True
mid.cfg.pixel_dropout_prob = 1.0
scan()
assert not mid.data.valid_mask.any()
assert torch.count_nonzero(mid.get_observation()) == 0
mid.cfg.enable_sensor_noise = False

# A trainable point encoder consumes the actual sensor output without inf/NaN.
encoder = torch.nn.Sequential(torch.nn.Linear(4, 16), torch.nn.ELU(), torch.nn.Linear(16, 3)).to(args.device)
optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)
for _ in range(3):
    scan()
    optimizer.zero_grad()
    action = encoder(mid.get_observation().detach()).mean(1)
    loss = action.square().mean()
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in encoder.parameters())
    optimizer.step()
print('OMNI_RESULT=' + json.dumps({'moving_box_delta_m': movement.tolist(),
    'usd_stale_delta_m': (stale_end-stale_start).flatten().tolist(),
    'observation_shape': list(obs.shape), 'partial_reset': 'passed', 'encoder_backward': 'passed',
    'mount_transform': 'passed', 'dropout': 'passed'}))
app.close()
