"""Bounded multi-environment sensor timing, not a PPO training benchmark."""
import argparse
import json
import time
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument('--num_envs', type=int, default=32)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch
import warp as wp
import isaaclab.sim as sim_utils
from isaacsim.core.utils.prims import create_prim
from LidarSensor.example.isaaclab.isaaclab.sensors import LidarSensor, LidarSensorCfg, LivoxPatternCfg

sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=.01, device=args.device))
ground = sim_utils.GroundPlaneCfg()
ground.func('/World/ground', ground, translation=(0., 0., -2.))
for i in range(args.num_envs):
    root = f'/World/envs/env_{i}'
    create_prim(root, 'Xform')
    for name, size, position in [('Sensor', (.01, .01, .01), (0., i * 3., 0.)),
                                  ('Box', (1., 1., 1.), (3., i * 3., 0.))]:
        cfg = sim_utils.CuboidCfg(size=size, rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.), collision_props=sim_utils.CollisionPropertiesCfg())
        cfg.func(root + '/' + name, cfg, translation=position)
sensor = LidarSensor(LidarSensorCfg(prim_path='/World/envs/env_.*/Sensor', mesh_prim_paths=['/World/ground'],
    dynamic_env_mesh_prim_paths=['{ENV_REGEX_NS}/Box'], max_distance=10., update_period=.01,
    pattern_cfg=LivoxPatternCfg(sensor_type='mid360', samples=2000, downsample=4)))
sim.reset()
elapsed = []
torch.cuda.reset_peak_memory_stats()
for step in range(45):
    sim.step(render=False)
    torch.cuda.synchronize()
    start = time.perf_counter()
    sensor.update(.01, force_recompute=True)
    obs = sensor.get_observation()
    torch.cuda.synchronize()
    if step >= 15:
        elapsed.append((time.perf_counter() - start) * 1000)
    assert obs.shape == (args.num_envs, 500, 4)
    assert torch.isfinite(obs).all()
    assert torch.allclose(obs, obs[0:1].expand_as(obs), atol=1e-3), 'Translated identical environments disagree'
ordered = sorted(elapsed)
print('BATCH_RESULT=' + json.dumps({'num_envs': args.num_envs, 'rays_per_env': 500,
    'measured_steps': len(elapsed), 'sensor_wall_ms_median': ordered[len(ordered)//2],
    'sensor_wall_ms_p95': ordered[int(.95*(len(ordered)-1))],
    'torch_peak_allocated_mib': torch.cuda.max_memory_allocated()/2**20,
    'warp_runtime': wp.__version__, 'finite_observations': True,
    'note': 'Shared busy GPU; excludes physics and policy; Torch memory excludes Warp/Sim allocations'}), flush=True)
app.close()
