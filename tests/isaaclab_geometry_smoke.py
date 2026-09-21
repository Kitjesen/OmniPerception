"""Overlapping environments, heterogeneous geometry and complete USD transforms."""
import argparse
import json
import math
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch
from pxr import UsdGeom
from isaacsim.core.utils.prims import create_prim, get_prim_at_path
import isaaclab.sim as sim_utils
from LidarSensor.example.isaaclab.isaaclab.sensors import LidarSensor, LidarSensorCfg, LivoxPatternCfg

sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01, device=args.device))
sensors = {}

def make_sensor(case):
    pattern = LivoxPatternCfg(sensor_type='mid360', use_simple_grid=True, horizontal_line_num=1,
        vertical_line_num=1, horizontal_fov_deg_min=0, horizontal_fov_deg_max=0,
        vertical_fov_deg_min=0, vertical_fov_deg_max=0)
    return LidarSensor(LidarSensorCfg(prim_path=f'/World/{case}/env_.*/Sensor', mesh_prim_paths=[],
        dynamic_env_mesh_prim_paths=['{ENV_REGEX_NS}/Object'], max_distance=10.,
        mesh_exclude_paths=['{ENV_REGEX_NS}/Object/Emitter'] if case == 'emitter' else [],
        update_period=0.01, pattern_cfg=pattern))

create_prim('/World/InstanceTemplate', 'Xform')
template = create_prim('/World/InstanceTemplate/mesh', 'Cube')
UsdGeom.Cube(template).GetSizeAttr().Set(1.0)

for case in ('overlap', 'size', 'transform', 'capsule', 'cone', 'cylinder', 'instance', 'emitter'):
    for env in range(2):
        root = f'/World/{case}/env_{env}'
        create_prim(root, 'Xform')
        create_prim(root + '/Sensor', 'Xform', translation=(0, 1 if case == 'transform' else 0, 0))
        if case == 'emitter':
            create_prim(root + '/Object', 'Xform')
            prim = create_prim(root + '/Object/Emitter', 'Cube')
            UsdGeom.Cube(prim).GetSizeAttr().Set(.2)
            prim = create_prim(root + '/Object/Body', 'Cube', translation=(3, 0, 0))
            UsdGeom.Cube(prim).GetSizeAttr().Set(1.0)
        elif case == 'instance':
            prim = create_prim(root + '/Object', 'Xform', translation=(3, 0, 0))
            prim.GetReferences().AddInternalReference('/World/InstanceTemplate')
            prim.SetInstanceable(True)
        elif case == 'transform':
            create_prim(root + '/Object', 'Xform', translation=(3, 0, 0),
                        orientation=(math.sqrt(.5), 0, 0, math.sqrt(.5)))
            prim = create_prim(root + '/Object/mesh', 'Cube', translation=(1, 0, 0), scale=(2, 1, .5))
            UsdGeom.Cube(prim).GetSizeAttr().Set(1.0)
        else:
            kind = case.capitalize() if case in ('capsule', 'cone', 'cylinder') else 'Cube'
            x = 1 if case == 'overlap' and env == 1 else 3
            prim = create_prim(root + '/Object', kind, translation=(x, 0, 0))
            shape = getattr(UsdGeom, kind)(prim)
            if kind == 'Cube':
                shape.GetSizeAttr().Set(2.0 if case == 'size' and env == 1 else 1.0)
            else:
                shape.GetRadiusAttr().Set(.5)
                shape.GetHeightAttr().Set(2.)
                shape.GetAxisAttr().Set('X' if env else 'Z')
    sensors[case] = make_sensor(case)
sim.reset()
sim.step(render=False)
results = {}
expected = {'overlap': [2.5, .5], 'size': [2.5, 2.], 'transform': [2.5, 2.5],
            'capsule': [2.5, 1.5], 'cone': [2.75, 2.], 'cylinder': [2.5, 2.],
            'instance': [2.5, 2.5], 'emitter': [2.5, 2.5]}
for case, sensor in sensors.items():
    sensor.update(.01, force_recompute=True)
    actual = sensor.data.distances.flatten()
    assert torch.allclose(actual, torch.tensor(expected[case], device=args.device), atol=.025), (case, actual)
    results[case] = actual.tolist()

instance = get_prim_at_path('/World/instance/env_1/Object')
for op in UsdGeom.Xformable(instance).GetOrderedXformOps():
    if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
        op.Set((4., 0., 0.))
sensors['instance'].update(.01, force_recompute=True)
assert torch.allclose(sensors['instance'].data.distances.flatten(), torch.tensor([2.5, 3.5], device=args.device))
results['instance_motion'] = sensors['instance'].data.distances.flatten().tolist()

# Reset-time geometry randomization is explicit and rebuilds the changed shape.
UsdGeom.Cube(get_prim_at_path('/World/size/env_1/Object')).GetSizeAttr().Set(3.0)
sensors['size'].refresh_geometry()
sensors['size'].update(.01, force_recompute=True)
assert torch.allclose(sensors['size'].data.distances.flatten(), torch.tensor([2.5, 1.5], device=args.device))
results['refresh'] = sensors['size'].data.distances.flatten().tolist()

# Invalid configured geometry is an error, never an invented plane or missing obstacle.
sensor = sensors['size']
original_paths = sensor.cfg.dynamic_env_mesh_prim_paths
sensor.cfg.dynamic_env_mesh_prim_paths = ['{ENV_REGEX_NS}/DoesNotExist']
try:
    sensor.refresh_geometry()
except ValueError as error:
    assert 'matched no prims' in str(error)
else:
    raise AssertionError('Missing configured geometry was silently skipped')
sensor.cfg.dynamic_env_mesh_prim_paths = original_paths
sensor.refresh_geometry()
results['missing_path'] = 'rejected'
print('GEOMETRY_RESULT=' + json.dumps(results), flush=True)
app.close()
