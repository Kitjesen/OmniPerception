"""Instance-local meshes and environment-filtered GPU ray queries for LiDAR."""
import re

import numpy as np
import torch
import trimesh
import warp as wp
from pxr import Gf, Usd, UsdGeom, UsdPhysics
from isaacsim.core.prims import XFormPrim
from isaacsim.core.utils.prims import get_prim_at_path
import isaaclab.sim as sim_utils
from isaaclab.utils.warp import convert_to_warp_mesh


@wp.kernel
def cast_instances(starts: wp.array2d(dtype=wp.vec3), directions: wp.array2d(dtype=wp.vec3),
                   env_ids: wp.array(dtype=wp.int64), table: wp.array2d(dtype=wp.int32),
                   counts: wp.array(dtype=wp.int32), meshes: wp.array(dtype=wp.uint64),
                   positions: wp.array(dtype=wp.vec3), rotations: wp.array(dtype=wp.quat),
                   max_distance: float, distances: wp.array2d(dtype=float)):
    batch, ray = wp.tid()
    env = env_ids[batch]
    closest = max_distance
    found = bool(False)
    for slot in range(counts[env]):
        instance = table[env, slot]
        q = rotations[instance]
        start = wp.quat_rotate_inv(q, starts[batch, ray] - positions[instance])
        direction = wp.quat_rotate_inv(q, directions[batch, ray])
        hit = wp.mesh_query_ray(meshes[instance], start, direction, closest)
        if hit.result:
            closest = hit.t
            found = True
    if found:
        distances[batch, ray] = closest
    else:
        distances[batch, ray] = wp.inf


def environment_root(path):
    match = re.search(r"^(.*?/env_\d+)(?:/|$)", path)
    return match.group(1) if match else None


def raw_mesh(prim):
    """Extract untransformed geometry. Apply the complete USD transform once later."""
    kind = prim.GetTypeName()
    if kind == "Mesh":
        mesh = UsdGeom.Mesh(prim)
        vertices = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
        counts = mesh.GetFaceVertexCountsAttr().Get()
        indices = mesh.GetFaceVertexIndicesAttr().Get()
        faces, offset = [], 0
        for count in counts:
            if count not in (3, 4):
                raise ValueError(f"Triangulate polygons with {count} vertices before loading {prim.GetPath()}")
            face = indices[offset:offset + count]
            faces.append([face[0], face[1], face[2]])
            if count == 4:
                faces.append([face[0], face[2], face[3]])
            offset += count
        return vertices, np.asarray(faces, dtype=np.int32).reshape(-1)
    if kind == "Cube":
        size = UsdGeom.Cube(prim).GetSizeAttr().Get()
        mesh = trimesh.creation.box(extents=[size] * 3)
    elif kind == "Sphere":
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=UsdGeom.Sphere(prim).GetRadiusAttr().Get())
    elif kind in ("Cylinder", "Capsule", "Cone"):
        shape = getattr(UsdGeom, kind)(prim)
        radius, height = shape.GetRadiusAttr().Get(), shape.GetHeightAttr().Get()
        mesh = getattr(trimesh.creation, kind.lower())(radius=radius, height=height)
        mesh.vertices -= (mesh.bounds[0] + mesh.bounds[1]) * 0.5
        axis = shape.GetAxisAttr().Get()
        if axis == "X":
            mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
        elif axis == "Y":
            mesh.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
    elif kind == "Plane":
        # Match the existing raycaster's effectively infinite ground plane.
        vertices = np.array([[-1e6, -1e6, 0], [1e6, -1e6, 0], [1e6, 1e6, 0], [-1e6, 1e6, 0]])
        axis = prim.GetAttribute("axis").Get() or "Z"
        if axis == "X":
            vertices = vertices[:, [2, 1, 0]]
        elif axis == "Y":
            vertices = vertices[:, [0, 2, 1]]
        return vertices, np.array([0, 1, 2, 0, 2, 3], dtype=np.int32)
    else:
        raise ValueError(f"Unsupported LiDAR geometry {kind}: {prim.GetPath()}")
    return np.asarray(mesh.vertices), np.asarray(mesh.faces, dtype=np.int32).reshape(-1)


class IsolatedGeometry:
    def __init__(self, sensor):
        self.sensor = sensor
        self.device = sensor.device
        cache = UsdGeom.XformCache()
        sensor_paths = list(sensor._view.prim_paths)
        owners = [environment_root(path) for path in sensor_paths]
        roots = list(dict.fromkeys(owner for owner in owners if owner is not None))
        excluded = []
        for path in sensor.cfg.mesh_exclude_paths:
            expanded = [path.replace("{ENV_REGEX_NS}", root) for root in roots] if "{ENV_REGEX_NS}" in path else [path]
            for item in expanded:
                if not get_prim_at_path(item).IsValid():
                    raise ValueError(f"Excluded LiDAR geometry does not exist: {item}")
                excluded.append(item.rstrip("/"))
        paths = list(sensor.cfg.mesh_prim_paths)
        for pattern in sensor.cfg.dynamic_env_mesh_prim_paths:
            if "{ENV_REGEX_NS}" in pattern:
                paths.extend(pattern.replace("{ENV_REGEX_NS}", root) for root in roots)
            else:
                paths.append(pattern)
        geometries = {}
        for pattern in paths:
            matches = sim_utils.find_matching_prims(pattern)
            if not matches:
                raise ValueError(f"Configured LiDAR path matched no prims: {pattern}")
            for root in matches:
                found = False
                for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
                    if prim.IsA(UsdGeom.Gprim):
                        path = str(prim.GetPath())
                        if any(path == item or path.startswith(item + "/") for item in excluded):
                            continue
                        geometries[str(prim.GetPath())] = prim
                        found = True
                if not found:
                    raise ValueError(f"No geometry under configured LiDAR path: {root.GetPath()}")
        if not geometries:
            raise ValueError("LiDAR requires explicitly configured geometry")
        self.mesh_paths = tuple(geometries)
        self.meshes, anchors, mesh_owners = [], [], []
        for path, prim in geometries.items():
            vertices, indices = raw_mesh(prim)
            if vertices.size == 0 or indices.size == 0:
                raise ValueError(f"Empty LiDAR mesh: {path}")
            anchor = prim
            while anchor.IsValid() and not anchor.HasAPI(UsdPhysics.RigidBodyAPI):
                anchor = anchor.GetParent()
            if not anchor.IsValid():
                anchor = prim
                # Isaac views cannot target an instance proxy. Bake the child
                # transform relative to its editable instance root instead.
                while anchor.IsInstanceProxy():
                    anchor = anchor.GetParent()
            # Gf matrices use row vectors; extract a rigid pose and bake scale/shear
            # and the full descendant transform into local vertices exactly once.
            world = cache.GetLocalToWorldTransform(anchor)
            transform = Gf.Transform(world)
            rigid = Gf.Matrix4d().SetRotate(transform.GetRotation())
            rigid.SetTranslateOnly(transform.GetTranslation())
            relative = cache.GetLocalToWorldTransform(prim) * rigid.GetInverse()
            homogeneous = np.column_stack((vertices, np.ones(len(vertices))))
            local_vertices = (homogeneous @ np.asarray(relative))[:, :3]
            self.meshes.append(convert_to_warp_mesh(local_vertices, indices, device=self.device))
            anchors.append(str(anchor.GetPath()))
            mesh_owners.append(environment_root(path))
        anchor_paths = list(dict.fromkeys(anchors))
        self.view = XFormPrim(anchor_paths, reset_xform_properties=False)
        order = {path: i for i, path in enumerate(self.view.prim_paths)}
        self.anchor_indices = torch.tensor([order[path] for path in anchors], device=self.device)
        rows = [[i for i, owner in enumerate(mesh_owners) if owner is None or owner == own] for own in owners]
        width = max(map(len, rows))
        if not width:
            raise ValueError("No geometry belongs to the sensor environments")
        table = np.zeros((len(rows), width), dtype=np.int32)
        for i, row in enumerate(rows):
            table[i, :len(row)] = row
        self.table = wp.array(table, dtype=wp.int32, device=self.device)
        self.counts = wp.array([len(row) for row in rows], dtype=wp.int32, device=self.device)
        self.mesh_ids = wp.array([mesh.id for mesh in self.meshes], dtype=wp.uint64, device=self.device)

    def cast(self, starts, directions, env_ids):
        positions, rotations = self.sensor._get_live_poses(self.view, None)
        positions = positions[self.anchor_indices].contiguous()
        rotations = rotations[self.anchor_indices][:, [1, 2, 3, 0]].contiguous()
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).contiguous()
        distances = torch.empty(starts.shape[:2], device=self.device)
        wp.launch(cast_instances, dim=starts.shape[:2], inputs=[
            wp.from_torch(starts.contiguous(), dtype=wp.vec3), wp.from_torch(directions.contiguous(), dtype=wp.vec3),
            wp.from_torch(ids), self.table, self.counts, self.mesh_ids,
            wp.from_torch(positions, dtype=wp.vec3), wp.from_torch(rotations, dtype=wp.quat),
            self.sensor.cfg.max_distance, wp.from_torch(distances)], device=self.device)
        return distances
