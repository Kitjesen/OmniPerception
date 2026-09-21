# Thunder V4 self-occlusion integration

USD instance proxies are now traversed when collecting LiDAR meshes. A proxy
without a physical ancestor is anchored to its editable instance root because
Isaac Sim views cannot bind directly to the proxy path. Physical robot link
ancestors continue to use their live physics poses.

`LidarSensorCfg.mesh_exclude_paths` names exact subtrees to exclude and expands
`{ENV_REGEX_NS}` per environment. Use it for the emitting LiDAR's own housing,
not to erase robot self-occlusion. A nonexistent exclusion is reported rather
than silently using a different geometry configuration.

For the Thunder URDF used by robot_lab, optical self-occlusion queries select
`{ENV_REGEX_NS}/Robot/.*/visuals` and exclude
`{ENV_REGEX_NS}/Robot/base_link/visuals/lidar1_Link`. The physics collision boxes
are deliberately coarse and can enclose the ray origin; they remain enabled in
the physics engine but are not the optical surface model. Camera, body, legs,
and the rear LiDAR visual meshes remain visible to the ray queries.

Validation on Isaac Sim 5.0 / GPU 7 (RTX 3090), 2026-09-22:

- Existing overlapping-environment, geometry-size, transform, primitive,
  geometry-refresh and missing-path assertions passed.
- New instanced-cube rays returned 2.5 m in both environments; moving one
  instance changed only its return to 3.5 m.
- Excluding the emitter housing preserved the 2.5 m robot-body obstacle.
- AME-2 Thunder integration ran two PPO iterations on each of plane and rough
  terrain, with actual environment points entering the actor's map.

The standalone geometry script printed all passing results but subsequently
stalled in application shutdown; the external timeout bounds cleanup. This is
not a claim of normal process exit. The Thunder training runs completed and
saved their integration results separately.
