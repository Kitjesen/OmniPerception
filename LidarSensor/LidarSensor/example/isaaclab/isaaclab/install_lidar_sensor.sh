#!/usr/bin/env bash
# The source-checkout integration keeps the installed Isaac Lab untouched.
set -eu
cat >&2 <<'MESSAGE'
The legacy installer is disabled in this fork: it overwrites Isaac Lab modules
and cannot install the isolated LiDAR backend correctly.

From the OmniPerception repository root, use:
  export PYTHONPATH="$PWD/LidarSensor:${PYTHONPATH:-}"
  python tests/isaaclab_geometry_smoke.py --headless --device cuda:0

Import LidarSensor from LidarSensor.example.isaaclab.isaaclab.sensors.
See docs/isaacsim5_rl.md. No files in the target Isaac Lab directory were changed.
MESSAGE
exit 2
