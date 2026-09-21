# LiDAR Sensor Integration for IsaacLab

> Kitjesen fork: use [the isolated source integration](../../../../../docs/isaacsim5_rl.md).
> The legacy installer below is disabled because it overwrites installed Isaac Lab
> modules. `LidarSensor` now uses per-environment instance geometry; the retained
> standalone RayCaster/Camera examples do not use that backend.

This directory contains all the necessary files to integrate LiDAR sensors (including Livox sensors) into IsaacLab. The implementation supports both traditional grid-based LiDAR patterns and realistic Livox scan patterns using precomputed `.npy` files.

## 📁 Directory Structure

```
isaaclab/
├── sensors/
│   ├── lidar_sensor.py              # Main LiDAR sensor implementation
│   ├── lidar_sensor_cfg.py          # LiDAR sensor configuration
│   ├── lidar_sensor_data.py         # LiDAR sensor data container
│   └── ray_caster/
│       └── patterns/
│           ├── patterns.py          # Extended patterns with Livox support
│           └── patterns_cfg.py      # Pattern configurations
├── scripts/
│   └── examples/
│       └── simple_lidar_integration.py  # Example usage and benchmark script
├── scan_patterns/
│   ├── avia.npy                     # Livox Avia scan pattern
│   ├── HAP.npy                      # Livox HAP scan pattern
│   ├── horizon.npy                  # Livox Horizon scan pattern
│   ├── mid360.npy                   # Livox Mid-360 scan pattern
│   ├── mid40.npy                    # Livox Mid-40 scan pattern
│   ├── mid70.npy                    # Livox Mid-70 scan pattern
│   └── tele.npy                     # Livox Tele scan pattern
├── benchmark_lidar.sh               # Performance benchmark script
└── README.md                        # This file

# Scan patterns are now unified in: ../../sensor_pattern/sensor_lidar/scan_mode/
# Contains: avia.npy, HAP.npy, horizon.npy, mid360.npy, mid40.npy, mid70.npy, tele.npy
```

## 🚀 Quick Start

### Step 1: Installation Options

**Option A: Automated Installation (Recommended)**

# Follow the automated installation script
```bash
# Use the provided installation script
./install_lidar_sensor.sh /path/to/your/IsaacLab
```


**Option B: Manual Installation**
```bash
# Navigate to your IsaacLab directory
cd /path/to/your/IsaacLab

# Copy sensor implementations
cp /path/to/this/directory/sensors/lidar_sensor*.py source/isaaclab/isaaclab/sensors/

# Copy pattern implementations
cp /path/to/this/directory/sensors/ray_caster/patterns/patterns*.py source/isaaclab/isaaclab/sensors/ray_caster/patterns/

# Copy scan pattern files from unified location (for local fallback)
mkdir -p source/isaaclab/isaaclab/sensors/ray_caster/patterns/scan_patterns
cp /path/to/this/directory/../../../sensor_pattern/sensor_lidar/scan_mode/*.npy source/isaaclab/isaaclab/sensors/ray_caster/patterns/scan_patterns/

# Copy example script
cp /path/to/this/directory/scripts/examples/simple_lidar_integration.py scripts/examples/
```

### Step 2: Update IsaacLab Imports

Add the following imports to the appropriate `__init__.py` files:

#### In `source/isaaclab/isaaclab/sensors/__init__.py`:
```python
from .lidar_sensor import LidarSensor
from .lidar_sensor_cfg import LidarSensorCfg
from .lidar_sensor_data import LidarSensorData
```

#### In `source/isaaclab/isaaclab/sensors/ray_caster/patterns/__init__.py`:
```python
from .patterns_cfg import LivoxPatternCfg
from .patterns import livox_pattern
```

### Step 3: Run the Example

```bash
# Basic example with 2 environments
./isaaclab.sh -p scripts/demos/simple_lidar_integration.py --enable_lidar

# Performance benchmark
./isaaclab.sh -p scripts/demos/simple_lidar_integration.py --num_envs 1024 --enable_lidar --benchmark_steps 1000 --headless
```

## 🔧 Usage Guide

### Basic LiDAR Sensor Configuration

```python
from isaaclab.sensors import LidarSensorCfg
from isaaclab.sensors.ray_caster.patterns import LivoxPatternCfg

# Simple grid-based LiDAR
lidar_sensor = LidarSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base",
    pattern_cfg=LivoxPatternCfg(
        sensor_type="mid360",
        use_simple_grid=True,
        vertical_line_num=32,
        horizontal_line_num=64,
    ),
    max_distance=20.0,
    min_range=0.2,
    return_pointcloud=True,
    mesh_prim_paths=["/World/ground"],
)

# Realistic Livox sensor with .npy patterns
livox_sensor = LidarSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base",
    pattern_cfg=LivoxPatternCfg(
        sensor_type="mid360",
        use_simple_grid=False,  # Use realistic patterns
        samples=8000,
        downsample=1,
    ),
    max_distance=50.0,
    min_range=0.1,
    return_pointcloud=True,
    enable_sensor_noise=True,
    random_distance_noise=0.02,
    mesh_prim_paths=["/World/ground"],
)
```

### Supported Livox Sensor Types

| Sensor Type | Horizontal FOV | Vertical FOV | Max Samples | Pattern File |
|-------------|----------------|--------------|-------------|--------------|
| `avia`      | 70.4°          | 77.2°        | 24,000      | avia.npy     |
| `horizon`   | 81.7°          | 25.1°        | 24,000      | horizon.npy  |
| `HAP`       | 81.7°          | 25.1°        | 45,300      | HAP.npy      |
| `mid360`    | 360°           | 59°          | 20,000      | mid360.npy   |
| `mid40`     | 81.7°          | 25.1°        | 24,000      | mid40.npy    |
| `mid70`     | 70.4°          | 70.4°        | 10,000      | mid70.npy    |
| `tele`      | 14.5°          | 16.1°        | 24,000      | tele.npy     |

### Accessing LiDAR Data

```python
# In your environment step function
def _update_buffers_impl(self, env_ids):
    # Get distance measurements
    distances = self.scene["lidar_sensor"].get_distances(env_ids)
    
    # Get point cloud (if enabled)
    if self.scene["lidar_sensor"].cfg.return_pointcloud:
        pointcloud = self.scene["lidar_sensor"].get_pointcloud(env_ids)
    
    # Access raw sensor data
    sensor_data = self.scene["lidar_sensor"].data
    hit_points = sensor_data.ray_hits_w[env_ids]  # World coordinates
    positions = sensor_data.pos_w[env_ids]        # Sensor positions
    orientations = sensor_data.quat_w[env_ids]    # Sensor orientations
```

## 🏗️ Implementation Details

### Key Features

1. **Multiple Pattern Support**:
   - Simple grid patterns for basic LiDAR simulation
   - Realistic Livox patterns loaded from `.npy` files
   - Dynamic pattern updates for time-varying scan patterns

2. **Performance Optimizations**:
   - Vectorized ray direction updates (no Python loops)
   - Configurable sample counts for performance tuning
   - Optional pointcloud generation
   - Noise and visualization can be disabled for benchmarking

3. **Coordinate System Compatibility**:
   - Proper conversion between LidarSensor coordinate system (x=forward, y=left, z=up)
   - and IsaacLab's ray-casting coordinate system

4. **Realistic Sensor Behavior**:
   - Sensor noise simulation
   - Pixel dropout modeling
   - Range limiting and validation

### Architecture

```
LidarSensor (extends RayCaster)
├── LidarSensorData (data container)
├── LidarSensorCfg (configuration)
└── Pattern Generation
    ├── livox_pattern() (main pattern function)
    ├── _livox_simple_grid_pattern() (grid fallback)
    └── _livox_scan_pattern() (realistic patterns from .npy)
```

## 📊 Performance Benchmarking

Use the provided benchmark script to test performance:

```bash
# Run automated benchmark comparing with/without LiDAR
./benchmark_lidar.sh

# Manual benchmarking
./isaaclab.sh -p scripts/examples/simple_lidar_integration.py --num_envs 1024 --benchmark_steps 1000 --headless
./isaaclab.sh -p scripts/examples/simple_lidar_integration.py --num_envs 1024 --benchmark_steps 1000 --enable_lidar --headless
```

Expected performance impact with 1024 environments and Livox Mid-360 (8000 rays):
- Total rays: ~8.2 million per step
- Performance overhead: 20-40% depending on hardware

## 🔧 Configuration Options

### LidarSensorCfg Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_distance` | float | 100.0 | Maximum detection range (m) |
| `min_range` | float | 0.0 | Minimum detection range (m) |
| `return_pointcloud` | bool | False | Generate 3D point cloud |
| `pointcloud_in_world_frame` | bool | True | Point cloud coordinates |
| `enable_sensor_noise` | bool | False | Add realistic noise |
| `random_distance_noise` | float | 0.0 | Gaussian noise std dev |
| `pixel_dropout_prob` | float | 0.0 | Probability of pixel dropout |
| `update_frequency` | float | 50.0 | Sensor update rate (Hz) |

### LivoxPatternCfg Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sensor_type` | str | "mid360" | Livox sensor model |
| `use_simple_grid` | bool | False | Use grid instead of .npy |
| `samples` | int | 8000 | Number of rays per scan |
| `downsample` | int | 1 | Downsampling factor |
| `rolling_window_start` | int | 0 | Starting index in pattern |

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all files are copied to correct locations and `__init__.py` files are updated
2. **Pattern File Not Found**: Check scan patterns are available in either location:
   - Primary: `sensor_pattern/sensor_lidar/scan_mode/` (unified location)
   - Fallback: `source/isaaclab/isaaclab/sensors/ray_caster/patterns/scan_patterns/` (local copy)
3. **Performance Issues**: Reduce `samples` count or increase `downsample` factor
4. **Memory Issues**: Disable `return_pointcloud` for large numbers of environments

### Debug Tips

```python
# Enable debug visualization
lidar_sensor = LidarSensorCfg(
    debug_vis=True,  # Shows ray directions in viewer
    # ... other config
)

# Check sensor initialization
print(f"LiDAR rays: {lidar_sensor.num_rays}")
print(f"Pattern type: {type(lidar_sensor.cfg.pattern_cfg)}")
```

## 📝 File Descriptions

### Core Implementation Files

- **`lidar_sensor.py`**: Main LiDAR sensor class extending RayCaster with LiDAR-specific functionality
- **`lidar_sensor_cfg.py`**: Configuration dataclass for LiDAR sensors
- **`lidar_sensor_data.py`**: Data container for sensor measurements and state
- **`patterns.py`**: Extended pattern generation including Livox support
- **`patterns_cfg.py`**: Configuration for all pattern types including LivoxPatternCfg

### Example and Test Files

- **`simple_lidar_integration.py`**: Complete example showing LiDAR integration in a quadruped environment
- **`benchmark_lidar.sh`**: Automated performance comparison script

### Data Files

- **`*.npy`**: Precomputed Livox scan patterns for realistic sensor simulation

## 🎯 Next Steps

1. **Integration**: Copy files to your IsaacLab installation
2. **Testing**: Run the example script to verify functionality
3. **Customization**: Modify configurations for your specific use case
4. **Optimization**: Use benchmarking to tune performance parameters

For questions or issues, refer to the IsaacLab documentation or the original LidarSensor implementation.
