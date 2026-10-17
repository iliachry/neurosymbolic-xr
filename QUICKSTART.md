# Quick Start Guide

This guide will help you get started with the Neurosymbolic XR Framework.

## Installation

```bash
# Clone the repository
git clone https://github.com/iliachry/neurosymbolic-xr.git
cd neurosymbolic-xr

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

## Quick Examples

### 1. Spatial Reasoning Engine

```python
from src.systems import create_spatial_reasoning_engine
import numpy as np

# Initialize engine
engine = create_spatial_reasoning_engine({
    'gaussian': {'max_gaussians': 100000, 'device': 'cuda'},
    'segmentation': {'model_size': 'large', 'device': 'cuda'},
    'grounding': {'model_name': 'ViT-L/14', 'device': 'cuda'},
    'device': 'cuda'
})

# Capture and understand scene
images = [...]  # List of RGB images
camera_poses = [...]  # List of 4x4 transforms
camera_intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

symbolic_scene = engine.capture_and_understand_scene(
    images=images,
    camera_poses=camera_poses,
    camera_intrinsics=camera_intrinsics,
    num_iterations=10000
)

# Query scene
results = engine.query_scene("Find all chairs near the table")
print(f"Found {len(results)} objects")

# Find objects by properties
red_chairs = engine.find_objects(
    category="chair",
    properties={'color': 'red'}
)
```

### 2. Hybrid SLAM

```python
from src.systems import create_hybrid_slam

# Initialize SLAM
slam = create_hybrid_slam({
    'vocab_path': 'ORBvoc.txt',
    'config_path': 'config.yaml',
    'use_imu': True,
    'device': 'cuda'
})

# Track frame
image = ...  # RGB image
timestamp = 0.033  # seconds
imu_data = [...]  # List of (timestamp, accel, gyro)

pose = slam.track_frame(image, timestamp, imu_data)
if pose:
    print(f"Position: {pose.position}")
    print(f"Orientation: {pose.orientation}")

# Query semantic memory
nearby_objects = slam.query_semantic_memory(
    position=pose.position,
    radius=2.0
)

# Save map
slam.save_map("maps/my_venue")
```

### 3. Training Assistant

```python
from src.systems import (
    create_training_assistant,
    create_example_procedure,
    HandPose,
    HandGesture
)
import numpy as np

# Initialize assistant
assistant = create_training_assistant({
    'device': 'cuda',
    'use_prolog': True
})

# Start procedure
procedure = create_example_procedure()
assistant.start_procedure(procedure)

# Update with tracking data
left_hand = HandPose(
    timestamp=0.033,
    hand_type='left',
    joints=np.random.randn(21, 3),
    gesture=HandGesture.GRAB,
    confidence=0.95
)

result = assistant.update(
    left_hand=left_hand,
    world_state={'base_component_position': [0.5, 1.0, -0.3]}
)

print(f"Status: {result['status']}")
print(f"Feedback: {result['feedback']}")
```

### 4. Asset Generator

```python
from src.systems import (
    create_asset_generator,
    AssetSpecification,
    AssetType,
    GeometricConstraint,
    ConstraintType
)

# Initialize generator
generator = create_asset_generator({
    'model_name': 'stabilityai/stable-diffusion-2-1',
    'device': 'cuda',
    'use_constraints': True
})

# Create specification
spec = AssetSpecification(
    asset_type=AssetType.OBJECT_3D,
    description="modern office chair with wheels",
    constraints=[
        GeometricConstraint(
            type=ConstraintType.DIMENSION,
            parameters={'width': 0.6, 'height': 1.0, 'depth': 0.6},
            priority=1.0
        )
    ],
    style="professional, ergonomic"
)

# Generate asset
asset = generator.generate_asset(spec)
print(f"Generated: {asset.asset_id}")
print(f"Vertices: {len(asset.mesh)}")
print(f"Texture: {asset.texture.shape}")

# Generate variations
variations = generator.generate_variations(spec, num_variations=4)
```

### 5. Poetics Integration

```python
from src.integration import create_poetics_integration

# Initialize integration
integration = create_poetics_integration({
    'supabase_url': 'https://your-project.supabase.co',
    'supabase_key': 'your-api-key',
    'neural_config': {
        'gaussian': {'max_gaussians': 100000},
        'segmentation': {'model_size': 'large'},
        'grounding': {'model_name': 'ViT-L/14'},
        'slam': {'use_imu': True},
        'training': {'use_prolog': True},
        'asset_gen': {'model_name': 'stabilityai/stable-diffusion-2-1'}
    },
    'device': 'cuda'
})

# Start session
session = await integration.start_session(
    venue_id='venue_001',
    user_id='user_123',
    mode='exploration'
)

# Track pose
pose = await integration.track_user_pose(
    session_id=session.session_id,
    image=image,
    timestamp=timestamp
)

# Query scene
results = await integration.query_venue_scene(
    session_id=session.session_id,
    query="Where are the exits?"
)

# End session
summary = await integration.end_session(session.session_id)
```

## Running the Demo

```bash
# Run comprehensive demonstration
python demo.py
```

## Configuration

Create a `config.yaml` file:

```yaml
device: cuda

gaussian:
  max_gaussians: 500000
  sh_degree: 3

segmentation:
  model_size: large
  use_video_mode: false

grounding:
  model_name: ViT-L/14

slam:
  vocab_path: ORBvoc.txt
  config_path: slam_config.yaml
  use_imu: true

training:
  use_prolog: true

asset_gen:
  model_name: stabilityai/stable-diffusion-2-1
  use_constraints: true

poetics:
  supabase_url: ${SUPABASE_URL}
  supabase_key: ${SUPABASE_KEY}
```

## Performance Tips

1. **GPU Memory**: Reduce `max_gaussians` if running out of memory
2. **Real-time Performance**: Use `model_size: 'small'` for segmentation
3. **SLAM Accuracy**: Enable IMU for better tracking
4. **Asset Generation**: Lower `num_inference_steps` for faster generation

## Troubleshooting

### CUDA Out of Memory
```python
# Reduce Gaussian count
gaussian_config = {'max_gaussians': 100000}

# Use smaller models
segmentation_config = {'model_size': 'small'}
```

### Slow Performance
```python
# Disable heavy systems
session = await integration.start_session(
    venue_id='venue_001',
    user_id='user_123',
    mode='training'  # Only activates necessary systems
)
```

### Import Errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

## Next Steps

- Read [`ARCHITECTURE.md`](ARCHITECTURE.md) for technical details
- Explore [`demo.py`](demo.py) for complete examples
- Check API documentation in source code docstrings

## Support

- GitHub Repository: https://github.com/iliachry/neurosymbolic-xr
- GitHub Issues: https://github.com/iliachry/neurosymbolic-xr/issues
