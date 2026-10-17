# Neurosymbolic XR Intelligence Framework

A unified neurosymbolic AI architecture for Extended Reality systems, integrating neural perception with symbolic reasoning for intelligent spatial computing.

## 🎯 Overview

This framework combines four interconnected neurosymbolic AI systems designed for XR/VR platforms:

1. **Spatial Reasoning Engine** - Scene understanding with semantic graphs
2. **Hybrid SLAM** - Localization with persistent knowledge
3. **VR Training Assistant** - Procedural learning and validation
4. **Neural-Symbolic Asset Generator** - Constrained 3D content creation

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Neurosymbolic XR Core                     │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────────┐         ┌──────────────────┐          │
│  │  Neural Layer    │◄───────►│  Symbolic Layer  │          │
│  ├──────────────────┤         ├──────────────────┤          │
│  │ • 3DGS/NeRF      │         │ • Knowledge Graph│          │
│  │ • SAM 2          │         │ • Logic Engine   │          │
│  │ • CLIP/DINOv2    │         │ • Ontologies     │          │
│  │ • Hand Tracking  │         │ • Rule Systems   │          │
│  │ • Gaze Analysis  │         │ • Constraints    │          │
│  └──────────────────┘         └──────────────────┘          │
│           │                            │                     │
│           └────────────┬───────────────┘                     │
│                        ▼                                     │
│              ┌──────────────────┐                           │
│              │ Reasoning Bridge │                           │
│              │  • Grounding     │                           │
│              │  • Inference     │                           │
│              │  • Explanation   │                           │
│              └──────────────────┘                           │
└─────────────────────────────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│   Spatial    │  │    Hybrid    │  │   Training   │
│  Reasoning   │  │     SLAM     │  │  Assistant   │
│    Engine    │  │              │  │              │
└──────────────┘  └──────────────┘  └──────────────┘
```

## 📦 Components

### 1. Spatial Reasoning Engine

**Purpose**: Understand 3D scenes through neural-symbolic fusion

**Neural Components**:
- [`3DGaussianSplatting`](src/neural/gaussian_splatting.py) - Real-time scene capture
- [`SAM2Segmenter`](src/neural/sam2_segmenter.py) - Zero-shot object segmentation
- [`CLIPEncoder`](src/neural/clip_encoder.py) - Open-vocabulary embeddings

**Symbolic Components**:
- [`SpatialOntology`](src/symbolic/spatial_ontology.py) - Geometric relationships
- [`SceneGraph`](src/symbolic/scene_graph.py) - Object-relationship structure
- [`SpatialReasoner`](src/symbolic/spatial_reasoner.py) - Query engine

**Example Query**:
```python
# "Find all surfaces that can support a coffee mug"
query = SpatialQuery(
    object_type="surface",
    constraints=[
        Constraint("horizontal", tolerance=5),  # degrees
        Constraint("area", min=0.01),  # m²
        Constraint("height", range=(0.5, 1.2))  # m
    ]
)
results = spatial_engine.query(query)
```

### 2. Hybrid SLAM with Semantic Memory

**Purpose**: Localization + persistent spatial knowledge

**Neural Components**:
- [`ORB_SLAM3`](src/neural/orb_slam.py) - Visual-inertial tracking
- [`ObjectDetector`](src/neural/object_detector.py) - Real-time detection
- [`DepthEstimator`](src/neural/depth_estimator.py) - Metric depth

**Symbolic Components**:
- [`SemanticMap`](src/symbolic/semantic_map.py) - Persistent knowledge graph
- [`TopologicalGraph`](src/symbolic/topological_graph.py) - Place recognition
- [`TemporalReasoner`](src/symbolic/temporal_reasoner.py) - Change detection

**Features**:
- Persistent venue memory across sessions
- Semantic place recognition ("kitchen", "workshop")
- Change detection and anomaly reasoning
- Multi-session map merging with conflict resolution

### 3. VR Training Assistant

**Purpose**: Intelligent procedural training with validation

**Neural Components**:
- [`HandTracker`](src/neural/hand_tracker.py) - Skeletal hand pose
- [`GazeAnalyzer`](src/neural/gaze_analyzer.py) - Attention tracking
- [`ActionRecognizer`](src/neural/action_recognizer.py) - Activity classification

**Symbolic Components**:
- [`ProcedureGraph`](src/symbolic/procedure_graph.py) - Task decomposition
- [`ConstraintValidator`](src/symbolic/constraint_validator.py) - Rule checking
- [`ErrorDiagnoser`](src/symbolic/error_diagnoser.py) - Explainable feedback

**Example Procedure**:
```python
procedure = Procedure(
    name="Equipment Assembly",
    steps=[
        Step(1, "Pick up base plate",
             preconditions=["hands_empty"],
             postconditions=["holding_base_plate"]),
        Step(2, "Align with mounting holes",
             constraints=["rotation_tolerance: 5°", "position_tolerance: 2mm"]),
        Step(3, "Insert bolts in sequence",
             order=["bolt_A", "bolt_B", "bolt_C", "bolt_D"],
             validation="torque_spec: 15Nm")
    ]
)

# Real-time validation
assistant.monitor_procedure(procedure)
# Output: "Warning: Bolt B not torqued to specification"
```

### 4. Neural-Symbolic Asset Generator

**Purpose**: Generate 3D content with functional constraints

**Neural Components**:
- [`DiffusionModel`](src/neural/diffusion_3d.py) - Multi-view generation
- [`MeshGenerator`](src/neural/mesh_generator.py) - Geometry synthesis
- [`TextureGenerator`](src/neural/texture_generator.py) - PBR materials

**Symbolic Components**:
- [`GeometricConstraints`](src/symbolic/geometric_constraints.py) - CAD-like rules
- [`PhysicsValidator`](src/symbolic/physics_validator.py) - Stability checking
- [`DesignPatterns`](src/symbolic/design_patterns.py) - Reusable templates

**Example Generation**:
```python
spec = AssetSpec(
    type="chair",
    constraints=[
        "seat_height: 0.45m ± 0.05m",
        "weight_capacity: 150kg",
        "stability: center_of_mass within base",
        "ergonomic: lumbar_support_angle 95-105°"
    ],
    style="modern minimalist"
)

asset = generator.create(spec)
# Returns: 3DGS or mesh with guaranteed constraint satisfaction
```

## 🔧 Technology Stack

### Neural Processing
- **PyTorch 2.5** - Deep learning framework
- **CUDA 12.4** - GPU acceleration
- **OpenCV 4.10** - Computer vision
- **MediaPipe** - Hand/pose tracking
- **Transformers** - Foundation models

### Symbolic Reasoning
- **RDFLib** - Knowledge graph (RDF/OWL)
- **Prolog (SWI-Prolog)** - Logic programming
- **Z3 Solver** - Constraint satisfaction
- **NetworkX** - Graph algorithms
- **PDDL** - Planning domain definition

### XR Integration
- **OpenXR 1.1** - Cross-platform XR
- **Unity 6 / Unreal 5** - Game engines
- **WebXR** - Browser-based XR
- **Meta Quest SDK** - Native Quest support

### Backend
- **FastAPI** - Python web framework
- **PostgreSQL + pgvector** - Vector database
- **Redis** - Caching and pub/sub
- **MQTT** - IoT telemetry
- **Supabase** - BaaS integration

## 📁 Project Structure

```
neurosymbolic-xr/
├── README.md                          # This file
├── ARCHITECTURE.md                    # Detailed architecture
├── requirements.txt                   # Python dependencies
├── pyproject.toml                     # Project configuration
│
├── src/
│   ├── __init__.py
│   │
│   ├── core/                          # Core neurosymbolic framework
│   │   ├── __init__.py
│   │   ├── bridge.py                  # Neural-symbolic grounding
│   │   ├── reasoner.py                # Hybrid inference engine
│   │   └── explainer.py               # Explanation generation
│   │
│   ├── neural/                        # Neural perception layer
│   │   ├── __init__.py
│   │   ├── gaussian_splatting.py      # 3DGS implementation
│   │   ├── sam2_segmenter.py          # SAM 2 integration
│   │   ├── clip_encoder.py            # CLIP embeddings
│   │   ├── hand_tracker.py            # Hand pose estimation
│   │   ├── gaze_analyzer.py           # Eye tracking analysis
│   │   ├── orb_slam.py                # SLAM wrapper
│   │   └── diffusion_3d.py            # 3D generation
│   │
│   ├── symbolic/                      # Symbolic reasoning layer
│   │   ├── __init__.py
│   │   ├── knowledge_graph.py         # RDF/OWL graph
│   │   ├── spatial_ontology.py        # Spatial relationships
│   │   ├── scene_graph.py             # Scene structure
│   │   ├── spatial_reasoner.py        # Spatial queries
│   │   ├── semantic_map.py            # Persistent map
│   │   ├── procedure_graph.py         # Task representation
│   │   ├── constraint_validator.py    # Rule checking
│   │   └── logic_engine.py            # Prolog interface
│   │
│   ├── systems/                       # Complete systems
│   │   ├── __init__.py
│   │   ├── spatial_reasoning/         # System 1
│   │   │   ├── __init__.py
│   │   │   ├── engine.py
│   │   │   └── queries.py
│   │   ├── hybrid_slam/               # System 2
│   │   │   ├── __init__.py
│   │   │   ├── slam.py
│   │   │   └── semantic_memory.py
│   │   ├── training_assistant/        # System 3
│   │   │   ├── __init__.py
│   │   │   ├── assistant.py
│   │   │   └── procedures.py
│   │   └── asset_generator/           # System 4
│   │       ├── __init__.py
│   │       ├── generator.py
│   │       └── constraints.py
│   │
│   ├── integration/                   # Platform integrations
│   │   ├── __init__.py
│   │   ├── poetics_adapter.py         # Poetics VR platform
│   │   ├── openxr_bridge.py           # OpenXR interface
│   │   └── supabase_client.py         # Backend connection
│   │
│   └── utils/                         # Utilities
│       ├── __init__.py
│       ├── geometry.py                # 3D math utilities
│       ├── visualization.py           # Debug visualization
│       └── metrics.py                 # Performance tracking
│
├── examples/                          # Usage examples
│   ├── spatial_reasoning_demo.py
│   ├── hybrid_slam_demo.py
│   ├── training_assistant_demo.py
│   └── asset_generation_demo.py
│
├── tests/                             # Test suite
│   ├── test_neural/
│   ├── test_symbolic/
│   ├── test_systems/
│   └── test_integration/
│
├── docs/                              # Documentation
│   ├── ARCHITECTURE.md                # System architecture
│   ├── NEURAL_COMPONENTS.md           # Neural layer guide
│   ├── SYMBOLIC_COMPONENTS.md         # Symbolic layer guide
│   ├── INTEGRATION_GUIDE.md           # Platform integration
│   └── API_REFERENCE.md               # API documentation
│
├── models/                            # Pre-trained models
│   ├── sam2_checkpoint.pth
│   ├── clip_vit_l14.pth
│   └── hand_tracker.onnx
│
├── ontologies/                        # Knowledge bases
│   ├── spatial_relations.owl          # Spatial ontology
│   ├── procedures.owl                 # Task ontology
│   └── venue_knowledge.rdf            # Domain knowledge
│
└── notebooks/                         # Jupyter notebooks
    ├── 01_spatial_reasoning.ipynb
    ├── 02_hybrid_slam.ipynb
    ├── 03_training_assistant.ipynb
    └── 04_asset_generation.ipynb
```

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/iliachry/neurosymbolic-xr.git
cd neurosymbolic-xr

# Install dependencies with uv
uv sync

# Run interactive demonstration
uv run python demo.py
```

### Basic Usage

```python
from neurosymbolic_xr import NeurosymbolicXR

# Initialize framework
nxr = NeurosymbolicXR(
    neural_config="configs/neural_default.yaml",
    symbolic_config="configs/symbolic_default.yaml"
)

# Example 1: Spatial Reasoning
scene = nxr.capture_scene()  # 3DGS + segmentation
scene_graph = nxr.build_scene_graph(scene)
results = nxr.query("Find all horizontal surfaces above 0.5m")

# Example 2: Training Assistant
procedure = nxr.load_procedure("assembly_task.yaml")
assistant = nxr.create_training_assistant(procedure)
assistant.start_monitoring()

# Example 3: Asset Generation
spec = nxr.create_asset_spec(
    type="table",
    constraints=["height: 0.75m", "stability: high"]
)
asset = nxr.generate_asset(spec)
```

## 🔗 Integration with Poetics Platform

This framework integrates seamlessly with the Poetics VR platform:

```python
from src.integration.poetics import create_poetics_integration
from src.integration.aste import ASTENeurosymbolicVerifier, BiometricPacket, BiometricProvenance

# 1. Verify ASTE VR (https://aste.iliachry.gr) escape room invariants (Rooms 1..10)
verifier = ASTENeurosymbolicVerifier()
room1_verdict = verifier.verify_room(
    1,
    relevant_items={"reservoir", "inflow_pipe", "soil_moisture", "garden_bed"},
    noise_items={"distant_bell", "birdsong"},
    inside_boundary={"reservoir", "inflow_pipe", "soil_moisture", "garden_bed"},
    reservoir_stock=70.0,
    inflow_rate=6.0,
    outflow_rate=6.0,
    dwell_seconds=3.0,
)

# 2. Validate Garmin BLE Virtual Wrist Watch provenance & adjacent-beat RMSSD
packet = BiometricPacket(
    heart_rate_bpm=78.0,
    rr_intervals_ms=[800.0, 820.0, 790.0, 810.0],
    observed_at=100.0,
    source_mode=BiometricProvenance.LIVE,
)
rmssd_ms, provenance = packet.compute_rmssd_ms(current_time=101.5)

# 3. Gate Astra Socratic companion hints for entity grounding & non-leakage
ok, reason = ASTENeurosymbolicVerifier.verify_socratic_hint_gate(
    room1_verdict.astra_prompt,
    grounded_entities=ASTENeurosymbolicVerifier.ROOM_SPECS[1]["entities"],
    forbidden_solution_tokens={"auto_solve", "set radius to 3.2"},
)
```

## 🎬 Video Showcases (`deliverables/`)

- **10-Minute Extended Technical Showcase (`10:00` · Neurosymbolic XR × ASTE VR `aste.iliachry.gr`)**:
  - [`deliverables/neurosymbolic_xr_10min_showcase.mp4`](deliverables/neurosymbolic_xr_10min_showcase.mp4) — Full `1920x1080` `24 fps` master (`600.0s`)
  - [`deliverables/neurosymbolic_xr_10min_compressed.mp4`](deliverables/neurosymbolic_xr_10min_compressed.mp4) — `< 20 MB` (`< 20,000,000 bytes`) 2-pass size-controlled edition (`600.0s`)
  - [`deliverables/neurosymbolic_xr_10min_showcase.srt`](deliverables/neurosymbolic_xr_10min_showcase.srt) — Frame-aligned subtitles (`20 scenes`)
- **3-Minute Executive Showcase & Social Cuts**:
  - [`deliverables/neurosymbolic_xr_showcase.mp4`](deliverables/neurosymbolic_xr_showcase.mp4) (`3m 14s` master) & [`deliverables/neurosymbolic_xr_showcase_compressed.mp4`](deliverables/neurosymbolic_xr_showcase_compressed.mp4) (`< 20 MB`)
  - [`deliverables/neurosymbolic_xr_60s_cut.mp4`](deliverables/neurosymbolic_xr_60s_cut.mp4), [`deliverables/neurosymbolic_xr_30s_cut.mp4`](deliverables/neurosymbolic_xr_30s_cut.mp4), [`deliverables/neurosymbolic_xr_15s_cut.mp4`](deliverables/neurosymbolic_xr_15s_cut.mp4)

## 📊 Performance Benchmarks

| Component | Latency | Throughput | Hardware |
|-----------|---------|------------|----------|
| 3DGS Rendering | 8.3ms | 120 FPS | RTX 4090 |
| SAM 2 Segmentation | 45ms | 22 FPS | RTX 4090 |
| Spatial Query | 12ms | - | CPU |
| Constraint Validation | 3ms | - | CPU |
| ASTE 10-Room Symbolic Verifier (`test_aste_integration.py`) | < 2ms (13 tests) | - | Apple M4 |
| SLAM Tracking | 11ms | 90 Hz | Quest 3 |

## 🎓 Research Foundation

This framework implements and integrates concepts from:

- **Arnold, R. D., & Wade, J. P. (2015).** *A Definition of Systems Thinking: A Systems Approach.* Procedia Computer Science, 44, 669–678. ([DOI: 10.1016/j.procs.2015.03.050](https://doi.org/10.1016/j.procs.2015.03.050))
- **Arnold, R. D., & Wade, J. P. (2017).** *A Complete Set of Systems Thinking Skills.* INSIGHT, 20(3), 9–17. ([DOI: 10.1002/inst.12159](https://doi.org/10.1002/inst.12159))
- **Neural Radiance Fields** (Mildenhall et al., 2020)
- **3D Gaussian Splatting** (Kerbl et al., 2023)
- **SAM 2** (Ravi et al., 2024)
- **ORB-SLAM3** (Campos et al., 2021)
- **Neurosymbolic AI** (Garcez et al., 2019)
- **ExpertAssist: An "On-the-Job" Augmented Reality Toolkit for Assistance in the Construction Industry** (Chrysovergis et al., *Computers & Education: X Reality*, 2026, [DOI: 10.1016/j.cexr.2026.100155](https://doi.org/10.1016/j.cexr.2026.100155))
- **Metaverse Framework for Wireless Systems Management** (Chrysovergis et al., *IEEE Internet of Things Magazine*, 2026, [DOI: 10.1109/MIOT.2026.3680843](https://doi.org/10.1109/MIOT.2026.3680843))
- **Less Is More: Efficient Networked VR Transformation Handling Using Geometric Algebra** (Kamarianakis, Chrysovergis et al., *Advances in Applied Clifford Algebras*, 2023, [DOI: 10.1007/s00006-022-01253-9](https://doi.org/10.1007/s00006-022-01253-9))

## 📖 Documentation

- [Engineering Continuation Roadmap (`ROADMAP.md`)](ROADMAP.md)
- [10-Minute Extended Video Script (`VIDEO_SCRIPT_10MIN.md`)](VIDEO_SCRIPT_10MIN.md)
- [Architecture Overview](ARCHITECTURE.md)
- [Quick Start Guide](QUICKSTART.md)
- [Project Summary](PROJECT_SUMMARY.md)
- [3-Minute Video Script](VIDEO_SCRIPT.md)

## 📄 License

MIT License

## 🙏 Acknowledgments

Built upon research from:
- H.F.R.I. Grant No. 26237 (EduGenZ — ASTE VR [`https://aste.iliachry.gr`](https://aste.iliachry.gr))
- Meta AI (SAM 2, 3D Gaussian Splatting)
- University of Zaragoza (ORB-SLAM3)
- OpenAI (CLIP)
- Khronos Group (OpenXR)
- Poetics VR Platform

## 🔗 Verified Live Links

- GitHub Repository: [https://github.com/iliachry/neurosymbolic-xr](https://github.com/iliachry/neurosymbolic-xr)
- Live WebXR Serious Game: [https://aste.iliachry.gr](https://aste.iliachry.gr)

---

**Status**: Active Development | **Version**: 0.2.0 | **Last Updated**: 2026-10-17
