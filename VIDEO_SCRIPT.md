# Video Showcase Script: Neurosymbolic XR Framework

**Duration**: 3-4 minutes  
**Style**: Technical demonstration with voiceover and screen recordings  
**Target Audience**: VR developers, AI researchers, Poetics platform users

---

## Scene 1: Opening Title (0:00 - 0:10)

### Visual
- Animated title card with framework logo
- Background: Abstract neural network morphing into symbolic graph
- Text overlay: "Neurosymbolic XR Framework"
- Subtitle: "Bridging Neural Perception with Symbolic Reasoning for VR"

### Voiceover
*"Introducing the Neurosymbolic XR Framework—a breakthrough in Extended Reality AI that combines the power of deep learning with logical reasoning."*

### Technical Notes
- Use particle effects for neural→symbolic transition
- Color scheme: Blue (neural) → Gold (symbolic)

---

## Scene 2: Architecture Overview (0:10 - 0:30)

### Visual
- Animated architecture diagram from README.md
- Layers appear sequentially:
  1. Sensor Layer (cameras, IMU, hand tracking)
  2. Neural Layer (3DGS, SAM 2, CLIP)
  3. Grounding Bridge (bidirectional arrows)
  4. Symbolic Layer (RDF, Prolog, Z3)
  5. Application Layer (Poetics VR)

### Voiceover
*"The framework features a layered architecture. Sensors feed into neural networks for perception, which are grounded into symbolic representations for reasoning. This bidirectional bridge enables both bottom-up understanding and top-down validation."*

### Technical Notes
- Animate data flow with glowing particles
- Highlight each layer as mentioned
- Show code snippets from bridge.py

---

## Scene 3: System 1 - Spatial Reasoning (0:30 - 1:00)

### Visual - Part A: Scene Capture
- Split screen showing:
  - Left: Multi-view RGB images of a VR venue
  - Right: 3D Gaussian point cloud being optimized
- Progress bar showing optimization iterations
- Gaussians splitting and densifying in real-time

### Visual - Part B: Segmentation & Grounding
- SAM 2 segmentation masks appearing over objects
- CLIP labels appearing: "chair", "table", "laptop"
- Confidence scores displayed

### Visual - Part C: Scene Graph
- 3D view transitions to symbolic graph visualization
- Nodes: Entities (chair, table, laptop)
- Edges: Relations (on, near, above)
- RDF triples displayed in sidebar

### Voiceover
*"System One: Spatial Reasoning. We capture the scene with 3D Gaussian Splatting, segment objects with SAM 2, and ground them to language with CLIP. The result? A queryable scene graph that understands both geometry and semantics."*

### Demo Query
- Terminal shows: `query("Find all objects on the table")`
- Results highlight: laptop, book, cup
- 3D visualization shows spatial relationships

### Technical Notes
- Screen recording of demo.py spatial reasoning section
- Use OBS to capture terminal + 3D viewer
- Add subtle glow effects to highlighted objects

---

## Scene 4: System 2 - Hybrid SLAM (1:00 - 1:30)

### Visual - Part A: Real-time Tracking
- First-person VR view moving through venue
- Overlay showing:
  - Current pose (position + orientation)
  - Tracking time: 16ms
  - Map points being detected
- Green wireframe showing tracked features

### Visual - Part B: Semantic Memory
- Split screen:
  - Left: Live camera view
  - Right: Semantic memory visualization
- Objects appear in memory with metadata:
  - "Door" - First seen: 2.5s ago, Observations: 15
  - "Stage" - First seen: 10.2s ago, Observations: 42
- Knowledge graph grows as objects are discovered

### Visual - Part C: Map Persistence
- Fade to black, then fade in
- New session loads previous map
- Relocalization successful
- Semantic memory restored

### Voiceover
*"System Two: Hybrid SLAM. Visual-inertial tracking provides precise localization at 60 FPS. But unlike traditional SLAM, we maintain semantic memory—objects are remembered across sessions, building a persistent understanding of the environment."*

### Technical Notes
- Record with VR headset or simulate with Unity
- Overlay graphics using After Effects
- Show trajectory path in 3D

---

## Scene 5: System 3 - Training Assistant (1:30 - 2:00)

### Visual - Part A: Procedure Display
- VR view with UI overlay showing:
  - Procedure: "Basic Assembly Task"
  - Current step: "Pick up the base component"
  - Timer: 00:12
- Hand tracking visualization (skeleton overlay)

### Visual - Part B: Action Validation
- User's hand approaches object
- Distance indicator: 0.08m
- Gesture recognized: "GRAB"
- Prolog reasoning shown in sidebar:
  ```prolog
  ?- can_grab(base_component).
  true.
  ```
- Checkmarks appear:
  - ✓ Preconditions satisfied
  - ✓ Action valid
  - ✓ Constraints satisfied

### Visual - Part C: Feedback
- Success message: "✓ Step completed!"
- Progress bar advances: 1/4 → 2/4
- Next instruction appears
- Performance metrics update

### Voiceover
*"System Three: VR Training Assistant. Hand and gaze tracking feed into Prolog-based procedural reasoning. Each action is validated against logical preconditions and constraints, providing real-time feedback to guide users through complex procedures."*

### Technical Notes
- Capture VR training session
- Add UI overlays in post-production
- Show Prolog reasoning in code window

---

## Scene 6: System 4 - Asset Generator (2:00 - 2:30)

### Visual - Part A: Specification
- UI showing asset specification form:
  - Type: 3D Object
  - Description: "modern office chair with wheels"
  - Constraints:
    - Dimensions: 0.6m × 1.0m × 0.6m
    - Symmetry: Y-axis
    - Functional: Sittable, Movable

### Visual - Part B: Constraint Validation
- Z3 solver visualization:
  ```
  Checking constraints...
  ✓ Dimensions satisfiable
  ✓ Symmetry achievable
  ✓ Constraints valid
  ```

### Visual - Part C: Generation
- Diffusion model progress:
  - Text encoding
  - Denoising steps (progress bar)
  - 3D mesh generation
  - Texture synthesis
- Time-lapse showing asset appearing
- Final result: Photorealistic office chair

### Visual - Part D: Variations
- Grid showing 4 variations of the chair
- Different styles but same constraints
- User can select preferred version

### Voiceover
*"System Four: Neural-Symbolic Asset Generator. Natural language descriptions are combined with geometric constraints. Z3 validates feasibility, then Stable Diffusion generates the asset. The result? AI-created content that satisfies both aesthetic and functional requirements."*

### Technical Notes
- Screen recording of generation process
- Speed up diffusion steps (time-lapse)
- 3D viewer for final asset

---

## Scene 7: Poetics Integration (2:30 - 2:50)

### Visual - Part A: Platform Overview
- Poetics VR platform interface
- Venue management dashboard
- Session list with active users

### Visual - Part B: Unified API
- Code editor showing integration:
```python
integration = create_poetics_integration(config)
session = await integration.start_session(
    venue_id='main_theater',
    mode='exploration'
)
```

### Visual - Part C: Real-time Updates
- Split screen showing:
  - Left: VR user exploring venue
  - Right: Dashboard with live updates:
    - Pose tracking
    - Scene entities discovered
    - Semantic memory growing
    - Event log scrolling

### Visual - Part D: Database
- Supabase dashboard showing:
  - Sessions table
  - Venue states
  - Generated assets
  - Semantic memory

### Voiceover
*"All four systems integrate seamlessly with the Poetics VR platform. A unified API manages sessions, tracks users, stores semantic memory in Supabase, and provides real-time events. Everything you need for intelligent VR experiences."*

### Technical Notes
- Screen recording of Poetics dashboard
- Split-screen with VR view
- Highlight database updates

---

## Scene 8: Performance Metrics (2:50 - 3:05)

### Visual
- Animated performance dashboard:
  - Tracking: 16ms (60 FPS) ✓
  - Scene Understanding: 8.3s ✓
  - Asset Generation: 10.2s ✓
  - Training Validation: 3ms ✓
- Bar charts comparing to baseline
- Green checkmarks for all targets met

### Voiceover
*"Performance is critical for XR. The framework achieves sub-20 millisecond tracking latency, real-time scene understanding, and fast asset generation—all while running complex AI models."*

### Technical Notes
- Animated charts using Motion Graphics
- Use green for "passing" metrics

---

## Scene 9: Technical Stack (3:05 - 3:20)

### Visual
- Animated tech stack visualization:
  - **Neural**: 3D Gaussian Splatting, SAM 2, CLIP, Stable Diffusion
  - **Symbolic**: RDFLib, Prolog, Z3
  - **Platform**: Supabase, FastAPI, OpenXR
- Logos appear with brief descriptions

### Voiceover
*"Built on cutting-edge technologies: 3D Gaussian Splatting for rendering, SAM 2 for segmentation, CLIP for grounding, and Stable Diffusion for generation. Symbolic reasoning with RDF, Prolog, and Z3. All integrated with Supabase and OpenXR."*

### Technical Notes
- Use official logos where available
- Arrange in layered architecture

---

## Scene 10: Call to Action (3:20 - 3:40)

### Visual
- GitHub repository page (`https://github.com/iliachry/neurosymbolic-xr`)
- Architecture and Quickstart documentation in repository
- Quick start code snippet
- Repository module overview

### Text Overlays
```
🚀 Get Started Today
💻 GitHub: https://github.com/iliachry/neurosymbolic-xr
📐 Architecture: ARCHITECTURE.md
🚀 Quickstart: QUICKSTART.md
```

### Voiceover
*"Ready to build intelligent VR experiences? The Neurosymbolic XR Framework is open source and production-ready. Visit the GitHub repository for the source code, architecture documentation, and interactive demo. Transform your VR applications with the power of neurosymbolic AI."*

### Technical Notes
- Show GitHub stars/forks counter
- Animate social proof metrics

---

## Scene 11: Closing (3:40 - 3:50)

### Visual
- Return to opening animation
- Neural network → Symbolic graph transition
- Framework logo
- Fade to black

### Text Overlay
```
Neurosymbolic XR Framework
Bridging Perception and Reasoning for Extended Reality

Built with ❤️ for the Poetics VR Platform
```

### Voiceover
*"Neurosymbolic XR Framework. The future of intelligent Extended Reality."*

---

## Production Notes

### Recording Requirements

1. **Screen Recordings**
   - Terminal running `demo.py`
   - Code editor showing key files
   - 3D visualization tools
   - Supabase dashboard
   - Poetics platform interface

2. **VR Footage**
   - First-person view with hand tracking
   - Training procedure execution
   - Scene exploration with SLAM
   - Asset placement in venue

3. **Graphics**
   - Architecture diagrams (from README.md)
   - Performance charts
   - Tech stack visualization
   - UI overlays for VR footage

### Software Needed

- **Recording**: OBS Studio, VR capture tools
- **Editing**: DaVinci Resolve or Adobe Premiere
- **Graphics**: After Effects for overlays
- **3D Visualization**: Blender or Unity for scene graphs
- **Audio**: Professional voiceover recording

### Style Guide

- **Colors**: 
  - Neural (blue): #1E90FF
  - Symbolic (gold): #FFD700
  - Success (green): #00FF00
  - Background: Dark theme (#1a1a1a)

- **Fonts**:
  - Titles: Montserrat Bold
  - Body: Inter Regular
  - Code: Fira Code

- **Transitions**: 
  - Smooth fades (0.5s)
  - Particle effects for data flow
  - Minimal motion blur

### Music

- **Intro/Outro**: Upbeat electronic (120 BPM)
- **Demo Sections**: Ambient tech background (80 BPM)
- **Ensure**: Royalty-free or licensed music

### Voiceover

- **Tone**: Professional, enthusiastic, clear
- **Pace**: 150-160 words per minute
- **Recording**: Studio quality, minimal background noise

---

## Alternative: Animated Explainer Version

For a fully animated version without live footage:

1. Use motion graphics for all visuals
2. Animated architecture diagrams
3. Stylized representations of systems
4. Code snippets with syntax highlighting
5. Abstract visualizations of AI processes

**Tools**: After Effects, Blender, Cinema 4D

---

## Distribution

- **YouTube**: Primary platform, 4K upload
- **Twitter**: 1-minute teaser clip
- **LinkedIn**: Professional audience
- **GitHub**: Embed in README.md
- **Poetics Platform**: Feature on homepage

---

## Timeline Estimate

- **Pre-production**: 2-3 days (storyboarding, asset gathering)
- **Recording**: 1-2 days (screen captures, VR footage)
- **Editing**: 3-4 days (cutting, effects, audio)
- **Review & Revisions**: 1-2 days
- **Total**: ~1-2 weeks

---

## Budget Estimate (Professional Production)

- **Voiceover Artist**: $200-400
- **Video Editor**: $500-1000
- **Motion Graphics**: $300-600
- **Music Licensing**: $50-100
- **Total**: $1,050-2,100

*Note: Can be done in-house to reduce costs*

---

This script provides a complete blueprint for creating a professional showcase video of the Neurosymbolic XR Framework. Adjust timing and content based on your specific needs and target audience.
