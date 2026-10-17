# Neurosymbolic XR × ASTE VR × Poetics Platform — Engineering Continuation Roadmap

**Repositories & Live Endpoints**
- **Neurosymbolic XR Framework**: [`https://github.com/iliachry/neurosymbolic-xr`](https://github.com/iliachry/neurosymbolic-xr)
- **ASTE VR (Adaptive Systems Thinking Escape Ecology)**: [`https://aste.iliachry.gr`](https://aste.iliachry.gr) (H.F.R.I. Grant No. 26237 — EduGenZ)

---

## 1. Architectural Vision

This roadmap bridges the **Neurosymbolic XR Framework** (3D Gaussian Splatting, SAM 2, CLIP, RDF/OWL Scene Graphs, SWI-Prolog, Z3 SMT Solver, and Hybrid ORB-SLAM3) with **ASTE VR** (`https://aste.iliachry.gr` — 10 embodied 6DoF WebXR escape chambers operationalizing the Arnold & Wade 2015, 2017 systems thinking taxonomy) and the **Poetics Location-Based VR (LBVR) Venue Platform**.

Every escape chamber enforces the shared formal neurosymbolic contract:

$$
(o_t, e_t) \xrightarrow{G_\theta} F_t \xrightarrow{R_r, C_r} (\text{decision}, \text{proof}) \xrightarrow{\pi_r} \text{Room Response / Grounded Astra Socratic Prompt}
$$

where engine interaction events ($e_t$) and multimodal sensor observations ($o_t$) are grounded into typed predicates ($F_t$) carrying explicit provenance, units, and validity windows (`true | false | unknown`), verified by room-specific rules ($R_r, C_r$) before triggering pedagogical interventions ($\pi_r$).

---

## 2. Five-Phase Continuation Roadmap

### Phase 1 — Shared Ontology, Evidence Contract & Formal Room Verifiers (Weeks 1–3)
- **Implemented (`src/integration/aste.py` & `tests/test_aste_integration.py`)**:
  - Formal symbolic verifiers across all **10 ASTE Escape Rooms** (`ASTENeurosymbolicVerifier.verify_room(1..10)`), enforcing $\operatorname{Complete}_r = \operatorname{ValidEvidence}_r \land \operatorname{TaskConstraints}_r \land \operatorname{RequiredTrace}_r$.
  - Three-valued truth semantics (`TruthValue.TRUE`, `FALSE`, `UNKNOWN`) preserving epistemic uncertainty when telemetry is incomplete or stale.
- **Next Milestones**:
  - Extend `ontologies/xr_spatial.owl` with `ontologies/aste_systems.owl` (`EscapeRoom`, `SystemsThinkingSkill`, `PhysiologicalState`, `SocraticTurn`, `CausalEdge`).
  - Host a resolved, permanent JSON-LD context directly on `https://aste.iliachry.gr/context/v1.jsonld` (replacing unresolved placeholder contexts) with SHACL validation shapes.

### Phase 2 — Real-Time WebXR Perception & Grounding Bridge (Weeks 3–7)
- Expose `GroundingBridge` and `ASTENeurosymbolicVerifier` over a low-latency WebSocket/FastAPI endpoint integrated with ASTE's backend server.
- **Dual-Mode Ingestion**:
  - *WebXR Virtual Chamber Mode (`aste.iliachry.gr`)*: Direct ingestion of Three.js scene graph poses, 6DoF controller/hand interactions (`Interactable_BoundaryRing`, `Interactable_PrimaryLever`, etc.), and Runge-Kutta stock-flow trajectories into `GroundedFact` triples.
  - *Physical LBVR Venue Mode (Poetics)*: Full `3DGS -> SAM 2 -> CLIP` open-vocabulary spatial capture and `HybridSLAM` (ORB-SLAM3 + semantic loop closure) for physical room colocation.

### Phase 3 — Biometric-Aware Symbolic Layer & Garmin BLE Provenance (Weeks 7–11)
- **Implemented (`BiometricPacket` in `src/integration/aste.py`)**:
  - Explicit four-state provenance tracking (`live`, `simulated`, `stale`, `unavailable`) for the Garmin BLE Heart Rate Service (`0x180D`) and Virtual Wrist Watch HUD.
  - Strict adjacent-beat RMSSD computation ($250\text{ ms} \le \text{RR}_i, \text{RR}_{i+1} \le 2000\text{ ms}$) that never bridges across dropped beats or fabricates physiological values when stale ($>5.0\text{ s}$).
- **Next Milestones**:
  - Connect Prolog rules (`cognitive_load(high) :- provenance(live), rmssd_drop_pct(D), D > 20, dwell_sec(T), T > 90.`) to Z3 adaptive scaffolding constraints with full unsat-core explanations in the Educator Cockpit.

### Phase 4 — Astra Socratic Companion as a Symbolically Gated Agent (Weeks 11–15)
- **Implemented (`verify_socratic_hint_gate` in `src/integration/aste.py`)**:
  - Every Astra utterance must (1) reference at least one grounded entity in the active room's scene graph, (2) pass the non-leakage guard blocking direct solution tokens, and (3) terminate as a Socratic question (`?`).
- **Next Milestones**:
  - Connect the GX10 Blackwell LLM (`Qwen3.6-35B-A3B`) to generate candidate Socratic turns conditioned on `RoomVerificationResult.violated_rules`, filtered deterministically by `verify_socratic_hint_gate` before WebXR TTS synthesis.

### Phase 5 — Multi-Pod Poetics LBVR Venue Deployment & Empirical Validation (Weeks 15–20)
- Isolate per-session and per-learner state across multi-pod Poetics VR deployments (`VenueContext` / `SessionContext`) with CRDT scene-graph synchronization.
- Run empirical pre/post systems thinking evaluations anchored to the Arnold & Wade (2015, 2017) framework, explicitly distinguishing *in-game chamber completion traces* from *externally validated psychometric competency scores*.

---

## 3. Formal Neurosymbolic Mapping Across All 10 ASTE Rooms

| Room | Title | Arnold & Wade Alignment | Formal Symbolic Invariant & Verification Rule |
|---:|---|---|---|
| **1** | **Forgotten Boundary** | Skills 2.1–2.2: Recognize Systems & Maintain Boundaries | $I(B) = \frac{\lvert R \cap B \rvert + \lvert N \cap B^c \rvert}{\lvert R \rvert + \lvert N \rvert} \ge 0.70 \land \lvert S - 70 \rvert \le 4 \land \lvert q_{\text{in}} - q_{\text{out}} \rvert \le 0.5$ |
| **2** | **Hall of Scale & Quantity** | Skill 2.3: Differentiate & Quantify Elements | $\operatorname{Comparable}(q_1, q_2) \Rightarrow \operatorname{SameKind}(q_1, q_2) \land \operatorname{CompatibleUnits}(q_1, q_2)$ (rejects $24\text{ cm}^2$ vs $20\text{ cm}$) |
| **3** | **River of Relationships** | Skills 3.1–3.2: Identify & Characterize Relationships | $\operatorname{sign}(x \to y) = \operatorname{sign}(\partial f_y / \partial x)$; requires Mediator path and $\text{Outflow} \xrightarrow{-1} \text{Stock}$ |
| **4** | **Echo Forest: Feedback & Delays** | Skills 3.3–3.4: Feedback Loops & Delays | $P(C) = \prod_{e \in C} \operatorname{sign}(e)$ ($+1$ reinforcing, $-1$ balancing) $\land t_{\text{wait}} \ge \tau_{\text{delay}}$ |
| **5** | **Valley of Time: Stocks & Flows** | Skills 4.1–4.3: Past Behavior, Prediction & Change | $S_{k+1} = S_k + \Delta t \left(\sum q_{\text{in},k} - \sum q_{\text{out},k}\right)$ with pre-reveal trajectory prediction |
| **6** | **Prism Mountain: Perspectives** | Skill 1.1: Multiple Perspectives | Articulated stakeholder concerns $\forall p \in \{\text{Pupil}, \text{Teacher}, \text{Designer}\} \land \operatorname{JustifiedTradeoff}$ |
| **7** | **Garden of Models** | Skill 1.5: Mental Models & Abstraction | $L(M) = \operatorname{Error}_{\text{held-out}}(M) + \lambda \operatorname{Complexity}(M) \le 0.35 \land R_{\text{essential}} \subseteq M$ |
| **8** | **Seed of Change: Leverage Points** | Skill 4.4: Use Leverage Points | $a^* = \arg\max_{a \in A_{\text{feasible}}} [J(x^{\operatorname{do}(a)}_{0:H}) - \lambda \operatorname{Cost}(a)] \land \operatorname{Structural}(a^*)$ |
| **9** | **Emergence Chamber** | Skills 1.1–4.4: Integrated Transfer | $\sum_i u_i \le C_{\max} \land \operatorname{MacroStability}(h(x_t)) \ge 0.80 \land \operatorname{AblationCompared}$ |
| **10** | **Reflection Observatory** | Metacognitive Consolidation & Educator Cockpit | $\operatorname{AcceptedClaim}(c) \Rightarrow \exists E \subseteq E_{\text{valid}}: \operatorname{Supports}(E, c) \land \operatorname{RevisedBelief}$ |

---

## 4. Verified Primary References

1. **Arnold, R. D., & Wade, J. P. (2015).** *A Definition of Systems Thinking: A Systems Approach.* Procedia Computer Science, 44, 669–678. DOI: [`10.1016/j.procs.2015.03.050`](https://doi.org/10.1016/j.procs.2015.03.050)
2. **Arnold, R. D., & Wade, J. P. (2017).** *A Complete Set of Systems Thinking Skills.* INSIGHT, 20(3), 9–17. DOI: [`10.1002/inst.12159`](https://doi.org/10.1002/inst.12159)
3. **ExpertAssist (2026).** *Computers & Education: X Reality.* DOI: [`10.1016/j.cexr.2026.100155`](https://doi.org/10.1016/j.cexr.2026.100155)
4. **Metaverse Framework for Wireless Systems Management (2026).** *IEEE Internet of Things Magazine.* DOI: [`10.1109/MIOT.2026.3680843`](https://doi.org/10.1109/MIOT.2026.3680843)
5. **Less is More: Efficient Networked VR Transformation Handling Using Geometric Algebra (2022).** *Advances in Applied Clifford Algebras.* DOI: [`10.1007/s00006-022-01253-9`](https://doi.org/10.1007/s00006-022-01253-9)
