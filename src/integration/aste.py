"""
ASTE VR (Adaptive Systems Thinking Escape Ecology) Neurosymbolic Integration.

Connects the Neurosymbolic XR Framework (RDF/OWL Scene Graph, Prolog Reasoning,
Z3 SMT Constraint Verification) with ASTE VR (https://aste.iliachry.gr):
- Formal room-level symbolic grounding and verification across all 10 escape rooms
  aligned with Arnold & Wade (2015, 2017) systems thinking skills.
- Garmin BLE Virtual Wrist Watch physiological state provenance (`live`, `simulated`,
  `stale`, `unavailable`) with strict adjacent-RR-interval RMSSD validation.
- Astra Socratic AI companion symbolic grounding gate (ensuring hints reference
  verified room entities without leaking direct puzzle answers).
- Educator Cockpit evidence provenance verification (distinguishing task completion
  from validated competency claims).

References:
- Arnold, R. D., & Wade, J. P. (2015). DOI: 10.1016/j.procs.2015.03.050
- Arnold, R. D., & Wade, J. P. (2017). DOI: 10.1002/inst.12159
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


class TruthValue(str, Enum):
    """Three-valued symbolic truth state preserving epistemic uncertainty."""

    TRUE = "true"
    FALSE = "false"
    UNKNOWN = "unknown"


class BiometricProvenance(str, Enum):
    """Explicit sensor provenance state for the Garmin BLE Virtual Wrist Watch."""

    LIVE = "live"
    SIMULATED = "simulated"
    STALE = "stale"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class GroundedFact:
    """Typed symbolic assertion with explicit provenance, units, and validity window."""

    session_id: str
    room_id: int
    scenario_version: str
    event_id: str
    entity_id: str
    predicate: str
    value: Any
    unit: str
    source: str
    observed_at: float
    valid_until: float
    model_version: str = "aste-ns-1.0"
    confidence: float = 1.0
    evidence_refs: Tuple[str, ...] = ()

    def is_fresh(self, current_time: float) -> bool:
        return self.observed_at <= current_time <= self.valid_until


@dataclass
class BiometricPacket:
    """Physiological telemetry packet from the Garmin BLE / WebSocket bridge."""

    heart_rate_bpm: Optional[float]
    rr_intervals_ms: List[float] = field(default_factory=list)
    observed_at: Optional[float] = None
    source_mode: BiometricProvenance = BiometricProvenance.UNAVAILABLE
    device_id: str = "garmin-ble-hrs-0x180d"
    staleness_threshold_sec: float = 5.0

    def resolve_provenance(self, current_time: float) -> BiometricProvenance:
        if (
            self.source_mode == BiometricProvenance.UNAVAILABLE
            or self.heart_rate_bpm is None
            or self.observed_at is None
        ):
            return BiometricProvenance.UNAVAILABLE
        age = current_time - self.observed_at
        if age < 0.0 or age > self.staleness_threshold_sec:
            return BiometricProvenance.STALE
        return self.source_mode

    def compute_rmssd_ms(
        self, current_time: float
    ) -> Tuple[Optional[float], BiometricProvenance]:
        """
        Compute RMSSD strictly from adjacent RR intervals (i, i+1) where BOTH
        consecutive beats lie within the physiological range [250ms, 2000ms].
        Never bridges across dropped/invalid beats.
        """
        prov = self.resolve_provenance(current_time)
        if prov in (BiometricProvenance.UNAVAILABLE, BiometricProvenance.STALE):
            return None, prov
        adjacent_diffs_sq: List[float] = []
        for i in range(len(self.rr_intervals_ms) - 1):
            rr_a = self.rr_intervals_ms[i]
            rr_b = self.rr_intervals_ms[i + 1]
            if 250.0 <= rr_a <= 2000.0 and 250.0 <= rr_b <= 2000.0:
                adjacent_diffs_sq.append((rr_b - rr_a) ** 2)
        if not adjacent_diffs_sq:
            return None, prov
        rmssd = math.sqrt(sum(adjacent_diffs_sq) / len(adjacent_diffs_sq))
        return round(rmssd, 2), prov


@dataclass
class RoomVerificationResult:
    """Formal verification verdict for an ASTE escape room."""

    room_id: int
    room_name: str
    skills: Tuple[str, ...]
    status: TruthValue
    constraints_satisfied: bool
    trace_satisfied: bool
    score_metric: float
    explanation: str
    astra_prompt: str
    violated_rules: List[str] = field(default_factory=list)
    evidence_refs: List[str] = field(default_factory=list)


class ASTENeurosymbolicVerifier:
    """
    Formal symbolic verifier for the 10 ASTE VR escape chambers.
    Enforces: Complete_r = ValidEvidence_r AND TaskConstraints_r AND RequiredTrace_r.
    All score_metric values are normalized to [0.0, 100.0].
    """

    ROOM_SPECS: Dict[int, Dict[str, Any]] = {
        1: {
            "name": "Forgotten Boundary",
            "skills": ("2.1 Recognize Systems", "2.2 Maintain Boundaries"),
            "astra": "What changes if soil moisture is left outside your system boundary?",
            "entities": {"reservoir", "inflow_pipe", "soil_moisture", "garden_bed", "boundary"},
        },
        2: {
            "name": "Hall of Scale and Quantity",
            "skills": ("2.3 Differentiate and Quantify Elements",),
            "astra": "What physical quantity kind does each numeral '10' actually measure?",
            "entities": {"quantity", "count", "length", "time", "unit"},
        },
        3: {
            "name": "River of Relationships",
            "skills": ("3.1 Identify Relationships", "3.2 Characterize Relationships"),
            "astra": "If outflow increases while inflow stays fixed, what happens to the stock?",
            "entities": {"valve", "mediator", "inflow", "outflow", "stock"},
        },
        4: {
            "name": "Echo Forest: Feedback and Delays",
            "skills": ("3.3 Identify Feedback Loops", "3.4 Characterize Feedback Loops"),
            "astra": "Has your previous intervention had time to propagate across the delay lag?",
            "entities": {"feedback", "loop", "delay", "lag", "intervention"},
        },
        5: {
            "name": "Valley of Time: Stocks and Flows",
            "skills": ("4.1 Past System Behavior", "4.2 Predict Future Behavior", "4.3 Response to Change"),
            "astra": "Does equal inflow and outflow mean the reservoir is empty, or simply constant?",
            "entities": {"reservoir", "inflow", "outflow", "stock", "trajectory"},
        },
        6: {
            "name": "Prism Mountain: Perspectives",
            "skills": ("1.1 Consider Multiple Perspectives",),
            "astra": "Who benefits from this system policy, and which stakeholder bears its cost?",
            "entities": {"pupil", "teacher", "designer", "stakeholder", "policy"},
        },
        7: {
            "name": "Garden of Models",
            "skills": ("1.5 Use Mental Models and Abstraction",),
            "astra": "Which causal branch can you prune without losing held-out explanatory power?",
            "entities": {"model", "branch", "variable", "explanatory", "abstraction"},
        },
        8: {
            "name": "Seed of Change: Leverage Points",
            "skills": ("4.4 Use Leverage Points",),
            "astra": "Does this intervention alter the system's feedback structure or only today's symptom?",
            "entities": {"intervention", "leverage", "feedback", "structure", "symptom"},
        },
        9: {
            "name": "Emergence Chamber",
            "skills": ("1.1-4.4 Integrated Systems Transfer",),
            "astra": "Why did locally sensible district actions produce an undesirable macro water shortage?",
            "entities": {"district", "water", "aquifer", "macro", "allocation"},
        },
        10: {
            "name": "Reflection Observatory",
            "skills": ("Metacognitive Consolidation & Evidence Provenance",),
            "astra": "Which specific disconfirming observation caused you to revise your mental model?",
            "entities": {"observation", "model", "evidence", "claim", "revision"},
        },
    }

    def verify_room(self, room_id: int, **kwargs: Any) -> RoomVerificationResult:
        """Unified dispatcher across all 10 ASTE escape rooms."""
        dispatch: Dict[int, Callable[..., RoomVerificationResult]] = {
            1: self.verify_room_1_boundary,
            2: self.verify_room_2_scale,
            3: self.verify_room_3_relationships,
            4: self.verify_room_4_feedback_delays,
            5: self.verify_room_5_stocks_flows,
            6: self.verify_room_6_perspectives,
            7: self.verify_room_7_models,
            8: self.verify_room_8_leverage,
            9: self.verify_room_9_emergence,
            10: self.verify_room_10_observatory,
        }
        if room_id not in dispatch:
            raise ValueError(f"Unsupported ASTE room_id={room_id}; expected 1..10.")
        return dispatch[room_id](**kwargs)

    def verify_room_1_boundary(
        self,
        relevant_items: Set[str],
        noise_items: Set[str],
        inside_boundary: Set[str],
        reservoir_stock: float,
        inflow_rate: float,
        outflow_rate: float,
        dwell_seconds: float,
        has_rationale: bool = True,
    ) -> RoomVerificationResult:
        """
        Room 1: I(B) = (|R cap B| + |N cap B^c|) / (|R| + |N|) >= 0.70,
        |stock - 70| <= 4, |inflow - outflow| <= 0.5, dwell >= 2.0s.
        """
        spec = self.ROOM_SPECS[1]
        total = len(relevant_items) + len(noise_items)
        if total == 0:
            return RoomVerificationResult(
                1, spec["name"], spec["skills"], TruthValue.UNKNOWN,
                False, False, 0.0, "Missing boundary entity sets.", spec["astra"]
            )
        tp = len(relevant_items.intersection(inside_boundary))
        tn = len(noise_items.difference(inside_boundary))
        integrity = (tp + tn) / total
        stock_ok = abs(reservoir_stock - 70.0) <= 4.0
        flow_ok = abs(inflow_rate - outflow_rate) <= 0.5
        boundary_ok = integrity >= 0.70
        trace_ok = dwell_seconds >= 2.0 and has_rationale

        violated: List[str] = []
        if not boundary_ok:
            violated.append(f"boundary_integrity_below_threshold({integrity:.2f} < 0.70)")
        if not stock_ok:
            violated.append(f"reservoir_stock_out_of_band({reservoir_stock:.1f} != 70±4)")
        if not flow_ok:
            violated.append(f"net_flow_nonzero({inflow_rate:.1f} vs {outflow_rate:.1f})")
        if not trace_ok:
            violated.append("insufficient_equilibrium_dwell_or_missing_rationale")

        passed = boundary_ok and stock_ok and flow_ok and trace_ok
        return RoomVerificationResult(
            room_id=1,
            room_name=spec["name"],
            skills=spec["skills"],
            status=TruthValue.TRUE if passed else TruthValue.FALSE,
            constraints_satisfied=(boundary_ok and stock_ok and flow_ok),
            trace_satisfied=trace_ok,
            score_metric=round(max(0.0, min(100.0, integrity * 100.0)), 1),
            explanation=(
                f"Boundary integrity I(B)={integrity:.2f}, stock={reservoir_stock:.1f}, "
                f"net_flow={inflow_rate - outflow_rate:+.2f}, dwell={dwell_seconds:.1f}s."
            ),
            astra_prompt=spec["astra"],
            violated_rules=violated,
        )

    def verify_room_2_scale(
        self,
        classifications: Dict[str, str],
        expected_kinds: Dict[str, str],
        rejected_incompatible_comparison: bool,
    ) -> RoomVerificationResult:
        """
        Room 2: Comparable(q1, q2) => SameQuantityKind(q1, q2) AND CompatibleUnits(q1, q2).
        Reject comparing 24 cm^2 (area) with 20 cm (length).
        """
        spec = self.ROOM_SPECS[2]
        if (
            not classifications
            or not expected_kinds
            or set(classifications.keys()) != set(expected_kinds.keys())
        ):
            return RoomVerificationResult(
                2, spec["name"], spec["skills"], TruthValue.UNKNOWN,
                False, False, 0.0, "Incomplete quantity classifications.", spec["astra"]
            )
        correct_count = sum(
            1 for k, v in expected_kinds.items() if classifications.get(k) == v
        )
        all_classified = correct_count == len(expected_kinds)
        violated = []
        if not all_classified:
            violated.append(f"misclassified_quantity_kinds({correct_count}/{len(expected_kinds)})")
        if not rejected_incompatible_comparison:
            violated.append("accepted_incompatible_comparison(24_cm2_vs_20_cm)")

        passed = all_classified and rejected_incompatible_comparison
        return RoomVerificationResult(
            room_id=2,
            room_name=spec["name"],
            skills=spec["skills"],
            status=TruthValue.TRUE if passed else TruthValue.FALSE,
            constraints_satisfied=all_classified,
            trace_satisfied=rejected_incompatible_comparison,
            score_metric=round((correct_count / len(expected_kinds)) * 100.0, 1),
            explanation=(
                f"Classified {correct_count}/{len(expected_kinds)} quantities; "
                f"incompatible unit rejection={rejected_incompatible_comparison}."
            ),
            astra_prompt=spec["astra"],
            violated_rules=violated,
        )

    def verify_room_3_relationships(
        self,
        edges: Dict[Tuple[str, str], int],
        predicted_before_pulse: bool,
    ) -> RoomVerificationResult:
        """
        Room 3: Require mediator path ('Valve' -> 'Mediator' (+1), 'Mediator' -> 'Inflow' (+1)),
        positive 'Inflow' -> 'Stock' (+1), and negative 'Outflow' -> 'Stock' (-1).
        """
        spec = self.ROOM_SPECS[3]
        if not edges:
            return RoomVerificationResult(
                3, spec["name"], spec["skills"], TruthValue.UNKNOWN,
                False, False, 0.0, "No causal edges submitted.", spec["astra"]
            )
        required_edges = {
            ("Valve", "Mediator"): 1,
            ("Mediator", "Inflow"): 1,
            ("Inflow", "Stock"): 1,
            ("Outflow", "Stock"): -1,
        }
        violated = []
        matched = 0
        for edge, req_sign in required_edges.items():
            if edges.get(edge) == req_sign:
                matched += 1
            else:
                violated.append(f"invalid_edge_polarity({edge[0]}->{edge[1]}, expected={req_sign})")
        if not predicted_before_pulse:
            violated.append("missing_pre_pulse_prediction_trace")

        constraints_ok = matched == len(required_edges)
        passed = constraints_ok and predicted_before_pulse
        return RoomVerificationResult(
            room_id=3,
            room_name=spec["name"],
            skills=spec["skills"],
            status=TruthValue.TRUE if passed else TruthValue.FALSE,
            constraints_satisfied=constraints_ok,
            trace_satisfied=predicted_before_pulse,
            score_metric=round((matched / len(required_edges)) * 100.0, 1),
            explanation=f"Matched {matched}/{len(required_edges)} signed causal edges; pre-pulse prediction={predicted_before_pulse}.",
            astra_prompt=spec["astra"],
            violated_rules=violated,
        )

    def verify_room_4_feedback_delays(
        self,
        cycle_edge_signs: List[int],
        reported_loop_type: str,
        delay_lag_sec: float,
        waited_after_intervention_sec: float,
    ) -> RoomVerificationResult:
        """
        Room 4: Loop polarity P(C) = prod sign(e). P=+1 => reinforcing, P=-1 => balancing.
        Learner must wait >= delay_lag_sec to observe delayed stabilization without over-correcting.
        """
        spec = self.ROOM_SPECS[4]
        if not cycle_edge_signs or any(s not in (-1, 1) for s in cycle_edge_signs):
            return RoomVerificationResult(
                4, spec["name"], spec["skills"], TruthValue.UNKNOWN,
                False, False, 0.0, "Invalid or empty cycle edge signs.", spec["astra"]
            )
        prod = 1
        for s in cycle_edge_signs:
            prod *= s
        expected_type = "reinforcing" if prod == 1 else "balancing"
        polarity_ok = reported_loop_type.lower() == expected_type
        delay_ok = waited_after_intervention_sec >= delay_lag_sec

        violated = []
        if not polarity_ok:
            violated.append(f"loop_polarity_mismatch(reported={reported_loop_type}, actual={expected_type})")
        if not delay_ok:
            violated.append(f"premature_intervention_before_delay_lag({waited_after_intervention_sec:.1f}s < {delay_lag_sec:.1f}s)")

        passed = polarity_ok and delay_ok
        return RoomVerificationResult(
            room_id=4,
            room_name=spec["name"],
            skills=spec["skills"],
            status=TruthValue.TRUE if passed else TruthValue.FALSE,
            constraints_satisfied=polarity_ok,
            trace_satisfied=delay_ok,
            score_metric=100.0 if passed else (50.0 if polarity_ok else 0.0),
            explanation=f"Loop product P(C)={prod:+d} ({expected_type}); delay observation {waited_after_intervention_sec:.1f}s / {delay_lag_sec:.1f}s.",
            astra_prompt=spec["astra"],
            violated_rules=violated,
        )

    def verify_room_5_stocks_flows(
        self,
        initial_stock: float,
        inflows: List[float],
        outflows: List[float],
        dt: float,
        reported_trajectory: List[float],
        predicted_before_reveal: bool,
        target_stock: float = 90.0,
    ) -> RoomVerificationResult:
        """
        Room 5: Conservation S_{k+1} = S_k + dt * (q_in,k - q_out,k).
        """
        spec = self.ROOM_SPECS[5]
        if (
            not inflows
            or len(inflows) != len(outflows)
            or len(reported_trajectory) != len(inflows) + 1
        ):
            return RoomVerificationResult(
                5, spec["name"], spec["skills"], TruthValue.UNKNOWN,
                False, False, 0.0, "Mismatched stock-flow trajectory length.", spec["astra"]
            )
        s = initial_stock
        max_residual = max(0.0, abs(reported_trajectory[0] - s))
        for k in range(len(inflows)):
            s = s + dt * (inflows[k] - outflows[k])
            max_residual = max(max_residual, abs(reported_trajectory[k + 1] - s))

        conservation_ok = max_residual <= 1e-2
        target_ok = abs(s - target_stock) <= 3.0
        violated = []
        if not conservation_ok:
            violated.append(f"mass_conservation_residual_exceeded({max_residual:.3f})")
        if not target_ok:
            violated.append(f"terminal_stock_off_target({s:.1f} != {target_stock:.1f}±3)")
        if not predicted_before_reveal:
            violated.append("prediction_recorded_after_trajectory_reveal")

        passed = conservation_ok and target_ok and predicted_before_reveal
        return RoomVerificationResult(
            room_id=5,
            room_name=spec["name"],
            skills=spec["skills"],
            status=TruthValue.TRUE if passed else TruthValue.FALSE,
            constraints_satisfied=(conservation_ok and target_ok),
            trace_satisfied=predicted_before_reveal,
            score_metric=round(max(0.0, min(100.0, 100.0 - max_residual * 10.0)), 1),
            explanation=f"Terminal stock={s:.1f} (target {target_stock:.1f}), max conservation residual={max_residual:.4f}.",
            astra_prompt=spec["astra"],
            violated_rules=violated,
        )

    def verify_room_6_perspectives(
        self,
        stakeholder_concerns: Dict[str, str],
        documented_tradeoff: bool,
    ) -> RoomVerificationResult:
        """Room 6: Require explicit supported concerns for Pupil, Teacher, Designer + justified trade-off."""
        spec = self.ROOM_SPECS[6]
        if not stakeholder_concerns:
            return RoomVerificationResult(
                6, spec["name"], spec["skills"], TruthValue.UNKNOWN,
                False, False, 0.0, "No stakeholder concerns recorded.", spec["astra"]
            )
        required = {"Pupil", "Teacher", "Designer"}
        covered = {k for k in required if stakeholder_concerns.get(k, "").strip()}
        all_covered = covered == required
        violated = []
        if not all_covered:
            violated.append(f"missing_stakeholder_perspectives({sorted(required - covered)})")
        if not documented_tradeoff:
            violated.append("missing_justified_stakeholder_tradeoff")
        passed = all_covered and documented_tradeoff
        return RoomVerificationResult(
            room_id=6,
            room_name=spec["name"],
            skills=spec["skills"],
            status=TruthValue.TRUE if passed else TruthValue.FALSE,
            constraints_satisfied=all_covered,
            trace_satisfied=documented_tradeoff,
            score_metric=round((len(covered) / 3.0) * 100.0, 1),
            explanation=f"Stakeholders articulated: {len(covered)}/3; trade-off justified={documented_tradeoff}.",
            astra_prompt=spec["astra"],
            violated_rules=violated,
        )

    def verify_room_7_models(
        self,
        retained_variables: Set[str],
        essential_variables: Set[str],
        noise_variables: Set[str],
        held_out_error: float,
        revised_after_counterexample: bool,
        lambda_complexity: float = 0.05,
        max_loss_threshold: float = 0.35,
    ) -> RoomVerificationResult:
        """Room 7: L(M) = Error_held_out(M) + lambda * Complexity(M) <= max_loss_threshold."""
        spec = self.ROOM_SPECS[7]
        if not essential_variables:
            return RoomVerificationResult(
                7, spec["name"], spec["skills"], TruthValue.UNKNOWN,
                False, False, 0.0, "Empty essential variable specification.", spec["astra"]
            )
        kept_essential = essential_variables.issubset(retained_variables)
        pruned_noise = len(retained_variables.intersection(noise_variables)) == 0
        loss = held_out_error + lambda_complexity * len(retained_variables)
        loss_ok = (held_out_error <= 0.15) and (loss <= max_loss_threshold)
        violated = []
        if not kept_essential:
            violated.append("pruned_causally_essential_variable")
        if not pruned_noise:
            violated.append("retained_spurious_noise_variable")
        if not loss_ok:
            violated.append(f"model_adequacy_loss_exceeded(L={loss:.3f}, err={held_out_error:.2f})")
        if not revised_after_counterexample:
            violated.append("missing_model_revision_after_counterexample")

        passed = kept_essential and pruned_noise and loss_ok and revised_after_counterexample
        return RoomVerificationResult(
            room_id=7,
            room_name=spec["name"],
            skills=spec["skills"],
            status=TruthValue.TRUE if passed else TruthValue.FALSE,
            constraints_satisfied=(kept_essential and pruned_noise and loss_ok),
            trace_satisfied=revised_after_counterexample,
            score_metric=round(max(0.0, min(100.0, (1.0 - loss) * 100.0)), 1),
            explanation=f"Model loss L(M)={loss:.3f} (error={held_out_error:.2f}, vars={len(retained_variables)}).",
            astra_prompt=spec["astra"],
            violated_rules=violated,
        )

    def verify_room_8_leverage(
        self,
        interventions: Dict[str, Tuple[float, float, bool]],
        selected_action: str,
        predicted_long_horizon: bool,
        lambda_cost: float = 0.5,
    ) -> RoomVerificationResult:
        """
        Room 8: a* = argmax_{a} [J(x^{do(a)}_{0:H}) - lambda * Cost(a)].
        interventions maps action_id -> (long_horizon_gain_J, cost, is_structural).
        Normalized score_metric in [0.0, 100.0] relative to optimal net utility.
        """
        spec = self.ROOM_SPECS[8]
        if not interventions or selected_action not in interventions:
            return RoomVerificationResult(
                8, spec["name"], spec["skills"], TruthValue.UNKNOWN,
                False, False, 0.0, "Unknown intervention selection.", spec["astra"]
            )
        scores = {
            a: j - lambda_cost * c for a, (j, c, _) in interventions.items()
        }
        best_action = max(scores, key=lambda k: scores[k])
        best_score = scores[best_action]
        selected_score = scores[selected_action]
        is_structural = interventions[selected_action][2]
        optimal_ok = (selected_action == best_action) and is_structural

        violated = []
        if selected_action != best_action:
            violated.append(f"suboptimal_leverage_point(selected={selected_action}, optimal={best_action})")
        if not is_structural:
            violated.append("selected_symptomatic_parameter_instead_of_structural_rule")
        if not predicted_long_horizon:
            violated.append("missing_long_horizon_trajectory_prediction")

        passed = optimal_ok and predicted_long_horizon
        norm_score = (
            (selected_score / best_score) * 100.0
            if best_score > 0
            else (100.0 if optimal_ok else 0.0)
        )
        return RoomVerificationResult(
            room_id=8,
            room_name=spec["name"],
            skills=spec["skills"],
            status=TruthValue.TRUE if passed else TruthValue.FALSE,
            constraints_satisfied=optimal_ok,
            trace_satisfied=predicted_long_horizon,
            score_metric=round(max(0.0, min(100.0, norm_score)), 1),
            explanation=f"Selected '{selected_action}' (net utility={selected_score:.1f} vs optimal '{best_action}'={best_score:.1f}).",
            astra_prompt=spec["astra"],
            violated_rules=violated,
        )

    def verify_room_9_emergence(
        self,
        district_allocations: List[float],
        total_capacity: float,
        macro_gini_stability: float,
        compared_ablation_condition: bool,
    ) -> RoomVerificationResult:
        """
        Room 9: Coupled urban-water allocation sum(u_i) <= total_capacity,
        macro observable stability >= 0.80, and explicit comparison against uncoordinated greedy ablation.
        """
        spec = self.ROOM_SPECS[9]
        if not district_allocations or total_capacity <= 0.0:
            return RoomVerificationResult(
                9, spec["name"], spec["skills"], TruthValue.UNKNOWN,
                False, False, 0.0, "Missing district allocations or invalid capacity.", spec["astra"]
            )
        total_used = sum(district_allocations)
        capacity_ok = total_used <= total_capacity + 1e-5
        macro_ok = macro_gini_stability >= 0.80
        violated = []
        if not capacity_ok:
            violated.append(f"aquifer_overdraft({total_used:.1f} > {total_capacity:.1f})")
        if not macro_ok:
            violated.append(f"emergent_macro_instability({macro_gini_stability:.2f} < 0.80)")
        if not compared_ablation_condition:
            violated.append("missing_local_vs_global_ablation_comparison")

        passed = capacity_ok and macro_ok and compared_ablation_condition
        return RoomVerificationResult(
            room_id=9,
            room_name=spec["name"],
            skills=spec["skills"],
            status=TruthValue.TRUE if passed else TruthValue.FALSE,
            constraints_satisfied=(capacity_ok and macro_ok),
            trace_satisfied=compared_ablation_condition,
            score_metric=round(max(0.0, min(100.0, macro_gini_stability * 100.0)), 1),
            explanation=f"Urban-water total={total_used:.1f}/{total_capacity:.1f}, macro stability={macro_gini_stability:.2f}.",
            astra_prompt=spec["astra"],
            violated_rules=violated,
        )

    def verify_room_10_observatory(
        self,
        claims: Dict[str, List[str]],
        valid_session_events: Set[str],
        revised_belief_recorded: bool,
    ) -> RoomVerificationResult:
        """
        Room 10: AcceptedClaim(c) => exists E: Supports(E, c) AND ValidProvenance(E).
        Preserves qualification: chamber completion != validated psychometrics without external assessment.
        """
        spec = self.ROOM_SPECS[10]
        if not claims:
            return RoomVerificationResult(
                10, spec["name"], spec["skills"], TruthValue.UNKNOWN,
                False, False, 0.0, "No reflection claims submitted.", spec["astra"]
            )
        verified_claims = 0
        cited_valid_events: Set[str] = set()
        violated = []
        for claim_id in sorted(claims.keys()):
            refs = claims[claim_id]
            if refs and all(r in valid_session_events for r in refs):
                verified_claims += 1
                cited_valid_events.update(refs)
            else:
                violated.append(f"unsubstantiated_claim({claim_id})")
        if not revised_belief_recorded:
            violated.append("missing_disconfirming_observation_belief_revision")

        all_supported = verified_claims == len(claims)
        passed = all_supported and revised_belief_recorded
        return RoomVerificationResult(
            room_id=10,
            room_name=spec["name"],
            skills=spec["skills"],
            status=TruthValue.TRUE if passed else TruthValue.FALSE,
            constraints_satisfied=all_supported,
            trace_satisfied=revised_belief_recorded,
            score_metric=round((verified_claims / len(claims)) * 100.0, 1),
            explanation=(
                f"Evidence-backed claims: {verified_claims}/{len(claims)}; "
                "note: chamber completion is behavioral trace evidence, not standalone psychometric certification."
            ),
            astra_prompt=spec["astra"],
            violated_rules=violated,
            evidence_refs=sorted(cited_valid_events),
        )

    @staticmethod
    def _contains_term(text: str, term: str) -> bool:
        pattern = r"\b" + re.escape(term.strip().lower()).replace(r"\_", r"[\s_]+") + r"\b"
        return bool(re.search(pattern, text.lower()))

    @classmethod
    def verify_socratic_hint_gate(
        cls,
        utterance: str,
        grounded_entities: Set[str],
        forbidden_solution_tokens: Set[str],
    ) -> Tuple[bool, str]:
        """
        Symbolic gate for Astra Socratic utterances:
        1. Must reference at least one grounded entity in the current room's scene graph
           (whole-token / phrase match, deterministic sorted order).
        2. Must NOT leak any forbidden direct solution token (information-gain guard).
        3. Must end as a Socratic inquiry (ends with '?').
        """
        stripped = utterance.strip()
        referenced = [
            e for e in sorted(grounded_entities) if cls._contains_term(stripped, e)
        ]
        if not referenced:
            return False, "ungrounded_utterance: no scene-graph entity referenced"
        leaked = [
            tok
            for tok in sorted(forbidden_solution_tokens)
            if cls._contains_term(stripped, tok)
        ]
        if leaked:
            return False, f"solution_leakage_blocked: contains {leaked}"
        if not stripped.endswith("?"):
            return False, "non_socratic_form: utterance must end with '?'"
        return True, f"grounded_on({referenced[0]})_and_non_leaking"
