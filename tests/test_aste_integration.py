"""
Unit tests for ASTE VR (https://aste.iliachry.gr) Neurosymbolic Integration.

Verifies positive mastery traces, negative counterexample rejections, and
epistemic UNKNOWN states across all 10 ASTE Systems Thinking Escape Rooms,
all 4 Garmin BLE provenance states (including dropped-beat RMSSD adjacency),
GroundedFact freshness, and all branches of the Astra Socratic hint gate.
"""

import unittest
from src.integration.aste import (
    ASTENeurosymbolicVerifier,
    BiometricPacket,
    BiometricProvenance,
    GroundedFact,
    TruthValue,
)


class TestASTENeurosymbolicIntegration(unittest.TestCase):
    def setUp(self) -> None:
        self.verifier = ASTENeurosymbolicVerifier()

    def test_grounded_fact_freshness(self) -> None:
        fact = GroundedFact(
            session_id="sess-01",
            room_id=1,
            scenario_version="1.0",
            event_id="ev-001",
            entity_id="Interactable_BoundaryRing",
            predicate="BoundaryRadius",
            value=3.2,
            unit="m",
            source="webxr_engine",
            observed_at=10.0,
            valid_until=20.0,
            evidence_refs=("ev-001",),
        )
        self.assertTrue(fact.is_fresh(15.0))
        self.assertFalse(fact.is_fresh(9.5))
        self.assertFalse(fact.is_fresh(20.5))

    def test_room_1_boundary(self) -> None:
        rel = {"reservoir", "inflow_pipe", "soil_moisture", "garden_bed"}
        noise = {"distant_bell", "birdsong"}
        res_ok = self.verifier.verify_room(
            1,
            relevant_items=rel,
            noise_items=noise,
            inside_boundary={"reservoir", "inflow_pipe", "soil_moisture", "garden_bed"},
            reservoir_stock=70.0,
            inflow_rate=6.0,
            outflow_rate=6.0,
            dwell_seconds=3.0,
        )
        self.assertEqual(res_ok.status, TruthValue.TRUE)
        self.assertEqual(res_ok.violated_rules, [])

        res_bad = self.verifier.verify_room_1_boundary(
            rel, noise, {"reservoir", "distant_bell", "birdsong"},
            reservoir_stock=70.0, inflow_rate=6.0, outflow_rate=6.0, dwell_seconds=3.0
        )
        self.assertEqual(res_bad.status, TruthValue.FALSE)
        self.assertTrue(any("boundary_integrity_below_threshold" in r for r in res_bad.violated_rules))

        res_unk = self.verifier.verify_room_1_boundary(
            set(), set(), set(), 70.0, 6.0, 6.0, 3.0
        )
        self.assertEqual(res_unk.status, TruthValue.UNKNOWN)

    def test_room_2_scale(self) -> None:
        expected = {"cube_10_apples": "count", "cube_10_meters": "length", "cube_10_seconds": "time"}
        res_ok = self.verifier.verify_room(
            2, classifications=expected, expected_kinds=expected, rejected_incompatible_comparison=True
        )
        self.assertEqual(res_ok.status, TruthValue.TRUE)

        res_bad = self.verifier.verify_room_2_scale(expected, expected, rejected_incompatible_comparison=False)
        self.assertEqual(res_bad.status, TruthValue.FALSE)
        self.assertIn("accepted_incompatible_comparison(24_cm2_vs_20_cm)", res_bad.violated_rules)

        res_unk = self.verifier.verify_room_2_scale({}, expected, True)
        self.assertEqual(res_unk.status, TruthValue.UNKNOWN)

    def test_room_3_relationships(self) -> None:
        edges_ok = {
            ("Valve", "Mediator"): 1,
            ("Mediator", "Inflow"): 1,
            ("Inflow", "Stock"): 1,
            ("Outflow", "Stock"): -1,
        }
        self.assertEqual(
            self.verifier.verify_room(3, edges=edges_ok, predicted_before_pulse=True).status,
            TruthValue.TRUE,
        )
        edges_bad = dict(edges_ok)
        edges_bad[("Outflow", "Stock")] = 1
        res_bad = self.verifier.verify_room_3_relationships(edges_bad, predicted_before_pulse=True)
        self.assertEqual(res_bad.status, TruthValue.FALSE)
        self.assertIn("invalid_edge_polarity(Outflow->Stock, expected=-1)", res_bad.violated_rules)

        self.assertEqual(
            self.verifier.verify_room_3_relationships({}, True).status,
            TruthValue.UNKNOWN,
        )

    def test_room_4_feedback_delays(self) -> None:
        res_ok = self.verifier.verify_room(
            4,
            cycle_edge_signs=[1, 1, -1],
            reported_loop_type="balancing",
            delay_lag_sec=4.0,
            waited_after_intervention_sec=4.5,
        )
        self.assertEqual(res_ok.status, TruthValue.TRUE)

        res_bad = self.verifier.verify_room_4_feedback_delays(
            [1, 1, -1], "balancing", delay_lag_sec=4.0, waited_after_intervention_sec=1.5
        )
        self.assertEqual(res_bad.status, TruthValue.FALSE)
        self.assertTrue(any("premature_intervention_before_delay_lag" in r for r in res_bad.violated_rules))

        self.assertEqual(
            self.verifier.verify_room_4_feedback_delays([0], "balancing", 4.0, 4.5).status,
            TruthValue.UNKNOWN,
        )

    def test_room_5_stocks_flows(self) -> None:
        res_ok = self.verifier.verify_room(
            5,
            initial_stock=70.0,
            inflows=[10.0, 10.0, 10.0, 10.0],
            outflows=[5.0, 5.0, 5.0, 5.0],
            dt=1.0,
            reported_trajectory=[70.0, 75.0, 80.0, 85.0, 90.0],
            predicted_before_reveal=True,
        )
        self.assertEqual(res_ok.status, TruthValue.TRUE)

        res_bad = self.verifier.verify_room_5_stocks_flows(
            70.0, [10.0, 10.0, 10.0, 10.0], [5.0, 5.0, 5.0, 5.0], 1.0,
            [70.0, 80.0, 85.0, 88.0, 90.0], predicted_before_reveal=True
        )
        self.assertEqual(res_bad.status, TruthValue.FALSE)
        self.assertTrue(any("mass_conservation_residual_exceeded" in r for r in res_bad.violated_rules))

        self.assertEqual(
            self.verifier.verify_room_5_stocks_flows(70.0, [], [], 1.0, [70.0], True).status,
            TruthValue.UNKNOWN,
        )

    def test_room_6_perspectives(self) -> None:
        r6_ok = self.verifier.verify_room(
            6,
            stakeholder_concerns={
                "Pupil": "Engagement",
                "Teacher": "Curriculum alignment",
                "Designer": "6DoF comfort",
            },
            documented_tradeoff=True,
        )
        self.assertEqual(r6_ok.status, TruthValue.TRUE)

        r6_bad = self.verifier.verify_room_6_perspectives({"Pupil": "Fun"}, documented_tradeoff=False)
        self.assertEqual(r6_bad.status, TruthValue.FALSE)
        self.assertIn("missing_justified_stakeholder_tradeoff", r6_bad.violated_rules)

        self.assertEqual(
            self.verifier.verify_room_6_perspectives({}, True).status,
            TruthValue.UNKNOWN,
        )

    def test_room_7_models(self) -> None:
        r7_ok = self.verifier.verify_room(
            7,
            retained_variables={"inflow", "outflow", "soil_absorption"},
            essential_variables={"inflow", "outflow", "soil_absorption"},
            noise_variables={"cloud_color", "ambient_chime"},
            held_out_error=0.06,
            revised_after_counterexample=True,
        )
        self.assertEqual(r7_ok.status, TruthValue.TRUE)

        r7_bad = self.verifier.verify_room_7_models(
            retained_variables={"inflow", "cloud_color"},
            essential_variables={"inflow", "outflow", "soil_absorption"},
            noise_variables={"cloud_color", "ambient_chime"},
            held_out_error=0.06,
            revised_after_counterexample=True,
        )
        self.assertEqual(r7_bad.status, TruthValue.FALSE)
        self.assertIn("pruned_causally_essential_variable", r7_bad.violated_rules)
        self.assertIn("retained_spurious_noise_variable", r7_bad.violated_rules)

        self.assertEqual(
            self.verifier.verify_room_7_models(set(), set(), set(), 0.05, True).status,
            TruthValue.UNKNOWN,
        )

    def test_room_8_leverage(self) -> None:
        interventions = {
            "symptom_buffer": (25.0, 10.0, False),
            "feedback_gain_rule": (92.0, 15.0, True),
        }
        r8_ok = self.verifier.verify_room(
            8,
            interventions=interventions,
            selected_action="feedback_gain_rule",
            predicted_long_horizon=True,
        )
        self.assertEqual(r8_ok.status, TruthValue.TRUE)
        self.assertEqual(r8_ok.score_metric, 100.0)

        r8_bad = self.verifier.verify_room_8_leverage(interventions, "symptom_buffer", True)
        self.assertEqual(r8_bad.status, TruthValue.FALSE)
        self.assertIn("selected_symptomatic_parameter_instead_of_structural_rule", r8_bad.violated_rules)

        self.assertEqual(
            self.verifier.verify_room_8_leverage({}, "unknown", True).status,
            TruthValue.UNKNOWN,
        )

    def test_room_9_emergence(self) -> None:
        r9_ok = self.verifier.verify_room(
            9,
            district_allocations=[30.0, 32.0, 33.0],
            total_capacity=100.0,
            macro_gini_stability=0.88,
            compared_ablation_condition=True,
        )
        self.assertEqual(r9_ok.status, TruthValue.TRUE)

        r9_bad = self.verifier.verify_room_9_emergence([45.0, 45.0, 45.0], 100.0, 0.42, False)
        self.assertEqual(r9_bad.status, TruthValue.FALSE)
        self.assertTrue(any("aquifer_overdraft" in r for r in r9_bad.violated_rules))

        self.assertEqual(
            self.verifier.verify_room_9_emergence([], 100.0, 0.9, True).status,
            TruthValue.UNKNOWN,
        )

    def test_room_10_observatory(self) -> None:
        r10_ok = self.verifier.verify_room(
            10,
            claims={"claim_1": ["ev_r1_boundary", "ev_r4_delay"]},
            valid_session_events={"ev_r1_boundary", "ev_r4_delay", "ev_r8_leverage"},
            revised_belief_recorded=True,
        )
        self.assertEqual(r10_ok.status, TruthValue.TRUE)
        self.assertEqual(r10_ok.evidence_refs, ["ev_r1_boundary", "ev_r4_delay"])

        r10_bad = self.verifier.verify_room_10_observatory(
            {"claim_1": ["fabricated_event_id"]},
            {"ev_r1_boundary"},
            revised_belief_recorded=True,
        )
        self.assertEqual(r10_bad.status, TruthValue.FALSE)
        self.assertIn("unsubstantiated_claim(claim_1)", r10_bad.violated_rules)

        self.assertEqual(
            self.verifier.verify_room_10_observatory({}, {"ev_r1"}, True).status,
            TruthValue.UNKNOWN,
        )

    def test_garmin_biometric_provenance_and_rmssd_adjacency(self) -> None:
        # 1. LIVE packet with dropped beat (3000ms) in the middle:
        # Adjacent valid pairs are (800, 820) -> diff 20, and (790, 810) -> diff 20 -> RMSSD = 20.0
        pkt = BiometricPacket(
            heart_rate_bpm=78.0,
            rr_intervals_ms=[800.0, 820.0, 3000.0, 790.0, 810.0],
            observed_at=100.0,
            source_mode=BiometricProvenance.LIVE,
        )
        rmssd, prov = pkt.compute_rmssd_ms(current_time=102.0)
        self.assertEqual(prov, BiometricProvenance.LIVE)
        self.assertEqual(rmssd, 20.0)

        # 2. SIMULATED packet
        pkt_sim = BiometricPacket(
            heart_rate_bpm=72.0,
            rr_intervals_ms=[810.0, 820.0],
            observed_at=100.0,
            source_mode=BiometricProvenance.SIMULATED,
        )
        rmssd_sim, prov_sim = pkt_sim.compute_rmssd_ms(current_time=101.0)
        self.assertEqual(prov_sim, BiometricProvenance.SIMULATED)
        self.assertEqual(rmssd_sim, 10.0)

        # 3. STALE packet (age > 5.0s or future timestamp)
        rmssd_stale, prov_stale = pkt.compute_rmssd_ms(current_time=108.0)
        self.assertEqual(prov_stale, BiometricProvenance.STALE)
        self.assertIsNone(rmssd_stale)
        _, prov_future = pkt.compute_rmssd_ms(current_time=99.0)
        self.assertEqual(prov_future, BiometricProvenance.STALE)

        # 4. UNAVAILABLE packet
        pkt_unavail = BiometricPacket(heart_rate_bpm=None, observed_at=100.0)
        rmssd_unavail, prov_unavail = pkt_unavail.compute_rmssd_ms(current_time=101.0)
        self.assertEqual(prov_unavail, BiometricProvenance.UNAVAILABLE)
        self.assertIsNone(rmssd_unavail)

    def test_socratic_hint_gate_all_branches(self) -> None:
        # 1. Grounded & non-leaking Socratic question for all 10 ROOM_SPECS prompts
        for room_id, spec in ASTENeurosymbolicVerifier.ROOM_SPECS.items():
            ok, reason = ASTENeurosymbolicVerifier.verify_socratic_hint_gate(
                spec["astra"],
                grounded_entities=spec["entities"],
                forbidden_solution_tokens={"auto_solve", "set radius to 3.2"},
            )
            self.assertTrue(ok, f"Room {room_id} Astra prompt failed gate: {reason}")

        # 2. Ungrounded utterance (word-boundary check: 'stockpile' must not match 'stock')
        ok_ungrounded, reason_ungrounded = ASTENeurosymbolicVerifier.verify_socratic_hint_gate(
            "Why did you build a stockpile today?",
            grounded_entities={"stock", "inflow"},
            forbidden_solution_tokens={"set valve to 6"},
        )
        self.assertFalse(ok_ungrounded)
        self.assertIn("ungrounded_utterance", reason_ungrounded)

        # 3. Solution leakage blocked
        ok_leak, reason_leak = ASTENeurosymbolicVerifier.verify_socratic_hint_gate(
            "Why not set radius to 3.2 on the boundary right now?",
            grounded_entities={"boundary", "reservoir"},
            forbidden_solution_tokens={"set radius to 3.2"},
        )
        self.assertFalse(ok_leak)
        self.assertIn("solution_leakage_blocked", reason_leak)

        # 4. Non-Socratic statement (does not end with '?')
        ok_stmt, reason_stmt = ASTENeurosymbolicVerifier.verify_socratic_hint_gate(
            "Observe the reservoir and boundary carefully.",
            grounded_entities={"reservoir", "boundary"},
            forbidden_solution_tokens={"set radius to 3.2"},
        )
        self.assertFalse(ok_stmt)
        self.assertIn("non_socratic_form", reason_stmt)


if __name__ == "__main__":
    unittest.main()
