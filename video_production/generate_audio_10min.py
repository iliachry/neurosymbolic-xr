#!/usr/bin/env python3
"""
10-Minute (600.0s) Audio & Voiceover Generator for Neurosymbolic XR x ASTE VR Showcase.

Generates:
1. 20 neural voiceover segments via edge-tts (en-US-AndrewNeural)
2. 600.0-second stereo ambient/electronic soundtrack with sidechain ducking
3. Exact frame-aligned SRT subtitles (deliverables/neurosymbolic_xr_10min_showcase.srt)
4. Timing manifest (video_production/scene_timings_10min.json)
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import wave
import numpy as np

ROOT = Path(__file__).resolve().parent
AUDIO_DIR = ROOT / "audio_10min"
DELIVERABLES_DIR = ROOT.parent / "deliverables"
VOICE = "en-US-AndrewNeural"
SAMPLE_RATE = 44100
TOTAL_DURATION = 600.0

SCENES_10MIN = [
    {
        "id": 1,
        "key": "cold_open",
        "title": "Scene 1: Neurosymbolic XR × ASTE VR",
        "duration": 25.0,
        "shot": "intro",
        "text": (
            "Welcome to the ten-minute technical showcase of the Neurosymbolic XR Framework "
            "integrated with ASTE VR at aste.iliachry.gr. Developed under H.F.R.I. Grant 26237 "
            "EduGenZ, this architecture unites neural spatial perception and formal symbolic verification "
            "with ten embodied WebXR escape chambers grounded in the Arnold and Wade systems thinking taxonomy."
        ),
    },
    {
        "id": 2,
        "key": "evidence_contract",
        "title": "Scene 2: Formal Neurosymbolic Evidence Contract",
        "duration": 25.0,
        "shot": "intro",
        "text": (
            "Purely neural VR assistants hallucinate ungrounded hints and lack verifiable state provenance. "
            "Our shared neurosymbolic contract transforms sensor observations and authoritative WebXR engine events "
            "into typed predicates with explicit units, timestamps, and three-valued truth semantics—true, false, "
            "or unknown—ensuring every room verdict is backed by a formal proof trace."
        ),
    },
    {
        "id": 3,
        "key": "spatial_and_slam",
        "title": "Scene 3: Systems 1 & 2 — Spatial Reasoning & Hybrid SLAM",
        "duration": 30.0,
        "shot": "r1",
        "text": (
            "In physical Location-Based VR deployments, System One reconstructs scenes with 3D Gaussian Splatting, "
            "isolates objects via SAM 2, and aligns open-vocabulary concepts with CLIP to populate an RDF and OWL "
            "knowledge graph. System Two pairs ORB-SLAM3 geometric tracking with semantic loop closure, maintaining "
            "persistent object memory across multi-room sessions."
        ),
    },
    {
        "id": 4,
        "key": "training_and_z3",
        "title": "Scene 4: Systems 3 & 4 — Prolog Assistant & Z3 Asset Generator",
        "duration": 30.0,
        "shot": "r2",
        "text": (
            "System Three combines temporal hand-pose recognition with SWI-Prolog inference to verify procedural "
            "prerequisites and diagnose root-cause errors in real time. System Four couples diffusion-based 3D "
            "synthesis with the Z3 SMT constraint solver, guaranteeing that generated props satisfy bounding-box, "
            "support-surface, and collision invariants before entering the scene."
        ),
    },
    {
        "id": 5,
        "key": "aste_constellation",
        "title": "Scene 5: Enter ASTE VR — Macrocosm Curriculum Constellation",
        "duration": 25.0,
        "shot": "intro",
        "text": (
            "Let us now enter ASTE VR at aste.iliachry.gr, captured via deterministic Director Mode on the Three.js "
            "WebXR engine. The Macrocosm Constellation Hub maps ten progressive escape chambers to Arnold and Wade's "
            "2015 and 2017 systems thinking skills, pairing first-person 6DoF locomotion with tactical radar, "
            "causal loop diagrams, and the Garmin virtual smartwatch."
        ),
    },
    {
        "id": 6,
        "key": "room_1_boundary",
        "title": "Scene 6: Room 1 — Forgotten Boundary (Skills 2.1–2.2)",
        "duration": 35.0,
        "shot": "r1",
        "text": (
            "Room One, The Forgotten Boundary, targets Skills 2.1 and 2.2: recognizing systems and maintaining "
            "boundaries. The learner rotates the central Vernier Handwheel to expand the boundary radius from "
            "2.2 to 3.2 meters, enclosing relevant elements—reservoir, inflow pipe, soil moisture, and garden bed—"
            "while excluding environmental noise. Our symbolic verifier enforces boundary integrity I of B at least "
            "0.70 alongside balanced inflow and outflow at 70 units, rejecting counterexamples where the tank is "
            "filled under an invalid boundary."
        ),
    },
    {
        "id": 7,
        "key": "room_2_scale",
        "title": "Scene 7: Room 2 — Hall of Scale & Quantity (Skill 2.3)",
        "duration": 35.0,
        "shot": "r2",
        "text": (
            "Room Two, The Hall of Scale and Quantity, operationalizes Skill 2.3: differentiating and quantifying "
            "elements. Three deceptive pedestals display the numeral ten across incompatible physical quantity kinds: "
            "discrete item count, continuous length in meters, and duration in seconds. The neurosymbolic verifier "
            "enforces dimensional homogeneity—two quantities are comparable only if they share quantity kind and "
            "compatible units—and requires explicit rejection of false equivalences such as comparing 24 square "
            "centimeters of area against 20 centimeters of length."
        ),
    },
    {
        "id": 8,
        "key": "room_3_relationships",
        "title": "Scene 8: Room 3 — River of Relationships (Skills 3.1–3.2)",
        "duration": 35.0,
        "shot": "r3",
        "text": (
            "Room Three, The River of Relationships, trains Skills 3.1 and 3.2: identifying and characterizing "
            "causal relationships. Using the Relation Weaver console, the player inserts the intermediate mediator "
            "node and assigns signed causal polarities matching the partial-derivative Jacobian of the underlying "
            "system. Positive inflow links accumulate stock, whereas the outflow-to-stock edge carries negative "
            "polarity. Any reversed arrow, missing mediator, or unpredicted pulse trajectory is deterministically "
            "flagged by the symbolic rule engine."
        ),
    },
    {
        "id": 9,
        "key": "room_4_feedback",
        "title": "Scene 9: Room 4 — Echo Forest: Feedback & Delays (Skills 3.3–3.4)",
        "duration": 35.0,
        "shot": "r4",
        "text": (
            "Room Four, Echo Forest, addresses Skills 3.3 and 3.4: feedback loop polarity and propagation delays. "
            "The learner inspects closed causal cycles where loop polarity equals the product of signed edges—plus "
            "one for runaway reinforcing loops and minus one for stabilizing balancing loops. Crucially, the "
            "Rotary Delay Dial introduces phase lag tau. The verifier requires the learner to wait across the full "
            "delay horizon after intervening, rejecting premature over-corrections that trigger oscillation."
        ),
    },
    {
        "id": 10,
        "key": "room_5_stocks_flows",
        "title": "Scene 10: Room 5 — Valley of Time: Stocks & Flows (Skills 4.1–4.3)",
        "duration": 35.0,
        "shot": "r5",
        "text": (
            "Room Five, The Valley of Time, targets Skills 4.1 through 4.3: dynamic accumulation and trajectory "
            "prediction. Through the Chronos Lens, the learner integrates discrete stock conservation—S at k plus one "
            "equals S at k plus delta-t times net inflow minus outflow. Our verification contract mandates that the "
            "learner record a forward prediction before revealing the future trajectory, while Astra probes a classic "
            "misconception: equal inflow and outflow imply a constant stock level, not an empty reservoir."
        ),
    },
    {
        "id": 11,
        "key": "room_6_perspectives",
        "title": "Scene 11: Room 6 — Prism Mountain: Perspectives (Skill 1.1)",
        "duration": 35.0,
        "shot": "r6",
        "text": (
            "Room Six, Prism Mountain, cultivates Skill 1.1: considering multiple perspectives. Rotating the "
            "tri-faceted Perspective Prism reframes the chamber across three distinct stakeholder lenses: Pupil "
            "engagement, Teacher curriculum alignment, and Designer 6DoF ergonomic comfort. Simply cycling the "
            "three optical lenses is insufficient for symbolic mastery; the verifier requires supported concerns "
            "for all three stakeholders and an explicit trade-off justification balancing competing objectives."
        ),
    },
    {
        "id": 12,
        "key": "room_7_models",
        "title": "Scene 12: Room 7 — Garden of Models (Skill 1.5)",
        "duration": 35.0,
        "shot": "r7",
        "text": (
            "Room Seven, The Garden of Models, trains Skill 1.5: mental models and abstraction. Using the Model "
            "Seed Kit, the learner prunes spurious environmental noise branches—such as ambient chimes and cloud "
            "color—while preserving causally essential variables like inflow, outflow, and soil absorption. The "
            "symbolic evaluator scores model adequacy via held-out prediction error plus a complexity penalty "
            "lambda, rejecting any model that prunes a causally essential mechanism."
        ),
    },
    {
        "id": 13,
        "key": "room_8_leverage",
        "title": "Scene 13: Room 8 — Seed of Change: Leverage Points (Skill 4.4)",
        "duration": 35.0,
        "shot": "r8",
        "text": (
            "Room Eight, The Seed of Change, focuses on Skill 4.4: high-leverage structural interventions. Learners "
            "compare candidate policy actions across a multi-step horizon under matched initial conditions and budget "
            "constraints. Rather than rewarding low-impact symptomatic buffers, the Z3-backed verifier identifies the "
            "optimal structural intervention that modifies feedback gain rules, maximizing long-horizon stability net "
            "of intervention cost."
        ),
    },
    {
        "id": 14,
        "key": "room_9_emergence",
        "title": "Scene 14: Room 9 — Emergence Chamber (Skills 1.1–4.4 Transfer)",
        "duration": 35.0,
        "shot": "r9",
        "text": (
            "Room Nine, The Emergence Chamber, serves as an integrated transfer assessment across all systems "
            "thinking skills. Managing a coupled multi-district urban water network with hidden aquifer delays, "
            "learners discover how locally greedy district withdrawals trigger counter-intuitive macro depletion. "
            "Verification requires keeping total extraction within aquifer capacity, achieving macro Gini stability "
            "above 0.80, and comparing the coordinated policy against an uncoordinated ablation baseline."
        ),
    },
    {
        "id": 15,
        "key": "room_10_observatory",
        "title": "Scene 15: Room 10 — Reflection Observatory & Provenance",
        "duration": 35.0,
        "shot": "r10",
        "text": (
            "Room Ten, The Reflection Observatory, consolidates metacognitive evidence across the entire expedition. "
            "Every reflection claim submitted by the learner must resolve to verified session event identifiers in "
            "the provenance log and document at least one belief revision triggered by a disconfirming observation. "
            "Critically, our audit contract preserves scientific rigor: in-game chamber completion is recorded as "
            "behavioral trace evidence, distinct from externally validated psychometric competency."
        ),
    },
    {
        "id": 16,
        "key": "garmin_biometrics",
        "title": "Scene 16: Garmin BLE Virtual Wrist Watch & Biometric Provenance",
        "duration": 30.0,
        "shot": "r4",
        "text": (
            "Throughout all ten chambers, the Virtual Wrist Watch integrates physiological telemetry from the Garmin "
            "BLE Heart Rate Service over WebSocket. To prevent fabricated physiological claims, our BiometricPacket "
            "verifier enforces four explicit provenance states: live, simulated, stale beyond five seconds, and "
            "unavailable. Heart-rate variability RMSSD is computed strictly from adjacent RR intervals between 250 "
            "and 2000 milliseconds, never bridging across dropped beats."
        ),
    },
    {
        "id": 17,
        "key": "astra_and_cockpit",
        "title": "Scene 17: Symbolically Gated Astra Companion & Educator Cockpit",
        "duration": 30.0,
        "shot": "r10",
        "text": (
            "When a learner stalls or violates a room invariant, the Astra Socratic companion generates scaffolding "
            "gated by our symbolic verifier. Every candidate utterance must reference a grounded entity in the active "
            "room's scene graph, pass a whole-word non-leakage check blocking direct solution tokens, and end as a "
            "Socratic question. Simultaneously, the Educator Cockpit and Poetics venue adapter maintain isolated "
            "per-session telemetry streams for teachers and multi-pod classrooms."
        ),
    },
    {
        "id": 18,
        "key": "tests_and_roadmap",
        "title": "Scene 18: Counterexample Verification, Benchmarks & 5-Phase Roadmap",
        "duration": 30.0,
        "shot": "r8",
        "text": (
            "All ten room verifiers, biometric provenance transitions, and Socratic gate branches are verified by "
            "automated unit tests in test_aste_integration.py, rejecting every synthetic counterexample in under two "
            "milliseconds. Our five-phase continuation roadmap extends this foundation with the shared aste_systems "
            "OWL ontology, real-time WebSocket grounding, Z3 adaptive puzzle scheduling, GX10 Blackwell LLM Socratic "
            "generation, and multi-pod Poetics venue trials."
        ),
    },
    {
        "id": 19,
        "key": "verified_references",
        "title": "Scene 19: Verified Primary Literature & Live URL Manifest",
        "duration": 15.0,
        "shot": "intro",
        "text": (
            "Every URL and publication cited in this showcase is live and verified online: the open-source repository "
            "at github.com/iliachry/neurosymbolic-xr, the live WebXR platform at aste.iliachry.gr, and foundational "
            "peer-reviewed DOIs by Arnold and Wade."
        ),
    },
    {
        "id": 20,
        "key": "closing_card",
        "title": "Scene 20: Neurosymbolic XR × ASTE VR — Closing",
        "duration": 10.0,
        "shot": "intro",
        "text": (
            "Neurosymbolic XR and ASTE VR: where neural perception meets formal symbolic verification for embodied "
            "systems thinking."
        ),
    },
]


def run_tts(text: str, mp3_path: Path, wav_path: Path) -> float:
    if not wav_path.exists():
        subprocess.run(
            [
                "uvx",
                "edge-tts",
                "--voice",
                VOICE,
                "--rate=+4%",
                "--text",
                text,
                "--write-media",
                str(mp3_path),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(mp3_path),
                "-af",
                "silenceremove=start_periods=1:start_silence=0.05:start_threshold=-45dB,areverse,silenceremove=start_periods=1:start_silence=0.08:start_threshold=-45dB,areverse",
                "-ar",
                str(SAMPLE_RATE),
                "-ac",
                "1",
                str(wav_path),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    with wave.open(str(wav_path), "rb") as wf:
        return wf.getnframes() / float(wf.getframerate())


def load_wav_mono(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wf:
        raw = wf.readframes(wf.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def synthesize_music(duration: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Synthesize a rich 600s stereo electronic/ambient soundtrack."""
    n = int(round(duration * sr))
    t = np.arange(n, dtype=np.float32) / sr
    left = np.zeros(n, dtype=np.float32)
    right = np.zeros(n, dtype=np.float32)

    # Chord progression cycling every 16 seconds (A minor -> F major -> C major -> G major)
    chords = [
        (110.0, 220.0, 261.63, 329.63),
        (87.31, 174.61, 220.0, 261.63),
        (130.81, 261.63, 329.63, 392.00),
        (98.00, 196.00, 246.94, 293.66),
    ]
    chord_len = int(4.0 * sr)
    for start in range(0, n, chord_len):
        end = min(n, start + chord_len)
        seg_t = t[start:end] - t[start]
        seg_len = (end - start) / sr
        env = np.minimum(1.0, seg_t / 0.4) * np.minimum(1.0, (seg_len - seg_t) / 0.4)
        c = chords[(start // chord_len) % len(chords)]
        pad_l = (
            0.35 * np.sin(2 * math.pi * c[0] * seg_t)
            + 0.25 * np.sin(2 * math.pi * c[1] * seg_t + 0.2)
            + 0.20 * np.sin(2 * math.pi * c[2] * seg_t)
            + 0.15 * np.sin(2 * math.pi * c[3] * seg_t + 0.5)
        ) * env
        pad_r = (
            0.35 * np.sin(2 * math.pi * c[0] * seg_t)
            + 0.25 * np.sin(2 * math.pi * c[1] * seg_t - 0.2)
            + 0.20 * np.sin(2 * math.pi * c[2] * seg_t + 0.3)
            + 0.15 * np.sin(2 * math.pi * c[3] * seg_t - 0.4)
        ) * env
        left[start:end] += pad_l * 0.14
        right[start:end] += pad_r * 0.14

    # Subtle rhythmic arpeggio pulse (100 BPM = 0.6s per beat)
    arp_notes = [220.0, 261.63, 329.63, 392.0, 440.0, 392.0, 329.63, 261.63]
    step_samples = int(0.3 * sr)
    for idx, start in enumerate(range(0, n, step_samples)):
        end = min(n, start + step_samples)
        st = t[start:end] - t[start]
        freq = arp_notes[idx % len(arp_notes)]
        pluck = np.sin(2 * math.pi * freq * st) * np.exp(-st * 7.5) * 0.045
        if idx % 2 == 0:
            left[start:end] += pluck
            right[start:end] += pluck * 0.6
        else:
            left[start:end] += pluck * 0.6
            right[start:end] += pluck

    # Smooth fade in/out
    fade_in = min(n, int(2.0 * sr))
    fade_out = min(n, int(3.0 * sr))
    left[:fade_in] *= np.linspace(0.0, 1.0, fade_in, dtype=np.float32)
    right[:fade_in] *= np.linspace(0.0, 1.0, fade_in, dtype=np.float32)
    left[-fade_out:] *= np.linspace(1.0, 0.0, fade_out, dtype=np.float32)
    right[-fade_out:] *= np.linspace(1.0, 0.0, fade_out, dtype=np.float32)
    return np.stack([left, right], axis=1)


def fmt_srt_time(sec: float) -> str:
    ms = int(round(sec * 1000))
    h = ms // 3600000
    ms %= 3600000
    m = ms // 60000
    ms %= 60000
    s = ms // 1000
    ms %= 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def main() -> None:
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    DELIVERABLES_DIR.mkdir(parents=True, exist_ok=True)

    total_samples = int(round(TOTAL_DURATION * SAMPLE_RATE))
    music = synthesize_music(TOTAL_DURATION, SAMPLE_RATE)
    vo_track = np.zeros(total_samples, dtype=np.float32)
    duck_gain = np.ones(total_samples, dtype=np.float32)

    cursor = 0.0
    timings = []
    srt_lines = []

    for s in SCENES_10MIN:
        sid = s["id"]
        dur = float(s["duration"])
        start = cursor
        end = cursor + dur
        mp3_p = AUDIO_DIR / f"scene_{sid:02d}.mp3"
        wav_p = AUDIO_DIR / f"scene_{sid:02d}.wav"

        raw_vo_dur = run_tts(s["text"], mp3_p, wav_p)
        # If raw VO exceeds (dur - 1.2s), time-stretch slightly with ffmpeg atempo so it fits comfortably
        max_vo = dur - 1.4
        if raw_vo_dur > max_vo:
            ratio = min(1.35, max(1.02, raw_vo_dur / max_vo))
            fitted_wav = AUDIO_DIR / f"scene_{sid:02d}_fit.wav"
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(wav_p),
                    "-filter:a",
                    f"atempo={ratio:.4f}",
                    "-ar",
                    str(SAMPLE_RATE),
                    "-ac",
                    "1",
                    str(fitted_wav),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            samples = load_wav_mono(fitted_wav)
        else:
            samples = load_wav_mono(wav_p)

        vo_dur = len(samples) / float(SAMPLE_RATE)
        vo_start = start + 0.6
        vo_end = min(end - 0.3, vo_start + vo_dur)

        idx0 = int(round(vo_start * SAMPLE_RATE))
        idx1 = min(total_samples, idx0 + len(samples))
        vo_track[idx0:idx1] += samples[: idx1 - idx0] * 0.92

        # Duck background music during voiceover
        pad = int(0.25 * SAMPLE_RATE)
        d0 = max(0, idx0 - pad)
        d1 = min(total_samples, idx1 + pad)
        duck_gain[d0:d1] = 0.28

        print(
            f"  ✓ Scene {sid:02d} [{start:6.1f}s - {end:6.1f}s] ({dur:4.1f}s) | VO: {vo_dur:4.1f}s | {s['title']}"
        )

        timings.append(
            {
                "id": sid,
                "key": s["key"],
                "title": s["title"],
                "shot": s["shot"],
                "start": round(start, 3),
                "end": round(end, 3),
                "duration": round(dur, 3),
                "vo_start": round(vo_start, 3),
                "vo_end": round(vo_end, 3),
                "vo_duration": round(vo_dur, 3),
                "vo_text": s["text"],
            }
        )

        srt_lines.append(
            f"{sid}\n{fmt_srt_time(vo_start)} --> {fmt_srt_time(vo_end)}\n{s['text']}\n"
        )
        cursor = end

    # Smooth ducking envelope
    kernel = np.ones(int(0.18 * SAMPLE_RATE), dtype=np.float32)
    kernel /= len(kernel)
    duck_smooth = np.convolve(duck_gain, kernel, mode="same")

    stereo = np.zeros((total_samples, 2), dtype=np.float32)
    stereo[:, 0] = np.clip(music[:, 0] * duck_smooth + vo_track, -0.98, 0.98)
    stereo[:, 1] = np.clip(music[:, 1] * duck_smooth + vo_track, -0.98, 0.98)

    mix_wav = AUDIO_DIR / "showcase_10min_mix.wav"
    pcm = (stereo * 32767.0).astype(np.int16)
    with wave.open(str(mix_wav), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())

    srt_path = DELIVERABLES_DIR / "neurosymbolic_xr_10min_showcase.srt"
    srt_path.write_text("\n".join(srt_lines), encoding="utf-8")

    manifest_path = ROOT / "scene_timings_10min.json"
    manifest_path.write_text(
        json.dumps(
            {
                "total_duration": TOTAL_DURATION,
                "audio_path": str(mix_wav),
                "srt_path": str(srt_path),
                "scenes": timings,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nGenerated 600.0s master audio -> {mix_wav}")
    print(f"Generated SRT subtitles      -> {srt_path}")
    print(f"Generated scene timings      -> {manifest_path}")


if __name__ == "__main__":
    main()
