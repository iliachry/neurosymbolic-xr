"""
Studio Audio & Voiceover Generator for the Neurosymbolic XR Framework Video Showcase.
Generates:
1. Neural TTS voiceovers (en-US-AndrewNeural) for all 11 scenes of the main video
   plus dedicated voiceovers for the 60s, 30s, and 15s social cuts.
2. Procedural stereo electronic/ambient soundtrack (120 BPM intro/outro, 80 BPM ambient demo sections)
   with automatic sidechain ducking under voiceover and subtle scene-transition chimes.
3. Exact scene timing metadata (scene_timings.json) and SRT subtitles.
"""

import asyncio
import json
import math
import subprocess
import wave
from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
AUDIO_DIR = BASE_DIR / "audio"
DELIVERABLES_DIR = BASE_DIR.parent / "deliverables"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
DELIVERABLES_DIR.mkdir(parents=True, exist_ok=True)

VOICE = "en-US-AndrewNeural"

MAIN_SCENES = [
    {
        "id": 1,
        "key": "opening",
        "title": "Opening Title",
        "min_duration": 10.0,
        "lead_in": 1.0,
        "vo": "Introducing the Neurosymbolic XR Framework—a breakthrough in Extended Reality AI that combines the power of deep learning with logical reasoning."
    },
    {
        "id": 2,
        "key": "architecture",
        "title": "Architecture Overview",
        "min_duration": 17.0,
        "lead_in": 0.8,
        "vo": "The framework features a layered architecture. Sensors feed into neural networks for perception, which are grounded into symbolic representations for reasoning. This bidirectional bridge enables both bottom-up understanding and top-down validation."
    },
    {
        "id": 3,
        "key": "spatial_reasoning",
        "title": "System 1: Spatial Reasoning",
        "min_duration": 22.0,
        "lead_in": 0.8,
        "vo": "System One: Spatial Reasoning. We capture the scene with 3D Gaussian Splatting, segment objects with SAM 2, and ground them to language with CLIP. The result? A queryable scene graph that understands both geometry and semantics."
    },
    {
        "id": 4,
        "key": "hybrid_slam",
        "title": "System 2: Hybrid SLAM",
        "min_duration": 22.0,
        "lead_in": 0.8,
        "vo": "System Two: Hybrid SLAM. Visual-inertial tracking provides precise localization at 60 frames per second. But unlike traditional SLAM, we maintain semantic memory—objects are remembered across sessions, building a persistent understanding of the environment."
    },
    {
        "id": 5,
        "key": "training_assistant",
        "title": "System 3: VR Training Assistant",
        "min_duration": 22.0,
        "lead_in": 0.8,
        "vo": "System Three: VR Training Assistant. Hand and gaze tracking feed into Prolog-based procedural reasoning. Each action is validated against logical preconditions and constraints, providing real-time feedback to guide users through complex procedures."
    },
    {
        "id": 6,
        "key": "asset_generator",
        "title": "System 4: Neural-Symbolic Asset Generator",
        "min_duration": 22.0,
        "lead_in": 0.8,
        "vo": "System Four: Neural-Symbolic Asset Generator. Natural language descriptions are combined with geometric constraints. Z3 validates feasibility, then Stable Diffusion generates the asset. The result? AI-created content that satisfies both aesthetic and functional requirements."
    },
    {
        "id": 7,
        "key": "poetics_integration",
        "title": "Poetics Platform Integration",
        "min_duration": 18.0,
        "lead_in": 0.8,
        "vo": "All four systems integrate seamlessly with the Poetics VR platform. A unified API manages sessions, tracks users, stores semantic memory in Supabase, and provides real-time events. Everything you need for intelligent VR experiences."
    },
    {
        "id": 8,
        "key": "performance_metrics",
        "title": "Performance Metrics",
        "min_duration": 14.0,
        "lead_in": 0.8,
        "vo": "Performance is critical for XR. The framework achieves sub-20 millisecond tracking latency, real-time scene understanding, and fast asset generation—all while running complex AI models."
    },
    {
        "id": 9,
        "key": "tech_stack",
        "title": "Technical Stack",
        "min_duration": 15.0,
        "lead_in": 0.8,
        "vo": "Built on cutting-edge technologies: 3D Gaussian Splatting for rendering, SAM 2 for segmentation, CLIP for grounding, and Stable Diffusion for generation. Symbolic reasoning with RDF, Prolog, and Z3. All integrated with Supabase and OpenXR."
    },
    {
        "id": 10,
        "key": "call_to_action",
        "title": "Call to Action",
        "min_duration": 16.0,
        "lead_in": 0.8,
        "vo": "Ready to build intelligent VR experiences? The Neurosymbolic XR Framework is open source and production-ready. Visit the GitHub repository for the source code, architecture documentation, and interactive demo. Transform your VR applications with the power of neurosymbolic AI."
    },
    {
        "id": 11,
        "key": "closing",
        "title": "Closing",
        "min_duration": 8.0,
        "lead_in": 0.8,
        "vo": "Neurosymbolic XR Framework. The future of intelligent Extended Reality."
    }
]

SOCIAL_CUTS = {
    "60s": {
        "target_duration": 60.0,
        "segments": [
            {"scene_id": 1, "duration": 6.0, "vo": "Introducing the Neurosymbolic XR Framework—bridging deep neural perception with symbolic logical reasoning for Extended Reality."},
            {"scene_id": 2, "duration": 7.0, "vo": "A bidirectional grounding bridge connects 3D neural sensors directly to formal knowledge graphs and logic solvers."},
            {"scene_id": 3, "duration": 9.0, "vo": "System One fuses 3D Gaussian Splatting, SAM 2, and CLIP into a queryable RDF spatial scene graph."},
            {"scene_id": 4, "duration": 8.0, "vo": "System Two delivers 16-millisecond Hybrid SLAM with persistent cross-session semantic memory."},
            {"scene_id": 5, "duration": 8.0, "vo": "System Three validates VR hand and gaze procedures in real time using Prolog inference."},
            {"scene_id": 6, "duration": 9.0, "vo": "And System Four combines Z3 geometric constraint solving with 3D diffusion for verified asset generation."},
            {"scene_id": 7, "duration": 7.0, "vo": "Fully integrated with the Poetics VR platform and Supabase for real-time venue intelligence."},
            {"scene_id": 11, "duration": 6.0, "vo": "Neurosymbolic XR Framework. Open-source and ready for the future of spatial computing."}
        ]
    },
    "30s": {
        "target_duration": 30.0,
        "segments": [
            {"scene_id": 1, "duration": 4.5, "vo": "Meet the Neurosymbolic XR Framework: where neural perception meets symbolic logic in VR."},
            {"scene_id": 3, "duration": 5.5, "vo": "Capture 3D scenes with Gaussian Splatting, SAM 2, and CLIP into live RDF scene graphs."},
            {"scene_id": 4, "duration": 5.5, "vo": "Track at 60 FPS with Hybrid SLAM and persistent semantic memory."},
            {"scene_id": 5, "duration": 5.5, "vo": "Validate VR training with Prolog rules and generate constrained 3D assets with Z3."},
            {"scene_id": 7, "duration": 4.5, "vo": "Powered by the Poetics VR platform with sub-20 millisecond latency."},
            {"scene_id": 11, "duration": 4.5, "vo": "Neurosymbolic XR. Build intelligent worlds today."}
        ]
    },
    "15s": {
        "target_duration": 15.0,
        "segments": [
            {"scene_id": 1, "duration": 3.5, "vo": "Neurosymbolic XR Framework: Neural perception meets symbolic reasoning."},
            {"scene_id": 3, "duration": 4.0, "vo": "3D Gaussian Splatting, SAM 2, CLIP, Prolog, and Z3 in one unified engine."},
            {"scene_id": 8, "duration": 4.0, "vo": "Four real-time AI systems built for the Poetics VR platform."},
            {"scene_id": 11, "duration": 3.5, "vo": "Experience the future of intelligent XR."}
        ]
    }
}


def run_tts(text: str, out_mp3: Path, rate: str = "+0%"):
    if out_mp3.exists() and out_mp3.stat().st_size > 1000:
        return
    cmd = [
        "uvx", "edge-tts",
        "--voice", VOICE,
        f"--rate={rate}",
        "--text", text,
        "--write-media", str(out_mp3)
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def mp3_to_wav_array(mp3_path: Path, sr: int = 44100) -> np.ndarray:
    wav_path = mp3_path.with_suffix(".wav")
    subprocess.run([
        "ffmpeg", "-y", "-i", str(mp3_path),
        "-ac", "1", "-ar", str(sr), "-f", "wav", str(wav_path)
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    with wave.open(str(wav_path), "rb") as wf:
        frames = wf.readframes(wf.getnframes())
        samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    # Trim leading/trailing silence below -45dB
    thresh = 0.006
    idx = np.where(np.abs(samples) > thresh)[0]
    if len(idx) > 0:
        start = max(0, idx[0] - int(0.05 * sr))
        end = min(len(samples), idx[-1] + int(0.1 * sr))
        samples = samples[start:end]
    return samples


def synthesize_background_music(total_duration: float, scene_boundaries: list[float], sr: int = 44100) -> np.ndarray:
    """
    Synthesizes a lush, modern stereo tech-showcase electronic ambient soundtrack.
    - Intro & Outro: 120 BPM subtle electronic pulse + shimmering neural-gold chord progression
    - Middle sections: 80 BPM warm ambient pads + gentle harmonic arpeggio
    - Scene transitions: subtle crystalline harmonic chime
    """
    n_samples = int(total_duration * sr)
    t = np.linspace(0, total_duration, n_samples, endpoint=False)
    left = np.zeros(n_samples, dtype=np.float32)
    right = np.zeros(n_samples, dtype=np.float32)

    # Chord progression in D minor / F major9 (futuristic, warm, Apple/NVIDIA keynote vibe)
    # Chords change every 6 seconds
    chords = [
        [146.83, 220.00, 261.63, 329.63, 440.00],  # Dm9 (D3, A3, C4, E4, A4)
        [116.54, 174.61, 220.00, 261.63, 392.00],  # Bbmaj9 (Bb2, F3, A3, C4, G4)
        [174.61, 261.63, 329.63, 392.00, 523.25],  # Fmaj7 (F3, C4, E4, G4, C5)
        [130.81, 196.00, 246.94, 293.66, 392.00],  # Cadd9/G (C3, G3, B3, D4, G4)
    ]

    chord_len = 6.0
    n_chords = int(math.ceil(total_duration / chord_len))

    for c_idx in range(n_chords):
        c_start = c_idx * chord_len
        c_end = min(total_duration, (c_idx + 1) * chord_len)
        i0 = int(c_start * sr)
        i1 = int(c_end * sr)
        if i1 <= i0:
            break
        ct = t[i0:i1] - c_start
        dur = c_end - c_start
        # Smooth crossfade envelope
        env = np.sin(np.pi * ct / dur) ** 0.7
        freqs = chords[c_idx % len(chords)]

        # Warm stereo pad with slight detune
        pad_l = np.zeros_like(ct)
        pad_r = np.zeros_like(ct)
        for k, f in enumerate(freqs):
            detune = 1.0 + 0.0018 * (k - 2)
            pad_l += (0.22 / len(freqs)) * np.sin(2 * np.pi * f * ct + k * 0.4)
            pad_r += (0.22 / len(freqs)) * np.sin(2 * np.pi * (f * detune) * ct + k * 0.9)
            # Soft octave shimmer
            if k >= 2:
                pad_l += 0.04 * np.sin(2 * np.pi * (f * 2.0) * ct) * (0.5 + 0.5 * np.sin(2 * np.pi * 0.25 * ct))
                pad_r += 0.04 * np.sin(2 * np.pi * (f * 2.001) * ct) * (0.5 + 0.5 * np.cos(2 * np.pi * 0.25 * ct))

        # Sub-bass foundation
        bass_f = freqs[0] * 0.5
        bass = 0.16 * np.sin(2 * np.pi * bass_f * ct) * env

        left[i0:i1] += (pad_l * env + bass).astype(np.float32)
        right[i0:i1] += (pad_r * env + bass).astype(np.float32)

    # Gentle arpeggiator (120 BPM sixteenth/eighth pulse in intro/outro, 80 BPM in middle)
    arp_notes = [293.66, 349.23, 440.00, 523.25, 659.25, 523.25, 440.00, 349.23]
    step_sec = 0.25  # 120 BPM eighth notes
    n_steps = int(total_duration / step_sec)
    for s_idx in range(n_steps):
        s_start = s_idx * step_sec
        i0 = int(s_start * sr)
        i1 = min(n_samples, i0 + int(0.42 * sr))
        if i1 <= i0:
            break
        st = np.arange(i1 - i0, dtype=np.float32) / sr
        f = arp_notes[s_idx % len(arp_notes)]
        # Softer in the middle demo section, more energetic at intro (first 12s) and outro (last 20s)
        is_intro_outro = (s_start < 12.0) or (s_start > total_duration - 22.0)
        amp = 0.065 if is_intro_outro else 0.028
        pluck_env = np.exp(-st * 7.5) * (1.0 - np.exp(-st * 120.0))
        note = amp * np.sin(2 * np.pi * f * st) * pluck_env
        pan = 0.5 + 0.35 * math.sin(s_idx * 0.7)
        left[i0:i1] += note * pan
        right[i0:i1] += note * (1.0 - pan)

    # Subtle crystalline chime at scene transitions
    for b_time in scene_boundaries:
        i0 = int(b_time * sr)
        i1 = min(n_samples, i0 + int(1.6 * sr))
        if i1 <= i0 or i0 < 0:
            continue
        bt = np.arange(i1 - i0, dtype=np.float32) / sr
        chime_env = np.exp(-bt * 3.2) * (1.0 - np.exp(-bt * 80.0))
        chime_l = 0.08 * (np.sin(2 * np.pi * 880.0 * bt) + 0.5 * np.sin(2 * np.pi * 1318.5 * bt)) * chime_env
        chime_r = 0.08 * (np.sin(2 * np.pi * 880.5 * bt) + 0.5 * np.sin(2 * np.pi * 1760.0 * bt)) * chime_env
        left[i0:i1] += chime_l
        right[i0:i1] += chime_r

    # Master fade in (1.0s) and fade out (2.0s)
    fade_in = min(n_samples, int(1.0 * sr))
    fade_out = min(n_samples, int(2.0 * sr))
    left[:fade_in] *= np.linspace(0, 1, fade_in, dtype=np.float32)
    right[:fade_in] *= np.linspace(0, 1, fade_in, dtype=np.float32)
    left[-fade_out:] *= np.linspace(1, 0, fade_out, dtype=np.float32)
    right[-fade_out:] *= np.linspace(1, 0, fade_out, dtype=np.float32)

    return np.stack([left, right], axis=1)


def save_stereo_wav(path: Path, stereo: np.ndarray, sr: int = 44100):
    clipped = np.clip(stereo, -0.98, 0.98)
    pcm = (clipped * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def format_srt_time(seconds: float) -> str:
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    return f"{hrs:02d}:{mins:02d}:{secs:02d},{ms:03d}"


def build_main_showcase_audio(sr: int = 44100):
    print("Generating neural voiceover for Main Showcase (11 scenes)...")
    scene_records = []
    vo_segments = []
    current_time = 0.0

    for scene in MAIN_SCENES:
        mp3_path = AUDIO_DIR / f"main_scene_{scene['id']:02d}.mp3"
        run_tts(scene["vo"], mp3_path, rate="+2%")
        vo_samples = mp3_to_wav_array(mp3_path, sr=sr)
        vo_dur = len(vo_samples) / sr
        lead_in = scene["lead_in"]
        tail_out = 1.1
        scene_dur = max(scene["min_duration"], round(lead_in + vo_dur + tail_out, 2))

        scene_info = {
            "id": scene["id"],
            "key": scene["key"],
            "title": scene["title"],
            "start": round(current_time, 3),
            "end": round(current_time + scene_dur, 3),
            "duration": round(scene_dur, 3),
            "vo_start": round(current_time + lead_in, 3),
            "vo_end": round(current_time + lead_in + vo_dur, 3),
            "vo_duration": round(vo_dur, 3),
            "vo_text": scene["vo"]
        }
        scene_records.append(scene_info)
        vo_segments.append((scene_info["vo_start"], vo_samples))
        print(f"  Scene {scene['id']:02d} ({scene['title']}): {scene_info['start']:.1f}s -> {scene_info['end']:.1f}s (VO: {vo_dur:.1f}s, Total: {scene_dur:.1f}s)")
        current_time += scene_dur

    total_duration = round(current_time, 3)
    n_samples = int(total_duration * sr)
    vo_track = np.zeros(n_samples, dtype=np.float32)
    duck_envelope = np.ones(n_samples, dtype=np.float32)

    for vo_start, samples in vo_segments:
        i0 = int(vo_start * sr)
        i1 = min(n_samples, i0 + len(samples))
        vo_track[i0:i1] = samples[:i1 - i0] * 0.92
        # Duck background music to 32% during active voiceover with smooth 0.35s ramp
        pad = int(0.35 * sr)
        d0 = max(0, i0 - pad)
        d1 = min(n_samples, i1 + pad)
        duck_envelope[d0:d1] = 0.32

    # Smooth ducking envelope with moving average
    kernel_size = int(0.25 * sr)
    kernel = np.ones(kernel_size, dtype=np.float32) / kernel_size
    duck_envelope = np.convolve(duck_envelope, kernel, mode="same")

    boundaries = [s["start"] for s in scene_records]
    music_stereo = synthesize_background_music(total_duration, boundaries, sr=sr)
    mixed = np.zeros_like(music_stereo)
    mixed[:, 0] = music_stereo[:, 0] * duck_envelope + vo_track
    mixed[:, 1] = music_stereo[:, 1] * duck_envelope + vo_track

    wav_out = AUDIO_DIR / "main_showcase_mix.wav"
    save_stereo_wav(wav_out, mixed, sr=sr)

    # Write SRT file
    srt_path = DELIVERABLES_DIR / "neurosymbolic_xr_showcase.srt"
    srt_lines = []
    for s in scene_records:
        srt_lines.append(f"{s['id']}")
        srt_lines.append(f"{format_srt_time(s['vo_start'])} --> {format_srt_time(s['vo_end'])}")
        srt_lines.append(s["vo_text"])
        srt_lines.append("")
    srt_path.write_text("\n".join(srt_lines), encoding="utf-8")

    return {
        "total_duration": total_duration,
        "audio_path": str(wav_out),
        "scenes": scene_records
    }


def build_social_cut_audio(cut_name: str, cut_spec: dict, sr: int = 44100):
    print(f"Generating neural voiceover & soundtrack for {cut_name} social cut...")
    target_duration = cut_spec["target_duration"]
    segments_out = []
    vo_segments = []
    current_time = 0.0

    # First generate TTS for all segments so we can scale durations accurately to hit exact target_duration
    raw_tts = []
    for idx, seg in enumerate(cut_spec["segments"]):
        mp3_path = AUDIO_DIR / f"cut_{cut_name}_seg_{idx+1:02d}.mp3"
        rate = "+8%" if cut_name in ("30s", "15s") else "+4%"
        run_tts(seg["vo"], mp3_path, rate=rate)
        samples = mp3_to_wav_array(mp3_path, sr=sr)
        raw_tts.append((seg, samples, len(samples) / sr))

    # Allocate exact segment durations summing to target_duration
    total_min_needed = sum(max(seg["duration"], vo_dur + 0.5) for seg, _, vo_dur in raw_tts)
    scale = target_duration / total_min_needed if total_min_needed > 0 else 1.0

    for idx, (seg, samples, vo_dur) in enumerate(raw_tts):
        base_dur = max(seg["duration"], vo_dur + 0.5) * scale
        # Ensure VO fits inside segment
        if vo_dur + 0.3 > base_dur:
            base_dur = vo_dur + 0.35
        seg_dur = round(base_dur, 3)
        if idx == len(raw_tts) - 1:
            seg_dur = round(max(vo_dur + 0.35, target_duration - current_time), 3)
        lead_in = max(0.15, min(0.45, (seg_dur - vo_dur) * 0.35))
        seg_info = {
            "index": idx + 1,
            "scene_id": seg["scene_id"],
            "start": round(current_time, 3),
            "end": round(current_time + seg_dur, 3),
            "duration": seg_dur,
            "vo_start": round(current_time + lead_in, 3),
            "vo_end": round(current_time + lead_in + vo_dur, 3),
            "vo_text": seg["vo"]
        }
        segments_out.append(seg_info)
        vo_segments.append((seg_info["vo_start"], samples))
        current_time += seg_dur

    actual_duration = round(current_time, 3)
    n_samples = int(actual_duration * sr)
    vo_track = np.zeros(n_samples, dtype=np.float32)
    duck_envelope = np.ones(n_samples, dtype=np.float32)

    for vo_start, samples in vo_segments:
        i0 = int(vo_start * sr)
        i1 = min(n_samples, i0 + len(samples))
        vo_track[i0:i1] = samples[:i1 - i0] * 0.92
        pad = int(0.25 * sr)
        duck_envelope[max(0, i0 - pad):min(n_samples, i1 + pad)] = 0.32

    kernel_size = int(0.18 * sr)
    kernel = np.ones(kernel_size, dtype=np.float32) / kernel_size
    duck_envelope = np.convolve(duck_envelope, kernel, mode="same")

    boundaries = [s["start"] for s in segments_out]
    music_stereo = synthesize_background_music(actual_duration, boundaries, sr=sr)
    mixed = np.zeros_like(music_stereo)
    mixed[:, 0] = music_stereo[:, 0] * duck_envelope + vo_track
    mixed[:, 1] = music_stereo[:, 1] * duck_envelope + vo_track

    wav_out = AUDIO_DIR / f"cut_{cut_name}_mix.wav"
    save_stereo_wav(wav_out, mixed, sr=sr)
    print(f"  Cut {cut_name}: Total duration = {actual_duration:.2f}s")

    return {
        "total_duration": actual_duration,
        "audio_path": str(wav_out),
        "segments": segments_out
    }


def main():
    main_meta = build_main_showcase_audio()
    cuts_meta = {}
    for cut_name, spec in SOCIAL_CUTS.items():
        cuts_meta[cut_name] = build_social_cut_audio(cut_name, spec)

    all_timings = {
        "main": main_meta,
        "cuts": cuts_meta
    }
    timings_path = BASE_DIR / "scene_timings.json"
    timings_path.write_text(json.dumps(all_timings, indent=2), encoding="utf-8")
    print(f"\nSaved complete timings & audio metadata to {timings_path}")


if __name__ == "__main__":
    main()
