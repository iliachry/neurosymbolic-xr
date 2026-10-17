#!/usr/bin/env python3
"""
Captures 1920x1080 3D WebGL frame sequences across all 11 shots (intro + Rooms 1..10)
from the ASTE VR WebXR client (/Users/iliaschrysovergis/GitHub/SeriousGames/webxr)
via deterministic Director Mode (`?director=1`) on Apple M4 Metal GPU.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import time
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "aste_frames"
SERIOUS_GAMES_DIR = Path("/Users/iliaschrysovergis/GitHub/SeriousGames")
WEBXR_DIR = SERIOUS_GAMES_DIR / "webxr"
FRAMES_PER_SHOT = 16
FPS = 24


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Copy additional reference captures from SeriousGames/docs/png
    docs_png = SERIOUS_GAMES_DIR / "docs" / "png"
    extra_pngs = [
        "educator_cockpit_live.png",
        "quest_live_screencap.png",
        "ASTEArchitecture.png",
        "Room10-Dashboard.png",
        "StoryboardAsAWhole.png",
    ]
    for name in extra_pngs:
        src = docs_png / name
        if src.exists():
            shutil.copy2(src, OUT_DIR / name)

    srv = subprocess.Popen(
        [
            "python3",
            "-m",
            "http.server",
            "8097",
            "--bind",
            "127.0.0.1",
            "--directory",
            str(WEBXR_DIR),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(0.6)

    manifest = {"fps": FPS, "frames_per_shot": FRAMES_PER_SHOT, "shots": []}
    t0 = time.time()
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                args=[
                    "--use-angle=metal",
                    "--enable-gpu",
                    "--ignore-gpu-blocklist",
                    "--disable-dev-shm-usage",
                ],
            )
            page = browser.new_page(
                viewport={"width": 1920, "height": 1080},
                device_scale_factor=1,
            )
            page.goto(
                "http://127.0.0.1:8097/index.html?director=1",
                wait_until="networkidle",
            )
            page.evaluate(
                f"window.director.enable({{width: 1920, height: 1080}}); window.director.fps = {FPS};"
            )
            shots_meta = page.evaluate(
                "window.director.shots.map((s, idx) => ({index: idx, id: s.id, room: s.room, duration: s.duration, tutorial: s.tutorial || []}))"
            )

            for s in shots_meta:
                idx = s["index"]
                shot_id = s["id"]
                total_frames = page.evaluate(f"window.director.beginShot({idx})")
                # Select FRAMES_PER_SHOT evenly spaced frame indices across [2, total_frames - 3]
                capture_indices = [
                    min(
                        total_frames - 1,
                        max(
                            1,
                            int(
                                round(
                                    2
                                    + (total_frames - 5)
                                    * (k / max(1, FRAMES_PER_SHOT - 1))
                                )
                            ),
                        ),
                    )
                    for k in range(FRAMES_PER_SHOT)
                ]

                current_f = 0
                saved_files = []
                for k, target_f in enumerate(capture_indices):
                    steps = max(1, target_f - current_f)
                    page.evaluate(
                        f"for (let s = 0; s < {steps}; s++) window.director.tick();"
                    )
                    current_f += steps
                    fname = f"{shot_id}_{k:02d}.jpg"
                    fpath = OUT_DIR / fname
                    buf = page.screenshot(type="jpeg", quality=90)
                    fpath.write_bytes(buf)
                    saved_files.append(fname)

                print(
                    f"  ✓ Captured {len(saved_files)} 1080p WebGL frames for shot '{shot_id}' (room {s['room']}, {total_frames} sim frames) [{time.time() - t0:.1f}s]"
                )
                manifest["shots"].append(
                    {
                        "index": idx,
                        "id": shot_id,
                        "room": s["room"],
                        "duration": s["duration"],
                        "files": saved_files,
                        "tutorial": s["tutorial"],
                    }
                )

            browser.close()
    finally:
        srv.terminate()

    manifest_path = OUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Done in {time.time() - t0:.1f}s -> {manifest_path}")


if __name__ == "__main__":
    main()
