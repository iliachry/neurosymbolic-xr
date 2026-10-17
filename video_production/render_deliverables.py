"""
High-Speed Headless Chromium + FFmpeg Video Renderer for Neurosymbolic XR Framework.
Renders:
1. deliverables/thumbnail_1920x1080.png
2. deliverables/scene_previews/scene_01..11.png
3. deliverables/neurosymbolic_xr_showcase.mp4 (Full 11-scene showcase, 1080p)
4. deliverables/neurosymbolic_xr_60s_cut.mp4 (60s social cut, 1080p)
5. deliverables/neurosymbolic_xr_30s_cut.mp4 (30s teaser cut, 1080p)
6. deliverables/neurosymbolic_xr_15s_cut.mp4 (15s stories/reels cut, 1080p)
"""

import asyncio
import base64
import json
import subprocess
import time
from pathlib import Path
from playwright.async_api import async_playwright

BASE_DIR = Path(__file__).resolve().parent
DELIVERABLES_DIR = BASE_DIR.parent / "deliverables"
PREVIEWS_DIR = DELIVERABLES_DIR / "scene_previews"
DELIVERABLES_DIR.mkdir(parents=True, exist_ok=True)
PREVIEWS_DIR.mkdir(parents=True, exist_ok=True)

FPS = 24  # Cinema 24fps for smooth motion graphics and fast rendering


async def render_video_mode(page, mode: str, duration: float, audio_path: str, out_mp4: Path):
    total_frames = int(round(duration * FPS))
    print(f"\n[Renderer] Starting '{mode}' -> {out_mp4.name} ({duration:.2f}s, {total_frames} frames @ {FPS}fps)...")
    t0 = time.time()

    cmd = [
        "ffmpeg", "-y",
        "-f", "image2pipe",
        "-vcodec", "mjpeg",
        "-r", str(FPS),
        "-i", "-",
        "-i", audio_path,
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "192k",
        "-movflags", "+faststart",
        "-shortest",
        str(out_mp4)
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    batch_size = 24  # 1 second of frames per batch
    for start_f in range(0, total_frames, batch_size):
        end_f = min(total_frames, start_f + batch_size)
        # Render batch inside Chromium and return concatenated base64 or array of data URLs
        b64_list = await page.evaluate(
            """async ({ startF, endF, fps, mode }) => {
                const canvas = document.getElementById('stage');
                const out = [];
                for (let f = startF; f < endF; f++) {
                    const t = f / fps;
                    window.renderAtTime(t, mode);
                    const dataUrl = canvas.toDataURL('image/jpeg', 0.92);
                    out.push(dataUrl.split(',')[1]);
                }
                return out;
            }""",
            {"startF": start_f, "endF": end_f, "fps": FPS, "mode": mode}
        )
        for b64_item in b64_list:
            proc.stdin.write(base64.b64decode(b64_item))

        if (end_f % (FPS * 15) == 0) or (end_f == total_frames):
            elapsed = time.time() - t0
            fps_rate = end_f / max(0.01, elapsed)
            print(f"  -> {end_f}/{total_frames} frames ({end_f / total_frames * 100:.0f}%) | {fps_rate:.1f} fps")

    proc.stdin.close()
    proc.wait()
    elapsed = time.time() - t0
    size_mb = out_mp4.stat().st_size / (1024 * 1024)
    print(f"[Renderer] ✓ Finished {out_mp4.name} in {elapsed:.1f}s ({size_mb:.2f} MB)")


async def main():
    timings_path = BASE_DIR / "scene_timings.json"
    timings = json.loads(timings_path.read_text(encoding="utf-8"))

    studio_url = f"file://{BASE_DIR / 'studio.html'}?headless=1"

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(viewport={"width": 1920, "height": 1080}, device_scale_factor=1)
        page = await context.new_page()
        await page.goto(studio_url)

        # Inject exact timings from scene_timings.json
        await page.evaluate("data => window.setTimelineData(data)", timings)

        # 1. Render Thumbnail (1920x1080 PNG)
        thumb_path = DELIVERABLES_DIR / "thumbnail_1920x1080.png"
        await page.evaluate("window.renderAtTime(0, 'thumbnail')")
        await page.locator("#stage").screenshot(path=str(thumb_path))
        print(f"[Renderer] ✓ Saved YouTube Thumbnail: {thumb_path}")

        # 2. Render Scene Preview Stills for all 11 scenes
        for s in timings["main"]["scenes"]:
            mid_t = s["start"] + s["duration"] * 0.62
            await page.evaluate(f"window.renderAtTime({mid_t}, 'main')")
            still_path = PREVIEWS_DIR / f"scene_{s['id']:02d}_{s['key']}.png"
            await page.locator("#stage").screenshot(path=str(still_path))
        print(f"[Renderer] ✓ Saved all 11 scene preview stills to {PREVIEWS_DIR}")

        # 3. Render 15s, 30s, 60s cuts and Main Showcase
        for cut_key, filename in [
            ("15s", "neurosymbolic_xr_15s_cut.mp4"),
            ("30s", "neurosymbolic_xr_30s_cut.mp4"),
            ("60s", "neurosymbolic_xr_60s_cut.mp4"),
        ]:
            cut_meta = timings["cuts"][cut_key]
            await render_video_mode(
                page,
                mode=cut_key,
                duration=cut_meta["total_duration"],
                audio_path=cut_meta["audio_path"],
                out_mp4=DELIVERABLES_DIR / filename
            )

        main_meta = timings["main"]
        await render_video_mode(
            page,
            mode="main",
            duration=main_meta["total_duration"],
            audio_path=main_meta["audio_path"],
            out_mp4=DELIVERABLES_DIR / "neurosymbolic_xr_showcase.mp4"
        )

        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
