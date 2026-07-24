"""Drives the live dashboard with Playwright to produce:
  - reports/figures/dashboard_screenshot.png  (hero shot for the README)
  - reports/figures/dashboard_demo.gif        (short interaction sequence)

Requires the Flask server to be running at localhost:8000 already.
"""
import sys
import time
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).parent.parent
FIG = ROOT / "reports" / "figures"
FIG.mkdir(exist_ok=True, parents=True)
FRAMES_DIR = Path("/tmp/dash_frames")
FRAMES_DIR.mkdir(exist_ok=True)

URL = "http://localhost:8000/"


def main():
    frames = []
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path="/opt/pw-browsers/chromium")
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.goto(URL, wait_until="networkidle", timeout=15000)
        page.wait_for_timeout(600)

        # pick a series with a visible story (Store 3 / Dept 5 -- used in the eval report too)
        page.select_option("#storeSel", "3")
        page.wait_for_timeout(150)
        page.select_option("#deptSel", "5")
        page.click("text=Load series")
        page.wait_for_timeout(700)

        frame1 = FRAMES_DIR / "f1.png"
        page.screenshot(path=str(frame1))
        frames.append(frame1)

        # hero screenshot (this exact state) for the README
        page.screenshot(path=str(FIG / "dashboard_screenshot.png"))

        # ask the agent something that shows off the anomaly + audit trail story
        page.fill("#chatInput", "Any anomalies for store 4 department 8?")
        frame2 = FRAMES_DIR / "f2.png"
        page.screenshot(path=str(frame2))
        frames.append(frame2)

        page.click("#chatlog ~ div button")
        page.wait_for_timeout(1200)
        frame3 = FRAMES_DIR / "f3.png"
        page.screenshot(path=str(frame3))
        frames.append(frame3)

        # expand the reasoning trace (audit log) -- the differentiator
        page.locator("#chatlog details summary").first.click(timeout=5000)
        page.wait_for_timeout(400)
        frame4 = FRAMES_DIR / "f4.png"
        page.screenshot(path=str(frame4))
        frames.append(frame4)

        # ask a second question to show conversational continuity
        page.fill("#chatInput", "What are the biggest declines year over year?")
        page.click("#chatlog ~ div button")
        page.wait_for_timeout(1200)
        frame5 = FRAMES_DIR / "f5.png"
        page.screenshot(path=str(frame5))
        frames.append(frame5)

        browser.close()

    # durations per frame (ms) -- linger longer on the "reveal" frames
    durations = [1600, 1200, 1600, 2400, 2600]
    imgs = [Image.open(f).convert("P", palette=Image.ADAPTIVE, colors=200) for f in frames]
    out_path = FIG / "dashboard_demo.gif"
    imgs[0].save(
        out_path, save_all=True, append_images=imgs[1:], duration=durations, loop=0, optimize=True,
    )
    print("Wrote", FIG / "dashboard_screenshot.png")
    print("Wrote", out_path, f"({len(frames)} frames)")


if __name__ == "__main__":
    main()
