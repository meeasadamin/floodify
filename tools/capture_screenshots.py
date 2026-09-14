"""Capture README screenshots from a running dashboard and check browser-only behaviour.

Uses the locally installed Microsoft Edge (or Chrome via --channel chrome), so no
browser download is needed. Start the app first:

    streamlit run app.py --server.headless true
    python tools/capture_screenshots.py --url http://localhost:8501

Besides screenshots, this verifies in a real browser what Streamlit's AppTest
cannot: the sidebar can be collapsed AND reopened.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from playwright.sync_api import Page, expect, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
TIMEOUT_MS = 90_000


def settle(page: Page) -> None:
    """Wait until Streamlit has finished the current script run."""
    page.wait_for_timeout(800)
    expect(page.locator('[data-testid="stStatusWidget"]')).to_have_count(0, timeout=TIMEOUT_MS)
    page.wait_for_timeout(1200)


def shoot(page: Page, out_dir: Path, name: str) -> None:
    path = out_dir / f"{name}.png"
    page.screenshot(path=str(path))
    print(f"saved {path.relative_to(ROOT)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8501")
    parser.add_argument("--out_dir", default=str(ROOT / "docs" / "screenshots"))
    parser.add_argument("--channel", default="msedge")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(channel=args.channel, headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1100}, color_scheme="dark")
        page.goto(args.url, wait_until="networkidle")
        expect(page.get_by_text("Floodify").first).to_be_visible(timeout=TIMEOUT_MS)
        settle(page)

        page.get_by_role("button", name="TILE-01", exact=True).click()
        expect(page.get_by_text("Hand-labeled ground truth").first).to_be_visible(timeout=TIMEOUT_MS)
        settle(page)
        shoot(page, out_dir, "01_flood_assessment")

        page.get_by_role("button", name="SYNTH-05", exact=True).click()
        expect(page.get_by_text("UNFAMILIAR INPUT").first).to_be_visible(timeout=TIMEOUT_MS)
        settle(page)
        shoot(page, out_dir, "02_input_guard")

        page.get_by_role("tab", name="Priority Triage").click()
        tiles = sorted((ROOT / "samples").glob("*.*"))
        page.locator('[data-testid="stFileUploaderDropzoneInput"]').last.set_input_files([str(t) for t in tiles])
        expect(page.locator('[data-testid="stProgress"]')).to_have_count(0, timeout=TIMEOUT_MS)
        # All tabs are in the DOM; only the active tab's dataframe is visible.
        expect(page.locator('[data-testid="stDataFrame"]:visible')).to_have_count(1, timeout=TIMEOUT_MS)
        settle(page)
        shoot(page, out_dir, "03_priority_triage")

        page.get_by_role("tab", name="Model Performance").click()
        expect(page.get_by_text("Balanced accuracy").first).to_be_visible(timeout=TIMEOUT_MS)
        settle(page)
        shoot(page, out_dir, "04_model_performance")

        page.get_by_role("tab", name="System & Provenance").click()
        expect(page.get_by_text("Checkpoint provenance").first).to_be_visible(timeout=TIMEOUT_MS)
        settle(page)
        shoot(page, out_dir, "05_system_provenance")

        # Regression check for the hidden-toolbar bug: collapse, then the reopen control must be visible.
        # The collapse button is only revealed while the pointer is over the sidebar.
        page.locator('[data-testid="stSidebar"]').hover()
        page.locator('[data-testid="stSidebarCollapseButton"] button').click()
        reopen = page.locator('[data-testid="stExpandSidebarButton"]')
        expect(reopen).to_be_visible(timeout=10_000)
        reopen.click()
        expect(page.get_by_role("button", name="TILE-01", exact=True)).to_be_visible(timeout=10_000)
        print("sidebar collapse/reopen: OK")

        browser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
