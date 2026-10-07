"""Capture dashboard screenshots (for the README and slides) and fail loudly on any browser error.

Start the server first (``python main.py serve --no-browser``), then:
    python scripts/screenshots.py [--url http://127.0.0.1:8000] [--theme light|dark] [--wait 20]

Needs Playwright (``pip install playwright && python -m playwright install chromium``).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "docs" / "screenshots"
TABS = ["radar", "index", "stress", "watch", "analyze", "results"]


def main() -> int:
    from playwright.sync_api import sync_playwright

    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--theme", choices=["light", "dark"], default="light")
    parser.add_argument("--wait", type=float, default=15.0, help="seconds to let the replay run before capturing")
    parser.add_argument("--width", type=int, default=1440)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": args.width, "height": 900}, color_scheme=args.theme, device_scale_factor=1.5)
        page.on("console", lambda m: errors.append(f"console.{m.type}: {m.text}") if m.type in ("error", "warning") else None)
        page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
        page.goto(args.url, wait_until="networkidle")
        time.sleep(args.wait)
        for tab in TABS:
            page.click(f'button[data-tab="{tab}"]')
            if tab == "analyze":
                page.click("#btn-analyze")
            time.sleep(4 if tab in ("analyze", "results") else 2.5)
            path = OUT / f"{tab}_{args.theme}.png"
            page.screenshot(path=str(path), full_page=True)
            print("saved", path.relative_to(OUT.parents[1]))
        if args.theme == "light":  # the risk memo is a print page with one (light) theme
            runs = page.evaluate("fetch('/api/stress/runs').then(r => r.json())")
            if runs:
                worst = max(runs, key=lambda r: r["trigger"]["impact_score"] if r.get("trigger") else 0)
                page.goto(f"{args.url}/api/stress/runs/{worst['run_id']}/memo", wait_until="networkidle")
                # With a local language model running, the first request queues the summary draft (~30 s).
                deadline = time.time() + 180
                while "drafting one" in page.content() and time.time() < deadline:
                    time.sleep(5)
                    page.reload(wait_until="networkidle")
                print("memo summary:", "language model" if "<h2>Executive summary</h2>" in page.content() else "template")
                path = OUT / "memo.png"
                page.screenshot(path=str(path), full_page=True)
                print("saved", path.relative_to(OUT.parents[1]))
        browser.close()
    for err in errors:
        print(err, file=sys.stderr)
    return 1 if any(e.startswith(("pageerror", "console.error")) for e in errors) else 0


if __name__ == "__main__":
    sys.exit(main())
