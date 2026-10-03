"""Render the Mermaid diagrams in docs/diagrams/*.mmd to PNG files next to them.

The PNGs are what docs/PROJECT_GUIDE.md shows, so the diagrams display everywhere (GitHub, the
VS Code preview, a PDF export); the .mmd files are the editable sources.

    .venv-train/Scripts/python.exe scripts/render_diagrams.py

Needs Playwright with Chromium (as scripts/screenshots.py does) and internet access: Mermaid is
loaded from jsDelivr.
"""

from __future__ import annotations

from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
DIAGRAMS = ROOT / "docs" / "diagrams"

PAGE = """<!doctype html>
<html><head><meta charset="utf-8">
<style>
  body { margin: 0; background: #ffffff; }
  #out { display: inline-block; padding: 28px; background: #ffffff; }
</style></head>
<body><div id="out"></div>
<script type="module">
  import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs";
  const font = '"Segoe UI", Inter, Arial, sans-serif';
  mermaid.initialize({
    startOnLoad: false, theme: "base", securityLevel: "loose", fontFamily: font,
    themeVariables: {
      fontFamily: font, fontSize: "15px",
      primaryColor: "#e7f0fd", primaryBorderColor: "#2a6fdb", primaryTextColor: "#1f2328",
      secondaryColor: "#f1f3f5", tertiaryColor: "#fafbfc", lineColor: "#5f6b7a",
      clusterBkg: "#fafbfc", clusterBorder: "#d0d7de", edgeLabelBackground: "#ffffff", titleColor: "#1f2328",
    },
    // wrappingWidth: lines break where the sources put <br/>, not at Mermaid's default 200 px
    flowchart: { useMaxWidth: false, htmlLabels: true, curve: "basis", padding: 12, nodeSpacing: 36, rankSpacing: 48,
                 wrappingWidth: 520 },
    timeline: { useMaxWidth: false },
    mindmap: { useMaxWidth: false, padding: 14 },
  });
  let n = 0;
  window.renderDiagram = async (code) => {
    const { svg } = await mermaid.render(`d${n++}`, code);
    document.getElementById("out").innerHTML = svg;
    await document.fonts.ready;
    return true;
  };
  window.mermaidReady = true;
</script></body></html>"""


def main() -> None:
    sources = sorted(DIAGRAMS.glob("*.mmd"))
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(device_scale_factor=2, viewport={"width": 2400, "height": 2000})
        page.set_content(PAGE)
        page.wait_for_function("window.mermaidReady === true", timeout=60_000)
        for src in sources:
            page.evaluate("code => window.renderDiagram(code)", src.read_text(encoding="utf-8"))
            out = src.with_suffix(".png")
            page.locator("#out").screenshot(path=str(out))
            print("rendered", out.relative_to(ROOT))
        browser.close()
    try:  # flat-colour diagrams lose nothing at 256 colours and shrink ~3x
        from PIL import Image
    except ImportError:
        return
    for src in sources:
        png = src.with_suffix(".png")
        Image.open(png).convert("RGB").quantize(colors=256, method=Image.Quantize.MEDIANCUT).save(png, optimize=True)


if __name__ == "__main__":
    main()
