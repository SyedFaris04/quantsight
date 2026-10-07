"""Read-only real-data route and XAI QA; isolated portfolio/error fixtures."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from playwright.sync_api import sync_playwright, expect, TimeoutError as BrowserTimeout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--browser", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:3000")
    parser.add_argument("--output", type=Path, default=Path("data/research/ui_xai_20261007/screens"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    findings = {"checked_at": datetime.now(timezone.utc).isoformat(), "routes": []}
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, executable_path=str(args.browser))
        page = browser.new_page(viewport={"width": 1440, "height": 1050})
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        for route in ["/", "/market", "/detail/AAPL", "/compare", "/backtesting", "/track-record", "/portfolio", "/game", "/leaderboard", "/settings", "/login"]:
            page.goto(args.base_url.rstrip("/") + route, wait_until="domcontentloaded")
            expect(page.locator("main h1")).to_be_visible(timeout=30000)
            # Data provider failure may be displayed; no claim that a provider was healthy.
            try: page.wait_for_load_state("networkidle", timeout=15000)
            except BrowserTimeout: pass
            page.evaluate("window.scrollTo(0, 0)")
            page.screenshot(path=str(args.output / ((route.replace("/", "_").strip("_") or "dashboard") + ".png")), full_page=True, animations="disabled")
            findings["routes"].append({"route": route, "heading": page.locator("main h1").inner_text()})
            page.set_viewport_size({"width": 390, "height": 844})
            # Responsive charts resize asynchronously through ResizeObserver.
            page.wait_for_function("document.documentElement.scrollWidth <= innerWidth", timeout=5000)
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth"), route
            page.set_viewport_size({"width": 1440, "height": 1050})
        page.goto(args.base_url.rstrip("/") + "/compare", wait_until="networkidle")
        study = page.get_by_role("region", name="Explanation method comparison")
        study.get_by_role("button", name="Open XAI study").click()
        expect(study).to_contain_text("LIME (historical neighborhood)", timeout=15000)
        expect(study).to_contain_text("2024-12-20")
        expect(study).to_contain_text("not prediction accuracy")
        table = study.get_by_role("region", name="XAI method measurements")
        expect(table.get_by_role("row")).to_have_count(3)
        study.get_by_label("Comparison model").select_option("lstm_finance")
        expect(study).to_contain_text("Day-window occlusion")
        expect(study).to_contain_text("Input window: 2024-12-06 to 2024-12-19")
        study.get_by_label("Comparison instrument").select_option("MSFT")
        expect(study).to_contain_text("MSFT")
        study.scroll_into_view_if_needed()
        page.screenshot(path=str(args.output / "xai_lstm.png"), full_page=True, animations="disabled")
        for width in [390, 320]:
            page.set_viewport_size({"width": width, "height": 844})
            page.wait_for_function("document.documentElement.scrollWidth <= innerWidth", timeout=5000)
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        page.route("**/research/xai-comparison", lambda r: r.fulfill(status=503, body="Unavailable"))
        study.get_by_role("button", name="Hide XAI study").click()
        study.get_by_role("button", name="Open XAI study").click()
        expect(study.get_by_role("alert")).to_be_visible()
        expect(table).to_have_count(0)
        page.unroute("**/research/xai-comparison")
        study.get_by_role("button", name="Retry XAI comparison").click()
        expect(study).to_contain_text("LIME (historical neighborhood)")
        # Legacy metrics failure must not hide independent prepared explanations.
        page.route("**/metrics", lambda r: r.fulfill(status=503, body="Unavailable"))
        page.reload(wait_until="networkidle")
        study.get_by_role("button", name="Open XAI study").click()
        expect(study).to_contain_text("LIME (historical neighborhood)")
        page.unroute("**/metrics")
        # Ephemeral browser context only: no user account, holdings or DB are modified.
        page.evaluate("localStorage.setItem('nuroquant_portfolio', JSON.stringify([{ticker:'AAPL',shares:10,buyPrice:100}]))")
        page.route("**/realtime/AAPL", lambda r: r.fulfill(content_type="application/json", body=json.dumps({"price":110,"source":"live","change_pct":10})))
        page.goto(args.base_url.rstrip("/") + "/portfolio", wait_until="networkidle")
        expect(page.get_by_text("Daily quote P&L", exact=True).locator("..")).to_contain_text("$100.00")
        expect(page.get_by_text("BUY MORE", exact=True)).to_have_count(0)
        expect(page.locator("main")).to_contain_text("2024-12-20")
        page.route("**/realtime/AAPL", lambda r: r.fulfill(content_type="application/json", body=json.dumps({"price":100,"source":"historical","change_pct":10})))
        page.reload(wait_until="networkidle")
        expect(page.get_by_text("Daily quote P&L", exact=True).locator("..")).to_contain_text("Unavailable")
        assert not errors, errors
        findings.update(xai_model_instrument_selection=True, xai_failure_retry_independent_metrics=True,
                        xai_mobile_no_overflow=[390,320], isolated_portfolio_math_and_old_quote=True,
                        uncaught_page_errors=errors)
        browser.close()
    (args.output / "workspace_verification.json").write_text(json.dumps(findings, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(findings))


if __name__ == "__main__":
    main()
