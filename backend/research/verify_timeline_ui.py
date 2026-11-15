"""Browser QA for the dashboard training-history switch and the /timeline page."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlparse

SECTIONS = ["What each version learned from", "Prediction quality on the same test days",
            "When a version says 60%, how often was it right?", "How often the two versions agree",
            "Trading simulation", "What each version relies on", "Crisis years, one year at a time",
            "Limits of this comparison"]


def verify(base_url, browser_path, output):
    from playwright.sync_api import sync_playwright, expect
    output.mkdir(parents=True, exist_ok=True)
    base = base_url.rstrip("/")
    result = {"checked_at": datetime.now(timezone.utc).isoformat(), "base_url": base_url}
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, **({"executable_path": str(browser_path)} if browser_path else {}))
        page = browser.new_page(viewport={"width": 1440, "height": 1050})
        errors, dashboards = [], {}
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("console", lambda m: m.type == "error" and errors.append(m.text))

        def capture(response):
            url = urlparse(response.url)
            if url.path.endswith("/dashboard") and response.status == 200:
                dashboards[parse_qs(url.query).get("timeline", ["2015"])[0]] = response.json()
        page.on("response", capture)
        page.goto(base + "/", wait_until="networkidle")
        group = page.get_by_role("radiogroup", name="Training history of the models shown")
        radio = lambda year: group.get_by_role("radio", name=f"Trained from {year}")
        expect(radio(2015)).to_have_attribute("aria-checked", "true")
        snapshot = page.get_by_role("region", name="Historical model snapshot")
        performance = page.get_by_role("region", name="Historical performance", exact=True)
        expect(snapshot).to_contain_text("models trained from 2015")
        expect(performance).to_contain_text(f'{dashboards["2015"]["backtest"]["ensemble"]["total_return"] * 100:.2f}%')
        with page.expect_response(lambda r: "/dashboard?timeline=2005" in r.url) as response:
            radio(2005).click()
        dashboards["2005"] = response.value.json()
        assert "timeline=2005" in page.url, page.url
        expect(snapshot).to_contain_text("models trained from 2005")
        expect(performance).to_contain_text(f'{dashboards["2005"]["backtest"]["ensemble"]["total_return"] * 100:.2f}%')
        result["kpis"] = {t: dashboards[t]["kpis"] for t in ["2015", "2005"]}
        page.screenshot(path=str(output / "dashboard_2005.png"), full_page=True, animations="disabled")
        page.reload(wait_until="networkidle")
        expect(radio(2005)).to_have_attribute("aria-checked", "true")
        radio(2005).focus()
        page.keyboard.press("ArrowLeft")
        expect(radio(2015)).to_have_attribute("aria-checked", "true")
        expect(radio(2015)).to_be_focused()
        page.wait_for_load_state("networkidle")
        assert "timeline=2015" in page.url, page.url
        page.goto(base + "/?timeline=1999", wait_until="networkidle")
        expect(radio(2015)).to_have_attribute("aria-checked", "true")
        expect(page.get_by_role("alert")).to_have_count(0)
        page.goto(base + "/timeline", wait_until="networkidle")
        for title in SECTIONS:
            expect(page.get_by_role("heading", name=title, exact=True)).to_be_visible()
        page.get_by_label("Model").first.select_option("lstm_sentiment")
        page.screenshot(path=str(output / "timeline_desktop.png"), full_page=True, animations="disabled")
        page.set_viewport_size({"width": 390, "height": 844})
        for path, name in [("/?timeline=2005", "dashboard_390.png"), ("/timeline", "timeline_390.png")]:
            page.goto(base + path, wait_until="networkidle")
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth"), path
            page.screenshot(path=str(output / name), full_page=True, animations="disabled")
        assert not errors, errors
        result.update(switch_changes_url_and_content=True, reload_keeps_choice=True, arrow_keys=True,
                      bad_param_falls_back=True, timeline_sections=len(SECTIONS), mobile_without_overflow=[390],
                      page_errors=errors)
        browser.close()
    (output / "verification.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:3000")
    parser.add_argument("--browser", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/research/timeline_ui"))
    args = parser.parse_args()
    verify(args.base_url, args.browser, args.output)
