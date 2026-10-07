"""Optional real-data browser QA, followed by isolated network fault cases."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse


def verify(base_url, browser_path, output):
    from playwright.sync_api import sync_playwright, expect
    output.mkdir(parents=True, exist_ok=True)
    result = {"checked_at": datetime.now(timezone.utc).isoformat(), "base_url": base_url}
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, **({"executable_path": str(browser_path)} if browser_path else {}))
        page = browser.new_page(viewport={"width": 1440, "height": 1050})
        errors, responses = [], {}
        page.on("pageerror", lambda e: errors.append(str(e)))
        def capture(response):
            path = urlparse(response.url).path
            if path in ["/dashboard", "/backtest/summary", "/market-sentiment"] and response.status == 200:
                responses[path] = response.json()
        page.on("response", capture)
        page.goto(base_url.rstrip("/") + "/", wait_until="networkidle")
        dashboard = responses["/dashboard"]
        expect(page.get_by_role("region", name="Historical model snapshot")).to_contain_text(dashboard["snapshot"]["latest_signal_date"])
        expect(page.get_by_role("region", name="Historical model snapshot")).to_contain_text(str(dashboard["kpis"]["total_tickers"]) + " stocks and ETFs")
        expect(page.get_by_text("Mean model confidence", exact=True)).to_have_count(0)
        instruments = page.get_by_role("region", name="Find an instrument", exact=True)
        expect(instruments.get_by_role("row")).to_have_count(9)
        search = page.get_by_role("searchbox", name="Search ticker or company")
        search.fill("Apple")
        expect(instruments.get_by_role("row")).to_have_count(2)
        expect(instruments.get_by_role("link")).to_have_attribute("href", "/detail/AAPL")
        search.fill("zz-no-instrument")
        expect(instruments).to_contain_text("No instruments match your search.")
        search.fill("")
        instruments.get_by_role("button", name="Show all instruments").click()
        expect(instruments.get_by_role("row")).to_have_count(45)
        instruments.get_by_role("button", name="Show fewer").click()
        performance = page.get_by_role("region", name="Historical performance", exact=True)
        summary = responses["/backtest/summary"]
        assert summary["available"]
        expect(performance).to_contain_text(summary["evaluation"]["start"])
        for strategy in summary["strategies"]:
            expect(performance).to_contain_text(f'{strategy["metrics"]["total_return"] * 100:.2f}%')
            expect(performance).to_contain_text(f'{strategy["metrics"]["sharpe"]:.2f}')
        expect(performance).to_contain_text("not an independent test")
        assert "/market-sentiment" not in responses, "Archive should load only on request"
        page.evaluate("window.scrollTo(0, 0)")
        page.screenshot(path=str(output / "desktop.png"), full_page=True, animations="disabled")
        page.get_by_role("button", name="Saved news archive", exact=True).click()
        expect(page.get_by_role("region", name="Saved news archive")).to_contain_text("This is not a live news feed.")
        sentiment = responses["/market-sentiment"]
        expect(page.get_by_role("region", name="Saved news archive")).to_contain_text(sentiment["window_end"])
        page.get_by_role("button", name="Saved news archive", exact=True).click()
        for width in [390, 320]:
            page.set_viewport_size({"width": width, "height": 844})
            expect(page.get_by_role("navigation", name="Primary navigation")).not_to_be_visible()
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
            page.get_by_role("button", name="Open navigation menu").click()
            expect(page.get_by_role("navigation", name="Primary navigation")).to_be_visible()
            page.keyboard.press("Escape")
            expect(page.get_by_role("button", name="Open navigation menu")).to_be_focused()
            page.screenshot(path=str(output / f"mobile_{width}.png"), full_page=True, animations="disabled")
        page.route("**/dashboard", lambda r: r.fulfill(status=503, body="Unavailable"))
        page.reload(wait_until="networkidle")
        expect(instruments.get_by_role("alert")).to_be_visible()
        expect(instruments.get_by_role("row")).to_have_count(0)
        expect(page.get_by_role("region", name="Historical model snapshot")).to_contain_text("Unavailable")
        expect(performance).to_contain_text("SPY benchmark")
        page.unroute("**/dashboard")
        instruments.get_by_role("button", name="Try again").click()
        expect(instruments.get_by_role("row")).to_have_count(9)
        page.route("**/backtest/summary", lambda r: r.fulfill(status=503, body="Unavailable"))
        page.reload(wait_until="networkidle")
        expect(performance.get_by_role("alert")).to_be_visible()
        expect(performance.get_by_role("row")).to_have_count(0)
        expect(instruments.get_by_role("row")).to_have_count(9)
        page.unroute("**/backtest/summary")
        performance.get_by_role("button", name="Try again").click()
        expect(performance).to_contain_text("SPY benchmark")
        legacy = dict(dashboard)
        legacy.pop("snapshot")
        legacy.pop("instruments")
        page.route("**/dashboard", lambda r: r.fulfill(content_type="application/json", body=json.dumps(legacy)))
        page.reload(wait_until="networkidle")
        expect(page.get_by_role("region", name="Historical model snapshot")).to_contain_text("Date unavailable")
        expect(instruments).to_contain_text("backend version")
        page.unroute("**/dashboard")
        page.route("**/market-sentiment", lambda r: r.fulfill(content_type="application/json", body='{"article_count":0}'))
        page.get_by_role("button", name="Saved news archive", exact=True).click()
        expect(page.get_by_role("region", name="Saved news archive")).to_contain_text("No dated headlines")
        assert not errors, errors
        result.update(actual_dates_counts_returns=True, search_and_links=True,
                      mobile_widths_without_overflow=[390, 320], keyboard_menu=True,
                      independent_failure_retry_unknown_dates=True, archive_on_demand=True, uncaught_page_errors=errors)
        browser.close()
    (output / "verification.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:3000")
    parser.add_argument("--browser", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/research/dashboard_ui"))
    args = parser.parse_args()
    verify(args.base_url, args.browser, args.output)
