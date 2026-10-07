"""Read-only forward page QA; additional scenarios use isolated browser fixtures."""
import argparse
import copy
import json
from pathlib import Path
import urllib.request

from playwright.sync_api import sync_playwright, expect

from forward_metrics import build_evidence
from market_calendar import schedule


def verify(base, api, executable, output):
    output.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(api + "/live-track-record?days=90", timeout=30) as response:
        real = json.load(response)
    assert real["available"], real.get("reason")
    findings = {"as_of": real["as_of"], "actual_resolved": real["total_resolved"],
                "actual_resolved_dates": real["evidence"]["resolved_forecast_dates"]}
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, executable_path=str(executable))
        page = browser.new_page(viewport={"width": 1440, "height": 1050})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(base + "/track-record", wait_until="networkidle")
        expect(page.get_by_role("heading", name="Forward results", exact=True)).to_be_visible()
        assessment = page.get_by_role("region", name="Forward evidence assessment")
        dates_count = real["evidence"]["resolved_forecast_dates"]
        if dates_count < real["evidence"]["intervals"]["minimum_resolved_forecast_dates"]:
            expect(assessment).to_contain_text(f"{dates_count} resolved forecast date")
        difference = real["evidence"]["metrics"]["accuracy_difference_pp"]
        if difference:
            expect(assessment).to_contain_text(f"{abs(difference):.1f} percentage points {'above' if difference > 0 else 'below'}")
        if real["evidence"]["collection"]["missing_sessions"]:
            expect(assessment).to_contain_text("Forecast collection has gaps")
        resolved = page.get_by_role("region", name="Resolved forecast records")
        expect(resolved.locator("tbody tr")).to_have_count(min(20, real["total_resolved"]))
        for i, row in enumerate(real["recent_resolved"]):
            cells = resolved.locator("tbody tr").nth(i).get_by_role("cell")
            expect(cells.nth(0)).to_contain_text(row["ticker"])
            expect(cells.nth(4)).to_have_text(f"{row['probability_up'] * 100:.1f}%")
            expect(cells.nth(6)).to_have_text("Correct" if row["correct"] else "Incorrect")
        with page.expect_download() as download_info:
            page.get_by_role("button", name="Export evaluation JSON").click()
        downloaded = download_info.value
        path = output / downloaded.suggested_filename
        downloaded.save_as(path)
        snapshot = json.loads(path.read_text())
        assert snapshot["evidence"] == real["evidence"]
        assert "recent" not in snapshot and "feature_values" not in json.dumps(snapshot)
        page.get_by_role("button", name="Pending", exact=True).click()
        pending = page.get_by_role("region", name="Pending forecast records")
        expect(pending.locator("tbody tr")).to_have_count(min(20, real["total_pending"]))
        page.get_by_role("button", name="Resolved", exact=True).click()
        page.get_by_text("Results by forecast date", exact=True).click()
        date_table = page.get_by_role("region", name="Forecast date results")
        expect(date_table.locator("tbody tr")).to_have_count(len(real["evidence"]["cohorts"]))
        page.get_by_text("Uncertainty and evaluation method", exact=True).click()
        expect(page.get_by_text("Uncertainty intervals are not available.", exact=True)).to_be_visible()
        page.screenshot(path=str(output / "actual_desktop.png"), full_page=True, animations="disabled")
        for width in [390, 320]:
            page.set_viewport_size({"width": width, "height": 844})
            page.wait_for_function("document.documentElement.scrollWidth <= innerWidth")
            date_table.focus()
            page.keyboard.press("ArrowRight")
            page.wait_for_function("document.querySelector('[aria-label=\"Forecast date results\"]').scrollLeft > 0")
            page.screenshot(path=str(output / f"actual_mobile_{width}.png"), full_page=True, animations="disabled")
        page.set_viewport_size({"width": 1440, "height": 1050})
        with page.expect_response(lambda r: "live-track-record?days=30" in r.url):
            page.get_by_label("Forecast window").select_option("30")
        expect(page.get_by_role("region", name="Resolved forecast records")).to_be_visible()
        with page.expect_download() as download_info:
            page.get_by_role("button", name="Export evaluation JSON").click()
        download_info.value.save_as(output / "window30.json")
        assert json.loads((output / "window30.json").read_text())["days"] == 30

        pattern = "**/live-track-record?days=*"
        def fixture(payload, status=200):
            page.route(pattern, lambda route: route.fulfill(status=status, content_type="application/json", body=json.dumps(payload)))
            page.reload(wait_until="networkidle")

        fixture({"detail": "Temporary outage"}, 503)
        expect(page.get_by_role("alert")).to_contain_text("could not be loaded")
        page.unroute(pattern)
        page.get_by_role("button", name="Retry forward results").click()
        expect(assessment).to_be_visible()
        fixture({"available": False, "reason": "Forward integrity check unavailable."})
        expect(page.get_by_text("Forward integrity check unavailable.", exact=True)).to_be_visible()
        page.unroute(pattern)

        empty = dict(real, total_logged=0, total_resolved=0, total_pending=0,
                     accuracy_pct=None, always_up_accuracy_pct=None, brier_score=None,
                     evidence=build_evidence([]), recent=[], recent_resolved=[], recent_pending=[])
        fixture(empty)
        expect(assessment).to_contain_text("0 resolved forecast dates")
        expect(page.get_by_role("region", name="Resolved forecast records")).to_contain_text("No records")
        page.unroute(pattern)

        old = copy.deepcopy(real)
        old["days"] = 90
        for key in ["evidence", "recent_resolved", "recent_pending"]:
            old.pop(key)
        fixture(old)
        expect(assessment).to_contain_text("Detailed date-level evidence is not available")
        expect(page.get_by_text("This backend returned a 90-day window", exact=False)).to_be_visible()
        page.set_viewport_size({"width": 320, "height": 844})
        page.wait_for_function("document.documentElement.scrollWidth <= innerWidth")
        page.unroute(pattern)

        dates = schedule("2024-01-01", "2024-05-01").index.strftime("%Y-%m-%d")
        rows = [{"ticker": ticker, "predicted_date": date, "resolved": True, "correct": True,
                 "predicted_signal": "BUY", "actual_signal": "BUY", "probability_up": .6,
                 "model_version": "fixture-model", "feature_version": "fixture-features", "calibration_method": "sigmoid"}
                for date in dates for ticker in ["A", "B"]]
        eligible = dict(real, evidence=build_evidence(rows))
        fixture(eligible)
        page.get_by_text("Uncertainty and evaluation method", exact=True).click()
        expect(page.get_by_role("region", name="Exploratory uncertainty intervals")).to_be_visible()
        page.unroute(pattern)
        rows.append(dict(rows[0], ticker="C", model_version="fixture-model-b"))
        mixed = dict(real, evidence=build_evidence(rows))
        fixture(mixed)
        expect(assessment).to_contain_text("combine different model or feature versions")
        page.unroute(pattern)

        page.reload(wait_until="networkidle")
        page.evaluate("() => { URL.createObjectURL = () => { throw new Error('fixture'); }; }")
        page.get_by_role("button", name="Export evaluation JSON").click()
        expect(page.get_by_role("alert")).to_contain_text("snapshot could not be downloaded")
        assert not errors, errors
        browser.close()
        findings.update(real_records_and_snapshot=True, window_control=True, mobile_widths=[390, 320],
                        keyboard_table_scroll=True, fixture_checks=["failure_retry", "unavailable", "empty", "older_backend", "eligible_intervals", "mixed_versions", "export_failure"],
                        uncaught_errors=errors)
    (output / "verification.json").write_text(json.dumps(findings, indent=2) + "\n")
    print(json.dumps(findings))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:3000")
    parser.add_argument("--api-url", default="http://127.0.0.1:8001")
    parser.add_argument("--browser", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/research/forward_evidence_20261007"))
    args = parser.parse_args()
    verify(args.base_url.rstrip("/"), args.api_url.rstrip("/"), args.browser, args.output)
