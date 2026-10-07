"""Browser QA against read-only model-health data and isolated UI fixtures."""
import argparse
import copy
import json
import urllib.request
from pathlib import Path
from playwright.sync_api import sync_playwright, expect


def key(group):
    return "|".join(group[field] for field in ("date", "model_version", "feature_version", "calibration_method"))


def verify(base, api, browser_path, output):
    output.mkdir(parents=True, exist_ok=True)
    real = json.load(urllib.request.urlopen(api + "/model-health?days=30", timeout=45))
    assert real["available"] and real["groups"], real.get("reason")
    (output / "actual-report.json").write_text(json.dumps(real, indent=2))
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, executable_path=str(browser_path))
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(base + "/model-health", wait_until="networkidle")
        expect(page.get_by_role("heading", name="Model health", exact=True)).to_be_visible()
        assessment = page.get_by_role("region", name="Monitoring assessment")
        expect(assessment).to_be_visible()
        expect(page.get_by_role("region", name="Collection freshness")).to_contain_text(f"{real['latest_session_coverage']}/{real['expected_tickers']}")
        select = page.get_by_label("Forecast date and version")
        expect(select.get_by_role("option")).to_have_count(len(real["groups"]))
        if len(real["groups"]) > 1:
            select.select_option(key(real["groups"][-1]))
            expect(select).to_have_value(key(real["groups"][-1]))
            page.get_by_role("button", name="Refresh monitoring").click()
            expect(assessment).to_be_visible()
            expect(select).to_have_value(key(real["groups"][-1]))
        select.select_option(key(real["groups"][0]))
        calibrated = page.get_by_role("region", name="Calibrated probability of UP")
        expect(calibrated).to_contain_text(f"largest tied group: {real['groups'][0]['calibrated_probability']['largest_tie_count']}/{real['groups'][0]['calibrated_probability']['n']}")
        with page.expect_download() as info:
            page.get_by_role("button", name="Export monitoring JSON").click()
        info.value.save_as(output / "export30.json")
        exported = json.loads((output / "export30.json").read_text())
        assert exported["groups"] == real["groups"] and exported["reference"] == real["reference"]
        assert exported["days"] == 30 and "feature_values" not in json.dumps(exported)
        page.get_by_text("Input ranges by feature", exact=True).click()
        table = page.get_by_role("region", name="Input range checks")
        expect(table.locator("tbody tr")).to_have_count(len(real["groups"][0]["input_ranges"]))
        for i, row in enumerate(real["groups"][0]["input_ranges"]):
            cells = table.locator("tbody tr").nth(i).get_by_role("cell")
            expect(cells.nth(0)).to_have_text(row["feature"])
            expect(cells.nth(2)).to_have_text(str(row["outside"]))
        page.evaluate("window.scrollTo(0, 0)")
        page.screenshot(path=str(output / "desktop.png"), full_page=True)
        for width in (390, 320):
            page.set_viewport_size({"width": width, "height": 844})
            page.wait_for_function("document.documentElement.scrollWidth <= innerWidth")
            table.focus()
            page.keyboard.press("ArrowRight")
            page.wait_for_function("document.querySelector('[aria-label=\"Input range checks\"]').scrollLeft > 0")
            page.evaluate("window.scrollTo(0, 0)")
            page.screenshot(path=str(output / f"mobile{width}.png"), full_page=True)
        page.set_viewport_size({"width": 1440, "height": 1000})
        with page.expect_response(lambda r: "model-health?days=7" in r.url):
            page.get_by_label("Monitoring window").select_option("7")
        expect(assessment).to_be_visible()
        with page.expect_download() as info:
            page.get_by_role("button", name="Export monitoring JSON").click()
        info.value.save_as(output / "export7.json")
        assert json.loads((output / "export7.json").read_text())["days"] == 7
        pattern = "**/model-health?days=*"
        def fixture(data, status=200):
            page.route(pattern, lambda route: route.fulfill(status=status, content_type="application/json", body=json.dumps(data)))
            page.reload(wait_until="networkidle")
        fixture({"detail": "fixture"}, 503)
        expect(page.get_by_role("alert")).to_contain_text("could not be loaded")
        page.unroute(pattern)
        page.get_by_role("button", name="Retry model health").click()
        expect(assessment).to_be_visible()
        fixture({"available": False, "reason": "Reference verification fixture unavailable."})
        expect(page.get_by_text("Reference verification fixture unavailable.", exact=True)).to_be_visible()
        page.unroute(pattern)
        empty = dict(real, records=0, forecast_dates=0, groups=[], latest_recorded_date=None)
        fixture(empty)
        expect(page.get_by_role("heading", name="No saved forecasts in this window")).to_be_visible()
        page.unroute(pattern)
        unsupported = copy.deepcopy(real)
        unsupported["groups"] = [dict(real["groups"][0], model_version="unknown-model", reference_supported=False, input_ranges=[])]
        fixture(unsupported)
        expect(assessment).to_contain_text("recorded model differs")
        page.get_by_text("Input ranges by feature", exact=True).click()
        expect(table).to_contain_text("No supported input reference")
        page.unroute(pattern)
        compression = copy.deepcopy(real)
        compression["groups"][0]["probability_status"] = "calibration_compression"
        fixture(compression)
        expect(assessment).to_contain_text("Calibrated probabilities are nearly identical")
        page.unroute(pattern)
        small = copy.deepcopy(real)
        small["groups"][0]["probability_status"] = "insufficient_sample"
        fixture(small)
        expect(assessment).to_contain_text("flags are withheld")
        page.unroute(pattern)
        issues = copy.deepcopy(real)
        issues["groups"][0]["quality"]["timestamp_problem_rows"] = 1
        fixture(issues)
        expect(assessment).to_contain_text("Record quality checks need review")
        page.unroute(pattern)
        page.reload(wait_until="networkidle")
        page.evaluate("() => { URL.createObjectURL = () => { throw new Error('fixture'); }; }")
        page.get_by_role("button", name="Export monitoring JSON").click()
        expect(page.get_by_role("alert")).to_contain_text("could not be downloaded")
        assert not errors, errors
        browser.close()
    findings = {"actual_records": real["records"], "actual_groups": len(real["groups"]), "mobile_widths": [390, 320],
                "verified": ["API_table_values", "version_selection_and_refresh", "window_export", "keyboard_scroll", "failure_retry", "unavailable_reference", "empty", "unsupported_model", "compression", "small_sample", "record_issues", "export_failure"], "uncaught_errors": errors}
    (output / "verification.json").write_text(json.dumps(findings, indent=2))
    print(json.dumps(findings))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:3000")
    parser.add_argument("--api-url", default="http://127.0.0.1:8001")
    parser.add_argument("--browser", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/research/model_health_20261007"))
    args = parser.parse_args()
    verify(args.base_url.rstrip("/"), args.api_url.rstrip("/"), args.browser, args.output)
