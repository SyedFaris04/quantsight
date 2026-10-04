"""Read-only browser QA for the prepared confidence audit; no fitting or account writes."""
import argparse
import json
from pathlib import Path
from urllib.parse import urlparse


def verify(base_url, browser_path, output):
    from playwright.sync_api import sync_playwright, expect
    output.mkdir(parents=True, exist_ok=True)
    result = {'base_url': base_url}
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, **({'executable_path': str(browser_path)} if browser_path else {}))
        page = browser.new_page(viewport={'width': 1440, 'height': 1100})
        errors, requests = [], []
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.on('request', lambda request: requests.append(request.url)
                if request.method == 'GET' and urlparse(request.url).path.endswith('/development-confidence') else None)
        page.goto(base_url.rstrip('/') + '/compare', wait_until='networkidle')
        summary = page.get_by_text('Does stronger confidence improve accuracy?', exact=True)
        expect(summary).to_be_visible()
        assert requests == [], 'The hidden confidence report must not load on initial navigation'
        with page.expect_response(lambda response: urlparse(response.url).path.endswith('/development-confidence')
                                  and response.request.method == 'GET') as received:
            summary.click()
        report = received.value.json()['report']
        section = page.get_by_role('region', name='Confidence and coverage audit', exact=True)
        candidate = page.get_by_label('Confidence candidate', exact=True)
        period = page.get_by_label('Confidence period', exact=True)
        table = page.get_by_role('region', name='Confidence threshold comparison', exact=True)
        for name, selected in report['periods'].items():
            period.select_option(name)
            for key, rows in selected['candidates'].items():
                candidate.select_option(key)
                expect(table.locator('tbody tr')).to_have_count(7)
                for i, row in enumerate(rows):
                    cells = table.locator('tbody tr').nth(i).get_by_role('cell')
                    expect(cells.nth(0)).to_contain_text(f"{row['retained_rows']:,}")
                    expect(cells.nth(1)).to_have_text(f"{row['coverage'] * 100:.2f}%")
                    expect(cells.nth(4)).to_have_text('—' if row['accuracy'] is None else f"{row['accuracy'] * 100:.2f}%")
                    expect(cells.nth(5)).to_have_text('—' if row['baseline_accuracy'] is None else f"{row['baseline_accuracy'] * 100:.2f}%")
                    expect(cells.nth(6)).to_have_text('—' if row['brier_score'] is None else f"{row['brier_score']:.6f}")
                    if row['small_sample']:
                        expect(table.locator('tbody tr').nth(i).get_by_text('Small sample', exact=True)).to_be_visible()
        period.select_option('pooled')
        candidate.select_option('xgboost_raw')
        summary.click()
        summary.click()
        expect(candidate).to_be_visible()
        assert len(requests) == 1, 'Reopening or changing selectors should reuse the loaded report'
        with page.expect_download() as received:
            page.get_by_role('button', name='Download confidence JSON', exact=True).click()
        download = received.value
        assert download.failure() is None
        assert download.suggested_filename == 'quantsight-development-confidence.json'
        path = output / download.suggested_filename
        download.save_as(path)
        assert json.loads(path.read_text(encoding='utf-8')) == report
        section.screenshot(path=str(output / 'desktop.png'), animations='disabled')
        for width in [390, 320]:
            page.set_viewport_size({'width': width, 'height': 844})
            page.wait_for_function('document.documentElement.scrollWidth <= innerWidth')
            table.evaluate('(element) => { element.scrollLeft = 0; }')
            table.focus()
            page.keyboard.press('ArrowRight')
            page.wait_for_function("""document.querySelector('[aria-label="Confidence threshold comparison"]').scrollLeft > 0""")
            section.screenshot(path=str(output / f'mobile_{width}.png'), animations='disabled')
        pattern = '**/research/development-confidence'
        for payload, status in [({'available': False}, 200), ({'detail': 'Temporary verification failure'}, 503)]:
            page.route(pattern, lambda route, request, body=payload, code=status: route.fulfill(
                status=code, content_type='application/json', body=json.dumps(body)))
            page.reload(wait_until='networkidle')
            summary.click()
            expect(section.get_by_text('The confidence audit is currently unavailable.', exact=False)).to_be_visible()
            expect(page.get_by_text('Do the models beat a simple baseline?', exact=True)).to_be_visible()
            page.unroute(pattern)
            page.get_by_role('button', name='Retry confidence audit', exact=True).click()
            expect(candidate).to_be_visible()
        page.route('**/research/development-confidence/export', lambda route: route.fulfill(status=503, body='Unavailable'))
        page.get_by_role('button', name='Download confidence JSON', exact=True).click()
        expect(section.get_by_role('alert')).to_have_text('The confidence report could not be downloaded. Please retry.')
        assert not errors, errors
        result.update(candidate_period_views=36, threshold_rows_checked=252, lazy_fetch=True,
                      export_matches_report=True, mobile_widths=[390, 320], keyboard_scrolling=True,
                      unavailable_retry_export_failure=True, uncaught_errors=errors)
        browser.close()
    (output / 'verification.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:3000')
    parser.add_argument('--browser', type=Path)
    parser.add_argument('--output', type=Path, default=Path('data/research/confidence_ui'))
    args = parser.parse_args()
    verify(args.base_url, args.browser, args.output)
