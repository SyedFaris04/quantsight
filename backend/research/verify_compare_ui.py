"""Optional browser verification of the real research report on AI Compare.

Requires the local backend/frontend plus Playwright and Chromium. Output stays in
the ignored research directory. No login, account mutation or model fitting occurs.
"""
import argparse
import json
from pathlib import Path


def verify(base_url, browser_path, output):
    from playwright.sync_api import sync_playwright, expect
    output.mkdir(parents=True, exist_ok=True)
    result = {}
    with sync_playwright() as p:
        options = {'headless': True}
        if browser_path:
            options['executable_path'] = str(browser_path)
        browser = p.chromium.launch(**options)
        page = browser.new_page(viewport={'width': 1440, 'height': 1100}, device_scale_factor=1)
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        with page.expect_response(lambda response: '/research/news-comparison' in response.url
                                  and response.request.method == 'GET') as received:
            page.goto(base_url.rstrip('/') + '/compare', wait_until='networkidle')
        report = received.value.json()['report']
        section = page.get_by_role('region', name='Does financial news improve prediction?')
        expect(section).to_be_visible()
        expect(section.get_by_role('row')).to_have_count(6)  # header, baseline, four variants
        expect(section.get_by_text('Constant training prior')).to_be_visible()
        select = page.get_by_label('Prediction model', exact=True)
        select.select_option('hist_gradient_boosting')
        expect(select).to_have_value('hist_gradient_boosting')
        expect(section.get_by_role('row')).to_have_count(6)
        tree = next(row for row in report['models'] if row['key'] == 'hist_gradient_boosting_finance_coverage_finbert')
        expect(section.get_by_role('row').filter(has_text='Finance + FinBERT').get_by_role('cell').last).to_have_text(
            f"{tree['metrics']['brier_score']:.6f}")
        select.select_option('logistic')
        linear = next(row for row in report['models'] if row['key'] == 'logistic_finance_coverage_finbert')
        expect(section.get_by_role('row').filter(has_text='Finance + FinBERT').get_by_role('cell').last).to_have_text(
            f"{linear['metrics']['brier_score']:.6f}")
        section.get_by_text('Method and limitations', exact=True).click()
        expect(section.get_by_text('The 2023 period was previously examined; this is not a fresh final test.')).to_be_visible()
        section.get_by_text('Method and limitations', exact=True).click()
        section.screenshot(path=str(output / 'research_desktop.png'))
        result['desktop_comparison_and_controls'] = True

        page.set_viewport_size({'width': 390, 'height': 844})
        page.wait_for_function("document.querySelector('aside').getBoundingClientRect().right <= 0")
        expect(select).to_be_visible()
        page.evaluate('window.scrollTo(0,0)')
        page.screenshot(path=str(output / 'research_mobile.png'), animations='disabled')
        sizes = page.evaluate('({width: innerWidth, document: document.documentElement.scrollWidth})')
        assert sizes['document'] <= sizes['width'], sizes
        result['mobile_no_page_overflow'] = True

        pattern = '**/research/news-comparison*'
        page.route(pattern, lambda route: route.fulfill(status=503, content_type='application/json',
                                                       body=json.dumps({'detail': 'UI verification: temporary failure'})))
        page.reload(wait_until='networkidle')
        expect(page.get_by_text('The research report is currently unavailable.')).to_be_visible()
        page.unroute(pattern)
        page.get_by_role('button', name='Retry', exact=True).click()
        expect(page.get_by_role('region', name='Does financial news improve prediction?')).to_be_visible()
        result['failed_request_and_retry_recovery'] = True
        assert not errors, errors
        result['uncaught_page_errors'] = errors
        browser.close()
    (output / 'verification.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:3000')
    parser.add_argument('--browser', type=Path)
    parser.add_argument('--output', type=Path, default=Path('data/research/ui_checks'))
    args = parser.parse_args()
    verify(args.base_url, args.browser, args.output)
