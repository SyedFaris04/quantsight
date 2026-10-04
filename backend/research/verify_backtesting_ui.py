"""Optional read-only browser checks using the actual published backtest.

Requires Playwright/Chromium and a running frontend/backend. Fault cases intercept
only this browser's requests. Downloads and screenshots remain in ignored data.
"""
import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse


def verify(base_url, browser_path, output):
    from playwright.sync_api import sync_playwright, expect
    output.mkdir(parents=True, exist_ok=True)
    result = {'checked_at': datetime.now(timezone.utc).isoformat(), 'base_url': base_url}
    with sync_playwright() as p:
        options = {'headless': True}
        if browser_path:
            options['executable_path'] = str(browser_path)
        browser = p.chromium.launch(**options)
        page = browser.new_page(viewport={'width': 1440, 'height': 1100})
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        with page.expect_response(lambda r: urlparse(r.url).path.endswith('/backtest')
                                  and r.request.method == 'GET') as response:
            page.goto(base_url.rstrip('/') + '/backtesting', wait_until='networkidle')
        payload = response.value.json()
        assert payload['available']
        report = payload['report']
        result['run_id'] = report['run_id']
        result['report_fresh'] = report['freshness']['matches_current_files']
        assert result['report_fresh'], 'Published inputs differ from report sources'
        select = page.get_by_role('combobox', name='Strategy', exact=True)
        expect(select).to_have_value('ensemble')
        main = page.locator('main')
        assert report['period_analysis']['available']
        analysis = report['period_analysis']['report']
        period_section = page.get_by_role('region', name='Performance over time', exact=True)
        for strategy in report['strategies']:
            select.select_option(strategy['key'])
            card = main.locator('p').filter(has_text=re.compile(r'^Return minus SPY$')).locator('..')
            expect(card.get_by_text(f"{strategy['metrics']['excess_total_return'] * 100:.2f} pp", exact=True)).to_be_visible()
            expect(page.get_by_role('button', name=strategy['name'], exact=True)).to_have_attribute('aria-pressed', 'true')
            periods = analysis['strategies'][strategy['key']]
            table = page.get_by_role('region', name='Year performance comparison', exact=True)
            for row in periods['years']:
                observed = table.get_by_role('row').filter(has=page.get_by_role('rowheader', name=row['period'], exact=True))
                expect(observed).to_contain_text(f"{row['start']} to {row['end']}")
                expect(observed).to_contain_text(f"{row['net_return'] * 100:.2f}%")
                expect(observed).to_contain_text(f"{row['difference_vs_spy'] * 100:.2f} pp")
            expect(period_section.get_by_text(f"{periods['summary']['months_beating_spy']} / {periods['summary']['months']}", exact=True)).to_be_visible()
        spy = next(s for s in report['strategies'] if s['key'] == 'spy')
        page.get_by_role('button', name=spy['name'], exact=True).click()
        expect(select).to_have_value('spy')
        expect(main.get_by_text('Prediction quality is a separate measurement', exact=True)).to_have_count(0)
        select.select_option('ensemble')
        expect(main.get_by_text('Prediction quality is a separate measurement', exact=True)).to_be_visible()
        page.get_by_text('How to read these results', exact=True).click()
        expect(page.get_by_text('Drawdown and Calmar', exact=True)).to_be_visible()
        page.get_by_text('How to read these results', exact=True).click()
        page.get_by_text('Methodology and study limitations', exact=True).click()
        expect(page.get_by_text(report['warnings'][0], exact=True)).to_be_visible()
        page.get_by_text('Methodology and study limitations', exact=True).click()
        result['strategy_controls_and_units'] = len(report['strategies'])
        months = analysis['strategies']['ensemble']['months']
        page.get_by_text(f'Monthly comparison · {len(months)} observed months', exact=True).click()
        monthly = page.get_by_role('region', name='Monthly performance comparison', exact=True)
        expect(monthly.locator('tbody tr')).to_have_count(len(months))
        for row in months:
            observed = monthly.get_by_role('row').filter(has=page.get_by_role('rowheader', name=row['period'], exact=True))
            expect(observed).to_contain_text(f"{row['start']} to {row['end']}")
            expect(observed).to_contain_text(f"{row['net_return'] * 100:.2f}%")
        result['period_comparisons'] = {'strategies': len(analysis['strategies']), 'months': len(months)}

        ensemble = next(s for s in report['strategies'] if s['key'] == 'ensemble')
        exports = [('metrics', 'All metrics CSV'), ('daily', 'Selected daily values CSV'),
                   ('trades', 'Selected trades CSV'), ('report', 'Full report JSON'), ('periods', 'Period report JSON')]
        for kind, label in exports:
            with page.expect_download() as received:
                page.get_by_role('button', name=label, exact=True).click()
            download = received.value
            assert download.failure() is None
            suffix = 'json' if kind in {'report', 'periods'} else 'csv'
            assert download.suggested_filename == f'quantsight-{kind}-ensemble.{suffix}'
            path = output / download.suggested_filename
            download.save_as(path)
            if kind in {'report', 'periods'}:
                assert json.loads(path.read_text(encoding='utf-8'))['run_id'] == report['run_id']
                if kind == 'periods':
                    assert json.loads(path.read_text(encoding='utf-8')) == analysis
            else:
                with path.open(encoding='utf-8-sig', newline='') as stream:
                    rows = list(csv.DictReader(stream))
                assert rows
                if kind == 'metrics':
                    assert len(rows) == len(report['strategies'])
                elif kind == 'daily':
                    assert len(rows) == report['evaluation']['sessions']
                    assert abs(float(rows[-1]['equity']) - ensemble['metrics']['final_equity']) < .000001
                else:
                    assert len(rows) == ensemble['trade_count']
        result['verified_exports'] = [kind for kind, _ in exports]
        page.evaluate('window.scrollTo(0,0)')
        page.screenshot(path=str(output / 'desktop.png'), animations='disabled')
        for width in [390, 320]:
            page.set_viewport_size({'width': width, 'height': 844})
            page.wait_for_function("document.querySelector('aside').getBoundingClientRect().right <= 0")
            # ResponsiveContainer updates its chart widths after the resize event.
            page.wait_for_function('document.documentElement.scrollWidth <= innerWidth')
            table = page.get_by_role('region', name='All strategy metrics', exact=True)
            table.evaluate('(element) => { element.scrollLeft = 0; }')
            table.focus()
            page.keyboard.press('ArrowRight')
            page.wait_for_function("document.querySelector('[aria-label=\"All strategy metrics\"]').scrollLeft > 0")
            table.screenshot(path=str(output / f'table_{width}.png'), animations='disabled')
            for name in ['Year performance comparison', 'Monthly performance comparison']:
                period_table = page.get_by_role('region', name=name, exact=True)
                period_table.evaluate('(element) => { element.scrollLeft = 0; }')
                period_table.focus()
                page.keyboard.press('ArrowRight')
                page.wait_for_function('(label) => document.querySelector(`[aria-label="${label}"]`).scrollLeft > 0', arg=name)
            period_section.screenshot(path=str(output / f'periods_{width}.png'), animations='disabled')
            page.evaluate('window.scrollTo(0,0)')
            page.screenshot(path=str(output / f'mobile_{width}.png'), animations='disabled')
        result['mobile_widths_without_overflow'] = [390, 320]
        result['keyboard_table_scrolling'] = True

        pattern = '**/backtest'
        def fixture(body, status=200):
            page.route(pattern, lambda route: route.fulfill(status=status, content_type='application/json', body=json.dumps(body)))
            page.reload(wait_until='networkidle')

        fixture({'detail': 'Browser verification temporary failure'}, 503)
        expect(page.get_by_role('alert')).to_contain_text('The evaluation could not be loaded.')
        page.unroute(pattern)
        page.get_by_role('button', name='Try again', exact=True).click()
        expect(select).to_be_visible()
        fixture({'available': False})
        expect(page.get_by_text('A historical evaluation has not been published yet.', exact=False)).to_be_visible()
        page.unroute(pattern)
        original_analysis = payload['report']['period_analysis']
        for reason in ['missing', 'stale', 'invalid']:
            payload['report']['period_analysis'] = {'available': False, 'reason': reason}
            fixture(payload)
            expect(page.get_by_text('The year and month breakdown is unavailable for this saved run.', exact=False)).to_be_visible()
            expect(page.get_by_role('button', name='Period report JSON', exact=True)).to_have_count(0)
            expect(main.locator('p').filter(has_text=re.compile(r'^Net total return$'))).to_be_visible()
            page.unroute(pattern)
        payload['report']['period_analysis'] = original_analysis
        payload['report']['freshness']['matches_current_files'] = False
        fixture(payload)
        expect(page.get_by_role('alert')).to_contain_text('Results need refreshing.')
        page.unroute(pattern)
        page.route('**/backtest/export*', lambda route: route.fulfill(status=503, body='Unavailable'))
        page.get_by_role('button', name='All metrics CSV', exact=True).click()
        expect(page.get_by_text('The export could not be downloaded. Please try again.', exact=True)).to_be_visible()
        result['error_retry_empty_stale_export_failure'] = True
        assert not errors, errors
        result['uncaught_page_errors'] = errors
        browser.close()
    (output / 'verification.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:3000')
    parser.add_argument('--browser', type=Path)
    parser.add_argument('--output', type=Path, default=Path('data/research/backtest_ui'))
    args = parser.parse_args()
    verify(args.base_url, args.browser, args.output)
