"""Optional dashboard browser checks against real API data and isolated faults."""
import argparse
from datetime import datetime, timezone
import json
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
        responses = {}
        def capture(response):
            path = urlparse(response.url).path
            if path in ['/dashboard', '/market-sentiment'] and response.status == 200:
                responses[path] = response.json()
        page.on('response', capture)
        page.goto(base_url.rstrip('/') + '/', wait_until='networkidle')
        dashboard = responses['/dashboard']
        sentiment = responses['/market-sentiment']
        assert dashboard['snapshot']['kind'] == 'historical_predictions'
        assert sentiment['source'] == 'saved_news_archive'
        snapshot = page.get_by_role('region', name='Historical model snapshot', exact=True)
        news = page.get_by_role('region', name='Saved news sentiment', exact=True)
        expect(snapshot).to_contain_text(dashboard['snapshot']['latest_signal_date'])
        expect(snapshot.get_by_text('Instruments covered', exact=True).locator('..').locator('dd')).to_have_text(str(dashboard['kpis']['total_tickers']))
        expect(news).to_contain_text(sentiment['window_end'])
        expect(news).to_contain_text(f"{sentiment['age_days']} days old")
        expect(news).to_contain_text('This is not a live news feed.')
        expect(page.get_by_text('Live signals', exact=True)).to_have_count(0)
        expect(page.get_by_text('Live', exact=True)).to_have_count(0)
        expect(page.get_by_text('AI Insight', exact=True)).to_have_count(0)
        paths = page.get_by_role('navigation', name='Explore the evidence')
        assert paths.locator('a').evaluate_all('(links) => links.map(a => a.getAttribute("href"))') == ['/backtesting', '/compare', '/track-record']
        candidates = page.get_by_role('region', name='Historical BUY candidates')
        expect(candidates.locator('a')).to_have_count(len(dashboard['top_opportunities']))
        for row, link in zip(dashboard['top_opportunities'], candidates.locator('a').all()):
            expect(link).to_have_attribute('href', '/detail/' + row['ticker'])
        news.get_by_text('Daily sentiment values', exact=True).click()
        expect(news.get_by_role('row')).to_have_count(1 + len(sentiment['trend']))
        news.get_by_text('Daily sentiment values', exact=True).click()
        page.screenshot(path=str(output / 'desktop.png'), animations='disabled', full_page=True)
        for width in [390, 320]:
            page.set_viewport_size({'width': width, 'height': 844})
            page.wait_for_function("document.querySelector('aside').getBoundingClientRect().right <= 0")
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
            page.screenshot(path=str(output / f'mobile_{width}.png'), animations='disabled', full_page=True)
        result.update(actual_dates_and_counts=True, mobile_widths_without_overflow=[390, 320], evidence_links=True)

        # A failed snapshot must not create fabricated zero KPIs or an empty-market claim.
        page.route('**/dashboard', lambda route: route.fulfill(status=503, body='Unavailable'))
        page.reload(wait_until='networkidle')
        expect(snapshot.get_by_role('alert')).to_contain_text('could not be loaded')
        expect(snapshot.locator('dd')).to_have_count(0)
        expect(candidates).to_contain_text('Candidates are unavailable')
        expect(news).to_contain_text(sentiment['window_end'])
        page.unroute('**/dashboard')
        snapshot.get_by_role('button', name='Try again').click()
        expect(snapshot.locator('dd')).to_have_count(4)

        page.route('**/market-sentiment', lambda route: route.fulfill(status=503, body='Unavailable'))
        page.reload(wait_until='networkidle')
        expect(news.get_by_role('alert')).to_contain_text('could not be loaded')
        expect(snapshot.locator('dd')).to_have_count(4)
        page.unroute('**/market-sentiment')
        news.get_by_role('button', name='Try again').click()
        expect(news).to_contain_text(sentiment['window_end'])

        # Older backend deployments omit provenance. Never invent a date in that case.
        legacy = dict(dashboard)
        legacy.pop('snapshot')
        page.route('**/dashboard', lambda route: route.fulfill(content_type='application/json', body=json.dumps(legacy)))
        page.reload(wait_until='networkidle')
        expect(snapshot).to_contain_text('Date unavailable')
        page.unroute('**/dashboard')
        page.route('**/market-sentiment', lambda route: route.fulfill(content_type='application/json', body=json.dumps({'article_count': 0})))
        page.reload(wait_until='networkidle')
        expect(news).to_contain_text('No dated headlines are available to score.')
        expect(news.locator('dd')).to_have_count(0)
        result['isolated_failures_retry_unknown_dates_empty_news'] = True
        assert not errors, errors
        result['uncaught_page_errors'] = errors
        browser.close()
    (output / 'verification.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:3000')
    parser.add_argument('--browser', type=Path)
    parser.add_argument('--output', type=Path, default=Path('data/research/dashboard_ui'))
    args = parser.parse_args()
    verify(args.base_url, args.browser, args.output)
