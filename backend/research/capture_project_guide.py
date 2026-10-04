"""Capture QuantSight in an isolated guest browser for the illustrated PDF guide."""
import argparse
import json
from pathlib import Path

ROUTES = ['/', '/market', '/detail/AAPL', '/compare', '/leaderboard', '/backtesting',
          '/track-record', '/portfolio', '/game', '/settings', '/login']


def inventory(base_url, output, browser_path):
    from playwright.sync_api import sync_playwright
    output.mkdir(parents=True, exist_ok=True)
    inventory = {}
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, executable_path=str(browser_path))
        context = browser.new_context(viewport={'width': 1060, 'height': 900}, device_scale_factor=2)
        page = context.new_page()
        for route in ROUTES:
            page.goto(base_url + route, wait_until='networkidle', timeout=60000)
            page.locator('main h1').first.wait_for(timeout=30000)
            page.wait_for_timeout(1200)
            name = route.strip('/').replace('/', '_') or 'dashboard'
            main = page.locator('main')
            headings = main.locator('h1,h2,h3,summary').evaluate_all("""els => els.map(e => ({
              text:e.innerText, tag:e.tagName, y:Math.round(e.getBoundingClientRect().top+scrollY),
              parent:e.parentElement.className
            }))""")
            inventory[route] = {'headings': headings, 'text': main.inner_text(),
                                'buttons': main.get_by_role('button').all_text_contents()}
            page.screenshot(path=str(output / f'{name}_overview.png'), animations='disabled')
            print(name, flush=True)
        browser.close()
    (output / 'inventory.json').write_text(json.dumps(inventory, indent=2), encoding='utf-8')


def feature_capture(base_url, output, browser_path, extras=False):
    """Capture actual UI states, without authentication or persistent account writes."""
    from playwright.sync_api import sync_playwright
    output.mkdir(parents=True, exist_ok=True)
    records = json.loads((output / 'captures.json').read_text(encoding='utf-8')) if extras else {}
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, executable_path=str(browser_path))
        context = browser.new_context(viewport={'width': 1060, 'height': 900}, device_scale_factor=2)
        page = context.new_page()

        def save(name, locator=None):
            page.wait_for_timeout(800)
            target = locator or page.locator('main')
            target.screenshot(path=str(output / f'{name}.png'), animations='disabled', timeout=60000,
                              style="button[aria-label='Open AI Assistant'] { visibility: hidden !important; }")
            bb = target.bounding_box()
            records[name] = {'text': target.inner_text(), 'width': bb['width'],
                'height': bb['height'], 'route': page.url.replace(base_url, ''),
                'headings': target.locator('h1,h2,h3,summary').evaluate_all(
                    "els=>els.map(e=>({text:e.innerText,y:e.getBoundingClientRect().top}))")}
            # Heading positions are measured relative to this captured element.
            for item in records[name]['headings']:
                item['y'] = round(item['y'] - bb['y'])
            print(name, flush=True)

        def go(route):
            page.goto(base_url + route, wait_until='networkidle', timeout=60000)
            page.locator('main h1').first.wait_for(timeout=30000)
            page.wait_for_timeout(1000)

        if extras:
            go('/track-record')
            save('track-record')
            go('/detail/AAPL')
            for label in ['RSI', 'MACD']:
                page.get_by_role('button', name=label, exact=True).click()
                save('detail_chart_' + label.lower())
            page.get_by_role('button', name='Learn', exact=True).click()
            page.locator('main button').last.click()
            page.locator('main .card button').nth(1).click()
            save('detail_learn_open')
            go('/compare')
            page.locator('main select').last.select_option('AAPL')
            save('compare_ticker')
            page.get_by_text('Probability reliability, uncertainty and method', exact=True).click()
            save('compare_method')
            go('/backtesting')
            page.get_by_text('Monthly comparison', exact=False).first.click()
            save('backtest_months')
            page.get_by_text('Methodology and study limitations', exact=True).click()
            save('backtest_method')
            go('/portfolio')
            page.get_by_role('button', name='+ Add Stock', exact=True).click()
            page.locator('main select').first.select_option('AAPL')
            page.get_by_placeholder('e.g. 10', exact=True).fill('10')
            page.get_by_placeholder('e.g. 150.00', exact=True).fill('200')
            page.get_by_role('button', name='Add to Portfolio', exact=True).click()
            try:
                page.get_by_text('Loading\u2026', exact=True).first.wait_for(state='hidden', timeout=55000)
            except Exception:
                pass
            save('portfolio_example')
            go('/market')
            page.get_by_role('button', name='Switch to Live Mode', exact=True).click()
            try:
                page.get_by_text('Fetching live data...', exact=True).wait_for(state='hidden', timeout=95000)
            except Exception:
                pass
            save('market_live')
            browser.close()
            (output / 'captures.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
            return

        for route in ROUTES:
            go(route)
            save(route.strip('/').replace('/', '_') or 'dashboard')
        go('/detail/AAPL')
        for tab, name in [('Technical', 'technical'), ('Sentiment & Emotion', 'sentiment'),
                          ('History', 'history'), ('Learn', 'learn')]:
            page.get_by_role('button', name=tab, exact=True).click()
            save('detail_' + name)
        page.get_by_role('button', name='Overview', exact=True).click()
        page.get_by_text('All 30 inputs and verification details', exact=True).click()
        save('detail_attribution_expanded')

        go('/compare')
        page.get_by_text('Does stronger confidence improve accuracy?', exact=False).first.click()
        page.wait_for_timeout(1800)
        save('compare_confidence')
        go('/portfolio')
        page.get_by_role('button', name='+ Add Stock', exact=True).click()
        page.locator('main select').first.select_option('AAPL')
        page.get_by_placeholder('e.g. 10', exact=True).fill('10')
        page.get_by_placeholder('e.g. 150.00', exact=True).fill('200')
        save('portfolio_form')
        page.get_by_role('button', name='Add to Portfolio', exact=True).click()
        page.wait_for_timeout(5000)
        save('portfolio_example')
        go('/game')
        page.get_by_role('button', name='Start Game', exact=False).click()
        page.wait_for_timeout(1600)
        save('game_question')
        # Answers are compared with the model label. A fresh guest never syncs to Supabase.
        page.get_by_role('button', name='BUY', exact=False).last.click()
        page.wait_for_timeout(1600)
        save('game_result')
        go('/login')
        page.get_by_role('button', name='Create one', exact=True).click()
        save('signup')
        go('/detail/AAPL')
        page.get_by_role('button', name='Open AI Assistant', exact=True).click()
        panel = page.locator('div.fixed').filter(has=page.get_by_text('QuantSight Assistant', exact=True))
        save('chat_welcome', panel)
        page.get_by_role('textbox', name='Message to AI Assistant').fill(
            'In three short bullets, explain the saved AAPL signal, its date and one limitation.')
        page.get_by_role('button', name='Send', exact=True).click()
        try:
            page.get_by_role('button', name='Send', exact=True).wait_for(timeout=95000)
        except Exception:
            pass
        save('chat_answer', panel)
        browser.close()
    (output / 'captures.json').write_text(json.dumps(records, indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:3000')
    parser.add_argument('--output', type=Path, default=Path('data/research/documentation_20261005'))
    parser.add_argument('--browser', type=Path, default=Path('C:/Users/syeda/AppData/Local/ms-playwright/chromium-1228/chrome-win64/chrome.exe'))
    parser.add_argument('--features', action='store_true')
    parser.add_argument('--extras', action='store_true')
    args = parser.parse_args()
    if args.features or args.extras:
        feature_capture(args.base_url, args.output, args.browser, extras=args.extras)
    else:
        inventory(args.base_url, args.output, args.browser)
