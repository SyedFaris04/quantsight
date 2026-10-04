"""Production-build browser QA against a local backend; never submits mutations.

Start backend on 8000 and Vite preview on 4173 after npm run build.
The local build's API origin defaults to the existing developer setting, port 8001.
"""
import asyncio
import json
from pathlib import Path
from urllib.parse import urlsplit

from playwright.async_api import async_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/research/loading_performance'
PREVIEW = 'http://127.0.0.1:4173'
CHROMIUM = Path.home() / 'AppData/Local/ms-playwright/chromium-1228/chrome-win64/chrome.exe'


async def verify():
    OUT.mkdir(parents=True, exist_ok=True)
    async with async_playwright() as playwright:
        options = {'executable_path': str(CHROMIUM)} if CHROMIUM.exists() else {}
        browser = await playwright.chromium.launch(headless=True, **options)
        context = await browser.new_context(viewport={'width': 1440, 'height': 1000})

        async def local_api(route):
            if route.request.method != 'GET':
                await route.fulfill(status=405, body='No mutations in browser verification')
                return
            parsed = urlsplit(route.request.url)
            response = await context.request.get('http://127.0.0.1:8000' + parsed.path + ('?' + parsed.query if parsed.query else ''), timeout=60000)
            await route.fulfill(response=response, headers={**response.headers, 'access-control-allow-origin': '*'})

        await context.route('http://127.0.0.1:8001/**', local_api)
        page = await context.new_page()
        errors, scripts = [], []
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.on('request', lambda request: scripts.append(urlsplit(request.url).path) if request.resource_type == 'script' else None)
        await page.goto(PREVIEW, wait_until='domcontentloaded')
        await page.get_by_text('Instruments covered', exact=True).wait_for(timeout=60000)
        home_scripts = list(scripts)
        assert len([name for name in home_scripts if name.startswith('/assets/')]) == 1, home_scripts
        assert not any('CategoricalChart' in name or 'Compare-' in name or 'Detail-' in name for name in home_scripts)
        await page.get_by_role('button', name='Open AI Assistant').click()
        await page.get_by_role('textbox', name='Message to AI Assistant').fill('Unsent draft survives navigation')
        await page.locator('a[href="/market"]').first.click()
        await page.wait_for_url('**/market')
        await page.get_by_role('heading', name='Market', exact=True).wait_for()
        assert await page.get_by_role('textbox', name='Message to AI Assistant').input_value() == 'Unsent draft survives navigation'
        await page.get_by_role('button', name='Close AI Assistant').click()
        headings = {}
        expected_headings = {'/compare': 'Model Comparison', '/leaderboard': 'Model Leaderboard',
                             '/track-record': 'Live Track Record', '/backtesting': 'Backtesting',
                             '/portfolio': 'My Portfolio', '/game': 'Stock Prediction Game', '/settings': 'Settings'}
        for path, expected in expected_headings.items():
            await page.locator(f'a[href="{path}"]').first.click()
            await page.wait_for_url('**' + path)
            await page.locator('main h1').filter(has_text=expected).wait_for(timeout=60000)
            headings[path] = await page.locator('main h1').inner_text()
            assert 'could not be displayed' not in headings[path]
        for path in ['/detail/AAPL', '/login']:
            await page.goto(PREVIEW + path, wait_until='domcontentloaded')
            await page.locator('main h1').wait_for(timeout=60000)
            headings[path] = await page.locator('main h1').inner_text()
            assert 'could not be displayed' not in headings[path]
        assert not errors, errors
        print('All routes load; homepage omits chart bundles; chat draft persists.', flush=True)

        # A failed dynamic import stays inside the page boundary; navigation survives.
        fault = await context.new_page()
        await fault.route('**/assets/Compare-*.js', lambda route: route.abort())
        await fault.goto(PREVIEW + '/compare', wait_until='domcontentloaded')
        await fault.get_by_role('heading', name='This page could not be displayed').wait_for()
        await fault.screenshot(path=str(OUT / 'page-download-error.png'), full_page=True)
        await fault.locator('a[href="/"]').first.click()
        await fault.get_by_role('heading', name='Understand the signal. Check the evidence.').wait_for()
        await fault.locator('a[href="/compare"]').first.click()
        await fault.get_by_role('button', name='Reload page').wait_for()
        await fault.unroute('**/assets/Compare-*.js')
        await fault.get_by_role('button', name='Reload page').click()
        await fault.get_by_role('heading', name='Model Comparison').wait_for()
        await fault.get_by_text('32,472', exact=True).wait_for(timeout=60000)
        await fault.close()
        print('Failed page download recovers through navigation and explicit reload.', flush=True)

        waiting = await context.new_page()
        release_chunk = asyncio.Event()
        async def slow_chunk(route):
            await release_chunk.wait()
            await route.continue_()
        await waiting.route('**/assets/Settings-*.js', slow_chunk)
        await waiting.goto(PREVIEW, wait_until='domcontentloaded')
        await waiting.locator('a[href="/settings"]').first.click()
        await waiting.get_by_text('Loading page…', exact=True).wait_for(timeout=10000)
        release_chunk.set()
        await waiting.get_by_role('heading', name='Settings', exact=True).wait_for()
        await waiting.close()
        print('Page loading indicator appears during a delayed route download.', flush=True)

        # Real Axios deadline: a slow request becomes a readable timeout, then retry succeeds.
        delayed = await context.new_page()
        pending = asyncio.Event()
        async def stall(route):
            await pending.wait()
            try:
                await route.abort()
            except Exception:
                pass  # The browser may already have cancelled its timed-out request.
        await delayed.route('**:8001/dashboard', stall)
        await delayed.goto(PREVIEW, wait_until='domcontentloaded')
        await delayed.get_by_text('Still waiting for the snapshot.', exact=False).wait_for(timeout=12000)
        await delayed.get_by_text('The server did not respond within 30 seconds.', exact=False).wait_for(timeout=30000)
        pending.set()
        await delayed.unroute('**:8001/dashboard')
        await delayed.get_by_role('button', name='Try again', exact=True).first.click()
        await delayed.get_by_text('Instruments covered', exact=True).wait_for(timeout=60000)
        for width in [390, 320]:
            await delayed.set_viewport_size({'width': width, 'height': 844})
            await delayed.wait_for_function('document.documentElement.scrollWidth <= innerWidth')
            await delayed.screenshot(path=str(OUT / f'dashboard-{width}.png'), full_page=True)
        print('Slow-server notice, actual 30-second timeout, retry and mobile checks passed.', flush=True)
        result = {'homepage_scripts': home_scripts, 'route_headings': headings,
                  'chat_draft_preserved': True, 'chunk_failure_recovery': True,
                  'delayed_chunk_loading_notice': True,
                  'actual_timeout_and_retry': True, 'mobile_widths': [390, 320], 'page_errors': errors}
        (OUT / 'browser-verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        await browser.close()


if __name__ == '__main__':
    asyncio.run(verify())
