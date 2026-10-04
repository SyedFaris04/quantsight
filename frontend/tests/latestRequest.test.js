import test from 'node:test';
import assert from 'node:assert/strict';
import { createLatestRequest, describeApiError } from '../src/hooks/latestRequest.js';

function harness() {
  let state = {};
  const requests = [];
  const events = [];
  const runner = createLatestRequest((endpoint, options) => new Promise((resolve, reject) => {
    requests.push({ endpoint, ...options, resolve, reject });
  }), patch => { events.push(patch); state = { ...state, ...patch }; });
  return { runner, requests, events, state: () => state };
}

test('older completions cannot replace new data or end its loading state', async () => {
  const h = harness();
  const first = h.runner.run('/stock/AAPL');
  const second = h.runner.run('/stock/TSLA');
  assert.equal(h.requests[0].signal.aborted, true);
  // Deliberately simulate a transport that ignores AbortSignal.
  h.requests[0].resolve({ data: 'AAPL' });
  await first;
  assert.equal(h.state().loading, true);
  assert.equal(h.state().data, null);
  h.requests[1].resolve({ data: 'TSLA' });
  await second;
  assert.equal(h.state().data, 'TSLA');
  assert.equal(h.state().loading, false);
});

test('manual retries of the same endpoint also supersede earlier requests', async () => {
  const h = harness();
  const first = h.runner.run('/dashboard');
  const second = h.runner.run('/dashboard');
  h.requests[1].resolve({ data: 'latest' });
  await second;
  h.requests[0].reject({ code: 'ECONNABORTED' });
  await first;
  assert.equal(h.state().data, 'latest');
  assert.equal(h.state().error, null);
});

test('cancellation prevents all further state publication', async () => {
  const h = harness();
  const pending = h.runner.run('/dashboard');
  h.runner.cancel();
  const count = h.events.length;
  h.requests[0].resolve({ data: 'unmounted' });
  await pending;
  assert.equal(h.requests[0].signal.aborted, true);
  assert.equal(h.events.length, count);
});

test('disabled endpoints clear data and cancel pending work without a request', async () => {
  const h = harness();
  const first = h.runner.run('/dashboard');
  await h.runner.run(null);
  h.requests[0].resolve({ data: 'old' });
  await first;
  assert.equal(h.requests.length, 1);
  assert.deepEqual(h.state(), { endpoint: null, data: null, error: null, loading: false, isSlow: false });
});

test('slow indication belongs only to the active request and clears on completion', async t => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const h = harness();
  const first = h.runner.run('/first');
  t.mock.timers.tick(7000);
  assert.equal(h.state().isSlow, false);
  const second = h.runner.run('/second');
  t.mock.timers.tick(1000);
  assert.equal(h.state().isSlow, false);
  t.mock.timers.tick(7000);
  assert.equal(h.state().isSlow, true);
  h.requests[1].resolve({ data: 'ready' });
  await second;
  h.requests[0].resolve({ data: 'old' });
  await first;
  assert.equal(h.state().isSlow, false);
  assert.equal(h.state().data, 'ready');
});

test('network and rate-limit failures do not automatically send more requests', async () => {
  for (const error of [{ code: 'ERR_NETWORK' }, { response: { status: 429 } }, { code: 'ECONNABORTED' }]) {
    const h = harness();
    const pending = h.runner.run('/dashboard');
    h.requests[0].reject(error);
    await pending;
    assert.equal(h.requests.length, 1);
    assert.equal(h.state().loading, false);
    assert.equal(typeof h.state().error, 'string');
    assert.ok(h.state().error.length > 0);
  }
});

test('error messages are readable strings for timeout, HTTP and validation responses', () => {
  assert.match(describeApiError({ code: 'ECONNABORTED' }), /30 seconds/);
  assert.match(describeApiError({ response: { status: 503, data: '<html>private proxy response</html>' } }), /temporarily unavailable/);
  assert.match(describeApiError({ response: { status: 429 } }), /wait/);
  assert.equal(describeApiError({ response: { status: 404, data: { detail: 'No saved prediction exists.' } } }), 'No saved prediction exists.');
  assert.match(describeApiError({ response: { data: { detail: [{ msg: 'Invalid' }] } } }), /not accepted/);
  assert.match(describeApiError({ response: { data: { detail: { nested: 'invalid' } } } }), /could not be completed/);
});
