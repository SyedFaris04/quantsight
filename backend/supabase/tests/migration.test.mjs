// Executes the actual migration in an isolated PostgreSQL WASM instance.
// No network, Supabase credentials, or production rows are used by these tests.
import { PGlite } from '@electric-sql/pglite';
import { readFile } from 'node:fs/promises';
import assert from 'node:assert/strict';
import { before, after, test } from 'node:test';

let db;
const table = 'public.live_predictions_v2';
const migration = await readFile(new URL('../002_live_predictions_v2.sql', import.meta.url), 'utf8');
before(async () => {
  db = new PGlite();
  await db.exec(`create role anon; create role authenticated; create role service_role bypassrls;
    grant usage on schema public to anon, authenticated, service_role;
    create table public.live_predictions (id integer primary key, note text);
    insert into public.live_predictions values (1, 'preserve legacy evidence');`);
  await db.exec(migration);
});
after(async () => { await db?.close(); });

function fixture(ticker, past = false) {
  const now = Date.now(), day = 86400000;
  const cutoff = now - (past ? 8 : 1) * day;
  const iso = ms => new Date(ms).toISOString();
  return {
    ticker, predicted_date: iso(cutoff).slice(0, 10),
    target_date: iso(cutoff + 7 * day).slice(0, 10), target_close_at: iso(cutoff + 7 * day),
    horizon_sessions: 5, protocol_version: 'nyse-close-5-v2', model_version: 'test-model',
    feature_version: 'test-features', feature_values: { rsi: 50 },
    data_cutoff_at: iso(cutoff), data_fetched_at: iso(cutoff + 3600000),
    generated_at: iso(cutoff + 3700000), created_at: iso(cutoff + 3800000),
    record_before: iso(cutoff + 2 * day), predicted_signal: 'BUY', confidence: 60,
    probability_up: .6, raw_probability_up: .7, calibration_method: 'isotonic',
    price_basis: 'yahoo_auto_adjust', price_at_prediction: 100,
  };
}
async function insert(row, suffix = '') {
  const columns = Object.keys(row);
  // Column names come solely from the fixed test fixture, never external input.
  return db.query(`insert into ${table} (${columns.join(',')}) values
    (${columns.map((_, i) => '$' + (i + 1)).join(',')}) ${suffix} returning *`,
    Object.values(row).map(value => typeof value === 'object' ? JSON.stringify(value) : value));
}
async function asRole(role, action) {
  assert.ok(['anon', 'authenticated', 'service_role'].includes(role));
  await db.exec(`set role ${role}`);
  try { return await action(); } finally { await db.exec('reset role'); }
}
const rejectsCode = (action, code) => assert.rejects(action, error => error.code === code);

test('migration is repeatable and preserves legacy data', async () => {
  await db.exec(migration);
  assert.equal((await db.query('select note from live_predictions where id=1')).rows[0].note,
    'preserve legacy evidence');
});
test('server timestamps inserts; service role can record only once', async () => {
  const row = fixture('ONCE');
  const first = await asRole('service_role', () => insert(row));
  assert.ok(new Date(first.rows[0].created_at).getTime() > new Date(row.created_at).getTime());
  const duplicate = await asRole('service_role', () => insert(row,
    'on conflict (ticker,predicted_date,protocol_version) do nothing'));
  assert.equal(duplicate.rows.length, 0);
  assert.equal((await db.query(`select count(*)::int as n from ${table} where ticker='ONCE'`)).rows[0].n, 1);
});
for (const role of ['anon', 'authenticated']) {
  test(`${role} can read but cannot insert, update or delete`, async () => {
    await asRole(role, async () => {
      assert.ok((await db.query(`select id from ${table}`)).rows.length >= 1);
      await rejectsCode(() => insert(fixture('CLIENT')), '42501');
      await rejectsCode(() => db.exec(`update ${table} set confidence=99`), '42501');
      await rejectsCode(() => db.exec(`delete from ${table}`), '42501');
    });
  });
  test(`${role} RLS also denies writes if table privileges are accidentally broadened`, async () => {
    await db.exec(`grant insert, update, delete on ${table} to ${role}`);
    try {
      await asRole(role, async () => {
        await rejectsCode(() => insert(fixture('RLS')), '42501');
        assert.equal((await db.query(`update ${table} set confidence=99 returning id`)).rows.length, 0);
        assert.equal((await db.query(`delete from ${table} returning id`)).rows.length, 0);
      });
    } finally { await db.exec(`revoke insert, update, delete on ${table} from ${role}`); }
  });
}
test('forecast values cannot be rewritten even by the serving role', async () => {
  await asRole('service_role', () => rejectsCode(
    () => db.exec(`update ${table} set probability_up=.9 where ticker='ONCE'`), 'P0001'));
});
test('late and prematurely recorded forecasts are rejected', async () => {
  await rejectsCode(() => insert(fixture('LATE', true)), '23514');
  const early = fixture('EARLY');
  early.data_cutoff_at = new Date(Date.now() - 60000).toISOString();
  early.data_fetched_at = early.data_cutoff_at;
  early.generated_at = early.data_cutoff_at;
  await rejectsCode(() => insert(early), 'P0001');
});
test('invalid horizon, probability and pending outcome data are rejected', async () => {
  for (const extra of [{ horizon_sessions: 1 }, { probability_up: 1.5 }, { actual_close: 105 }]) {
    await rejectsCode(() => insert({ ...fixture('INVALID'), ...extra }), '23514');
  }
});
test('future outcomes cannot be resolved early or on the wrong target date', async () => {
  const row = (await insert(fixture('FUTURE'))).rows[0];
  for (const target of [row.target_date, '2001-01-01']) {
    await rejectsCode(() => db.query(`update ${table} set resolved=true, resolved_date=$1,
      resolved_at=now(), actual_close=105, resolution_entry_close=100, actual_return=.05,
      actual_signal='BUY', correct=true where id=$2`, [target, row.id]), '23514');
  }
});
test('exact due outcome resolves once and is then immutable', async () => {
  // Simulate a forecast recorded days ago. Bypass only the INSERT trigger during
  // fixture setup; restore it before exercising resolution and immutability.
  await db.exec(`alter table ${table} disable trigger live_prediction_v2_guard`);
  let row;
  try { row = (await insert(fixture('DUE', true))).rows[0]; }
  finally { await db.exec(`alter table ${table} enable trigger live_prediction_v2_guard`); }
  await asRole('service_role', async () => {
    const resolved = await db.query(`update ${table} set resolved=true, resolved_date=target_date,
      resolved_at='2000-01-01', actual_close=105, resolution_entry_close=100, actual_return=.05,
      actual_signal='BUY', correct=true where id=$1 and not resolved returning *`, [row.id]);
    assert.equal(resolved.rows.length, 1);
    assert.ok(new Date(resolved.rows[0].resolved_at).getTime() > Date.now() - 60000);
    assert.equal((await db.query(`update ${table} set resolved=true where id=$1 and not resolved returning id`, [row.id])).rows.length, 0);
    await rejectsCode(() => db.query(`update ${table} set actual_close=106 where id=$1`, [row.id]), 'P0001');
  });
});
