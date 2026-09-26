-- Also apply 002_live_predictions_v2.sql for the current five-session tracker.
-- backend/supabase/schema.sql
-- ─────────────────────────────────────────────────────────────────────────
-- Version-controlled source of truth for QuantSight's Supabase schema.
--
-- Previously these tables existed only as SQL pasted once into Supabase's
-- SQL Editor (see CHECKLIST.md) with no copy checked into the repo — so
-- there was no way to verify the constraints prediction_tracker.py and
-- Portfolio.jsx/Game.jsx assume (e.g. the live_predictions uniqueness
-- guarantee "unique constraint makes this idempotent") actually exist in
-- the live database, or to recreate the project from scratch.
--
-- `create table if not exists` / `create policy` guarded with a DO block
-- below, so this is safe to re-run against an already-provisioned project
-- without erroring on things that already exist.
-- ─────────────────────────────────────────────────────────────────────────

-- ── live_predictions ───────────────────────────────────────────────────────
-- Forward-looking, tamper-evident prediction track record (prediction_tracker.py).
-- Only the backend's service-role key can INSERT/UPDATE (bypasses RLS
-- entirely) — the RLS policy below only ever grants public SELECT, so no
-- client (including QuantSight's own frontend) can write or edit a row.
create table if not exists live_predictions (
  id                    uuid primary key default gen_random_uuid(),
  ticker                text not null,
  predicted_date        date not null,
  predicted_signal      text not null check (predicted_signal in ('BUY', 'SELL')),
  confidence            numeric not null,
  price_at_prediction   numeric not null,
  resolved              boolean not null default false,
  resolved_date         date,
  actual_close          numeric,
  actual_signal         text check (actual_signal in ('BUY', 'SELL')),
  correct               boolean,
  -- Insert timestamp is what makes this auditable rather than merely
  -- write-restricted: it lets anyone verify predicted_date reflects when a
  -- row was actually created, not a backdated value entered later by
  -- whoever holds the service-role key.
  created_at            timestamptz not null default now(),
  unique (ticker, predicted_date)
);

alter table live_predictions enable row level security;

do $$
begin
  if not exists (
    select 1 from pg_policies
    where tablename = 'live_predictions' and policyname = 'public read live_predictions'
  ) then
    create policy "public read live_predictions" on live_predictions
      for select using (true);
  end if;
end $$;

create index if not exists idx_live_predictions_ticker_date
  on live_predictions (ticker, predicted_date desc);


-- ── portfolio_holdings ─────────────────────────────────────────────────────
-- Per-account portfolio storage (Portfolio.jsx). Guest mode (localStorage)
-- remains available for anyone not signed in; this is only used when
-- logged in, RLS-scoped so each user only ever sees their own rows.
create table if not exists portfolio_holdings (
  id         uuid primary key default gen_random_uuid(),
  user_id    uuid not null references auth.users(id) on delete cascade,
  ticker     text not null,
  shares     numeric not null,
  buy_price  numeric not null,
  created_at timestamptz not null default now(),
  unique (user_id, ticker)
);

alter table portfolio_holdings enable row level security;

do $$
begin
  if not exists (
    select 1 from pg_policies
    where tablename = 'portfolio_holdings' and policyname = 'own holdings'
  ) then
    create policy "own holdings" on portfolio_holdings
      for all using (auth.uid() = user_id) with check (auth.uid() = user_id);
  end if;
end $$;


-- ── game_progress ──────────────────────────────────────────────────────────
-- One row per signed-in user (Game.jsx), upserted on every score change.
-- RLS scopes writes/reads to the owning user; a second public-read policy
-- allows the cross-user leaderboard (top high scores) without exposing
-- anything beyond what's already shown there.
create table if not exists game_progress (
  user_id    uuid primary key references auth.users(id) on delete cascade,
  score      integer not null default 0,
  streak     integer not null default 0,
  high_score integer not null default 0,
  total      integer not null default 0,
  correct    integer not null default 0,
  history    jsonb not null default '[]',
  updated_at timestamptz not null default now()
);

alter table game_progress enable row level security;

do $$
begin
  if not exists (
    select 1 from pg_policies
    where tablename = 'game_progress' and policyname = 'own progress'
  ) then
    create policy "own progress" on game_progress
      for all using (auth.uid() = user_id) with check (auth.uid() = user_id);
  end if;
  if not exists (
    select 1 from pg_policies
    where tablename = 'game_progress' and policyname = 'public read leaderboard'
  ) then
    create policy "public read leaderboard" on game_progress
      for select using (true);
  end if;
end $$;
