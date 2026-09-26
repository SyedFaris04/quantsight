-- Run once in the project's Supabase SQL Editor before deploying tracker v2.
-- Additive migration: legacy live_predictions rows are retained unchanged.
begin;
create table if not exists public.live_predictions_v2 (
  id uuid primary key default gen_random_uuid(),
  ticker text not null,
  predicted_date date not null,
  target_date date not null check (target_date > predicted_date),
  target_close_at timestamptz not null,
  horizon_sessions integer not null check (horizon_sessions = 5),
  protocol_version text not null check (protocol_version = 'nyse-close-5-v2'),
  model_version text not null,
  feature_version text not null,
  feature_values jsonb not null check (jsonb_typeof(feature_values) = 'object'),
  data_cutoff_at timestamptz not null,
  data_fetched_at timestamptz not null,
  generated_at timestamptz not null,
  record_before timestamptz not null,
  created_at timestamptz not null default now(),
  predicted_signal text not null check (predicted_signal in ('BUY', 'SELL')),
  confidence double precision not null check (confidence between 0 and 100),
  probability_up double precision not null check (probability_up between 0 and 1),
  raw_probability_up double precision not null check (raw_probability_up between 0 and 1),
  calibration_method text not null,
  price_basis text not null check (price_basis = 'yahoo_auto_adjust'),
  price_at_prediction double precision not null check (price_at_prediction > 0),
  resolved boolean not null default false,
  resolved_date date,
  resolved_at timestamptz,
  actual_close double precision,
  resolution_entry_close double precision,
  actual_return double precision,
  actual_signal text check (actual_signal in ('BUY', 'SELL')),
  correct boolean,
  unique (ticker, predicted_date, protocol_version),
  check (data_cutoff_at <= data_fetched_at and data_fetched_at <= generated_at),
  check (generated_at <= created_at and created_at < record_before),
  check (((not resolved and resolved_date is null and resolved_at is null
          and actual_close is null and resolution_entry_close is null
          and actual_return is null and actual_signal is null and correct is null)
         or (resolved and resolved_date = target_date and resolved_date is not null
          and resolved_at >= target_close_at + interval '20 minutes' and resolved_at is not null
          and actual_close > 0 and resolution_entry_close > 0
          and actual_close is not null and resolution_entry_close is not null
          and actual_return is not null and actual_signal is not null and correct is not null)) is true)
);

alter table public.live_predictions_v2 enable row level security;
revoke all on public.live_predictions_v2 from anon, authenticated;
grant select on public.live_predictions_v2 to anon, authenticated;
grant all on public.live_predictions_v2 to service_role;
drop policy if exists "public read live_predictions_v2" on public.live_predictions_v2;
create policy "public read live_predictions_v2" on public.live_predictions_v2 for select using (true);
create index if not exists idx_live_v2_pending on public.live_predictions_v2(target_date) where not resolved;
create index if not exists idx_live_v2_date on public.live_predictions_v2(predicted_date desc, id);

-- Freeze forecast fields during ordinary API writes; database owners remain trusted.
create or replace function public.guard_live_prediction_v2() returns trigger
language plpgsql set search_path = public as $$
begin
  if TG_OP = 'INSERT' then
    NEW.created_at := clock_timestamp();
    if NEW.resolved or NEW.created_at < NEW.data_cutoff_at + interval '20 minutes' then
      raise exception 'Forecast must be pending and recorded after the completed close';
    end if;
  elsif OLD.resolved or
    (to_jsonb(NEW) - array['resolved','resolved_date','resolved_at','actual_close',
                          'resolution_entry_close','actual_return','actual_signal','correct'])
    is distinct from
    (to_jsonb(OLD) - array['resolved','resolved_date','resolved_at','actual_close',
                          'resolution_entry_close','actual_return','actual_signal','correct']) then
    raise exception 'Forecasts and resolved outcomes are immutable';
  end if;
  if TG_OP = 'UPDATE' and NEW.resolved then
    NEW.resolved_at := clock_timestamp();
  end if;
  return NEW;
end $$;
drop trigger if exists live_prediction_v2_guard on public.live_predictions_v2;
create trigger live_prediction_v2_guard before insert or update on public.live_predictions_v2
for each row execute function public.guard_live_prediction_v2();
commit;
