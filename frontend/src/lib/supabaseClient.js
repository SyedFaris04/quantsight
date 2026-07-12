/**
 * frontend/src/lib/supabaseClient.js
 * ─────────────────────────────────────────────────────────────────
 * Single shared Supabase client — handles both Auth (email login) and
 * direct Postgres reads/writes for portfolio_holdings
 * and game_progress. Row Level Security on those tables (see the SQL
 * in the plan) means every query here is automatically scoped to
 * whoever is signed in — no manual user_id filtering needed client-side.
 *
 * VITE_SUPABASE_URL / VITE_SUPABASE_ANON_KEY come from a Supabase
 * project's Settings → API page. The anon key is safe to expose in
 * frontend code — RLS is what actually protects the data, not secrecy
 * of this key.
 * ─────────────────────────────────────────────────────────────────
 */

import { createClient } from "@supabase/supabase-js";

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL;
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY;

if (!supabaseUrl || !supabaseAnonKey) {
  console.warn(
    "VITE_SUPABASE_URL / VITE_SUPABASE_ANON_KEY are not set — " +
    "login and per-account storage will not work. Guest mode " +
    "(localStorage) is unaffected."
  );
}

export const supabase = createClient(
  supabaseUrl || "https://placeholder.supabase.co",
  supabaseAnonKey || "placeholder-anon-key"
);
