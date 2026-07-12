/**
 * frontend/src/context/AuthContext.jsx
 * ─────────────────────────────────────────────────────────────────
 * Wraps the app, exposes the current Supabase auth state and the
 * sign-in/sign-up/sign-out actions used by Login.jsx and Sidebar.jsx.
 *
 * `user` is null for guest visitors — every page that reads it should
 * treat that as "use localStorage / guest mode", not an error state.
 * See Portfolio.jsx / Game.jsx for the logged-in-vs-guest branch.
 * ─────────────────────────────────────────────────────────────────
 */

import { createContext, useContext, useEffect, useState } from "react";
import { supabase } from "../lib/supabaseClient";

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [session, setSession] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    supabase.auth.getSession().then(({ data: { session } }) => {
      setSession(session);
      setLoading(false);
    });

    const { data: { subscription } } = supabase.auth.onAuthStateChange(
      (_event, session) => {
        setSession(session);
      }
    );

    return () => subscription.unsubscribe();
  }, []);

  function signInWithGoogle() {
    return supabase.auth.signInWithOAuth({
      provider: "google",
      options: { redirectTo: window.location.origin },
    });
  }

  function signInWithEmail(email, password) {
    return supabase.auth.signInWithPassword({ email, password });
  }

  function signUpWithEmail(email, password) {
    return supabase.auth.signUp({ email, password });
  }

  function signOut() {
    return supabase.auth.signOut();
  }

  const value = {
    session,
    user: session?.user ?? null,
    loading,
    signInWithGoogle,
    signInWithEmail,
    signUpWithEmail,
    signOut,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within an AuthProvider");
  return ctx;
}
