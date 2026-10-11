import { useEffect, useState } from "react";
import { clearSession, isSessionValid, saveSession } from "./auth";
import { getAuthStatus, unlockAccount } from "./api";

export default function PinGate({ children }) {
  const [unlocked, setUnlocked] = useState(() => isSessionValid());
  const [locked, setLocked] = useState(true);
  const [statusReady, setStatusReady] = useState(false);
  const [pin, setPin] = useState("");
  const [error, setError] = useState("");
  const [shake, setShake] = useState(false);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const status = await getAuthStatus();
        if (cancelled) return;
        const enabled = status?.unlock_enabled !== false;
        setLocked(!enabled);
        if (!enabled) {
          clearSession();
          setUnlocked(false);
        }
      } catch {
        // Legacy API has no /api/auth/status — still show the PIN form.
        if (cancelled) return;
        setLocked(false);
      } finally {
        if (!cancelled) setStatusReady(true);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!unlocked) return;
    const check = () => {
      if (!isSessionValid()) {
        setUnlocked(false);
        setPin("");
      }
    };
    const onVis = () => {
      if (document.visibilityState === "visible") check();
    };
    document.addEventListener("visibilitychange", onVis);
    const id = window.setInterval(check, 60_000);
    return () => {
      document.removeEventListener("visibilitychange", onVis);
      window.clearInterval(id);
    };
  }, [unlocked]);

  if (unlocked) return children;

  async function onSubmit(e) {
    e.preventDefault();
    if (locked || !pin) return;
    setBusy(true);
    setError("");
    try {
      const data = await unlockAccount(pin);
      saveSession({
        owner: data.owner,
        token: data.token,
        admin: Boolean(data.admin),
        tester: Boolean(data.tester),
        saga: Boolean(data.saga),
      });
      setPin("");
      setUnlocked(true);
    } catch (err) {
      clearSession();
      const msg = String(err.message || err || "");
      if (/disabled/i.test(msg)) {
        setLocked(true);
        setError("Unlock is disabled");
      } else {
        setError("Wrong PIN");
      }
      setShake(true);
      setPin("");
      window.setTimeout(() => setShake(false), 420);
    } finally {
      setBusy(false);
    }
  }

  return (
    <>
      <div className="noise" aria-hidden="true" />
      <main className="pin-shell">
        <div className={`pin-card${shake ? " pin-shake" : ""}`}>
          <div className="pin-brand-row">
            <span className="app-mark" aria-hidden="true">
              W
            </span>
            <p className="pin-brand">Wan</p>
          </div>
          <p className="pin-tag">Create. Transform. Wow.</p>
          {!statusReady || locked ? (
            <>
              <h1 className="pin-title">Access locked</h1>
              <p className="pin-sub">
                Unlock is disabled. No PIN will open this app.
              </p>
            </>
          ) : (
            <>
              <h1 className="pin-title">Enter PIN</h1>
              <p className="pin-sub">Private library · unlock for 24 hours</p>
              <form className="pin-form" onSubmit={onSubmit}>
                <input
                  className="pin-input"
                  type="password"
                  inputMode="numeric"
                  autoComplete="one-time-code"
                  pattern="[0-9]*"
                  maxLength={16}
                  value={pin}
                  onChange={(e) => {
                    setPin(e.target.value.replace(/\D/g, ""));
                    setError("");
                  }}
                  placeholder="••••"
                  autoFocus
                  aria-label="PIN"
                  disabled={busy}
                />
                {error ? (
                  <p className="pin-error" role="alert">
                    {error}
                  </p>
                ) : null}
                <button type="submit" className="pin-submit" disabled={!pin || busy}>
                  {busy ? "Unlocking…" : "Unlock"}
                </button>
              </form>
            </>
          )}
        </div>
      </main>
    </>
  );
}
