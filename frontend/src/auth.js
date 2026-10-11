const SESSION_KEY = "wan_device_session_v4";
const SESSION_MS = 24 * 60 * 60 * 1000;
const LEGACY_SESSION_KEYS = [
  "wan_device_session_v1",
  "wan_device_session_v2",
  "wan_device_session_v3",
];

function readSession() {
  try {
    const raw = localStorage.getItem(SESSION_KEY);
    if (!raw) return null;
    const data = JSON.parse(raw);
    const unlockedAt = Number(data?.unlockedAt);
    if (!Number.isFinite(unlockedAt)) return null;
    if (Date.now() - unlockedAt >= SESSION_MS) {
      clearSession();
      return null;
    }
    if (!data?.owner || !data?.token) return null;
    return data;
  } catch {
    return null;
  }
}

export function isSessionValid() {
  return Boolean(readSession());
}

export function getSession() {
  return readSession();
}

export function getOwnerId() {
  return readSession()?.owner || null;
}

export function isAdminOwner() {
  return Boolean(readSession()?.admin);
}

export function isTesterOwner() {
  return Boolean(readSession()?.tester);
}

export function isSagaOwner() {
  return Boolean(readSession()?.saga);
}

export function getAccessToken() {
  return readSession()?.token || null;
}

export function saveSession({ owner, token, admin = false, tester = false, saga = false }) {
  if (!owner || !token) return;
  localStorage.setItem(
    SESSION_KEY,
    JSON.stringify({
      owner,
      token,
      admin: Boolean(admin),
      tester: Boolean(tester),
      saga: Boolean(saga),
      unlockedAt: Date.now(),
    })
  );
  for (const key of LEGACY_SESSION_KEYS) {
    localStorage.removeItem(key);
  }
}

export function clearSession() {
  localStorage.removeItem(SESSION_KEY);
  for (const key of LEGACY_SESSION_KEYS) {
    localStorage.removeItem(key);
  }
}
