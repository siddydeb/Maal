"""Owner isolation for private generation libraries.

PINs are stored only as bcrypt hashes in env ``WAN_PINS`` (never in git
or frontend JS). Generate hashes with ``frontend/scripts/hash_pin.js``
(bcryptjs). Empty / unset → unlock is disabled.
"""

from __future__ import annotations

import hashlib
import hmac
import re
from typing import Optional

import bcrypt

from backend.config import get_settings

# bcryptjs / Python bcrypt: $2a$ / $2b$ / $2y$ + cost + 53-char checksum
_BCRYPT_HASH = re.compile(r"^\$2[aby]\$\d{2}\$[./A-Za-z0-9]{53}$")


def parse_pin_map(raw: str) -> dict[str, str]:
    """Parse ``bcryptHash:owner,bcryptHash:owner``. Plaintext PINs are ignored."""
    out: dict[str, str] = {}
    for part in (raw or "").split(","):
        part = part.strip()
        if not part or ":" not in part:
            continue
        secret, owner = part.rsplit(":", 1)
        secret, owner = secret.strip(), owner.strip()
        if owner and _BCRYPT_HASH.match(secret):
            out[secret] = owner
    return out


def _pin_map() -> dict[str, str]:
    return parse_pin_map(get_settings().wan_pins)


def owner_ids() -> frozenset[str]:
    return frozenset(_pin_map().values())


def unlock_enabled() -> bool:
    return bool(_pin_map()) and bool((get_settings().wan_auth_secret or "").strip())


def is_admin_owner(owner: Optional[str]) -> bool:
    admin = (get_settings().wan_admin_owner or "").strip()
    return bool(owner) and bool(admin) and owner == admin


def is_tester_owner(owner: Optional[str]) -> bool:
    tester = (get_settings().wan_tester_owner or "").strip()
    return bool(owner) and bool(tester) and owner == tester


def is_saga_owner(owner: Optional[str]) -> bool:
    saga = (get_settings().wan_saga_owner or "").strip()
    return bool(owner) and bool(saga) and owner == saga


def _secret() -> bytes:
    raw = (get_settings().wan_auth_secret or "").strip()
    if not raw:
        return b""
    return raw.encode("utf-8")


def owner_for_pin(pin: str) -> Optional[str]:
    """Check submitted PIN against every stored bcrypt hash. None if locked/wrong."""
    submitted = str(pin or "").strip().encode("utf-8")
    if not submitted:
        return None
    mapping = _pin_map()
    if not mapping:
        return None
    found: Optional[str] = None
    for hashed, owner in mapping.items():
        try:
            if bcrypt.checkpw(submitted, hashed.encode("utf-8")):
                found = owner
        except ValueError:
            continue
    return found


SCRIPT_KEY_PREFIX = "sk_saga_"


def _token_versions() -> dict[str, str]:
    out: dict[str, str] = {}
    for part in (get_settings().wan_token_versions or "").split(","):
        owner, _, version = part.strip().partition(":")
        if owner.strip() and version.strip():
            out[owner.strip()] = version.strip()
    return out


def make_token(owner: str) -> str:
    secret = _secret()
    if not secret:
        raise ValueError("auth secret not configured")
    if owner not in owner_ids():
        raise ValueError("unknown owner")
    version = _token_versions().get(owner)
    signed = f"{owner}:v{version}" if version else owner
    sig = hmac.new(secret, signed.encode("utf-8"), hashlib.sha256).hexdigest()[:40]
    return f"{owner}.{sig}"


def script_key_digest(key: str) -> str:
    return hashlib.sha256(key.strip().encode("utf-8")).hexdigest()


def owner_from_script_key(key: str) -> Optional[str]:
    """A long-lived Saga script key -> the Saga owner (which can only use the Saga studio)."""
    saga = (get_settings().wan_saga_owner or "").strip()
    digests = [d.strip().lower() for d in (get_settings().wan_saga_script_keys or "").split(",") if d.strip()]
    if not saga or not digests or not key.startswith(SCRIPT_KEY_PREFIX):
        return None
    digest = script_key_digest(key)
    return saga if any(hmac.compare_digest(digest, d) for d in digests) else None


def owner_from_token(token: Optional[str]) -> Optional[str]:
    if not token or not isinstance(token, str):
        return None
    token = token.strip()
    if token.lower().startswith("bearer "):
        token = token[7:].strip()
    if token.startswith(SCRIPT_KEY_PREFIX):
        return owner_from_script_key(token)
    if not _secret():
        return None
    parts = token.split(".", 1)
    if len(parts) != 2:
        return None
    owner, _sig = parts
    if owner not in owner_ids():
        return None
    try:
        expected = make_token(owner)
    except ValueError:
        return None
    if not hmac.compare_digest(expected, token):
        return None
    return owner
