"""PIN map parsing and lockout — no production PINs in fixtures."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import bcrypt

from backend.owners import parse_pin_map


def _hash(pin: str) -> str:
    return bcrypt.hashpw(pin.encode("utf-8"), bcrypt.gensalt(rounds=4)).decode("ascii")


class ParsePinMapTests(unittest.TestCase):
    def test_empty_is_locked(self):
        self.assertEqual(parse_pin_map(""), {})
        self.assertEqual(parse_pin_map("   "), {})
        self.assertEqual(parse_pin_map(None or ""), {})

    def test_ignores_plaintext_and_junk(self):
        self.assertEqual(parse_pin_map("nocolon, :missingpin, pinonly:"), {})
        self.assertEqual(parse_pin_map("1111:u1, 2222:u2"), {})

    def test_parses_bcrypt_pairs(self):
        h1 = _hash("1111")
        h2 = _hash("2222")
        self.assertEqual(
            parse_pin_map(f"{h1}:u1, {h2}:u2"),
            {h1: "u1", h2: "u2"},
        )


class UnlockLockoutTests(unittest.TestCase):
    def test_no_env_means_disabled(self):
        from backend import owners

        class _S:
            wan_pins = ""
            wan_auth_secret = ""
            wan_admin_owner = "uadmin"
            wan_tester_owner = "utester"

        with patch.object(owners, "get_settings", return_value=_S()):
            self.assertFalse(owners.unlock_enabled())
            self.assertIsNone(owners.owner_for_pin("0000"))
            self.assertIsNone(owners.owner_from_token("u1.deadbeef"))

    def test_tester_owner_flag(self):
        from backend import owners

        hashed = _hash("5555")

        class _S:
            wan_pins = f"{hashed}:utester"
            wan_auth_secret = "test-secret-not-for-prod"
            wan_admin_owner = "uadmin"
            wan_tester_owner = "utester"

        with patch.object(owners, "get_settings", return_value=_S()):
            self.assertEqual(owners.owner_for_pin("5555"), "utester")
            self.assertTrue(owners.is_tester_owner("utester"))
            self.assertFalse(owners.is_tester_owner("uadmin"))
            self.assertFalse(owners.is_admin_owner("utester"))

    def test_saga_owner_flag(self):
        from backend import owners

        class _S:
            wan_saga_owner = "usaga"

        with patch.object(owners, "get_settings", return_value=_S()):
            self.assertTrue(owners.is_saga_owner("usaga"))
            self.assertFalse(owners.is_saga_owner("utester"))
            self.assertFalse(owners.is_saga_owner(None))

    def test_bcrypt_match_and_miss(self):
        from backend import owners

        hashed = _hash("1111")

        class _S:
            wan_pins = f"{hashed}:u1"
            wan_auth_secret = "test-secret-not-for-prod"
            wan_admin_owner = "uadmin"
            wan_tester_owner = "utester"

        with patch.object(owners, "get_settings", return_value=_S()):
            self.assertTrue(owners.unlock_enabled())
            self.assertEqual(owners.owner_for_pin("1111"), "u1")
            self.assertIsNone(owners.owner_for_pin("9999"))
            self.assertIsNone(owners.owner_for_pin("1111:u1"))

    def test_token_version_revokes_old_tokens(self):
        from backend import owners

        class _S:
            wan_pins = f"{_hash('1111')}:usaga,{_hash('2222')}:u2"
            wan_auth_secret = "test-secret-not-for-prod"
            wan_token_versions = ""

        settings = _S()
        with patch.object(owners, "get_settings", return_value=settings):
            old_saga, old_u2 = owners.make_token("usaga"), owners.make_token("u2")
            settings.wan_token_versions = "usaga:2"
            self.assertIsNone(owners.owner_from_token(old_saga))
            self.assertEqual(owners.owner_from_token(old_u2), "u2")
            self.assertEqual(owners.owner_from_token(owners.make_token("usaga")), "usaga")

    def test_saga_script_key(self):
        from backend import owners

        key = "sk_saga_" + "x" * 40

        class _S:
            wan_pins = ""
            wan_auth_secret = ""
            wan_saga_owner = "usaga"
            wan_saga_script_keys = f"abc, {owners.script_key_digest(key)}"

        settings = _S()
        with patch.object(owners, "get_settings", return_value=settings):
            self.assertEqual(owners.owner_from_token(key), "usaga")
            self.assertEqual(owners.owner_from_token(f"Bearer {key}"), "usaga")
            self.assertIsNone(owners.owner_from_token(key + "y"))
            self.assertIsNone(owners.owner_from_token("x" * 40))
            settings.wan_saga_script_keys = "abc"
            self.assertIsNone(owners.owner_from_token(key))


if __name__ == "__main__":
    unittest.main()
