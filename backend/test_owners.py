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


if __name__ == "__main__":
    unittest.main()
