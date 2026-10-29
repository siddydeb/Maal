#!/usr/bin/env python3
"""Push local main to both GitHub repos that Render deploys from.

  wellswenger-svg/Maal  (origin, primary API, Render account 1)  token: github=
  siddydeb/Maal         (failover API, Render account 2)          token: GITHUB2=

Tokens come from gitignored tokens&cmd and are passed per command as an auth
header — never written to git config or Windows Credential Manager, so whatever
GitHub login is cached on this PC (e.g. another account) is not used.

Usage (repo root):
  python scripts/push_both.py            # normal push
  python scripts/push_both.py --force    # after a history rewrite
  python scripts/push_both.py --only origin|backup
"""

from __future__ import annotations

import argparse
import base64
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TOKENS_FILE = REPO_ROOT / "tokens&cmd"
TARGETS = {
    "origin": ("https://github.com/wellswenger-svg/Maal.git", "github"),
    "backup": ("https://github.com/siddydeb/Maal.git", "github2"),
}


def _tokens() -> dict[str, str]:
    out: dict[str, str] = {}
    for line in TOKENS_FILE.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        line = line.strip()
        if "=" in line and not line.startswith(("#", ";")):
            k, v = line.split("=", 1)
            out.setdefault(k.strip().lower(), v.strip())
    return out


def push(url: str, token: str, *, force: bool) -> int:
    auth = base64.b64encode(f"x-access-token:{token}".encode()).decode()
    cmd = [
        "git",
        # Empty helper list: ignore cached Windows / gh logins for this push.
        "-c", "credential.helper=",
        "-c", f"http.https://github.com/.extraheader=AUTHORIZATION: basic {auth}",
        "push",
    ]
    if force:
        cmd.append("--force")
    cmd += [url, "main:main"]
    return subprocess.run(cmd, cwd=REPO_ROOT).returncode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", choices=sorted(TARGETS))
    args = ap.parse_args()

    tokens = _tokens()
    failed = 0
    for name, (url, key) in TARGETS.items():
        if args.only and name != args.only:
            continue
        token = tokens.get(key)
        if not token:
            print(f"[{name}] tokens&cmd has no {key}= — skipped")
            failed += 1
            continue
        print(f"[{name}] pushing main -> {url}{' (force)' if args.force else ''}")
        if push(url, token, force=args.force) != 0:
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
