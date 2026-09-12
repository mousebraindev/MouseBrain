"""
The repo must never carry a key, a credential, a machine path or a person.

Two tests: the repo is clean, and the scanner is not asleep.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import privacy_scan  # noqa: E402


def test_repo_is_clean():
    hits = privacy_scan.scan()
    assert hits == [], "\n".join(f"{h[0]}:{h[1]} {h[2]}: {h[3]}" for h in hits)


def test_scanner_catches_what_it_claims_to(tmp_path):
    """The bait is assembled at runtime, never written out as a literal: a test
    file full of realistic-looking secrets is exactly what must not ship."""
    planted = {
        "key.py": 'API_KEY = "' + "sk-" + "a" * 24 + '"',
        "path.py": 'HOME = r"C:' + chr(92) + "Users" + chr(92) + 'someone"',
        "mail.md": "write to " + "someone.real" + "@" + "mailhost.example",
        "handle.md": "posted by " + "@" + "somebodysaccount",
        "wallet.md": "send it to " + "0x" + "5290840009852788" + "6E0F7030069857D2E4169EE7",
        "aws.py": 'ID = "' + "AKIA" + "IOSFODNN7EXAMPL" + 'E"',
    }
    for name, body in planted.items():
        (tmp_path / name).write_text(body, encoding="utf-8")

    hits = privacy_scan.scan([tmp_path])
    found = {Path(h[0]).name for h in hits}
    missed = set(planted) - found
    assert not missed, f"the scanner walked past: {sorted(missed)}"


def test_gitignore_blocks_the_obvious():
    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for pattern in (".env", "data/", "build/", "__pycache__/", "checkpoints/"):
        assert pattern in ignored, f"{pattern} is not ignored"
