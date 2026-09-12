"""
Refuse to ship anything personal.

Runs over every tracked file and fails on anything that looks like a secret, a
credential, a machine-specific path, or a person. It is wired into the test
suite and into CI, so this is not a promise anyone has to remember to keep.

    python tools/privacy_scan.py            # scan the repo
    python tools/privacy_scan.py path/...   # scan specific paths

Exit code 0 clean, 1 if anything was found.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Extensions worth reading as text. Binaries and data are skipped.
TEXT = {".py", ".md", ".txt", ".toml", ".cfg", ".ini", ".yml", ".yaml", ".json",
        ".js", ".ts", ".html", ".css", ".sh", ".bat", ".ps1", ".rst", ""}

SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", "node_modules", "data",
             "build", "dist", "checkpoints", ".pytest_cache", ".mypy_cache"}

RULES = [
    ("absolute home path",
     re.compile(r"(?:[A-Za-z]:[\\/]+Users[\\/]+|/home/|/Users/)[A-Za-z0-9._-]+", re.I)),
    ("windows user profile variable",
     re.compile(r"%USERPROFILE%|\$env:USERPROFILE", re.I)),
    ("email address",
     re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("social handle",
     re.compile(r"(?<![\w/])@[A-Za-z0-9_]{3,30}\b")),
    ("private key block",
     re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |PGP )?PRIVATE KEY-----")),
    ("api key assignment",
     re.compile(r"(?i)\b(api[_-]?key|secret|passwd|password|access[_-]?token|"
                r"bearer|auth[_-]?token|client[_-]?secret)\b\s*[:=]\s*"
                r"['\"][^'\"]{6,}['\"]")),
    ("aws access key id", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("slack or bot token", re.compile(r"\b(?:xox[baprs]-|bot[0-9]{6,}:)[A-Za-z0-9-]{10,}")),
    ("openai-style key", re.compile(r"\bsk-[A-Za-z0-9]{20,}\b")),
    ("github token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{5,}")),
    ("private ip", re.compile(r"\b(?:10\.\d{1,3}|192\.168|172\.(?:1[6-9]|2\d|3[01]))"
                              r"\.\d{1,3}\.\d{1,3}\b")),
    ("crypto address",
     re.compile(r"\b(?:0x[a-fA-F0-9]{40}|bc1[a-z0-9]{20,}|[13][a-km-zA-HJ-NP-Z1-9]{25,34})\b")),
    ("telegram or discord webhook",
     re.compile(r"(?i)(api\.telegram\.org/bot|discord(?:app)?\.com/api/webhooks)")),
]

# Phrases that are allowed to look like a hit. Keep this list short and boring:
# every entry here is a hole in the scan.
ALLOW = re.compile(
    r"(?:"
    r"example\.com"
    r"|your[_-]?key"
    r"|@classmethod|@property|@dataclass|@staticmethod|@abstractmethod|@pytest"
    r"|@param|@returns"
    r")", re.I)


def files(paths=None):
    if paths:
        for p in paths:
            p = Path(p)
            if p.is_dir():
                yield from (f for f in p.rglob("*") if f.is_file())
            elif p.is_file():
                yield p
        return
    try:  # prefer what git would actually publish
        out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True,
                             text=True, check=True).stdout.split("\n")
        for line in out:
            if line.strip():
                yield ROOT / line.strip()
        return
    except Exception:
        pass
    for f in ROOT.rglob("*"):
        if f.is_file() and not any(part in SKIP_DIRS for part in f.parts):
            yield f


def scan(paths=None):
    hits = []
    for f in files(paths):
        if not f.exists() or f.suffix.lower() not in TEXT:
            continue
        if any(part in SKIP_DIRS for part in f.parts):
            continue
        if f.name == "privacy_scan.py":
            continue                      # the rules themselves are not a leak
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if ALLOW.search(line):
                continue
            for name, rx in RULES:
                m = rx.search(line)
                if m:
                    rel = f.relative_to(ROOT) if ROOT in f.parents or f.parent == ROOT else f
                    hits.append((str(rel), i, name, m.group(0)[:60]))
    return hits


def main(argv=None) -> int:
    hits = scan(argv or None)
    if not hits:
        print("privacy scan: clean")
        return 0
    print(f"privacy scan: {len(hits)} finding(s)\n")
    for path, line, name, snippet in hits:
        print(f"  {path}:{line}  {name}: {snippet}")
    print("\nNothing personal ships. Remove these, or add a narrow exception to "
          "ALLOW in tools/privacy_scan.py and say why in the commit.")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
