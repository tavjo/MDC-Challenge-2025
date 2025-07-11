"""
Assemble the master PATTERNS dict:
1.  start with DEFAULT_PATTERNS from your existing code
2.  overlay any curated tweaks in citation_patterns.yaml  (optional)
3.  extend with selected Bioregistry regexes
4.  wrap everything with a global '>=1 digit, >=1 uppercase, len>=5' guard
"""
from __future__ import annotations
import json, re, requests, pathlib, yaml, datetime
from typing import Dict

# -------- 1.1  your in-repo defaults  -----------------
from src.update_patterns import DEFAULT_PATTERNS      # <- your current dict

# -------- 1.2  optional manual overrides  -------------
OVERRIDE_FILE = pathlib.Path("artifacts/citation_patterns.yaml")
overrides: Dict[str,str] = yaml.safe_load(OVERRIDE_FILE.read_text()) if OVERRIDE_FILE.exists() else {}

# -------- 1.3  Bioregistry slice  ---------------------
BIOREG_DIR = pathlib.Path("artifacts/bioregistry")
BIOREG_DIR.mkdir(parents=True, exist_ok=True)
BIOREG_JSON = BIOREG_DIR / f"registry_{datetime.date.today()}.json"

# Download with fallback to cached copy
if not BIOREG_JSON.exists():
    try:
        BIOREG_JSON.write_bytes(requests.get("https://bioregistry.io/registry.json", timeout=30).content)
    except requests.exceptions.RequestException as e:
        # Fallback to most recent cached copy
        cached_files = list(BIOREG_DIR.glob("registry_*.json"))
        if cached_files:
            most_recent = max(cached_files, key=lambda f: f.stat().st_mtime)
            import shutil
            shutil.copy2(most_recent, BIOREG_JSON)
        else:
            raise FileNotFoundError(f"Bioregistry download failed and no cached copy found: {e}")

bioreg = json.loads(BIOREG_JSON.read_text())

# keep only prefixes you *don't* already define
bioreg_subset = {
    k.upper(): v["pattern"]           # 'pattern' field documented in Bioregistry schema
    for k, v in bioreg.items()
    if k.upper() not in DEFAULT_PATTERNS and k.upper() not in overrides
}

# -------- 1.4  helper to add global guard  ------------
def _tighten(pattern_body: str) -> str:
    # ≥1 digit, ≥1 uppercase, min len 5 - use lookahead to avoid forcing preamble
    # Note: \b treats underscore as word boundary, so we use explicit boundary check
    return rf"(?=[A-Z]*\d)(?=[A-Z0-9_-]*[A-Z])[A-Z0-9_-]{{5,}}(?:{pattern_body})?"

# -------- 1.5  compile master dict with precedence ----
PATTERNS: Dict[str, re.Pattern] = {}

#   (a) defaults first  -> lowest priority (use as-is, don't re-wrap)
for key, pat in DEFAULT_PATTERNS.items():
    PATTERNS[key.upper()] = pat  # Keep compiled pattern as-is

#   (b) YAML overrides  -> overwrite defaults (apply _tighten to raw strings)
for key, body in overrides.items():
    PATTERNS[key.upper()] = re.compile(rf"(?<![A-Za-z0-9_]){_tighten(body)}(?![A-Za-z0-9_])", re.I)

#   (c) Bioregistry extras -> only if new key (apply _tighten to raw strings)
for key, body in bioreg_subset.items():
    PATTERNS[key.upper()] = re.compile(rf"(?<![A-Za-z0-9_]){_tighten(body)}(?![A-Za-z0-9_])", re.I) 