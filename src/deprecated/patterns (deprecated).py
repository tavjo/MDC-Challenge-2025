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
# from src.update_patterns import DEFAULT_PATTERNS      # <- your current dict

DEFAULT_PATTERNS = {
        # Existing patterns
        'DOI': re.compile(r'(?<![A-Za-z0-9_])10\.\d{4,9}/[-._;()/:A-Za-z0-9]+(?![A-Za-z0-9_])', re.IGNORECASE),
        'GEO_Series': re.compile(r'(?<![A-Za-z0-9_])GSE\d{3,6}(?![A-Za-z0-9_])', re.IGNORECASE),
        'GEO_Sample': re.compile(r'(?<![A-Za-z0-9_])GSM\d{3,8}(?![A-Za-z0-9_])', re.IGNORECASE),
        'SRA_Run': re.compile(r'(?<![A-Za-z0-9_])SRR\d{6,}(?![A-Za-z0-9_])', re.IGNORECASE),
        'PDB_ID': re.compile(r'(?<![A-Za-z0-9_])[0-9][A-Za-z0-9]{3}(?![A-Za-z0-9_])', re.IGNORECASE),
        'PDB_DOI': re.compile(r'(?<![A-Za-z0-9_])10\.2210/pdb[0-9][A-Za-z0-9]{3}/pdb(?![A-Za-z0-9_])', re.IGNORECASE),
        'ArrayExpress': re.compile(r'(?<![A-Za-z0-9_])E\-[A-Z]{3,6}\-\d+(?![A-Za-z0-9_])', re.IGNORECASE),
        'dbGaP': re.compile(r'(?<![A-Za-z0-9_])phs\d{6}(?:\.\d+)?(?:\.p\d+)?(?![A-Za-z0-9_])', re.IGNORECASE),
        'TCGA': re.compile(r'(?<![A-Za-z0-9_])TCGA\-[A-Z0-9]{2}\-[A-Z0-9]{4}(?:\-[A-Z0-9\-]+)?(?![A-Za-z0-9_])', re.IGNORECASE),
        'ENA_Project': re.compile(r'(?<![A-Za-z0-9_])PRJ[EDN][A-Z]\d{4,}(?![A-Za-z0-9_])', re.IGNORECASE),
        'ENA_Study': re.compile(r'(?<![A-Za-z0-9_])ERP\d{6,}(?![A-Za-z0-9_])', re.IGNORECASE),
        'ENA_Sample': re.compile(r'(?<![A-Za-z0-9_])SAM[EDN][A-Z]?\d{5,}(?![A-Za-z0-9_])', re.IGNORECASE),
        
        # New additions
        'SRA_Experiment': re.compile(r'(?<![A-Za-z0-9_])SRX\d{6,}(?![A-Za-z0-9_])', re.IGNORECASE),
        'SRA_Project': re.compile(r'(?<![A-Za-z0-9_])SRP\d{6,}(?![A-Za-z0-9_])', re.IGNORECASE),
        'SRA_Sample': re.compile(r'(?<![A-Za-z0-9_])SRS\d{6,}(?![A-Za-z0-9_])', re.IGNORECASE),
        'SRA_Study': re.compile(r'(?<![A-Za-z0-9_])SRA\d{5,}(?![A-Za-z0-9_])', re.IGNORECASE),
        'RefSeq_Chromosome': re.compile(r'(?<![A-Za-z0-9_])NC_\d{6,9}(?:\.\d+)?(?![A-Za-z0-9_])', re.IGNORECASE),
        'ENA_Run': re.compile(r'(?<![A-Za-z0-9_])ERR\d{6,}(?![A-Za-z0-9_])', re.IGNORECASE),
        'ENA_Experiment': re.compile(r'(?<![A-Za-z0-9_])ERX\d{6,}(?![A-Za-z0-9_])', re.IGNORECASE),
        'ENA_Sample2': re.compile(r'(?<![A-Za-z0-9_])ERS\d{6,}(?![A-Za-z0-9_])', re.IGNORECASE),
        'DDBJ_Run': re.compile(r'(?<![A-Za-z0-9_])DRR\d{6,}(?![A-Za-z0-9_])', re.IGNORECASE),
        'DDBJ_Experiment': re.compile(r'(?<![A-Za-z0-9_])DRX\d{6,}(?![A-Za-z0-9_])', re.IGNORECASE),
        'ENCODE_Assay': re.compile(r'(?<![A-Za-z0-9_])ENCSR[0-9A-Z]{6}(?![A-Za-z0-9_])', re.IGNORECASE),
        'PRIDE': re.compile(r'(?<![A-Za-z0-9_])PXD\d{6,}(?![A-Za-z0-9_])', re.IGNORECASE),
        'WIKIDATA': re.compile(r'(?<![A-Za-z0-9_])(?:Q|P|E|L)\d{1,8}(?![A-Za-z0-9_])', re.IGNORECASE),
        'OCID': re.compile(r'(?<![A-Za-z0-9_])\d{12}(?![A-Za-z0-9_])', re.IGNORECASE),
        'DOID': re.compile(r'(?<![A-Za-z0-9_])DOID:\d{4,7}(?![A-Za-z0-9_])', re.IGNORECASE),

        # from train labels
        ## small-molecule & protein resources
        "ChEMBL": re.compile(r"(?<![A-Za-z0-9_])CHEMBL\d{4,7}(?![A-Za-z0-9_])", re.IGNORECASE), # ✔ CHEMBL2031
        "InterPro": re.compile(r"(?<![A-Za-z0-9_])IPR\d{6}(?![A-Za-z0-9_])", re.IGNORECASE), # ✔ IPR000001
        "Pfam": re.compile(r"(?<![A-Za-z0-9_])PF\d{5}(?:\.\d+)?(?![A-Za-z0-9_])", re.IGNORECASE),# now allows optional version, e.g. PF00001.21
        "UniProt": re.compile(r"(?<![A-Za-z0-9_])(?:[A-NR-Z][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]{4}[A-Z0-9]{5})(?![A-Za-z0-9_])", re.IGNORECASE),# 6- & 10-char formats
        "CATH": re.compile(r"(?<![A-Za-z0-9_])\d+\.\d+\.\d+\.\d+(?![A-Za-z0-9_])", re.IGNORECASE), # 1.10.8.10
        "KEGG": re.compile(r"(?<![A-Za-z0-9_])K\d{5}(?![A-Za-z0-9_])", re.IGNORECASE), # K00001
        "dbSNP": re.compile(r"(?<![A-Za-z0-9_])rs\d{3,}(?![A-Za-z0-9_])", re.IGNORECASE),# rs12345

        ## sequence & genome repositories
        "Ensembl": re.compile(r"(?<![A-Za-z0-9_])ENS[A-Z]{0,5}[GTRP]\d{11}(?:\.\d+)?(?![A-Za-z0-9_])", re.IGNORECASE),#ENSG00000139618, ENSMUSG00000064341
        "RefSeq_mRNA": re.compile(r"(?<![A-Za-z0-9_])NM_\d{6,}(?:\.\d+)?(?![A-Za-z0-9_])", re.IGNORECASE),# NM_000546.6
        "RefSeq_Protein": re.compile(r"(?<![A-Za-z0-9_])NP_\d{6,}(?:\.\d+)?(?![A-Za-z0-9_])", re.IGNORECASE),# NP_000537.3
        "RefSeq_Genome": re.compile(r"(?<![A-Za-z0-9_])[A-Z]{2}_\d{6,}(?:\.\d+)?(?![A-Za-z0-9_])", re.IGNORECASE),# NC_000001.11
        "GenBank": re.compile(r"(?<![A-Za-z0-9_])[A-Z]{1,2}\d{5,8}(?:\.\d+)?(?![A-Za-z0-9_])", re.IGNORECASE),# M10051, AF231982.1
        "DDBJ": re.compile(r"(?<![A-Za-z0-9_])D\d{5}(?:-\d+)?(?:\.\d+)?(?![A-Za-z0-9_])", re.IGNORECASE),# D12345-2
        "GISAID": re.compile(r"(?<![A-Za-z0-9_])EPI_ISL_\d{6,}(?![A-Za-z0-9_])", re.IGNORECASE),# EPI_ISL_402124
        
        ## imaging, proteomics, models
        "EMPIAR": re.compile(r"(?<![A-Za-z0-9_])EMPIAR-\d{5,}(?![A-Za-z0-9_])", re.IGNORECASE),# EMPIAR-10028
        "PRIDE": re.compile(r"(?<![A-Za-z0-9_])PXD\d{6,}(?![A-Za-z0-9_])", re.IGNORECASE),# PXD000001
        "ExpressionAtlas": re.compile(r"(?<![A-Za-z0-9_])E\-PROT\-\d+(?![A-Za-z0-9_])", re.IGNORECASE),# E-PROT-17
        "BioModels": re.compile(r"(?<![A-Za-z0-9_])MODEL\d{10,13}(?![A-Za-z0-9_])", re.IGNORECASE),# MODEL1402200001
        
        ## antibodies, cell lines, others
        "HPA_Antibody": re.compile(r"(?<![A-Za-z0-9_])HPA\d{6}(?![A-Za-z0-9_])", re.IGNORECASE),# HPA002830
        "CAB_Antibody": re.compile(r"(?<![A-Za-z0-9_])CAB\d{6}(?![A-Za-z0-9_])", re.IGNORECASE),# CAB004270
        "Cellosaurus": re.compile(r"(?<![A-Za-z0-9_])CVCL_[A-Z0-9]{4,}(?![A-Za-z0-9_])", re.IGNORECASE),# CVCL_00000000
    }

# -------- 1.2  optional manual overrides  -------------
OVERRIDE_FILE = pathlib.Path("artifacts/citation_patterns.yaml")
overrides: Dict[str,str] = yaml.safe_load(OVERRIDE_FILE.read_text()) if OVERRIDE_FILE.exists() else {}

# -------- 1.3  Bioregistry slice  ---------------------
BIOREG_DIR = pathlib.Path("artifacts/bioregistry")
BIOREG_DIR.mkdir(parents=True, exist_ok=True)
BIOREG_JSON = BIOREG_DIR / f"registry_{datetime.date.today()}.json"

bioreg = {}
try:
    if not BIOREG_JSON.exists():
        try:
            BIOREG_JSON.write_bytes(requests.get("https://bioregistry.io/registry.json", timeout=30).content)
        except Exception as e:
            # Fallback to most recent cached copy
            cached_files = list(BIOREG_DIR.glob("registry_*.json"))
            if cached_files:
                most_recent = max(cached_files, key=lambda f: f.stat().st_mtime)
                import shutil
                shutil.copy2(most_recent, BIOREG_JSON)
            else:
                print(f"[patterns.py] Warning: Bioregistry download failed and no cached copy found: {e}")
                print("[patterns.py] Proceeding with only default and YAML override patterns.")
                BIOREG_JSON = None
    if BIOREG_JSON and BIOREG_JSON.exists():
        try:
            bioreg = json.loads(BIOREG_JSON.read_text())
        except Exception as e:
            print(f"[patterns.py] Warning: Failed to parse Bioregistry JSON: {e}")
            print("[patterns.py] Proceeding with only default and YAML override patterns.")
            bioreg = {}
except Exception as e:
    print(f"[patterns.py] Unexpected error loading Bioregistry: {e}")
    print("[patterns.py] Proceeding with only default and YAML override patterns.")
    bioreg = {}

# keep only prefixes you *don't* already define
bioreg_subset = {
    k.upper(): v["pattern"]
    for k, v in bioreg.items()
    if k.upper() not in DEFAULT_PATTERNS and k.upper() not in overrides
} if bioreg else {}

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