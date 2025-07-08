# update_patterns.py
import os
from .helpers import initialize_logging

filename = os.path.basename(__file__)
logger = initialize_logging(filename)

import requests, yaml, re, datetime, pathlib, json, gzip
REG_URL = "https://raw.githubusercontent.com/biopragmatics/bioregistry/main/bioregistry/data/bioregistry.json.gz"
data = json.loads(gzip.decompress(requests.get(REG_URL, timeout=30).content))

patterns = {}
allow_prefixes = {"geo", "sra", "ena.embl", "ena.run", "pdb", "encode", "pride", "doi", "pubmed", "dryad", "oci", "pubchem", "chembl", "biostudies", "ncbi.gc", "ncbi.resource", "ncbi.genome", "biosystems", "bioportal"}  # add/modify
for prefix, entry in data.items():
    if prefix not in allow_prefixes:
        continue
    pat = entry.get("pattern")  # Bioregistry field
    if not pat:
        continue
    try:
        re.compile(pat)         # validate
    except re.error:
        continue
    patterns[prefix.upper()] = {
        "regex": pat,
        "source": "bioregistry",
        "updated": datetime.date.today().isoformat(),
    }

pathlib.Path("artifacts/entity_patterns.yaml").write_text(
    yaml.safe_dump(patterns, sort_keys=True, allow_unicode=True)
)
print(f"Wrote {len(patterns)} patterns.")

# semantic_chunking.py
# from importlib.resources import files
import yaml, re, pathlib

def load_entity_patterns(local_path="artifacts/entity_patterns.yaml") -> dict[str, re.Pattern]:
    try:
        text = pathlib.Path(local_path).read_text()
        raw = yaml.safe_load(text)
        compiled = {k: re.compile(v["regex"], re.IGNORECASE) for k, v in raw.items()}
        logger.info(f"Loaded {len(compiled)} patterns from {local_path}")
        return compiled
    except Exception as e:
        logger.warning(f"Falling back to baked-in patterns – {e}")
        return {
        # Existing patterns
        'DOI': re.compile(r'\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+\b', re.IGNORECASE),
        'GEO_Series': re.compile(r'\bGSE\d{3,6}\b'),
        'GEO_Sample': re.compile(r'\bGSM\d{3,6}\b'),
        'SRA_Run': re.compile(r'\bSRR\d{5,}\b'),
        'PDB_ID': re.compile(r'\b[A-Za-z0-9]{4}\b'),
        'PDB_DOI': re.compile(r'\b10\.2210/pdb[A-Za-z0-9]{4}/pdb\b', re.IGNORECASE),
        'ArrayExpress': re.compile(r'\bE-[A-Z]+-\d+\b'),
        'dbGaP': re.compile(r'\bphs\d{6}\b'),
        'TCGA': re.compile(r'\bTCGA-[A-Z0-9-]+\b'),
        'ENA_Project': re.compile(r'\bPRJ[EDN][A-Z]\d+\b'),
        'ENA_Study': re.compile(r'\bERP\d{6,}\b'),
        'ENA_Sample': re.compile(r'\bSAM[EDN][A-Z]?\d+\b'),
        
        # New additions
        'SRA_Experiment': re.compile(r'\bSRX\d{5,}\b'),
        'SRA_Project': re.compile(r'\bSRP\d{5,}\b'),
        'SRA_Sample': re.compile(r'\bSRS\d{5,}\b'),
        'SRA_Study': re.compile(r'\bSRA\d{5,}\b'),
        'RefSeq_Chromosome': re.compile(r'\bNC_\d{6,}(?:\.\d+)?\b'),
        'ENA_Run': re.compile(r'\bERR\d{6,}\b'),
        'ENA_Experiment': re.compile(r'\bERX\d{6,}\b'),
        'ENA_Sample2': re.compile(r'\bERS\d{6,}\b'),
        'DDBJ_Run': re.compile(r'\bDRR\d{6,}\b'),
        'DDBJ_Experiment': re.compile(r'\bDRX\d{6,}\b'),
        'ENCODE_Assay': re.compile(r'\bENCSR[0-9A-Z]{6}\b'),
        'PRIDE': re.compile(r'\bPXD\d{6,}\b'),
    }

ENTITY_PATTERNS = load_entity_patterns()