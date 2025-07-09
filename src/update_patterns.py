# update_patterns.py
import os, sys

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from src.helpers import initialize_logging, timer_wrap

filename = os.path.basename(__file__)
logger = initialize_logging(filename)

import requests, yaml, re, datetime, pathlib, json
# REG_URL = 'https://bioregistry.io/api/registry'
REG_URL = "https://raw.githubusercontent.com/biopragmatics/bioregistry/main/exports/registry/registry.json"

DEFAULT_PATTERNS = {
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
        'WIKIDATA': re.compile(r'^(Q|P|E|L)\d+$'),
        'OCID': re.compile(r'^[0-9]{12}$'),
        'DOID': re.compile(r'^\d+$')
    }

@timer_wrap
def download_bioregistry_data(url: str = REG_URL):
    logger.info(f"Downloading bioregistry data from {url}")
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading bioregistry data: {e}")
        raise
    logger.info(f"Downloaded {len(data)} prefixes from bioregistry")

    # save full data
    pathlib.Path("artifacts/bioregistry_data.json").write_text(
        json.dumps(data, indent=2)
    )
    logger.info(f"Saved {len(data)} prefixes to artifacts/bioregistry_data.json")

def load_bioregistry_data(local_path: str = "artifacts/bioregistry_data.json") -> dict:
    data = json.loads(pathlib.Path(local_path).read_text())
    # Update the patterns dictionary to store pattern strings, not compiled regex
    patterns = {
        k.upper(): {
            "regex": v["pattern"],
            "source": "bioregistry", 
            "updated": datetime.date.today().isoformat(),
        }
        for k, v in data.items()
        if "pattern" in v and v["pattern"]
    }

    # This will now work since we're storing strings, not compiled regex objects
    pathlib.Path("artifacts/entity_patterns.yaml").write_text(
        yaml.safe_dump(patterns, sort_keys=True, allow_unicode=True)
    )
    print(f"Wrote {len(patterns)} patterns.")
    return patterns


@timer_wrap
def load_entity_patterns(local_path="artifacts/entity_patterns.yaml") -> dict[str, re.Pattern]:
    try:
        if not pathlib.Path(local_path).exists():
            logger.warning(f"File {local_path} does not exist. Downloading bioregistry data.")
            download_bioregistry_data()
            load_bioregistry_data()
            text = pathlib.Path("artifacts/entity_patterns.yaml").read_text()
        else:
            text = pathlib.Path(local_path).read_text()
        raw = yaml.safe_load(text)
        compiled = {k: re.compile(v["regex"], re.IGNORECASE) for k, v in raw.items()}
        logger.info(f"Loaded {len(compiled)} patterns from {local_path}")
        # Check if any of the default patterns are in the compiled patterns
        for k, v in DEFAULT_PATTERNS.items():
            if k in compiled:
                logger.info(f"Default pattern {k} is in the compiled patterns. Updating it.")
                compiled[k] = v
        return compiled
    except Exception as e:
        logger.warning(f"Falling back to baked-in patterns – {e}")
        return DEFAULT_PATTERNS

ENTITY_PATTERNS = load_entity_patterns()