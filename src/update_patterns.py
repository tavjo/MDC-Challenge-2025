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
        'DOID': re.compile(r'^\d+$'),

        # from train labels
        ## small-molecule & protein resources
        "ChEMBL": re.compile(r"\bCHEMBL\d+\b"), # ✔ CHEMBL2031
        "InterPro": re.compile(r"\bIPR\d{6}\b"), # ✔ IPR000001
        "Pfam": re.compile(r"\bPF\d{5}(?:\.\d+)?\b"),# now allows optional version, e.g. PF00001.21
        "UniProt": re.compile(r"\b(?:[A-NR-Z][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]{4}[A-Z0-9]{5})\b"),# 6- & 10-char formats
        "CATH": re.compile(r"\b\d+\.\d+\.\d+\.\d+\b"), # 1.10.8.10
        "KEGG": re.compile(r"\bK\d{5}\b"), # K00001
        "dbSNP": re.compile(r"\brs\d+\b"),# rs12345

        ## sequence & genome repositories
        "Ensembl": re.compile(r"\bENS[A-Z]{0,5}[GTRP]\d{11}\b"),#ENSG00000139618, ENSMUSG00000064341
        "RefSeq_mRNA": re.compile(r"\bNM_\d{6,}(?:\.\d+)?\b"),# NM_000546.6
        "RefSeq_Protein": re.compile(r"\bNP_\d{6,}(?:\.\d+)?\b"),# NP_000537.3
        "RefSeq_Genome": re.compile(r"\b[A-Z]{2}_\d{6,}(?:\.\d+)?\b"),# NC_000001.11
        "GenBank": re.compile(r"\b[A-Z]{1,2}\d{5,8}(?:\.\d+)?\b"),# M10051, AF231982.1
        "DDBJ": re.compile(r"\bD\d{5}(?:-\d+)?(?:\.\d+)?\b"),# D12345-2
        "GISAID": re.compile(r"\bEPI_ISL_\d+\b"),# EPI_ISL_402124
        
        ## imaging, proteomics, models
        "EMPIAR": re.compile(r"\bEMPIAR-\d{5,}\b"),# EMPIAR-10028
        "PRIDE": re.compile(r"\bPXD\d{6}\b"),# PXD000001
        "ExpressionAtlas": re.compile(r"\bE-PROT-\d+\b"),# E-PROT-17
        "BioModels": re.compile(r"\bMODEL\d{10,13}\b"),# MODEL1402200001
        
        ## antibodies, cell lines, others
        "HPA_Antibody": re.compile(r"\bHPA\d{6}\b"),# HPA002830
        "CAB_Antibody": re.compile(r"\bCAB\d{6}\b"),# CAB004270
        "Cellosaurus": re.compile(r"\bCVCL_[A-Z0-9]{4,}\b"),# CVCL_00000000
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
        return compiled
    except Exception as e:
        logger.error(f"Unable to load entity patterns from {local_path}. Falling back to baked-in patterns.")

@timer_wrap
def get_patterns_from_train_labels(compiled: dict[str, re.Pattern],labels_file: str = "Data/train_labels.csv") -> dict[str, re.Pattern]:
    """
    Get patterns from train labels file.
    """
    import pandas as pd
    labels_df = pd.read_csv(labels_file)
    get_unique_dataset_ids = labels_df[labels_df["dataset_id"] != "Missing"]["dataset_id"].unique()
    new_patterns = {}
    # determine if any of the dataset ids are in the default patterns
    for dataset_id in get_unique_dataset_ids:
        for k, pattern in DEFAULT_PATTERNS.items():
            if re.match(pattern, dataset_id):
                new_patterns[k.upper()] = pattern

    # add patterns from bioregistry
    # compiled = load_entity_patterns(local_path)
    for k, pattern in compiled.items():
        for dataset_id in get_unique_dataset_ids:
            if re.match(pattern, dataset_id):
                new_patterns[k.upper()] = pattern

    return new_patterns

@timer_wrap
def update_entity_patterns(local_path="artifacts/entity_patterns.yaml") -> dict[str, re.Pattern]:
    try:
        compiled = load_entity_patterns(local_path)
        new_patterns = get_patterns_from_train_labels(compiled)
        # Check if any of the default patterns are in the compiled patterns
        for k, v in new_patterns.items():
            if k.upper() in compiled:
                logger.info(f"Default pattern {k} is in the compiled patterns. Updating it.")
                new_patterns[k.upper()] = v
        return new_patterns
    except Exception as e:
        logger.warning(f"Falling back to baked-in patterns – {e}")
        return DEFAULT_PATTERNS


ENTITY_PATTERNS = update_entity_patterns()