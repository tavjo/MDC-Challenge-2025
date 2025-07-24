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


def is_too_broad(pattern_str: str) -> bool:
    """
    Check if a pattern is too broad and would match common English words.
    
    Args:
        pattern_str: The regex pattern string to check
        
    Returns:
        True if the pattern is too broad and should be filtered out
    """
    # Patterns that are too generic and would match common words
    broad_patterns = [
        r'^[a-z0-9]+$',           # Any lowercase + numbers
        r'^[A-Z_a-z]+$',          # Any uppercase + underscore + lowercase  
        r'^[0-9a-z_-]+$',         # Any numbers + lowercase + underscore + hyphen
        r'^\w+$',                 # Any word characters
        r'^[a-zA-Z0-9]+$',        # Any alphanumeric
        r'^[a-zA-Z0-9_-]+$',      # Any alphanumeric + underscore + hyphen
        r'^[a-zA-Z0-9_]+$',       # Any alphanumeric + underscore
        r'^[a-zA-Z0-9-]+$',       # Any alphanumeric + hyphen
        r'^[a-z0-9_-]+$',         # Any lowercase + numbers + underscore + hyphen
        r'^[A-Za-z0-9]+$',        # Any alphanumeric (case insensitive)
        r'^[A-Za-z0-9_-]+$',      # Any alphanumeric + underscore + hyphen (case insensitive)
        r'^[A-Za-z0-9_]+$',       # Any alphanumeric + underscore (case insensitive)
        r'^[A-Za-z0-9-]+$',       # Any alphanumeric + hyphen (case insensitive)
        r'^[A-Za-z]+$',           # Any letters only
        r'^[a-zA-Z_]+$',          # Any letters + underscore
        r'^[a-zA-Z-]+$',          # Any letters + hyphen
        r'^[a-zA-Z_-]+$',         # Any letters + underscore + hyphen
        r'^[a-z]+$',              # Any lowercase letters only
        r'^[A-Z]+$',              # Any uppercase letters only
        r'^[A-Za-z_]+$',          # Any letters + underscore (case insensitive)
        r'^[A-Za-z-]+$',          # Any letters + hyphen (case insensitive)
        r'^[A-Za-z_-]+$',         # Any letters + underscore + hyphen (case insensitive)
    ]
    
    return any(re.match(broad, pattern_str) for broad in broad_patterns)


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
    if not pathlib.Path(local_path).exists():
        logger.warning(f"File {local_path} does not exist. Downloading bioregistry data.")
        download_bioregistry_data()
    logger.info(f"Loading bioregistry data from {local_path}")
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
            load_bioregistry_data()
            text = pathlib.Path("artifacts/entity_patterns.yaml").read_text()
        else:
            text = pathlib.Path(local_path).read_text()
        raw = yaml.safe_load(text)
        
        # Filter and fix patterns
        filtered_patterns = {}
        skipped_count = 0
        
        for k, v in raw.items():
            pattern_str = v["regex"]
            
            # Skip overly broad patterns that would match common words
            if is_too_broad(pattern_str):
                logger.debug(f"Skipping overly broad pattern for {k}: {pattern_str}")
                skipped_count += 1
                continue
                
            # Convert ^...$ patterns to \b...\b patterns for word boundaries
            if pattern_str.startswith('^') and pattern_str.endswith('$'):
                pattern_str = r'\b' + pattern_str[1:-1] + r'\b'
            
            # Compile with case-insensitive flag
            try:
                compiled_pattern = re.compile(pattern_str, re.IGNORECASE)
                filtered_patterns[k] = compiled_pattern
            except re.error as e:
                logger.warning(f"Invalid regex pattern for {k}: {pattern_str} - {e}")
                skipped_count += 1
                continue
        
        logger.info(f"Loaded {len(filtered_patterns)} patterns from {local_path} (skipped {skipped_count} overly broad or invalid patterns)")
        return filtered_patterns
    except Exception as e:
        logger.error(f"Unable to load entity patterns from {local_path}. Falling back to baked-in patterns.")

@timer_wrap
def get_patterns_from_train_labels(compiled: dict[str, re.Pattern]=None,labels_file: str = "Data/train_labels.csv") -> dict[str, re.Pattern]:
    """
    Get patterns from train labels file.
    """
    import pandas as pd
    labels_df = pd.read_csv(labels_file)
    get_unique_dataset_ids = labels_df[labels_df["dataset_id"] != "Missing"]["dataset_id"].unique()
    # dataset_ids_str = "\n".join(get_unique_dataset_ids)
    new_patterns = {}
    # determine if any of the dataset ids are in the default patterns
    for dataset_id in get_unique_dataset_ids:
        # dataset_id_pattern = rf"\b{re.escape(dataset_id)}\b"
        for k, pattern in DEFAULT_PATTERNS.items():
            matches = re.match(pattern, dataset_id)
            if matches:
                new_patterns[k.upper()] = pattern

    # add patterns from bioregistry
    # compiled = load_entity_patterns(local_path)
    # for dataset_id in get_unique_dataset_ids:
    #     # dataset_id_pattern = rf"\b{re.escape(dataset_id)}\b"
    #     for k, pattern in compiled.items():
    #         matches = re.match(pattern, dataset_id)
    #         if matches:
    #             new_patterns[k.upper()] = pattern
    return new_patterns

@timer_wrap
def update_entity_patterns(local_path="artifacts/entity_patterns.yaml") -> dict[str, re.Pattern]:
    try:
        compiled = load_entity_patterns(local_path)
        # Check if any of the default patterns are in the compiled patterns
        for k, v in DEFAULT_PATTERNS.items():
            if k.upper() in compiled:
                logger.info(f"Default pattern {k} is in the compiled patterns. Updating it.")
                DEFAULT_PATTERNS[k.upper()] = v
        new_patterns = get_patterns_from_train_labels(compiled)
        return new_patterns
    except Exception as e:
        logger.warning(f"Falling back to baked-in patterns – {e}")
        return DEFAULT_PATTERNS


ENTITY_PATTERNS = update_entity_patterns()

if __name__ == "__main__":
    print(f"Number of patterns: {len(ENTITY_PATTERNS)}")
    print(f"Last 5 Patterns:\n{list(ENTITY_PATTERNS.items())[-5:]}")