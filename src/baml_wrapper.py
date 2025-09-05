from typing import List, Any

import sys
from pathlib import Path
import time

# sys.path.append("/kaggle/input/baml-citation-extractor/src")
# Allow importing sibling kaggle helpers/models when used as a standalone script
THIS_DIR = Path(__file__).parent
if str(THIS_DIR) not in sys.path:
    sys.path.append(str(THIS_DIR))

from baml_client import b as baml

def extract_cites(chunk_group: List[str]) -> List[Any]:
    """
    Call BAML once on a small list of chunk texts (e.g., up to 3).
    Returns a list[CitationEntity] for that group.
    """
    max_attempts = 2
    delay = 0.5
    for attempt in range(max_attempts):
        try:
            return baml.ExtractCitation(chunk_group)
        except Exception:
            if attempt == max_attempts - 1:
                raise
            time.sleep(delay * (2 ** attempt))


