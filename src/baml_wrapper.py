from typing import List, Any

import sys
from pathlib import Path

# sys.path.append("/kaggle/input/baml-citation-extractor/src")
# Allow importing sibling kaggle helpers/models when used as a standalone script
THIS_DIR = Path(__file__).parent
if str(THIS_DIR) not in sys.path:
    sys.path.append(str(THIS_DIR))

from baml_client import b as baml

def extract_cites(doc: List[str]) -> List[Any]:
    """
    Extract citation entities from the document text using the BAML client.
    """
    citations = baml.ExtractCitation(doc)
    return citations