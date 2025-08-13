from typing import List, Any

import sys
sys.path.append("/kaggle/input/baml-folders/src")

from baml_client import b as baml

def extract_cites(doc: List[str]) -> List[Any]:
    """
    Extract citation entities from the document text using the BAML client.
    """
    citations = baml.ExtractCitation(doc)
    return citations