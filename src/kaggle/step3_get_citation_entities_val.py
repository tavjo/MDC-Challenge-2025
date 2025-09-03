# src/get_citation_entities.py
"""
This script is used to get the citation entities from the documents.
It will use the train_labels.csv file to get the citation entities for each document.
It will output a json file with a list of citation entities for each document.
During inference, since there will be no train_labels.csv, we will create another function to extract citation entities from the document text using the regex patterns from `src/update_patterns.py`
This script is currently only meant to operate on PDF files.
"""

from typing import List, Optional, Tuple, Dict, Set
from pathlib import Path
import pandas as pd
import numpy as np
import os
import json


from pathlib import Path
import sys

# Allow importing sibling kaggle helpers/models when used as a standalone script
THIS_DIR = Path(__file__).parent
if str(THIS_DIR) not in sys.path:
    sys.path.append(str(THIS_DIR))

from helpers import timer_wrap, initialize_logging, extract_entities_baml
from models import CitationEntity, Chunk
from duckdb_utils import get_duckdb_helper


logger = initialize_logging()

base_tmp = "/kaggle/temp/"


artifacts = os.path.join(base_tmp, "artifacts")
DEFAULT_DUCKDB = os.path.join(artifacts, "mdc_challenge.db")

global_prototypes = "/kaggle/input/rf-model-metadata-files/feature_decomposition.pkl"
ds_prototypes = "/kaggle/input/rf-model-metadata-files/prototypes.pkl"

@timer_wrap
class UnknownCitationEntityExtractor:
    def __init__(self, 
                 model,
                 db_path: str = DEFAULT_DUCKDB,
                 top_ids_path: str = os.path.join(artifacts, "top_ids.json")):
        logger.info(f"Initializing CitationEntityExtractor (JSON-driven)")
        self.db_path = db_path
        self.citation_entities: List[CitationEntity] = []
        self.model = model
        self.top_ids_path = top_ids_path
        # Loaded state
        self.triplets: List[Tuple[Optional[str], str, Optional[str]]] = []
        self.chunks_by_id: Dict[str, Chunk] = {}
    

    # Removed: per-doc loading; now JSON-driven
    
    # Removed: prototype/global embedding loading; not needed in this step
    
    def _retrieve_context(self) -> List[Chunk]:
        """
        Load triples from JSON, then fetch all unique chunks referenced by any triple.
        Persists raw triples into self.triplets.
        """
        if not os.path.exists(self.top_ids_path):
            raise FileNotFoundError(f"top_ids JSON not found at {self.top_ids_path}")
        with open(self.top_ids_path, "r") as f:
            triples = json.load(f)
        # Expect list of [pre_id, anchor_id, post_id] with possible nulls
        # Normalize to Optional[str]
        norm_triples: List[Tuple[Optional[str], str, Optional[str]]] = []
        all_ids: Set[str] = set()
        for t in triples:
            if not isinstance(t, (list, tuple)) or len(t) != 3:
                continue
            pre, anchor, post = t[0], t[1], t[2]
            pre_id = pre if isinstance(pre, str) and pre else None
            anchor_id = anchor if isinstance(anchor, str) and anchor else None
            post_id = post if isinstance(post, str) and post else None
            if not anchor_id:
                continue
            norm_triples.append((pre_id, anchor_id, post_id))
            all_ids.add(anchor_id)
            if pre_id:
                all_ids.add(pre_id)
            if post_id:
                all_ids.add(post_id)

        self.triplets = norm_triples
        if not all_ids:
            return []

        db_helper = get_duckdb_helper(self.db_path)
        chunks = db_helper.get_chunks_by_chunk_ids(list(all_ids))
        db_helper.close()
        self.chunks_by_id = {c.chunk_id: c for c in chunks}
        return chunks

    # Removed: helper; retrieval now loads chunks directly

    # Simplified: just call _retrieve_context once
    def retrieve_context(self) -> List[Chunk]:
        return self._retrieve_context()
    
    def bulk_baml_extraction(self) -> List[CitationEntity]:
        """
        Group loaded triples by document, merge overlapping triples (>=2 shared IDs),
        ensure merged group token budget <= 1000, then call BAML per group.
        """
        chunks = self.retrieve_context()
        if not chunks:
            return []

        # Build per-doc map of triples
        triples_by_doc: Dict[str, List[Tuple[Optional[str], str, Optional[str]]]] = {}
        for pre_id, anchor_id, post_id in self.triplets:
            anchor_chunk = self.chunks_by_id.get(anchor_id)
            if anchor_chunk is None:
                continue
            doc_id = anchor_chunk.document_id
            triples_by_doc.setdefault(doc_id, []).append((pre_id, anchor_id, post_id))

        # Helper to compute token sum for a set of chunk IDs
        def token_sum_for(ids: Set[str]) -> int:
            total = 0
            for cid in ids:
                ch = self.chunks_by_id.get(cid)
                if ch is not None and getattr(ch, "chunk_metadata", None) is not None:
                    total += int(getattr(ch.chunk_metadata, "token_count", 0) or 0)
            return total

        # Merge triples that share >=2 chunk_ids
        def merge_triples(triples: List[Tuple[Optional[str], str, Optional[str]]]) -> List[List[str]]:
            groups: List[Set[str]] = []
            for pre_id, anchor_id, post_id in triples:
                ids = {x for x in [pre_id, anchor_id, post_id] if isinstance(x, str) and x}
                if not ids:
                    continue
                merged = False
                for g in groups:
                    if len(g.intersection(ids)) >= 2:
                        g.update(ids)
                        merged = True
                        break
                if not merged:
                    groups.append(set(ids))
            return [sorted(list(g)) for g in groups]

        # For each doc, merge and enforce 1000-token budget
        for doc_id, triples in triples_by_doc.items():
            merged_groups = merge_triples(triples)
            for group_ids in merged_groups:
                # Enforce 1000 token budget by dropping lowest-priority neighbors first
                # Priority: keep anchors (those that appear as anchors in any triple) then neighbors
                group_set: Set[str] = set(group_ids)
                anchors_in_group: Set[str] = {a for (_, a, _) in triples if a in group_set}
                neighbors_in_group: List[str] = [cid for cid in group_ids if cid not in anchors_in_group]

                # If over budget, drop neighbors first (stable order), then anchors last if necessary
                while token_sum_for(group_set) > 1000 and neighbors_in_group:
                    drop_id = neighbors_in_group.pop()
                    if drop_id in group_set:
                        group_set.remove(drop_id)
                while token_sum_for(group_set) > 1000 and anchors_in_group:
                    # Drop the last added anchor (least likely top-ranked)
                    drop_id = None
                    for cid in reversed(group_ids):
                        if cid in anchors_in_group:
                            drop_id = cid
                            break
                    if drop_id is None:
                        break
                    anchors_in_group.remove(drop_id)
                    if drop_id in group_set:
                        group_set.remove(drop_id)

                # Build ordered texts (anchor-first order where possible)
                ordered_ids: List[str] = []
                # Prefer anchor order from original triples, then any remaining neighbors
                seen: Set[str] = set()
                for _, a, _ in triples:
                    if a in group_set and a not in seen:
                        ordered_ids.append(a)
                        seen.add(a)
                for cid in group_ids:
                    if cid in group_set and cid not in seen:
                        ordered_ids.append(cid)
                        seen.add(cid)
                texts: List[str] = []
                for cid in ordered_ids:
                    ch = self.chunks_by_id.get(cid)
                    if ch is not None:
                        texts.append(ch.text)
                if not texts:
                    continue
                cites = extract_entities_baml(texts, doc_id)
            self.citation_entities.extend(cites)
        return self.citation_entities
    
    def extract_entities(self) -> List[CitationEntity]:
        """
        Extract citation entities from the document text using the known entities from the labels file or the regex patterns.
        """
        return self.bulk_baml_extraction()
        
    def save_entities(self) -> None:
        """
        Save the list of CitationEntity objects to a DuckDB table.
        """
        db_helper = get_duckdb_helper(self.db_path)
        db_helper.store_citations_batch(self.citation_entities)
        db_helper.close()
        logger.info(f"Saved {len(self.citation_entities)} entities to {self.db_path}")


    def load_entities(self) -> List[CitationEntity]:
        """
        Load citation entities from a JSON file of dicts,
        reconstructing each via Pydantic’s model_validate.
        """
        db_helper = get_duckdb_helper( self.db_path)
        citations = db_helper.get_all_citation_entities()
        db_helper.close()
        return [CitationEntity.model_validate(citation) for citation in citations]
    
    @timer_wrap
    def run_all(self) -> List[CitationEntity]:
        logger.info("Extracting citation entities")
        cites = self.extract_entities()
        logger.info(f"Saving {len(cites)} citation entities to DuckDb")
        self.save_entities()
        logger.info(f"Saved {len(self.citation_entities)} entities to {self.db_path}")
        return cites

# @timer_wrap
# def main():
#     print("Initializing UnknownCitationEntityExtractor")
#     extractor = UnknownCitationEntityExtractor(
#         draw_subset=False, 
#         subset_size=None, 
#         db_path=DEFAULT_DUCKDB, 
#         globals_path=None,
#         prototypes = prot,
#         k=10, 
#         max_workers=8,
#         model = embedder
#         )
#     print("Extracting citation entities")
#     cites = extractor.extract_entities()
#     print(f"Saving {len(cites)} citation entities to DuckDb")
#     extractor.save_entities()
#     print(f"Saved {len(extractor.citation_entities)} entities to {extractor.db_path}")


# if __name__ == "__main__":
#     main()