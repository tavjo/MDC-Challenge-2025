# src/construct_queries.py
"""
In this script, I will be constructing queries for the retriever.
These queries will be constructed thusly:
1. The exact chunk in which a dataset citation is found is retrieved from DuckDB.
2. It's neightbors are retrieved from DuckDB as well (denoted by the `prev_chunk_id` and `next_chunk_id` fields).
3. The query is constructed as follows:
    - The query is the concatenation of the chunk text and the text of its neighbors.
    - The query is then passed to the retriever.
    - The retriever returns the top-k (default: 3) chunks.
"""
import argparse
import requests
import json, os, sys
from api.utils.duckdb_utils import get_duckdb_helper
from src.models import RetrievalPayload

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

# Local imports
from src.models import RetrievalPayload, RetrievalResult, BatchRetrievalResult
from src.helpers import initialize_logging, timer_wrap
import requests

filename = os.path.basename(__file__)
logger = initialize_logging(filename)

@timer_wrap
def main():
    parser = argparse.ArgumentParser(
        description="Construct queries from dataset citations and call batch retriever API"
    )
    parser.add_argument(
        "--db-path", default="artifacts/mdc_challenge.db",
        help="Path to the DuckDB database file"
    )
    parser.add_argument(
        "--base-url", default="http://localhost:8000",
        help="Base URL for the retrieval API"
    )
    parser.add_argument(
        "--collection-name", required=True,
        help="ChromaDB collection name for retrieval"
    )
    parser.add_argument(
        "-k", type=int, default=3,
        help="Number of chunks to retrieve per dataset citation"
    )
    args = parser.parse_args()

    # Initialize DuckDB helper
    db_helper = get_duckdb_helper(args.db_path)

    # Load all dataset citations
    citations = db_helper.get_all_citation_entities()

    # Build mapping of dataset ID to its enriched query text
    query_texts: dict[str, list[str]] = {}
    for ce in citations:
        # Retrieve all chunks for the document
        chunks = db_helper.get_chunks_by_document_id(ce.document_id)
        # Find the chunk containing this dataset citation
        target = None
        for c in chunks:
            ents = c.chunk_metadata.citation_entities or []
            if any(ent.data_citation == ce.data_citation for ent in ents):
                target = c
                break
        if not target:
            print(f"Warning: no chunk found for citation {ce.data_citation} in document {ce.document_id}")
            continue

        # Fetch neighbor chunks
        neighbor_ids = [cid for cid in (
            target.chunk_metadata.previous_chunk_id,
            target.chunk_metadata.next_chunk_id
        ) if cid]
        neighbors = db_helper.get_chunks_by_chunk_ids(neighbor_ids)

        # Construct the query string
        parts = [target.text] + [n.text for n in neighbors]
        query_text = " ".join(parts)
        query_texts[ce.data_citation] = [query_text]

    # Call the batch retrieval API endpoint with Pydantic payload
    url = f"{args.base_url}/batch_retrieve_top_chunks"
    # Determine default max_workers as half the CPU count, minimum 1
    cpu_count = os.cpu_count() or 1
    default_workers = max(1, cpu_count // 2)
    payload_obj = RetrievalPayload(
        query_texts=query_texts,
        collection_name=args.collection_name,
        k=args.k,
        max_workers=default_workers,
    )
    payload_data = payload_obj.model_dump(exclude_none=True)
    response = requests.post(url, json=payload_data)
    if response.status_code != 200:
        print(f"Error: API call failed with status {response.status_code}: {response.text}")
        return
    # Prepare output directory under project root
    base_dir = os.getcwd()
    output_dir = os.path.join(base_dir, "reports", "retrieval")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "retrieval_results.json")

    results = BatchRetrievalResult.model_validate(response.json())

    if results.success:
        logger.info("✅ Retrieval completed successfully!")
        # save results to a json file in reports/retrieval
        with open(output_file, "w") as f:
            json.dump(results.model_dump(), f)
    else:
        logger.error(f"❌ Pipeline failed: {results.error}")
        # save results to a json file in reports/retrieval
        with open(output_file, "w") as f:
            json.dump(results.model_dump(), f) 

if __name__ == "__main__":
    main()