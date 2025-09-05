#!/usr/bin/env python3
from pathlib import Path
import sys
import pandas as pd


def main():
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    # Late import after sys.path manipulation
    from api.utils.duckdb_utils import DuckDBHelper

    # Input CSV as specified
    csv_path = (Path(__file__).parent / "../Data/train/train_data_chunks.csv").resolve()

    # Output parquet in the requested folder
    out_dir = repo_root / "artifacts" / "models_20250814_1652"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "new_prototypes.parquet"

    # Load
    df = pd.read_csv(csv_path)

    # Index by chunk_id
    if "chunk_id" not in df.columns:
        raise RuntimeError("CSV must contain a 'chunk_id' column")
    df = df.set_index("chunk_id")

    # Keep only RELEVANT (1)
    if "target" not in df.columns:
        raise RuntimeError("CSV must contain a 'target' column")
    df = df[df["target"] == 1]

    # Retrieve chunks from DuckDB and filter to those with >=1 data citation
    db = DuckDBHelper()  # default db path artifacts/mdc_challenge.db
    chunk_ids = df.index.tolist()
    chunks = db.get_chunks_by_chunk_ids(chunk_ids)
    cited_ids = {
        c.chunk_id
        for c in chunks
        if getattr(c, "chunk_metadata", None) is not None
        and getattr(c.chunk_metadata, "citation_entities", None)
        and len(c.chunk_metadata.citation_entities) > 0
    }

    if not cited_ids:
        print("No chunks with citation entities found; output will be empty.")

    df = df.loc[df.index.intersection(cited_ids)]

    # Keep only embedding columns (exactly 384)
    emb_cols = [f"emb_{i}" for i in range(384)]
    missing = [c for c in emb_cols if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"Missing expected embedding columns (first few): {missing[:5]} (missing total: {len(missing)})"
        )
    proto_df = df[emb_cols]

    if proto_df.shape[1] != 384:
        raise RuntimeError(f"Expected 384 embedding columns, got {proto_df.shape[1]}")

    # Save parquet with chunk_id index
    proto_df.to_parquet(out_path, index=True)
    print(f"Wrote {proto_df.shape[0]} x {proto_df.shape[1]} embeddings to {out_path}")


if __name__ == "__main__":
    main()


