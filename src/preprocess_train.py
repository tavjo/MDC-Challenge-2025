# src/preprocess_train.py

"""
Run preprocessing steps to get training data. 
0. EDA on train data files (optional)
1. Get document objects
2. Get citation objects
3. Chunk and embed documents
4. Build Dataset objects
5. Run neighborhood stats (optional)
6. Run global UMAP and PCA (optional)
7. Leiden Clustering
8. Dimensionality reduction
9. Save training data
"""

import os, sys

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
print(f"Project root: {project_root}")
sys.path.append(project_root)

from src.helpers import timer_wrap, initialize_logging


filename = os.path.basename(__file__)
logger = initialize_logging(filename)

DEFAULT_DUCKDB_PATH = "artifacts/mdc_challenge.db"

class PreprocessTrainingData:
    pass

