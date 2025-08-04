#!/usr/bin/env python3
"""
Helper script to import train_labels.csv and update Dataset table with PRIMARY/SECONDARY labels.
Maps dataset_id from labels to dataset_id in datasets table and updates dataset_type field.
"""

import os
import sys
import pandas as pd
from typing import Dict, List

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

# Local imports
from src.helpers import initialize_logging, timer_wrap
from api.utils.duckdb_utils import DuckDBHelper
from src.models import Dataset

filename = os.path.basename(__file__)
logger = initialize_logging(filename)

DEFAULT_DUCKDB_PATH = "artifacts/mdc_challenge.db"
DEFAULT_LABELS_PATH = "Data/train_labels.csv"


class DatasetLabelImporter:
    """
    Import PRIMARY/SECONDARY labels from train_labels.csv and update Dataset table.
    
    Creates a 1-to-1 mapping from dataset_id to dataset_type (PRIMARY/SECONDARY).
    Each dataset can only have one label, but articles can have multiple datasets with different labels.
    """
    
    def __init__(self, db_path: str = DEFAULT_DUCKDB_PATH, labels_path: str = DEFAULT_LABELS_PATH):
        self.db_path = db_path
        self.labels_path = labels_path
        self.db_helper = DuckDBHelper(db_path)
        
    def load_train_labels(self) -> Dict[str, str]:
        """
        Load train_labels.csv and create mapping from dataset_id to dataset_type.
        
        Returns:
            Dict mapping dataset_id to dataset_type (PRIMARY/SECONDARY)
        """
        try:
            logger.info(f"Loading train labels from: {self.labels_path}")
            
            # Load the CSV file
            df = pd.read_csv(self.labels_path)
            logger.info(f"Loaded {len(df)} rows from train_labels.csv")
            
            # Filter out rows with missing types and dataset_ids
            valid_df = df[
                df['type'].notna() & 
                (df['type'] != 'Missing') & 
                df['dataset_id'].notna() & 
                (df['dataset_id'] != 'Missing')
            ]
            logger.info(f"Found {len(valid_df)} rows with valid dataset types and IDs")
            
            # Create mapping from dataset_id to type (PRIMARY/SECONDARY)
            label_mapping = {}
            for _, row in valid_df.iterrows():
                dataset_id = row['dataset_id']
                dataset_type = row['type'].upper()  # Ensure uppercase
                
                if dataset_type in ['PRIMARY', 'SECONDARY']:
                    label_mapping[dataset_id] = dataset_type
                else:
                    logger.warning(f"Unknown dataset type '{dataset_type}' for dataset {dataset_id}")
            
            logger.info(f"Created label mapping for {len(label_mapping)} datasets")
            primary_count = sum(1 for v in label_mapping.values() if v == 'PRIMARY')
            secondary_count = sum(1 for v in label_mapping.values() if v == 'SECONDARY')
            logger.info(f"  - PRIMARY: {primary_count}")
            logger.info(f"  - SECONDARY: {secondary_count}")
            
            return label_mapping
            
        except Exception as e:
            logger.error(f"Failed to load train labels: {str(e)}")
            raise
    
    def update_dataset_labels(self, label_mapping: Dict[str, str]) -> bool:
        """
        Update Dataset table with PRIMARY/SECONDARY labels.
        
        Args:
            label_mapping: Dict mapping dataset_id to dataset_type
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Loading all datasets from database...")
            all_datasets = self.db_helper.get_all_datasets()
            logger.info(f"Found {len(all_datasets)} datasets in database")
            
            # Update dataset_type field based on mapping
            updated_datasets = []
            updated_count = 0
            
            for dataset in all_datasets:
                if dataset.dataset_id in label_mapping:
                    # Create new dataset with updated label
                    updated_dataset = Dataset(
                        dataset_id=dataset.dataset_id,
                        document_id=dataset.document_id,
                        total_tokens=dataset.total_tokens,
                        avg_tokens_per_chunk=dataset.avg_tokens_per_chunk,
                        total_char_length=dataset.total_char_length,
                        clean_text_length=dataset.clean_text_length,
                        cluster=dataset.cluster,
                        dataset_type=label_mapping[dataset.dataset_id],  # Update with new label
                        text=dataset.text
                    )
                    updated_datasets.append(updated_dataset)
                    updated_count += 1
                else:
                    # Keep existing dataset as is
                    updated_datasets.append(dataset)
            
            logger.info(f"Updating {updated_count} datasets with new labels...")
            
            # Bulk upsert updated datasets
            result = self.db_helper.bulk_upsert_datasets(updated_datasets)
            
            if result["success"]:
                logger.info(f"✅ Successfully updated {updated_count} datasets with labels")
                return True
            else:
                logger.error("❌ Failed to update datasets with labels")
                return False
                
        except Exception as e:
            logger.error(f"Failed to update dataset labels: {str(e)}")
            return False
    
    @timer_wrap
    def run_import(self) -> bool:
        """
        Run the complete import process: load labels + update datasets.
        
        Returns:
            True if successful, False otherwise
        """
        logger.info("🚀 Starting dataset label import...")
        
        try:
            # Step 1: Load train labels
            logger.info("📥 Loading train labels from CSV...")
            label_mapping = self.load_train_labels()
            
            if not label_mapping:
                logger.warning("No valid labels found in train_labels.csv")
                return False
            
            # Step 2: Update dataset table
            logger.info("🔄 Updating dataset table with labels...")
            success = self.update_dataset_labels(label_mapping)
            
            if success:
                logger.info("✅ Dataset label import completed successfully!")
            else:
                logger.error("❌ Dataset label import failed!")
            
            return success
            
        except Exception as e:
            logger.error(f"Import process failed with error: {str(e)}")
            return False


@timer_wrap
def main():
    """Main function to run the dataset label import."""
    
    logger.info("Initializing dataset label importer...")
    
    importer = DatasetLabelImporter(
        db_path=DEFAULT_DUCKDB_PATH,
        labels_path=DEFAULT_LABELS_PATH
    )
    
    # Run the import process
    success = importer.run_import()
    
    # Summary logging
    if success:
        logger.info("🎉 Dataset label import completed successfully!")
    else:
        logger.error("💥 Import failed - check the logs for details")


if __name__ == "__main__":
    main()