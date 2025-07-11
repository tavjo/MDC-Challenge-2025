# src/get_citation_entities.py
"""
This script is used to get the citation entities from the documents.
It will use the train_labels.csv file to get the citation entities for each document.
It will output a json file with a list of citation entities for each document.
During inference, since there will be no train_labels.csv, we will create another function to extract citation entities from the document text using the regex patterns from `src/update_patterns.py`
This script is currently only meant to operate on PDF files.
"""

import re
from typing import List, Optional
from pathlib import Path
import pandas as pd
import numpy as np
import sys, os, json

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
print(f"Project root: {project_root}")
sys.path.append(project_root)

from src.helpers import timer_wrap, initialize_logging, parallel_processing_decorator
from src.models import CitationEntity
from src.update_patterns import ENTITY_PATTERNS
from src.extract_pdf_text_unstructured import load_pdf_pages

filename = os.path.basename(__file__)
logger = initialize_logging(filename)

@timer_wrap
class CitationEntityExtractor:
    def __init__(self, 
                 data_dir: str = "Data", 
                 known_entities: bool = True,
                 labels_file: Optional[str] = "train_labels.csv", 
                 draw_subset: bool = False,
                 subset_size: Optional[int] = None):
        logger.info(f"Initializing CitationEntityExtractor with data directory: {data_dir}")
        self.data_dir = os.path.join(project_root, data_dir)
        self.known_entities = known_entities
        self.files_dir = os.path.join(self.data_dir,"train","PDF") # assumes similar directory structure as me. 
        self.pdf_files : List[str] = [os.path.join(self.files_dir, f) for f in os.listdir(self.files_dir) if f.endswith('.pdf')]
        if self.known_entities:
            logger.info(f"Using known entities from: {labels_file}")
            self.labels_path = os.path.join(self.data_dir, labels_file)
            logger.info(f"Loading labels from: {self.labels_path}")
            self.labels_df = pd.read_csv(self.labels_path)
            logger.info(f"Loaded {len(self.labels_df)} labels")
            self.missing_entities = None
            self.article_ids = self.labels_df["article_id"].unique()
        else:
            logger.info("No known entities provided. Using regex patterns to extract citation entities.")
            self.labels_path = None
            self.labels_df = pd.DataFrame(columns = ["article_id", "dataset_id"])
            self.patterns = ENTITY_PATTERNS
            self.article_ids = [Path(file).stem for file in self.pdf_files]
        if draw_subset:
            self.subset_size = subset_size if subset_size is not None else 20
            logger.info(f"Drawing subset of {self.subset_size} files from {len(self.pdf_files)} files.")
            # set seed
            np.random.seed(42)
            self.subset_ids = np.random.choice(self.pdf_files, self.subset_size, replace=False)
            self.pdf_files = self.subset_ids
        
        self.citation_entities: List[CitationEntity] = []

    @parallel_processing_decorator(batch_param_name="pages", batch_size=5, max_workers=8)
    def _get_known_entities(self, pages: List[str], article_id: str, dataset_ids: List[str]) -> List[CitationEntity]:
        """
        Get known entities from the document text using the known entities from the labels file.
        """
        num_pages = len(pages)
        
        # Track pages per dataset_id
        dataset_pages = {}  # dataset_id -> set of pages where it appears
        
        for idx, page in enumerate(pages):
            page_num = idx + 1
            logger.info(f"Processing page {page_num} of {num_pages}")
            
            for dataset_id in dataset_ids:
                pattern = rf"\b{re.escape(dataset_id)}\b"
                matches = re.findall(pattern, page)
                
                if matches:
                    # Initialize if not exists
                    if dataset_id not in dataset_pages:
                        dataset_pages[dataset_id] = set()
                    
                    # Add current page to this dataset's pages
                    dataset_pages[dataset_id].add(page_num)
        
        # Create one CitationEntity per unique dataset_id found
        entities = []
        for dataset_id, pages_set in dataset_pages.items():
            citation_entity = CitationEntity(
                data_citation=dataset_id,
                doc_id=article_id,
                pages=sorted(list(pages_set))  # Convert set to sorted list
            )
            entities.append(citation_entity)
        
        # Add to the main list
        self.citation_entities.extend(entities)
        
        return entities

    @parallel_processing_decorator(batch_param_name="pages", batch_size=5, max_workers=8)
    def _get_unknown_entities(self, pages: List[str], article_id: str) -> List[CitationEntity]:
        """
        Get unknown entities from the document text using the regex patterns from `src/update_patterns.py`
        """
        num_pages = len(pages)
        
        # Track pages per matched entity
        entity_pages = {}  # matched_entity -> set of pages where it appears
        
        for idx, page in enumerate(pages):
            page_num = idx + 1
            logger.info(f"Processing page {page_num} of {num_pages}")
            
            for name, pattern in self.patterns.items():
                matches = pattern.findall(page)
                logger.info(f"Found {len(matches)} matches for {name}")
                
                if matches:
                    for match in matches:
                        # Initialize if not exists
                        if match not in entity_pages:
                            entity_pages[match] = set()
                        
                        # Add current page to this entity's pages
                        entity_pages[match].add(page_num)
        
        # Create one CitationEntity per unique matched entity found
        entities = []
        for matched_entity, pages_set in entity_pages.items():
            citation_entity = CitationEntity(
                data_citation=matched_entity,
                doc_id=article_id,
                pages=sorted(list(pages_set))  # Convert set to sorted list
            )
            entities.append(citation_entity)
        
        # Add to the main list
        self.citation_entities.extend(entities)
        
        return entities

    def extract_known_entities(self) -> List[CitationEntity]:
        """
        Extract citation entities from the document text using the known entities from the labels file.
        """
        logger.info("Extracting known entities from the document text using the known entities from the labels file.")
        articles_ids = self.article_ids
        # create dictionary where keys are article IDs and values are lists of dataset IDs
        # maybe can group by article ID and then get the dataset IDs for each article
        # Filter out "Missing" values
        labels_dict = {}
        for article_id in articles_ids:
            dataset_ids = self.labels_df[
                (self.labels_df["article_id"] == article_id) & 
                (self.labels_df["dataset_id"] != "Missing")
            ]["dataset_id"].tolist()
            labels_dict[article_id] = dataset_ids
        # check if there are any pdf files
        if len(self.pdf_files) == 0:
            logger.warning("No PDF files found. Returning empty list.")
            return []
        for file in self.pdf_files:
            # check if the file exists
            if not os.path.exists(file):
                logger.warning(f"File not found: {file}")
                continue
            # extract the article id from the file name
            article_id = Path(file).stem
            if article_id not in articles_ids:
                logger.warning(f"Article ID {article_id} not found in the labels file. Skipping file: {file}")
                continue
            else:
                dataset_ids = labels_dict[article_id]
            # load the pdf file
            pages = load_pdf_pages(file)
            self._get_known_entities(pages=pages, article_id=article_id, dataset_ids=dataset_ids)
        #final check: compare length of citation entities with length of dataset IDs
        known_entities_count = len(self.labels_df[self.labels_df["dataset_id"] != "Missing"])
        if len(self.citation_entities) != known_entities_count:
            logger.warning(f"Number of extracted citation entities from pdfs ({len(self.citation_entities)}) does not match number of dataset IDs in labels file ({known_entities_count})")
            # check which dataset IDs are missing
            self.missing_entities = set(self.labels_df[self.labels_df["dataset_id"] != "Missing"]["dataset_id"].tolist()) - set([entity.data_citation for entity in self.citation_entities])
            logger.warning(f"Missing dataset IDs: {self.missing_entities}")
        else:
            logger.info("Known Entity extraction complete. No missing entities found.")
        return self.citation_entities
    
    
    def extract_unknown_entities(self) -> List[CitationEntity]:
        """
        Extract citation entities from the document text using the regex patterns from `src/update_patterns.py`
        """
        logger.info("Extracting unknown entities using regex patterns.")
        # check if there are any pdf files
        if len(self.pdf_files) == 0:
            logger.warning("No PDF files found. Returning empty list.")
            return []
        for file in self.pdf_files:
            # check if the file exists
            if not os.path.exists(file):
                logger.warning(f"File not found: {file}")
                continue
            # extract the article id from the file name
            article_id = Path(file).stem
            # load the pdf file
            pages = load_pdf_pages(file)
            self._get_unknown_entities(pages=pages, article_id=article_id)
        # get total number of entities extracted
        total_entities = len(self.citation_entities)
        # get total number of unique entities extracted
        unique_entities = len(set([entity.data_citation for entity in self.citation_entities]))
        logger.info(f"Entity extraction complete.\nTotal entities extracted: {total_entities}.\nTotal unique entities extracted: {unique_entities}")
        return self.citation_entities
    
    def extract_entities(self) -> List[CitationEntity]:
        """
        Extract citation entities from the document text using the known entities from the labels file or the regex patterns.
        """
        if self.known_entities:
            return self.extract_known_entities()
        else:
            return self.extract_unknown_entities()
    
    def export_entities(self, output_file: str = "citation_entities.json") -> None:
        """
        Export the list of CitationEntity objects as plain dicts,
        avoiding double-encoded JSON strings.
        """
        outfile = os.path.join(self.data_dir, output_file)
        logger.info(f"Exporting {len(self.citation_entities)} entities to {outfile}")
        # Use model_dump() to get a JSON-serializable dict directly
        to_export = [entity.model_dump() for entity in self.citation_entities]
        with open(outfile, "w") as f:
            json.dump(to_export, f, indent=4)


    def load_entities(self, input_file: str = "citation_entities.json") -> List[CitationEntity]:
        """
        Load citation entities from a JSON file of dicts,
        reconstructing each via Pydantic’s model_validate.
        """
        infile = os.path.join(self.data_dir, input_file)
        logger.info(f"Loading citation entities from {infile}")
        with open(infile, "r") as f:
            raw = json.load(f)  # List[dict]
        # Validate and instantiate each model in one step
        return [CitationEntity.model_validate(item) for item in raw]
    

if __name__ == "__main__":
    extractor = CitationEntityExtractor(data_dir="Data", known_entities=True, labels_file="train_labels.csv", draw_subset=True)
    extractor.extract_entities()
    extractor.export_entities()



            
        


