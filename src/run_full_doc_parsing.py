#!/usr/bin/env python3
"""
Step 5 Main Processor
Orchestrates the complete document parsing and section extraction pipeline
Following the MDC-Challenge-2025 Step 5 checklist
"""

import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, Any, List, Tuple
from datetime import datetime
import os

from document_parser import parse_document, create_document_entry
from models import Document

TEMP_SUFFIX = '.part'

def load_document_inventory(inventory_path: str = "Data/document_inventory.csv") -> pd.DataFrame:
    """
    Step 1: Load document inventory.
    
    Expected format from Step 4: CSV with columns including article_id, xml_path, source
    """
    inventory_df = pd.read_csv(inventory_path)
    
    print(f"✅ Loaded document inventory: {len(inventory_df)} files")
    print(f"📊 Columns: {list(inventory_df.columns)}")
    
    # Count by format type (TEI vs JATS)
    tei_count = len(inventory_df[inventory_df['source'] == 'grobid'])
    jats_count = len(inventory_df[inventory_df['source'].isna()])
    
    print(f"📊 TEI files (from Step 4): {tei_count}")
    print(f"📊 JATS files (existing): {jats_count}")
    print(f"📊 Total: {len(inventory_df)}")
    
    return inventory_df


def process_all_documents(inventory_df: pd.DataFrame) -> List[Tuple[Document, Dict[str, Any]]]:
    """
    Step 2: Process all documents using the parsing dispatcher.
    
    Returns dictionary of parsed documents.
    """
    # parsed_documents = []
    failed_documents = []
    res = []
    
    total_files = len(inventory_df)
    print(f"\n🔄 Processing {total_files} documents...")
    
    for idx, row in inventory_df.iterrows():
        article_id = row['article_id']
        xml_path = os.path.join("Data/train/XML", row['xml_path'])
        print(xml_path)
        
        source_type = row.get('source', None)
        
        if pd.isna(xml_path) or not Path(xml_path).exists():
            print(f"⚠️  Skipping {article_id}: XML file not found at {xml_path}")
            failed_documents.append(article_id)
            continue
        
        try:
            # Parse document
            sections = parse_document(Path(xml_path), source_type)
            
            if sections:
                # Create document entry
                entry, validation = create_document_entry(article_id, sections, Path(xml_path), source_type)
                # parsed_documents.append(entry)
                res.append((entry, validation))
                
                if idx % 50 == 0:  # Progress update every 50 files
                    print(f"✅ Processed {idx + 1}/{total_files}: {article_id} ({len(sections)} sections)")
            else:
                print(f"⚠️  Failed to parse {article_id}: No sections extracted")
                failed_documents.append(article_id)
                
        except Exception as e:
            print(f"❌ Error processing {article_id}: {e}")
            failed_documents.append(article_id)
    
    print(f"\n📊 Processing complete:")
    print(f"   ✅ Successfully parsed: {len(res)}")
    print(f"   ❌ Failed: {len(failed_documents)}")
    
    if failed_documents:
        print(f"   Failed files: {failed_documents[:10]}{'...' if len(failed_documents) > 10 else ''}")
    
    return res


def validate_parsed_corpus(parsed_documents: List[Tuple[Document, Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Step 3: Validate the parsed corpus according to checklist criteria.
    """
    validation_stats = {
        'total_documents': len(parsed_documents),
        'valid_documents': 0,
        'has_methods': 0,
        'has_results': 0,
        'has_data_availability': 0,
        'key_sections_coverage': 0,
        'sufficient_content': 0,
        'format_breakdown': {'TEI': 0, 'JATS': 0, 'UNKNOWN': 0},
        'validation_passed': 0
    }
    
    for doc, validation in parsed_documents:
        doc = doc.model_dump()
        # validation = doc['validation']
        
        if validation['validation_passed']:
            validation_stats['validation_passed'] += 1
        
        if validation['has_methods']:
            validation_stats['has_methods'] += 1
        
        if validation['has_results']:
            validation_stats['has_results'] += 1
        
        if validation['has_data_availability']:
            validation_stats['has_data_availability'] += 1
        
        if validation['key_sections_present']:
            validation_stats['key_sections_coverage'] += 1
        
        if validation['has_sufficient_content']:
            validation_stats['sufficient_content'] += 1
        
        # Format breakdown
        format_type = doc.get('format_type', 'UNKNOWN')
        if format_type in validation_stats['format_breakdown']:
            validation_stats['format_breakdown'][format_type] += 1
        else:
            validation_stats['format_breakdown']['UNKNOWN'] += 1
    
    # Calculate success rate
    total = validation_stats['total_documents']
    success_rate = validation_stats['validation_passed'] / total if total > 0 else 0
    
    print(f"\n📊 Validation Results:")
    print(f"   Total documents: {total}")
    print(f"   Validation passed: {validation_stats['validation_passed']} ({success_rate:.1%})")
    
    if total > 0:
        print(f"   Has methods: {validation_stats['has_methods']} ({validation_stats['has_methods']/total:.1%})")
        print(f"   Has results: {validation_stats['has_results']} ({validation_stats['has_results']/total:.1%})")
        print(f"   Has data availability: {validation_stats['has_data_availability']} ({validation_stats['has_data_availability']/total:.1%})")
        print(f"   Key sections coverage: {validation_stats['key_sections_coverage']} ({validation_stats['key_sections_coverage']/total:.1%})")
        print(f"   Sufficient content: {validation_stats['sufficient_content']} ({validation_stats['sufficient_content']/total:.1%})")
        
        print(f"\n📊 Format Breakdown:")
        for format_type, count in validation_stats['format_breakdown'].items():
            print(f"   {format_type}: {count} ({count/total:.1%})")
    else:
        print(f"   Has methods: {validation_stats['has_methods']} (0.0%)")
        print(f"   Has results: {validation_stats['has_results']} (0.0%)")
        print(f"   Has data availability: {validation_stats['has_data_availability']} (0.0%)")
        print(f"   Key sections coverage: {validation_stats['key_sections_coverage']} (0.0%)")
        print(f"   Sufficient content: {validation_stats['sufficient_content']} (0.0%)")
        
        print(f"\n📊 Format Breakdown:")
        for format_type, count in validation_stats['format_breakdown'].items():
            print(f"   {format_type}: {count} (0.0%)")
        print("   ⚠️  No documents processed - check file paths and XML availability")
    
    # Check success criteria from checklist
    success_threshold = 0.90  # 90% success rate target
    if success_rate >= success_threshold:
        print(f"✅ SUCCESS: {success_rate:.1%} success rate meets ≥90% target")
    else:
        print(f"⚠️  WARNING: {success_rate:.1%} success rate below 90% target")
    
    return validation_stats


def save_parsed_corpus(parsed_documents: List[Tuple[Document, Dict[str, Any]]], 
                      validation_stats: Dict[str, Any],
                      output_dir: str = "Data/train/parsed") -> None:
    """
    Step 4: Save the parsed corpus and summary.
    """
    docs = [doc for doc, _ in parsed_documents]
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save main pickle file
    pickle_path = output_path / f"parsed_documents.pkl{TEMP_SUFFIX}"
    with open(pickle_path, 'wb') as f:
        pickle.dump(docs, f)
    
    print(f"✅ Saved parsed corpus: {pickle_path}")

    
    # Create summary CSV
    summary_data = []
    for doc, validation in parsed_documents:
        # get list of section types in order
        sec_order = [section.section_type for section in sorted(doc.sections, key=lambda x: x.order)]
        n_sec_texts = sum(1 for section in doc.sections if section.text.strip())
        doc = doc.model_dump()
        summary_data.append({
            'doi': doc['doi'],
            'format_type': doc.get('format_type', 'UNKNOWN'),
            'source_type': doc.get('source_type', None),
            'conversion_source': doc.get('conversion_source', 'unknown'),  # 🆕 NEW
            'section_count': doc['section_count'],
            'section_order': sec_order,
            'total_char_length': doc['total_char_length'],
            'clean_text_length': doc['clean_text_length'],
            'sections_with_text': n_sec_texts,        # 🆕 NEW - useful summary
            'validation_passed': validation['validation_passed'],
            'has_methods': validation['has_methods'],
            'has_results': validation['has_results'],
            'has_data_availability': validation['has_data_availability'],
            'key_sections_present': validation['key_sections_present'],
            'has_sufficient_content': validation['has_sufficient_content'],
            'section_types': ','.join(validation['section_types']),
            'parsed_timestamp': doc['parsed_timestamp']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = output_path / f"parsed_documents_summary.csv{TEMP_SUFFIX}"
    summary_df.to_csv(summary_path, index=False)
    
    print(f"✅ Saved summary CSV: {summary_path}")
    
    # Save validation statistics
    stats_path = output_path / f"validation_stats.json{TEMP_SUFFIX}"
    import json
    with open(stats_path, 'w') as f:
        json.dump(validation_stats, f, indent=2)
    
    print(f"✅ Saved validation stats: {stats_path}")


def main():
    """
    Main Step 5 processing pipeline.
    """
    print("=== Step 5: Document Parsing & Section Extraction ===")
    print(f"Started at: {datetime.now()}")
    
    try:
        # Step 1: Load document inventory
        print("\n📋 Step 1: Load & Validate Document Inventory")
        inventory_df = load_document_inventory()
        
        # Step 2: Process all documents
        print("\n📋 Step 2: Document Parsing & Section Extraction")
        parsed_documents = process_all_documents(inventory_df)
        
        # Step 3: Validate corpus
        print("\n📋 Step 3: Corpus Validation")
        validation_stats = validate_parsed_corpus(parsed_documents)
        
        # Step 4: Save corpus
        print("\n📋 Step 4: Save Parsed Corpus")
        save_parsed_corpus(parsed_documents, validation_stats)
        
        print(f"\n✅ Step 5 completed successfully at: {datetime.now()}")
        print(f"📊 Final Results:")
        print(f"   Total processed: {len(parsed_documents)}")
        if len(parsed_documents) > 0:
            print(f"   Success rate: {validation_stats['validation_passed']/len(parsed_documents):.1%}")
        else:
            print(f"   Success rate: 0.0%")
        print(f"   Output saved to: Data/train/parsed/")
        
    except Exception as e:
        print(f"\n❌ Step 5 failed: {e}")
        raise


if __name__ == "__main__":
    main() 