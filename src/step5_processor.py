#!/usr/bin/env python3
"""
Step 5 Main Processor
Orchestrates the complete document parsing and section extraction pipeline
Following the MDC-Challenge-2025 Step 5 checklist
"""

import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
import os

from .document_parser import parse_document, create_document_entry, validate_document
from .xml_format_detector import detect_xml_format


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


def process_all_documents(inventory_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Step 2: Process all documents using the parsing dispatcher.
    
    Returns dictionary of parsed documents.
    """
    parsed_documents = {}
    failed_documents = []
    
    total_files = len(inventory_df)
    print(f"\n🔄 Processing {total_files} documents...")
    
    for idx, row in inventory_df.iterrows():
        article_id = row['article_id']
        xml_path = row['xml_path']
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
                entry = create_document_entry(article_id, sections, Path(xml_path), source_type)
                parsed_documents[article_id] = entry
                
                if idx % 50 == 0:  # Progress update every 50 files
                    print(f"✅ Processed {idx + 1}/{total_files}: {article_id} ({len(sections)} sections)")
            else:
                print(f"⚠️  Failed to parse {article_id}: No sections extracted")
                failed_documents.append(article_id)
                
        except Exception as e:
            print(f"❌ Error processing {article_id}: {e}")
            failed_documents.append(article_id)
    
    print(f"\n📊 Processing complete:")
    print(f"   ✅ Successfully parsed: {len(parsed_documents)}")
    print(f"   ❌ Failed: {len(failed_documents)}")
    
    if failed_documents:
        print(f"   Failed files: {failed_documents[:10]}{'...' if len(failed_documents) > 10 else ''}")
    
    return parsed_documents


def validate_parsed_corpus(parsed_documents: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
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
    
    for doi, doc in parsed_documents.items():
        validation = doc['validation']
        
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
    print(f"   Has methods: {validation_stats['has_methods']} ({validation_stats['has_methods']/total:.1%})")
    print(f"   Has results: {validation_stats['has_results']} ({validation_stats['has_results']/total:.1%})")
    print(f"   Has data availability: {validation_stats['has_data_availability']} ({validation_stats['has_data_availability']/total:.1%})")
    print(f"   Key sections coverage: {validation_stats['key_sections_coverage']} ({validation_stats['key_sections_coverage']/total:.1%})")
    print(f"   Sufficient content: {validation_stats['sufficient_content']} ({validation_stats['sufficient_content']/total:.1%})")
    
    print(f"\n📊 Format Breakdown:")
    for format_type, count in validation_stats['format_breakdown'].items():
        print(f"   {format_type}: {count} ({count/total:.1%})")
    
    # Check success criteria from checklist
    success_threshold = 0.90  # 90% success rate target
    if success_rate >= success_threshold:
        print(f"✅ SUCCESS: {success_rate:.1%} success rate meets ≥90% target")
    else:
        print(f"⚠️  WARNING: {success_rate:.1%} success rate below 90% target")
    
    return validation_stats


def save_parsed_corpus(parsed_documents: Dict[str, Dict[str, Any]], 
                      validation_stats: Dict[str, Any],
                      output_dir: str = "Data/train/parsed") -> None:
    """
    Step 4: Save the parsed corpus and summary.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save main pickle file
    pickle_path = output_path / "parsed_documents.pkl"
    with open(pickle_path, 'wb') as f:
        pickle.dump(parsed_documents, f)
    
    print(f"✅ Saved parsed corpus: {pickle_path}")
    
    # Create summary CSV
    summary_data = []
    for doi, doc in parsed_documents.items():
        validation = doc['validation']
        summary_data.append({
            'doi': doi,
            'format_type': doc.get('format_type', 'UNKNOWN'),
            'source_type': doc.get('source_type', None),
            'section_count': doc['section_count'],
            'total_char_length': doc['total_char_length'],
            'clean_text_length': doc['clean_text_length'],
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
    summary_path = output_path / "parsed_documents_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    
    print(f"✅ Saved summary CSV: {summary_path}")
    
    # Save validation statistics
    stats_path = output_path / "validation_stats.json"
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
        print(f"   Success rate: {validation_stats['validation_passed']/len(parsed_documents):.1%}")
        print(f"   Output saved to: Data/train/parsed/")
        
    except Exception as e:
        print(f"\n❌ Step 5 failed: {e}")
        raise


if __name__ == "__main__":
    main() 