#!/usr/bin/env python3
"""
PDF to XML Conversion Implementation (step 4)
Following the MDC-Challenge-2025 checklist
"""

import pandas as pd
import os
import sys
from pathlib import Path
from requests import post
import pdfplumber
from tqdm import tqdm
import time
from datetime import datetime

TEMP_SUFFIX = '.part'

PROJECT_ROOT = str(Path(__file__).parent.parent)

def main():
    print("=== PDF to XML Conversion for MDC-Challenge-2025 ===")
    print(f"Started at: {datetime.now()}")
    
    # Step 1: Load & Validate Conversion Candidates
    print("\n📋 Step 1: Load & Validate Conversion Candidates")
    cand, all_docs = load_conversion_candidates()
    
    # Step 2: Inventory Existing XML Files
    print("\n📋 Step 2: Inventory Existing XML Files")
    todo = inventory_existing_xml(cand)
    
    # Step 3: Implement Robust Conversion Function
    print("\n📋 Step 3: Conversion Functions Ready")
    
    # Step 4: Execute Batch Conversion
    print("\n📋 Step 4: Execute Batch Conversion")
    log = execute_batch_conversion(todo)
    
    # Step 5: Generate Conversion Report
    print("\n📋 Step 5: Generate Conversion Report")
    generate_conversion_report(log, cand, all_docs)
    
    # Step 6: Quality Validation
    print("\n📋 Step 6: Quality Validation")
    validate_conversions()
    
    print(f"\n✅ Conversion process completed at: {datetime.now()}")

def load_conversion_candidates(cand_path: Path = None):
    """Step 1: Load & Validate Conversion Candidates"""
    if cand_path is None:
        # Get the project root (parent of src directory)
        project_root = Path(__file__).parent.parent
        cand_path = project_root / "Data" / "conversion_candidates.csv"
    
    all_docs = pd.read_csv(cand_path)

    # filter for rows where convert_to_xml is True
    cand = all_docs[all_docs['convert_to_xml'] == True]
    
    # assert len(cand) == 124, f"Expected 124 candidates, got {len(cand)}"
    assert {'article_id', 'pdf_path'}.issubset(cand.columns), "Missing required columns"
    
    print(f"✅ Loaded {len(cand)} conversion candidates")
    print(f"   Columns: {list(cand.columns)}")
    
    return cand, all_docs

def inventory_existing_xml(cand):
    """Step 2: Inventory Existing XML Files"""
    xml_dir = Path("Data/train/XML")
    have_xml = {p.stem for p in xml_dir.glob("*.xml")}
    
    cand['already_has_xml'] = cand['article_id'].map(lambda aid: aid in have_xml)
    todo = cand[~cand['already_has_xml']]
    
    print(f"✅ Found {len(have_xml)} existing XML files")
    print(f"✅ {len(todo)} PDFs still need conversion")
    
    return todo

def convert_one(pdf_path: Path, out_xml: Path):
    """Step 3: Convert single PDF to XML using Grobid with pdfplumber fallback"""
    GROBID_URL = os.getenv("GROBID_URL", "http://localhost:8070")
    
    try:
        # Try Grobid first
        with pdf_path.open('rb') as f:
            r = post(f"{GROBID_URL}/api/processFulltextDocument",
                     files={'input': f}, timeout=120)
        if r.status_code == 200 and r.text.strip():
            out_xml.write_text(r.text, encoding='utf-8')
            return "grobid"
    except Exception as e:
        print(f"Grobid failed for {pdf_path.name}: {e}")

def execute_batch_conversion(todo):
    """Step 4: Execute Batch Conversion"""
    xml_dir = Path("Data/train/XML")
    xml_dir.mkdir(exist_ok=True)
    log = []
    
    print(f"Converting {len(todo)} PDFs...")
    
    for _, row in tqdm(todo.iterrows(), total=len(todo), desc="Converting PDFs"):
        # Handle both absolute and relative paths from CSV
        pdf_path_str = row['pdf_path']
        if pdf_path_str.startswith('/'):
            # Convert absolute path to relative
            pdf_path = Path(pdf_path_str.split('Data/')[-1])
            pdf_path = Path("Data") / pdf_path
        else:
            pdf_path = Path(pdf_path_str)
        
        # Use article_id for XML filename
        xml_path = xml_dir / f"{row['article_id']}.xml{TEMP_SUFFIX}"
        
        try:
            source = convert_one(pdf_path, xml_path)
            log.append({
                'article_id': row['article_id'],
                'pdf_path': row['pdf_path'],
                'xml_path': str(xml_path), 
                'source': source, 
                'error': None,
                'success': source != "failed",
                'has_primary': row['has_primary'],
                'has_secondary': row['has_secondary'],
                'label_count': row['label_count'],
                'has_missing': row['has_missing']
            })
        except Exception as e:
            log.append({
                'article_id': row['article_id'],
                'pdf_path': row['pdf_path'],
                'xml_path': None, 
                'source': None, 
                'error': str(e),
                'success': False,
                'has_primary': row['has_primary'],
                'has_secondary': row['has_secondary'],
                'label_count': row['label_count'],
                'has_missing': row['has_missing']
            })
        
        # Small delay to avoid overwhelming services
        time.sleep(0.1)
    
    return log

def generate_conversion_report(log, cand, all_docs):
    """Step 5: Generate Conversion Report"""
    # Handle empty log case (when no new conversions were needed)
    if log:
        conversion_log = pd.DataFrame(log)
    else:
        # Create empty DataFrame with expected columns
        conversion_log = pd.DataFrame(columns=[
            'article_id', 'pdf_path', 'xml_path', 'source', 'error', 
            'success', 'has_primary', 'has_secondary', 'label_count', 'processing_time', 'has_missing'
            #'processing_time', 'conversion_status', "convert_to_xml"
        ])
    
    # Get ALL existing XML files
    # pdf_path = all_docs['pdf_path'][0]
    xml_dir = Path(os.path.join(PROJECT_ROOT, "Data/train/XML"))
    all_xml_files = list(xml_dir.glob("*.xml")) + list(xml_dir.glob("*.xml.part"))
    
    # Create a lookup dictionary for conversion candidates metadata
    cand_dict = {row['article_id']: row for _, row in cand.iterrows()}
    all_docs_dict = {row['article_id']: row for _, row in all_docs.iterrows()}
    
    # Create entries for ALL existing XML files
    existing_entries = []
    for xml_file in all_xml_files:
        # xml_file = Path(xml_file).stem if str(xml_file).endswith(TEMP_SUFFIX) else Path(xml_file)
        print(xml_file)
        article_id = Path(xml_file).stem if not str(xml_file).endswith(TEMP_SUFFIX) else Path(str(xml_file).replace(TEMP_SUFFIX, '')).stem
        print(article_id)
        
        # Check if this file was also processed in this run (shouldn't happen, but just in case)
        if article_id not in [entry['article_id'] for entry in log]:
            # Check if we have metadata for this file from conversion_candidates.csv
            if article_id in all_docs_dict:
                # File from conversion candidates - include full metadata
                row = all_docs_dict[article_id]
                existing_entries.append({
                    'article_id': article_id,
                    'pdf_path': row['pdf_path'],
                    'xml_path': str(xml_file),
                    'source': None,  # No conversion source for existing files
                    'error': None,
                    'success': None,  # Blank to avoid inflating success rates
                    'has_primary': row['has_primary'],
                    'has_secondary': row['has_secondary'],
                    'label_count': row['label_count'],
                    'processing_time': None,
                    'has_missing': row['has_missing']
                })
            else:
                # File NOT from conversion candidates - basic info only
                existing_entries.append({
                    'article_id': article_id,
                    'pdf_path': None,  # No PDF path info available
                    'xml_path': str(xml_file),
                    'source': None,  # No conversion source for existing files
                    'error': None,
                    'success': None,  # Blank to avoid inflating success rates
                    'has_primary': None,  # No metadata available
                    'has_secondary': None,  # No metadata available
                    'label_count': None,  # No metadata available
                    'processing_time': None,  # No metadata available
                    'has_missing': None,
                })
    
    # Combine conversion log with existing entries
    existing_df = pd.DataFrame(existing_entries)
    if len(existing_df) > 0:
        complete_log = pd.concat([conversion_log, existing_df], ignore_index=True)
    else:
        complete_log = conversion_log
    
    # Save the complete inventory
    complete_log.to_csv("Data/document_inventory.csv", index=False)
    
    # Generate summary statistics (only for files that were actually processed)
    processed_files = conversion_log  # Only newly converted files
    if len(processed_files) > 0:
        successful = processed_files[processed_files['success'] == True]
        failed = processed_files[processed_files['success'] == False]
        grobid_success = processed_files[processed_files['source'] == 'grobid']
        fallback_success = processed_files[processed_files['source'] == 'pdfplumber']
        
        print(f"✅ {len(successful)} successful conversions")
        print(f"   📊 Grobid: {len(grobid_success)}")  
        print(f"   📊 Fallback: {len(fallback_success)}")
        print(f"⚠️ {len(failed)} failed conversions")
    else:
        print("✅ 0 new conversions needed (all files already had XML)")
    
    # Count files from conversion candidates vs other existing files
    from_candidates = sum(1 for _, row in complete_log.iterrows() 
                         if row['article_id'] in cand_dict)
    other_existing = len(complete_log) - from_candidates
    
    print(f"ℹ️  {len(complete_log)} total XML files in inventory")
    print(f"   📊 From conversion candidates: {from_candidates}")
    print(f"   📊 Other existing files: {other_existing}")
    
    # Coverage KPI
    calculate_coverage_kpi(cand)

def calculate_coverage_kpi(cand):
    """Calculate conversion coverage KPI"""
    total_candidates = len(cand)
    
    # Count existing XML files that match our candidates
    existing_xml = {p.stem for p in Path("Data/train/XML").glob("*.xml")}
    covered_candidates = sum(1 for article_id in cand['article_id'] 
                           if article_id in existing_xml)
    
    # Coverage KPI: covered_candidates / total_candidates >= 0.90
    coverage = covered_candidates / total_candidates
    
    print(f"📊 Coverage KPI: {coverage:.2%}")
    print(f"   Total candidates: {total_candidates}")
    print(f"   Covered candidates: {covered_candidates}")
    
    if coverage < 0.90:
        print("🚨 ALERT: Coverage below 90% threshold!")
    else:
        print("✅ Coverage meets 90% requirement")
    
    return coverage

def validate_conversions():
    """Step 6: Validation & Quality Checks"""
    xml_dir = Path("Data/train/XML")
    issues = []
    
    print("🔍 Validating XML files...")
    
    for xml_file in xml_dir.glob("*.xml"):
        if xml_file.stat().st_size < 1024:  # Less than 1KB (very small)
            issues.append(f"{xml_file.name}: File too small ({xml_file.stat().st_size} bytes)")
    
    if issues:
        print(f"⚠️ Found {len(issues)} small files that may need review:")
        for issue in issues[:10]:  # Show first 10
            print(f"   {issue}")
    else:
        print("✅ All XML files appear to have reasonable sizes")

def get_remaining_conversions():
    """Get list of PDFs that still need conversion (resumability)"""
    existing_xml = {p.stem for p in Path("Data/train/XML").glob("*.xml")}
    new_xml = {p.stem for p in Path("Data/train/XML").glob("*.xml.part")}
    all_docs = pd.read_csv("Data/conversion_candidates.csv")
    all_candidates = all_docs[all_docs['convert_to_xml'] == True]
    remaining = all_candidates[~all_candidates['article_id'].isin(existing_xml) & ~all_candidates['article_id'].isin(new_xml)]
    return remaining

if __name__ == "__main__":
    main() 