#!/usr/bin/env python3
"""
Enhanced Full Document Parsing Script with Comprehensive Reporting (step 5)

This script provides comprehensive document parsing and section extraction with structured 
report generation for the MDC Challenge 2025 dataset. It wraps the existing parsing logic 
with enhanced reporting capabilities similar to the pre-chunking EDA and PDF conversion scripts.

Features:
- Complete document parsing workflow with detailed progress tracking
- Multiple output formats (console, markdown, JSON)
- Performance metrics and quality analysis
- Error categorization and resolution guidance
- Integration with existing parsing pipeline
- Detailed reporting of all parsing steps
- Preservation of all existing outputs for downstream compatibility

Usage:
    python scripts/run_full_doc_parsing.py
    python scripts/run_full_doc_parsing.py --output-format markdown
    python scripts/run_full_doc_parsing.py --data-dir Data --reports-dir reports
"""

import os
import sys
import json
import argparse
import time
import psutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from helpers import initialize_logging, timer_wrap

# Import existing parsing functions (without modification)
sys.path.append(str(Path(__file__).parent.parent / "src"))
from run_full_doc_parsing import (
    load_document_inventory,
    # process_all_documents,
    validate_parsed_corpus,
    save_parsed_corpus
)
from models import Document, Section

TEMP_SUFFIX = '.part'


class DocumentParsingReporter:
    """
    Captures and formats outputs from document parsing process for comprehensive reporting.
    Based on EDAReporter and ConversionReporter but specialized for document parsing workflows.
    """    
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.start_time = datetime.now()
        self.sections = {}
        self.summary_stats = {}
        self.files_generated = []
        self.parsing_metrics = {
            'total_processed': 0,
            'successful_parses': 0,
            'failed_parses': 0,
            'tei_successes': 0,
            'jats_successes': 0,
            'unknown_format_attempts': 0,
            'processing_times': [],
            'memory_usage': [],
            'errors_by_type': {},
            'sections_by_type': {},
            'validation_failures': []
        }
        
    def add_section(self, section_name: str, content: Dict[str, Any]):
        """Add a section to the report with structured content."""
        self.sections[section_name] = {
            'timestamp': datetime.now(),
            'content': content
        }
    
    def add_summary_stat(self, key: str, value: Any):
        """Add a key statistic to the summary."""
        self.summary_stats[key] = value
    
    def add_generated_file(self, filepath: str, description: str):
        """Track files generated during parsing."""
        self.files_generated.append({
            'filepath': filepath,
            'description': description,
            'timestamp': datetime.now()
        })
    
    def add_parsing_metric(self, metric_name: str, value: Any):
        """Add parsing-specific metrics."""
        self.parsing_metrics[metric_name] = value
    
    def update_parsing_stats(self, doc_result: Optional[Tuple[Document, Dict[str, Any]]] = None, 
                           processing_time: Optional[float] = None, failed_doc_id: Optional[str] = None, 
                           error_details: Optional[str] = None):
        """Update parsing statistics based on processing results."""
        self.parsing_metrics['total_processed'] += 1
        
        # Record memory usage
        memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
        self.parsing_metrics['memory_usage'].append(memory_mb)
        
        if doc_result is not None:
            doc, validation = doc_result
            self.parsing_metrics['successful_parses'] += 1
            
            # Track format types
            if doc.format_type == 'TEI':
                self.parsing_metrics['tei_successes'] += 1
            elif doc.format_type == 'JATS':
                self.parsing_metrics['jats_successes'] += 1
            else:
                self.parsing_metrics['unknown_format_attempts'] += 1
            
            # Track section types
            for section in doc.sections:
                section_type = section.section_type
                self.parsing_metrics['sections_by_type'][section_type] = \
                    self.parsing_metrics['sections_by_type'].get(section_type, 0) + 1
            
            # Track validation failures
            if not validation['validation_passed']:
                self.parsing_metrics['validation_failures'].append({
                    'doi': doc.doi,
                    'reason': 'validation_failed',
                    'details': validation
                })
        else:
            self.parsing_metrics['failed_parses'] += 1
            if error_details:
                error_type = type(error_details).__name__
                self.parsing_metrics['errors_by_type'][error_type] = \
                    self.parsing_metrics['errors_by_type'].get(error_type, 0) + 1
                
                self.parsing_metrics['validation_failures'].append({
                    'doi': failed_doc_id or 'unknown',
                    'reason': 'parsing_failed',
                    'details': str(error_details)
                })
        
        if processing_time:
            self.parsing_metrics['processing_times'].append(processing_time)
    
    def calculate_performance_metrics(self):
        """Calculate performance metrics from collected data."""
        times = self.parsing_metrics['processing_times']
        memory = self.parsing_metrics['memory_usage']
        
        metrics = {}
        if times:
            metrics.update({
                'average_processing_time': sum(times) / len(times),
                'total_processing_time': sum(times),
                'fastest_parsing': min(times),
                'slowest_parsing': max(times),
                'processing_rate': len(times) / sum(times) if sum(times) > 0 else 0
            })
        
        if memory:
            metrics.update({
                'average_memory_mb': sum(memory) / len(memory),
                'peak_memory_mb': max(memory),
                'min_memory_mb': min(memory)
            })
        
        return metrics
    
    def generate_console_report(self) -> str:
        """Generate formatted console output."""
        report = []
        report.append("=" * 80)
        report.append("FULL DOCUMENT PARSING COMPREHENSIVE REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Data Directory: {self.data_dir}")
        report.append(f"Parsing Duration: {datetime.now() - self.start_time}")
        report.append("")
        
        # Executive Summary
        report.append("EXECUTIVE SUMMARY")
        report.append("-" * 40)
        for key, value in self.summary_stats.items():
            report.append(f"• {key}: {value}")
        report.append("")
        
        # Parsing Performance
        perf_metrics = self.calculate_performance_metrics()
        if perf_metrics:
            report.append("PARSING PERFORMANCE")
            report.append("-" * 40)
            for key, value in perf_metrics.items():
                if 'time' in key.lower():
                    report.append(f"• {key}: {value:.2f}s")
                elif 'memory' in key.lower():
                    report.append(f"• {key}: {value:.1f}MB")
                elif 'rate' in key.lower():
                    report.append(f"• {key}: {value:.2f} docs/sec")
                else:
                    report.append(f"• {key}: {value}")
            report.append("")
        
        # Format Distribution
        if self.parsing_metrics['successful_parses'] > 0:
            report.append("FORMAT DISTRIBUTION")
            report.append("-" * 40)
            total_success = self.parsing_metrics['successful_parses']
            report.append(f"• TEI documents: {self.parsing_metrics['tei_successes']} ({self.parsing_metrics['tei_successes']/total_success:.1%})")
            report.append(f"• JATS documents: {self.parsing_metrics['jats_successes']} ({self.parsing_metrics['jats_successes']/total_success:.1%})")
            report.append(f"• Unknown format: {self.parsing_metrics['unknown_format_attempts']} ({self.parsing_metrics['unknown_format_attempts']/total_success:.1%})")
            report.append("")
        
        # Section Analysis
        if self.parsing_metrics['sections_by_type']:
            report.append("SECTION TYPE ANALYSIS")
            report.append("-" * 40)
            sorted_sections = sorted(self.parsing_metrics['sections_by_type'].items(), 
                                   key=lambda x: x[1], reverse=True)
            for section_type, count in sorted_sections[:10]:  # Top 10 section types
                report.append(f"• {section_type}: {count}")
            report.append("")
        
        # Detailed Sections
        for section_name, section_data in self.sections.items():
            report.append(f"{section_name.upper()}")
            report.append("-" * len(section_name))
            
            content = section_data['content']
            if isinstance(content, dict):
                for key, value in content.items():
                    if isinstance(value, (list, dict)):
                        report.append(f"{key}: {str(value)[:100]}...")
                    else:
                        report.append(f"{key}: {value}")
            else:
                report.append(str(content))
            report.append("")
        
        # Generated Files
        if self.files_generated:
            report.append("GENERATED FILES")
            report.append("-" * 40)
            for file_info in self.files_generated:
                report.append(f"• {file_info['description']}: {file_info['filepath']}")
            report.append("")
        
        return "\n".join(report)
    
    def generate_markdown_report(self) -> str:
        """Generate markdown formatted report."""
        report = []
        report.append("# Full Document Parsing Comprehensive Report")
        report.append("")
        report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"**Data Directory:** `{self.data_dir}`")
        report.append(f"**Parsing Duration:** {datetime.now() - self.start_time}")
        report.append("")
        
        # Executive Summary
        report.append("## Executive Summary")
        report.append("")
        for key, value in self.summary_stats.items():
            report.append(f"- **{key}:** {value}")
        report.append("")
        
        # Parsing Performance
        perf_metrics = self.calculate_performance_metrics()
        if perf_metrics:
            report.append("## Parsing Performance")
            report.append("")
            for key, value in perf_metrics.items():
                if 'time' in key.lower():
                    report.append(f"- **{key}:** {value:.2f}s")
                elif 'memory' in key.lower():
                    report.append(f"- **{key}:** {value:.1f}MB")
                elif 'rate' in key.lower():
                    report.append(f"- **{key}:** {value:.2f} docs/sec")
                else:
                    report.append(f"- **{key}:** {value}")
            report.append("")
        
        # Format Distribution
        if self.parsing_metrics['successful_parses'] > 0:
            report.append("## Format Distribution")
            report.append("")
            total_success = self.parsing_metrics['successful_parses']
            report.append(f"- **TEI documents:** {self.parsing_metrics['tei_successes']} ({self.parsing_metrics['tei_successes']/total_success:.1%})")
            report.append(f"- **JATS documents:** {self.parsing_metrics['jats_successes']} ({self.parsing_metrics['jats_successes']/total_success:.1%})")
            report.append(f"- **Unknown format:** {self.parsing_metrics['unknown_format_attempts']} ({self.parsing_metrics['unknown_format_attempts']/total_success:.1%})")
            report.append("")
        
        # Section Type Analysis
        if self.parsing_metrics['sections_by_type']:
            report.append("## Section Type Analysis")
            report.append("")
            report.append("| Section Type | Count |")
            report.append("|--------------|-------|")
            sorted_sections = sorted(self.parsing_metrics['sections_by_type'].items(), 
                                   key=lambda x: x[1], reverse=True)
            for section_type, count in sorted_sections:
                report.append(f"| {section_type} | {count} |")
            report.append("")
        
        # Error Analysis
        if self.parsing_metrics['errors_by_type']:
            report.append("## Error Analysis")
            report.append("")
            for error_type, count in self.parsing_metrics['errors_by_type'].items():
                report.append(f"- **{error_type}:** {count} occurrences")
            report.append("")
        
        # Validation Failures
        if self.parsing_metrics['validation_failures']:
            report.append("## Validation Issues")
            report.append("")
            report.append(f"**Total validation failures:** {len(self.parsing_metrics['validation_failures'])}")
            report.append("")
            
            # Group by reason
            failure_reasons = {}
            for failure in self.parsing_metrics['validation_failures']:
                reason = failure['reason']
                failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
            
            for reason, count in failure_reasons.items():
                report.append(f"- **{reason}:** {count} documents")
            report.append("")
        
        # Detailed Sections
        for section_name, section_data in self.sections.items():
            report.append(f"## {section_name.replace('_', ' ').title()}")
            report.append("")
            
            content = section_data['content']
            if isinstance(content, dict):
                for key, value in content.items():
                    if isinstance(value, list) and len(value) <= 10:
                        report.append(f"**{key}:**")
                        for item in value:
                            report.append(f"  - {item}")
                    elif isinstance(value, dict) and len(value) <= 10:
                        report.append(f"**{key}:**")
                        for k, v in value.items():
                            report.append(f"  - {k}: {v}")
                    else:
                        report.append(f"**{key}:** {value}")
                report.append("")
        
        # Generated Files
        if self.files_generated:
            report.append("## Generated Files")
            report.append("")
            for file_info in self.files_generated:
                report.append(f"- **{file_info['description']}:** `{file_info['filepath']}`")
            report.append("")
        
        return "\n".join(report)
    
    def generate_json_summary(self) -> Dict[str, Any]:
        """Generate JSON summary of all parsing results."""
        return {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'data_directory': self.data_dir,
                'parsing_duration': str(datetime.now() - self.start_time),
                'parsing_steps_completed': len(self.sections)
            },
            'summary_statistics': self.summary_stats,
            'parsing_metrics': self.parsing_metrics,
            'performance_metrics': self.calculate_performance_metrics(),
            'detailed_sections': {
                name: data['content'] for name, data in self.sections.items()
            },
            'generated_files': self.files_generated
        }
    
    def save_reports(self, reports_dir: Optional[str] = None):
        """Save reports in multiple formats to reports directory."""
        if reports_dir is None:
            # Save reports to root/reports directory, not data directory
            reports_dir = Path(self.data_dir).parent / "reports"
        
        output_path = Path(reports_dir)
        # Ensure reports directory exists
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save markdown report
        md_file = output_path / f"full_doc_parsing_report_{timestamp}.md"
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(self.generate_markdown_report())
        
        # Save JSON summary
        json_file = output_path / f"full_doc_parsing_summary_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(self.generate_json_summary(), f, indent=2, default=str)
        
        return str(md_file), str(json_file)


@timer_wrap
def run_full_document_parsing(data_dir: str = "Data",
                             output_format: str = "all",
                             save_reports: bool = True,
                             reports_dir: str = None) -> DocumentParsingReporter:
    """
    Run comprehensive document parsing with enhanced reporting.
    
    Args:
        data_dir (str): Path to data directory
        output_format (str): Output format - 'console', 'markdown', 'json', or 'all'
        save_reports (bool): Whether to save reports to files
        reports_dir (str): Directory to save reports (default: reports/ in project root)
    
    Returns:
        DocumentParsingReporter: Reporter object with all parsing results
    """
    
    # Initialize logging and reporter
    filename = os.path.basename(__file__)
    logger = initialize_logging(filename)
    reporter = DocumentParsingReporter(data_dir)
    
    logger.info("Starting comprehensive document parsing...")
    reporter.add_section("parsing_start", {
        "message": "Document parsing process initiated",
        "data_directory": data_dir,
        "timestamp": datetime.now().isoformat()
    })
    
    try:
        # Step 1: Load & Validate Document Inventory
        logger.info("Step 1: Loading & validating document inventory...")
        step_start = time.time()
        
        inventory_df = load_document_inventory()
        
        step1_results = {
            "total_documents": len(inventory_df),
            "columns": list(inventory_df.columns),
            "inventory_file_loaded": "document_inventory.csv",
            "validation_passed": True,
            "step_duration": time.time() - step_start
        }
        
        # Count by format type
        tei_count = len(inventory_df[inventory_df['source'] == 'grobid'])
        jats_count = len(inventory_df[inventory_df['source'].isna()])
        
        step1_results.update({
            "tei_files_available": tei_count,
            "jats_files_available": jats_count,
            "format_distribution": f"TEI: {tei_count}, JATS: {jats_count}"
        })
        
        reporter.add_section("step_1_load_inventory", step1_results)
        reporter.add_summary_stat("Total Documents in Inventory", len(inventory_df))
        reporter.add_summary_stat("TEI Files Available", tei_count)
        reporter.add_summary_stat("JATS Files Available", jats_count)
        
        # Step 2: Process All Documents
        logger.info("Step 2: Processing all documents...")
        step_start = time.time()
        
        # Wrap the processing with enhanced tracking
        parsed_documents = []
        failed_documents = []
        
        total_files = len(inventory_df)
        logger.info(f"Processing {total_files} documents...")
        
        for idx, row in inventory_df.iterrows():
            doc_start = time.time()
            article_id = row['article_id']
            xml_path = row['xml_path']
            source_type = row.get('source', None)
            
            if pd.isna(xml_path) or not Path(xml_path).exists():
                logger.warning(f"Skipping {article_id}: XML file not found at {xml_path}")
                failed_documents.append(article_id)
                reporter.update_parsing_stats(
                    failed_doc_id=article_id, 
                    error_details="File not found",
                    processing_time=time.time() - doc_start
                )
                continue
            
            try:
                # Use existing parsing logic
                from document_parser import parse_document, create_document_entry
                
                sections = parse_document(Path(xml_path), source_type)
                
                if sections:
                    # Create document entry using existing function
                    entry, validation = create_document_entry(article_id, sections, Path(xml_path), source_type)
                    
                    # Validate using modern Pydantic
                    validated_entry = Document.model_validate(entry.model_dump())
                    
                    parsed_documents.append((validated_entry, validation))
                    
                    # Update reporter with success
                    reporter.update_parsing_stats(
                        doc_result=(validated_entry, validation),
                        processing_time=time.time() - doc_start
                    )
                    
                    if idx % 50 == 0:  # Progress update every 50 files
                        logger.info(f"Processed {idx + 1}/{total_files}: {article_id} ({len(sections)} sections)")
                else:
                    logger.warning(f"Failed to parse {article_id}: No sections extracted")
                    failed_documents.append(article_id)
                    reporter.update_parsing_stats(
                        failed_doc_id=article_id,
                        error_details="No sections extracted",
                        processing_time=time.time() - doc_start
                    )
                    
            except Exception as e:
                logger.error(f"Error processing {article_id}: {e}")
                failed_documents.append(article_id)
                reporter.update_parsing_stats(
                    failed_doc_id=article_id,
                    error_details=str(e),
                    processing_time=time.time() - doc_start
                )
        
        step2_results = {
            "total_processed": len(parsed_documents) + len(failed_documents),
            "successful_parses": len(parsed_documents),
            "failed_parses": len(failed_documents),
            "success_rate": len(parsed_documents) / total_files if total_files > 0 else 0,
            "step_duration": time.time() - step_start
        }
        
        if failed_documents:
            step2_results["failed_documents_sample"] = failed_documents[:10]
        
        reporter.add_section("step_2_document_processing", step2_results)
        reporter.add_summary_stat("Successfully Parsed Documents", len(parsed_documents))
        reporter.add_summary_stat("Failed Documents", len(failed_documents))
        reporter.add_summary_stat("Parsing Success Rate", f"{step2_results['success_rate']:.1%}")
        
        # Step 3: Validate Parsed Corpus
        logger.info("Step 3: Validating parsed corpus...")
        step_start = time.time()
        
        validation_stats = validate_parsed_corpus(parsed_documents)
        
        step3_results = {
            "validation_statistics": validation_stats,
            "validation_success_rate": validation_stats['validation_passed'] / validation_stats['total_documents'] if validation_stats['total_documents'] > 0 else 0,
            "key_sections_coverage": validation_stats['key_sections_coverage'] / validation_stats['total_documents'] if validation_stats['total_documents'] > 0 else 0,
            "step_duration": time.time() - step_start
        }
        
        reporter.add_section("step_3_corpus_validation", step3_results)
        reporter.add_summary_stat("Validation Success Rate", f"{step3_results['validation_success_rate']:.1%}")
        reporter.add_summary_stat("Key Sections Coverage", f"{step3_results['key_sections_coverage']:.1%}")
        
        # Step 4: Save Parsed Corpus (preserve existing functionality)
        logger.info("Step 4: Saving parsed corpus...")
        step_start = time.time()
        
        save_parsed_corpus(parsed_documents, validation_stats)
        
        # Track generated files
        output_dir = Path(data_dir) / "train" / "parsed"
        pickle_file = output_dir / "parsed_documents.pkl"
        summary_file = output_dir / "parsed_documents_summary.csv"
        stats_file = output_dir / "validation_stats.json"
        
        reporter.add_generated_file(str(pickle_file), "Parsed documents pickle file")
        reporter.add_generated_file(str(summary_file), "Document parsing summary CSV")
        reporter.add_generated_file(str(stats_file), "Validation statistics JSON")
        
        step4_results = {
            "corpus_saved": True,
            "output_directory": str(output_dir),
            "files_generated": 3,
            "step_duration": time.time() - step_start
        }
        
        reporter.add_section("step_4_save_corpus", step4_results)
        reporter.add_summary_stat("Output Files Generated", 3)
        
        # Step 5: Performance Analysis
        logger.info("Step 5: Analyzing performance metrics...")
        step_start = time.time()
        
        perf_metrics = reporter.calculate_performance_metrics()
        
        # Quality metrics
        format_distribution = {
            'TEI': reporter.parsing_metrics['tei_successes'],
            'JATS': reporter.parsing_metrics['jats_successes'],
            'Unknown': reporter.parsing_metrics['unknown_format_attempts']
        }
        
        step5_results = {
            "performance_metrics": perf_metrics,
            "format_distribution": format_distribution,
            "section_type_diversity": len(reporter.parsing_metrics['sections_by_type']),
            "validation_issues": len(reporter.parsing_metrics['validation_failures']),
            "step_duration": time.time() - step_start
        }
        
        reporter.add_section("step_5_performance_analysis", step5_results)
        reporter.add_summary_stat("Section Type Diversity", len(reporter.parsing_metrics['sections_by_type']))
        
        # Final summary statistics
        reporter.add_summary_stat("Total Processing Time", f"{sum(reporter.parsing_metrics['processing_times']):.2f}s")
        reporter.add_summary_stat("Average Processing Time", f"{perf_metrics.get('average_processing_time', 0):.2f}s")
        reporter.add_summary_stat("Parsing Steps Completed", 5)
        
        logger.info("Document parsing completed successfully!")
        
        # Generate and display reports based on output format
        if output_format in ['console', 'all']:
            print(reporter.generate_console_report())
        
        if save_reports and output_format in ['markdown', 'json', 'all']:
            md_file, json_file = reporter.save_reports(reports_dir)
            logger.info(f"Reports saved: {md_file}, {json_file}")
            reporter.add_generated_file(md_file, "Markdown parsing report")
            reporter.add_generated_file(json_file, "JSON parsing summary")
        
        return reporter
        
    except Exception as e:
        logger.error(f"Error during document parsing: {str(e)}")
        reporter.add_section("error", {
            "error_message": str(e),
            "error_type": type(e).__name__,
            "timestamp": datetime.now().isoformat()
        })
        raise


def main():
    """Command-line interface for full document parsing."""
    parser = argparse.ArgumentParser(
        description="Run comprehensive document parsing with report generation"
    )
    
    parser.add_argument(
        "--data-dir",
        default="Data",
        help="Path to data directory (default: Data)"
    )
    
    parser.add_argument(
        "--output-format",
        choices=['console', 'markdown', 'json', 'all'],
        default='all',
        help="Output format for reports (default: all)"
    )
    
    parser.add_argument(
        "--no-save",
        action='store_true',
        help="Don't save reports to files"
    )
    
    parser.add_argument(
        "--reports-dir",
        default=None,
        help="Directory to save reports (default: reports/ in project root)"
    )
    
    args = parser.parse_args()
    
    try:
        reporter = run_full_document_parsing(
            data_dir=args.data_dir,
            output_format=args.output_format,
            save_reports=not args.no_save,
            reports_dir=args.reports_dir
        )
        
        print(f"\n✅ Document parsing completed successfully!")
        print(f"📊 Total documents processed: {reporter.parsing_metrics['total_processed']}")
        print(f"✅ Successful parses: {reporter.parsing_metrics['successful_parses']}")
        print(f"❌ Failed parses: {reporter.parsing_metrics['failed_parses']}")
        
        if reporter.files_generated:
            print("\n📁 Generated files:")
            for file_info in reporter.files_generated:
                print(f"  • {file_info['description']}: {file_info['filepath']}")
        
    except Exception as e:
        print(f"❌ Document parsing failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main() 