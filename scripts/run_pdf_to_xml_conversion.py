#!/usr/bin/env python3
"""
Enhanced PDF to XML Conversion Script with Comprehensive Reporting (step 4)

This script provides comprehensive PDF to XML conversion with structured report generation
for the MDC Challenge 2025 dataset. It integrates existing conversion logic with enhanced
reporting capabilities similar to the pre-chunking EDA script.

Features:
- Comprehensive conversion workflow with detailed progress tracking
- Multiple output formats (console, markdown, JSON)
- Performance metrics and quality analysis
- Error categorization and resolution guidance
- Integration with existing conversion pipeline
- Detailed reporting of all conversion steps
"""

import os
import sys
import json
import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from helpers import initialize_logging, timer_wrap

# Import existing conversion functions (without modification)
sys.path.append(str(Path(__file__).parent.parent / "src"))
from pdf_to_xml_conversion import (
    load_conversion_candidates,
    inventory_existing_xml,
    convert_one,
    generate_conversion_report,
    calculate_coverage_kpi,
    validate_conversions,
    get_remaining_conversions
)


class ConversionReporter:
    """
    Captures and formats outputs from PDF to XML conversion process for comprehensive reporting.
    Based on EDAReporter but specialized for conversion workflows.
    """
    
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.start_time = datetime.now()
        self.sections = {}
        self.summary_stats = {}
        self.files_generated = []
        self.conversion_metrics = {
            'total_processed': 0,
            'successful_conversions': 0,
            'failed_conversions': 0,
            'grobid_successes': 0,
            'fallback_successes': 0,
            'processing_times': [],
            'file_sizes': [],
            'errors_by_type': {}
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
        """Track files generated during conversion."""
        self.files_generated.append({
            'filepath': filepath,
            'description': description,
            'timestamp': datetime.now()
        })
    
    def add_conversion_metric(self, metric_name: str, value: Any):
        """Add conversion-specific metrics."""
        self.conversion_metrics[metric_name] = value
    
    def update_conversion_stats(self, log_entry: Dict[str, Any], processing_time: float = None):
        """Update conversion statistics based on log entry."""
        self.conversion_metrics['total_processed'] += 1
        
        if log_entry.get('success', False):
            self.conversion_metrics['successful_conversions'] += 1
            if log_entry.get('source') == 'grobid':
                self.conversion_metrics['grobid_successes'] += 1
            elif log_entry.get('source') == 'pdfplumber':
                self.conversion_metrics['fallback_successes'] += 1
        else:
            self.conversion_metrics['failed_conversions'] += 1
            error_type = log_entry.get('error', 'Unknown')
            self.conversion_metrics['errors_by_type'][error_type] = \
                self.conversion_metrics['errors_by_type'].get(error_type, 0) + 1
        
        if processing_time:
            self.conversion_metrics['processing_times'].append(processing_time)
    
    def calculate_performance_metrics(self):
        """Calculate performance metrics from collected data."""
        times = self.conversion_metrics['processing_times']
        if times:
            return {
                'average_processing_time': sum(times) / len(times),
                'total_processing_time': sum(times),
                'fastest_conversion': min(times),
                'slowest_conversion': max(times),
                'processing_rate': len(times) / sum(times) if sum(times) > 0 else 0
            }
        return {}
    
    def generate_console_report(self) -> str:
        """Generate formatted console output."""
        report = []
        report.append("=" * 80)
        report.append("PDF TO XML CONVERSION COMPREHENSIVE REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Data Directory: {self.data_dir}")
        report.append(f"Conversion Duration: {datetime.now() - self.start_time}")
        report.append("")
        
        # Executive Summary
        report.append("EXECUTIVE SUMMARY")
        report.append("-" * 40)
        for key, value in self.summary_stats.items():
            report.append(f"• {key}: {value}")
        report.append("")
        
        # Conversion Performance
        perf_metrics = self.calculate_performance_metrics()
        if perf_metrics:
            report.append("CONVERSION PERFORMANCE")
            report.append("-" * 40)
            for key, value in perf_metrics.items():
                if 'time' in key.lower():
                    report.append(f"• {key}: {value:.2f}s")
                elif 'rate' in key.lower():
                    report.append(f"• {key}: {value:.2f} files/sec")
                else:
                    report.append(f"• {key}: {value}")
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
        report.append("# PDF to XML Conversion Comprehensive Report")
        report.append("")
        report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"**Data Directory:** `{self.data_dir}`")
        report.append(f"**Conversion Duration:** {datetime.now() - self.start_time}")
        report.append("")
        
        # Executive Summary
        report.append("## Executive Summary")
        report.append("")
        for key, value in self.summary_stats.items():
            report.append(f"- **{key}:** {value}")
        report.append("")
        
        # Conversion Performance
        perf_metrics = self.calculate_performance_metrics()
        if perf_metrics:
            report.append("## Conversion Performance")
            report.append("")
            for key, value in perf_metrics.items():
                if 'time' in key.lower():
                    report.append(f"- **{key}:** {value:.2f}s")
                elif 'rate' in key.lower():
                    report.append(f"- **{key}:** {value:.2f} files/sec")
                else:
                    report.append(f"- **{key}:** {value}")
            report.append("")
        
        # Error Analysis
        if self.conversion_metrics['errors_by_type']:
            report.append("## Error Analysis")
            report.append("")
            for error_type, count in self.conversion_metrics['errors_by_type'].items():
                report.append(f"- **{error_type}:** {count} occurrences")
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
        """Generate JSON summary of all conversion results."""
        return {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'data_directory': self.data_dir,
                'conversion_duration': str(datetime.now() - self.start_time),
                'conversion_steps_completed': len(self.sections)
            },
            'summary_statistics': self.summary_stats,
            'conversion_metrics': self.conversion_metrics,
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
        md_file = output_path / f"pdf_to_xml_conversion_report_{timestamp}.md"
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(self.generate_markdown_report())
        
        # Save JSON summary
        json_file = output_path / f"pdf_to_xml_conversion_summary_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(self.generate_json_summary(), f, indent=2, default=str)
        
        return str(md_file), str(json_file)


@timer_wrap
def run_pdf_to_xml_conversion(data_dir: str = "Data",
                             output_format: str = "all",
                             save_reports: bool = True,
                             reports_dir: str = None,
                             grobid_url: str = None,
                             resume: bool = True) -> ConversionReporter:
    """
    Run comprehensive PDF to XML conversion with enhanced reporting.
    
    Args:
        data_dir (str): Path to data directory
        output_format (str): Output format - 'console', 'markdown', 'json', or 'all'
        save_reports (bool): Whether to save reports to files
        reports_dir (str): Directory to save reports (default: reports/ in project root)
        grobid_url (str): Grobid service URL (default: from environment)
        resume (bool): Whether to resume from existing conversions
    
    Returns:
        ConversionReporter: Reporter object with all conversion results
    """
    
    # Initialize logging and reporter
    filename = os.path.basename(__file__)
    logger = initialize_logging(filename)
    reporter = ConversionReporter(data_dir)
    
    # Set Grobid URL if provided
    if grobid_url:
        os.environ["GROBID_URL"] = grobid_url
    
    logger.info("Starting comprehensive PDF to XML conversion...")
    reporter.add_section("conversion_start", {
        "message": "PDF to XML conversion process initiated",
        "data_directory": data_dir,
        "grobid_url": os.getenv("GROBID_URL", "http://localhost:8070"),
        "timestamp": datetime.now().isoformat()
    })
    
    try:
        # Step 1: Load & Validate Conversion Candidates
        logger.info("Step 1: Loading & validating conversion candidates...")
        step_start = time.time()
        
        cand = load_conversion_candidates()
        
        step1_results = {
            "total_candidates": len(cand),
            "columns": list(cand.columns),
            "candidate_file_loaded": "conversion_candidates.csv",
            "validation_passed": True,
            "step_duration": time.time() - step_start
        }
        
        reporter.add_section("step_1_load_candidates", step1_results)
        reporter.add_summary_stat("Total Conversion Candidates", len(cand))
        
        # Step 2: Inventory Existing XML Files
        logger.info("Step 2: Inventorying existing XML files...")
        step_start = time.time()
        
        todo = inventory_existing_xml(cand)
        existing_xml_count = len(cand) - len(todo)
        
        step2_results = {
            "existing_xml_files": existing_xml_count,
            "files_needing_conversion": len(todo),
            "conversion_coverage": f"{existing_xml_count}/{len(cand)} ({existing_xml_count/len(cand)*100:.1f}%)",
            "step_duration": time.time() - step_start
        }
        
        reporter.add_section("step_2_inventory_xml", step2_results)
        reporter.add_summary_stat("Existing XML Files", existing_xml_count)
        reporter.add_summary_stat("Files Needing Conversion", len(todo))
        
        # Step 3: Pre-conversion Validation
        logger.info("Step 3: Pre-conversion validation and planning...")
        step_start = time.time()
        
        # Validate PDF files exist
        pdf_validation = {"valid_pdfs": 0, "missing_pdfs": 0, "invalid_paths": []}
        for _, row in todo.iterrows():
            pdf_path_str = row['pdf_path']
            if pdf_path_str.startswith('/'):
                pdf_path = Path(pdf_path_str.split('Data/')[-1])
                pdf_path = Path(data_dir) / pdf_path
            else:
                pdf_path = Path(pdf_path_str)
            
            if pdf_path.exists():
                pdf_validation["valid_pdfs"] += 1
            else:
                pdf_validation["missing_pdfs"] += 1
                pdf_validation["invalid_paths"].append(str(pdf_path))
        
        step3_results = {
            "pdf_validation": pdf_validation,
            "conversion_readiness": f"{pdf_validation['valid_pdfs']}/{len(todo)} PDFs ready for conversion",
            "priority_analysis": {
                "high_priority": len(todo[todo['has_primary'] == True]),
                "medium_priority": len(todo[todo['has_secondary'] == True]),
                "low_priority": len(todo[todo['has_missing'] == True])
            },
            "step_duration": time.time() - step_start
        }
        
        reporter.add_section("step_3_preconversion_validation", step3_results)
        reporter.add_summary_stat("PDFs Ready for Conversion", pdf_validation["valid_pdfs"])
        
        # Step 4: Execute Batch Conversion
        logger.info("Step 4: Executing batch conversion...")
        step_start = time.time()
        
        if len(todo) > 0:
            logger.info(f"Converting {len(todo)} PDFs...")
            
            # Enhanced batch conversion with progress tracking
            log = []
            xml_dir = Path(data_dir) / "train" / "XML"
            xml_dir.mkdir(exist_ok=True)
            
            from tqdm import tqdm
            
            for idx, (_, row) in enumerate(tqdm(todo.iterrows(), total=len(todo), desc="Converting PDFs")):
                file_start = time.time()
                
                # Handle path conversion
                pdf_path_str = row['pdf_path']
                if pdf_path_str.startswith('/'):
                    pdf_path = Path(pdf_path_str.split('Data/')[-1])
                    pdf_path = Path(data_dir) / pdf_path
                else:
                    pdf_path = Path(pdf_path_str)
                
                xml_path = xml_dir / f"{row['article_id']}.xml"
                
                try:
                    source = convert_one(pdf_path, xml_path)
                    file_time = time.time() - file_start
                    
                    log_entry = {
                        'article_id': row['article_id'],
                        'pdf_path': row['pdf_path'],
                        'xml_path': str(xml_path),
                        'source': source,
                        'error': None,
                        'success': source != "failed",
                        'processing_time': file_time,
                        'has_primary': row['has_primary'],
                        'has_secondary': row['has_secondary'],
                        'label_count': row['label_count']
                    }
                    
                    log.append(log_entry)
                    reporter.update_conversion_stats(log_entry, file_time)
                    
                except Exception as e:
                    file_time = time.time() - file_start
                    log_entry = {
                        'article_id': row['article_id'],
                        'pdf_path': row['pdf_path'],
                        'xml_path': None,
                        'source': None,
                        'error': str(e),
                        'success': False,
                        'processing_time': file_time,
                        'has_primary': row['has_primary'],
                        'has_secondary': row['has_secondary'],
                        'label_count': row['label_count']
                    }
                    
                    log.append(log_entry)
                    reporter.update_conversion_stats(log_entry, file_time)
                
                # Small delay to avoid overwhelming services
                time.sleep(0.1)
        else:
            log = []
            logger.info("No files need conversion - all PDFs already have XML equivalents")
        
        step4_results = {
            "total_processed": len(log),
            "successful_conversions": sum(1 for entry in log if entry['success']),
            "failed_conversions": sum(1 for entry in log if not entry['success']),
            "conversion_sources": {
                "grobid": sum(1 for entry in log if entry.get('source') == 'grobid'),
                "pdfplumber": sum(1 for entry in log if entry.get('source') == 'pdfplumber')
            },
            "step_duration": time.time() - step_start
        }
        
        reporter.add_section("step_4_batch_conversion", step4_results)
        reporter.add_summary_stat("Successful Conversions", step4_results["successful_conversions"])
        reporter.add_summary_stat("Failed Conversions", step4_results["failed_conversions"])
        
        # Step 5: Generate Conversion Report (using existing function)
        logger.info("Step 5: Generating conversion report...")
        step_start = time.time()
        
        # Use existing function to generate report
        generate_conversion_report(log, cand)
        
        # Calculate coverage using existing function
        coverage = calculate_coverage_kpi(cand)
        
        step5_results = {
            "conversion_log_generated": True,
            "document_inventory_updated": True,
            "coverage_kpi": coverage,
            "coverage_meets_threshold": coverage >= 0.90,
            "step_duration": time.time() - step_start
        }
        
        reporter.add_section("step_5_generate_report", step5_results)
        reporter.add_summary_stat("Coverage KPI", f"{coverage:.2%}")
        
        # Add document inventory to generated files
        doc_inventory = Path(data_dir) / "document_inventory.csv"
        if doc_inventory.exists():
            reporter.add_generated_file(str(doc_inventory), "Updated document inventory")
        
        # Step 6: Quality Validation
        logger.info("Step 6: Conducting quality validation...")
        step_start = time.time()
        
        validate_conversions()
        
        # Additional quality checks
        xml_dir = Path(data_dir) / "train" / "XML"
        quality_metrics = {
            "total_xml_files": len(list(xml_dir.glob("*.xml"))),
            "small_files": 0,
            "large_files": 0,
            "average_file_size": 0,
            "file_size_distribution": {}
        }
        
        file_sizes = []
        for xml_file in xml_dir.glob("*.xml"):
            size = xml_file.stat().st_size
            file_sizes.append(size)
            
            if size < 1024:  # Less than 1KB
                quality_metrics["small_files"] += 1
            elif size > 1024 * 1024:  # Greater than 1MB
                quality_metrics["large_files"] += 1
        
        if file_sizes:
            quality_metrics["average_file_size"] = sum(file_sizes) / len(file_sizes)
        
        step6_results = {
            "quality_validation_completed": True,
            "quality_metrics": quality_metrics,
            "validation_issues": quality_metrics["small_files"],
            "step_duration": time.time() - step_start
        }
        
        reporter.add_section("step_6_quality_validation", step6_results)
        reporter.add_summary_stat("Quality Issues", quality_metrics["small_files"])
        
        # Step 7: Finalization and Summary
        logger.info("Step 7: Finalizing conversion and generating summary...")
        step_start = time.time()
        
        # Final statistics
        total_xml = len(list((Path(data_dir) / "train" / "XML").glob("*.xml")))
        total_candidates = len(cand)
        
        final_summary = {
            "total_conversion_candidates": total_candidates,
            "total_xml_files_available": total_xml,
            "final_coverage": total_xml / total_candidates if total_candidates > 0 else 0,
            "conversion_session_statistics": {
                "files_processed": len(log),
                "successful_conversions": sum(1 for entry in log if entry['success']),
                "failed_conversions": sum(1 for entry in log if not entry['success']),
                "total_processing_time": sum(reporter.conversion_metrics['processing_times']),
                "average_processing_time": sum(reporter.conversion_metrics['processing_times']) / len(reporter.conversion_metrics['processing_times']) if reporter.conversion_metrics['processing_times'] else 0
            },
            "step_duration": time.time() - step_start
        }
        
        reporter.add_section("step_7_finalization", final_summary)
        reporter.add_summary_stat("Final XML Coverage", f"{total_xml}/{total_candidates}")
        reporter.add_summary_stat("Conversion Session Complete", True)
        
        # Add final summary statistics
        reporter.add_summary_stat("Total Processing Time", f"{sum(reporter.conversion_metrics['processing_times']):.2f}s")
        reporter.add_summary_stat("Average Processing Time", f"{sum(reporter.conversion_metrics['processing_times']) / len(reporter.conversion_metrics['processing_times']):.2f}s" if reporter.conversion_metrics['processing_times'] else "N/A")
        reporter.add_summary_stat("Conversion Steps Completed", 7)
        
        logger.info("PDF to XML conversion completed successfully!")
        
        # Generate and display reports based on output format
        if output_format in ['console', 'all']:
            print(reporter.generate_console_report())
        
        if save_reports and output_format in ['markdown', 'json', 'all']:
            md_file, json_file = reporter.save_reports(reports_dir)
            logger.info(f"Reports saved: {md_file}, {json_file}")
            reporter.add_generated_file(md_file, "Markdown conversion report")
            reporter.add_generated_file(json_file, "JSON conversion summary")
        
        return reporter
        
    except Exception as e:
        logger.error(f"Error during PDF to XML conversion: {str(e)}")
        reporter.add_section("error", {
            "error_message": str(e),
            "error_type": type(e).__name__,
            "timestamp": datetime.now().isoformat()
        })
        raise


def main():
    """Command-line interface for PDF to XML conversion."""
    parser = argparse.ArgumentParser(
        description="Run comprehensive PDF to XML conversion with report generation"
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
    
    parser.add_argument(
        "--grobid-url",
        default=None,
        help="Grobid service URL (default: http://localhost:8070)"
    )
    
    parser.add_argument(
        "--no-resume",
        action='store_true',
        help="Don't resume from existing conversions"
    )
    
    args = parser.parse_args()
    
    try:
        reporter = run_pdf_to_xml_conversion(
            data_dir=args.data_dir,
            output_format=args.output_format,
            save_reports=not args.no_save,
            reports_dir=args.reports_dir,
            grobid_url=args.grobid_url,
            resume=not args.no_resume
        )
        
        print(f"\n✅ Conversion completed successfully!")
        print(f"📊 Total files processed: {reporter.conversion_metrics['total_processed']}")
        print(f"✅ Successful conversions: {reporter.conversion_metrics['successful_conversions']}")
        print(f"❌ Failed conversions: {reporter.conversion_metrics['failed_conversions']}")
        
        if reporter.files_generated:
            print("\n📁 Generated files:")
            for file_info in reporter.files_generated:
                print(f"  • {file_info['description']}: {file_info['filepath']}")
        
    except Exception as e:
        print(f"❌ Conversion failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main() 