# scripts/run_prechunking_eda.py

"""
Enhanced Pre-Chunking EDA Script with Comprehensive Report Generation (step 3)

This script mirrors the analysis performed in the label_doc_mapping.ipynb notebook
and generates structured reports of all findings and statistics.

Features:
- Comprehensive label and document analysis
- PDF→XML conversion workflow planning
- Quality checks and validation
- Multiple output formats (console, markdown, JSON)
- Detailed reporting of all analysis steps
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from label_mapper import LabelMapper
from helpers import initialize_logging, timer_wrap


class EDAReporter:
    """
    Captures and formats outputs from EDA analysis steps for comprehensive reporting.
    """
    
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.start_time = datetime.now()
        self.sections = {}
        self.summary_stats = {}
        self.files_generated = []
        
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
        """Track files generated during analysis."""
        self.files_generated.append({
            'filepath': filepath,
            'description': description,
            'timestamp': datetime.now()
        })
    
    def generate_console_report(self) -> str:
        """Generate formatted console output."""
        report = []
        report.append("=" * 80)
        report.append("PRE-CHUNKING EDA COMPREHENSIVE REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Data Directory: {self.data_dir}")
        report.append(f"Analysis Duration: {datetime.now() - self.start_time}")
        report.append("")
        
        # Executive Summary
        report.append("EXECUTIVE SUMMARY")
        report.append("-" * 40)
        for key, value in self.summary_stats.items():
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
        report.append("# Pre-Chunking EDA Comprehensive Report")
        report.append("")
        report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"**Data Directory:** `{self.data_dir}`")
        report.append(f"**Analysis Duration:** {datetime.now() - self.start_time}")
        report.append("")
        
        # Executive Summary
        report.append("## Executive Summary")
        report.append("")
        for key, value in self.summary_stats.items():
            report.append(f"- **{key}:** {value}")
        report.append("")
        
        # Detailed Sections
        for section_name, section_data in self.sections.items():
            report.append(f"## {section_name}")
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
        """Generate JSON summary of all analysis results."""
        return {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'data_directory': self.data_dir,
                'analysis_duration': str(datetime.now() - self.start_time),
                'analysis_steps_completed': len(self.sections)
            },
            'summary_statistics': self.summary_stats,
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
        md_file = output_path / f"prechunking_eda_report_{timestamp}.md"
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(self.generate_markdown_report())
        
        # Save JSON summary
        json_file = output_path / f"prechunking_eda_summary_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(self.generate_json_summary(), f, indent=2, default=str)
        
        return str(md_file), str(json_file)


@timer_wrap
def run_prechunking_eda(data_dir: str = "Data", 
                       output_format: str = "all",
                       save_reports: bool = True,
                       show_plots: bool = False,
                       reports_dir: str = None) -> EDAReporter:
    """
    Run comprehensive pre-chunking EDA following notebook analysis flow.
    
    Args:
        data_dir (str): Path to data directory
        output_format (str): Output format - 'console', 'markdown', 'json', or 'all'
        save_reports (bool): Whether to save reports to files
        show_plots (bool): Whether to display plots (for interactive use)
        reports_dir (str): Directory to save reports (default: reports/ in project root)
    
    Returns:
        EDAReporter: Reporter object with all analysis results
    """
    
    # Initialize logging and reporter
    filename = os.path.basename(__file__)
    logger = initialize_logging(filename)
    reporter = EDAReporter(data_dir)
    
    logger.info("Starting comprehensive pre-chunking EDA...")
    reporter.add_section("analysis_start", {
        "message": "Pre-chunking EDA analysis initiated",
        "data_directory": data_dir,
        "timestamp": datetime.now().isoformat()
    })
    
    try:
        # Step 1: Initialize LabelMapper and Load Labels
        logger.info("Step 1: Loading labels and initializing analysis...")
        mapper = LabelMapper(data_dir=data_dir)
        
        # Capture label loading results
        labels_df = mapper.load_labels("train_labels.csv")
        
        step1_results = {
            "total_label_entries": len(labels_df),
            "unique_articles": labels_df['article_id'].nunique(),
            "columns": list(labels_df.columns),
            "label_file_loaded": "train_labels.csv",
            "unique_articles_summary_created": len(mapper.unique_articles_df) if mapper.unique_articles_df is not None else 0
        }
        
        reporter.add_section("step_1_label_loading", step1_results)
        reporter.add_summary_stat("Total Label Entries", len(labels_df))
        reporter.add_summary_stat("Unique Articles", labels_df['article_id'].nunique())
        
        # Step 2: Unique Articles Analysis  
        logger.info("Step 2: Analyzing unique articles and multi-label distribution...")
        
        multi_label_count = len(labels_df) - len(labels_df['article_id'].unique())
        
        # Get label type distribution among unique articles
        unique_article_stats = {
            "articles_with_primary": mapper.unique_articles_df['has_primary'].sum() if mapper.unique_articles_df is not None else 0,
            "articles_with_secondary": mapper.unique_articles_df['has_secondary'].sum() if mapper.unique_articles_df is not None else 0,
            "articles_with_missing": mapper.unique_articles_df['has_missing'].sum() if mapper.unique_articles_df is not None else 0,
            "multi_label_articles": len(mapper.unique_articles_df[mapper.unique_articles_df['label_count'] > 1]) if mapper.unique_articles_df is not None else 0
        }
        
        step2_results = {
            "multi_label_entries": multi_label_count,
            "unique_article_statistics": unique_article_stats,
            "label_entries_vs_unique_articles": {
                "total_entries": len(labels_df),
                "unique_articles": labels_df['article_id'].nunique(),
                "difference": multi_label_count
            }
        }
        
        reporter.add_section("step_2_unique_articles_analysis", step2_results)
        reporter.add_summary_stat("Multi-label Articles", unique_article_stats.get("multi_label_articles", 0))
        
        # Step 3: Document Inventory
        logger.info("Step 3: Inventorying document paths and availability...")
        
        inventory_df = mapper.inventory_document_paths(pdf_dir="train/PDF", xml_dir="train/XML")
        
        # Capture inventory results
        pdf_count = inventory_df['pdf_available'].sum()
        xml_count = inventory_df['xml_available'].sum()
        fulltext_count = inventory_df['has_fulltext'].sum()
        
        step3_results = {
            "pdf_files_found": pdf_count,
            "xml_files_found": xml_count, 
            "articles_with_fulltext": fulltext_count,
            "fulltext_coverage_percentage": (fulltext_count / len(inventory_df)) * 100,
            "document_scanning_summary": {
                "total_articles": len(inventory_df),
                "pdf_availability": f"{pdf_count} ({pdf_count/len(inventory_df)*100:.1f}%)",
                "xml_availability": f"{xml_count} ({xml_count/len(inventory_df)*100:.1f}%)",
                "fulltext_availability": f"{fulltext_count} ({fulltext_count/len(inventory_df)*100:.1f}%)"
            }
        }
        
        reporter.add_section("step_3_document_inventory", step3_results)
        reporter.add_summary_stat("PDF Files Available", pdf_count)
        reporter.add_summary_stat("XML Files Available", xml_count)
        reporter.add_summary_stat("Full-text Coverage", f"{fulltext_count/len(inventory_df)*100:.1f}%")
        
        # Step 4: Conversion Workflow Analysis
        logger.info("Step 4: Analyzing PDF→XML conversion workflow requirements...")
        
        conversion_summary = mapper.get_conversion_summary()
        conversion_candidates = mapper.get_articles_needing_conversion()
        
        step4_results = {
            "conversion_workflow_summary": conversion_summary,
            "conversion_candidates_count": len(conversion_candidates),
            "conversion_priority_analysis": {
                "articles_with_primary_needing_conversion": conversion_candidates['has_primary'].sum() if len(conversion_candidates) > 0 else 0,
                "articles_with_secondary_needing_conversion": conversion_candidates['has_secondary'].sum() if len(conversion_candidates) > 0 else 0,
                "priority_candidates": len(conversion_candidates[conversion_candidates['has_primary'] | conversion_candidates['has_secondary']]) if len(conversion_candidates) > 0 else 0
            },
            "workflow_categorization": {
                "ready_for_processing": conversion_summary['both_available'] + conversion_summary['xml_only'],
                "needs_conversion": conversion_summary['pdf_only'],
                "cannot_process": conversion_summary['missing_both']
            }
        }
        
        reporter.add_section("step_4_conversion_workflow", step4_results)
        reporter.add_summary_stat("Articles Needing Conversion", len(conversion_candidates))
        reporter.add_summary_stat("Ready for Processing", conversion_summary['both_available'])
        
        # Step 5: Quality Checks
        logger.info("Step 5: Conducting comprehensive quality checks...")
        
        qc_results = mapper.conduct_basic_checks(show_plots=show_plots)
        
        step5_results = {
            "quality_check_summary": qc_results,
            "duplicate_analysis": {
                "duplicate_entries": qc_results.get('duplicate_entries', 0),
                "duplicate_article_entries": qc_results.get('duplicate_article_entries', 0)
            },
            "null_value_analysis": qc_results.get('null_counts', {}),
            "class_distribution": qc_results.get('class_distribution', {}),
            "class_percentages": qc_results.get('class_percentages', {}),
            "file_availability_check": qc_results.get('file_availability', {})
        }
        
        reporter.add_section("step_5_quality_checks", step5_results)
        
        # Add class distribution to summary
        class_dist = qc_results.get('class_distribution', {})
        for class_type, count in class_dist.items():
            reporter.add_summary_stat(f"{class_type} Labels", count)
        
        # Step 6: Summary Statistics
        logger.info("Step 6: Generating comprehensive summary statistics...")
        
        summary_stats = mapper.get_summary_stats()
        
        step6_results = {
            "comprehensive_statistics": summary_stats,
            "key_metrics": {
                "total_label_entries": summary_stats.get('total_label_entries', 0),
                "unique_articles": summary_stats.get('unique_articles', 0),
                "unique_datasets": summary_stats.get('unique_datasets', 0),
                "conversion_candidates": summary_stats.get('conversion_candidates', 0),
                "problematic_articles": summary_stats.get('problematic_articles', 0)
            }
        }
        
        reporter.add_section("step_6_summary_statistics", step6_results)
        
        # Step 7: Enhanced Inventory Analysis
        logger.info("Step 7: Performing enhanced document inventory analysis...")
        
        # Conversion status analysis
        conversion_status_counts = inventory_df['conversion_status'].value_counts().to_dict()
        
        # Label type distribution analysis
        label_type_summary = {
            'Primary Only': len(mapper.unique_articles_df[
                (mapper.unique_articles_df['has_primary']) & 
                (~mapper.unique_articles_df['has_secondary']) & 
                (~mapper.unique_articles_df['has_missing'])
            ]) if mapper.unique_articles_df is not None else 0,
            'Secondary Only': len(mapper.unique_articles_df[
                (~mapper.unique_articles_df['has_primary']) & 
                (mapper.unique_articles_df['has_secondary']) & 
                (~mapper.unique_articles_df['has_missing'])
            ]) if mapper.unique_articles_df is not None else 0,
            'Missing Only': len(mapper.unique_articles_df[
                (~mapper.unique_articles_df['has_primary']) & 
                (~mapper.unique_articles_df['has_secondary']) & 
                (mapper.unique_articles_df['has_missing'])
            ]) if mapper.unique_articles_df is not None else 0,
            'Multiple Labels': len(mapper.unique_articles_df[mapper.unique_articles_df['label_count'] > 1]) if mapper.unique_articles_df is not None else 0
        }
        
        # Priority analysis by conversion status
        priority_analysis = inventory_df.groupby(['conversion_status']).agg({
            'has_primary': 'sum',
            'has_secondary': 'sum', 
            'has_missing': 'sum',
            'article_id': 'count'
        }).to_dict()
        
        step7_results = {
            "conversion_status_distribution": conversion_status_counts,
            "label_type_distribution": label_type_summary,
            "priority_analysis_by_conversion_status": priority_analysis,
            "enhanced_metrics": {
                "total_articles_analyzed": len(inventory_df),
                "conversion_coverage": f"{len(conversion_candidates)}/{len(inventory_df)} articles need conversion"
            }
        }
        
        reporter.add_section("step_7_enhanced_inventory_analysis", step7_results)
        
        # Step 8: Export Operations
        logger.info("Step 8: Exporting conversion workflow files...")
        
        export_results = {
            "exports_performed": [],
            "file_locations": {}
        }
        
        # Export conversion candidates
        if len(conversion_candidates) > 0:
            mapper.export_conversion_candidates("conversion_candidates.csv")
            conv_file = Path(data_dir) / "conversion_candidates.csv"
            export_results["exports_performed"].append("conversion_candidates")
            export_results["file_locations"]["conversion_candidates"] = str(conv_file)
            reporter.add_generated_file(str(conv_file), "Articles needing PDF→XML conversion")
        
        # Export problematic articles
        mapper.export_missing_files("problematic_articles.txt")
        prob_file = Path(data_dir) / "problematic_articles.txt"
        export_results["exports_performed"].append("problematic_articles")
        export_results["file_locations"]["problematic_articles"] = str(prob_file)
        reporter.add_generated_file(str(prob_file), "Articles missing both PDF and XML")
        
        # Export document inventory
        inventory_file = Path(data_dir) / f"document_inventory.csv"
        inventory_df.to_csv(inventory_file, index=False)
        export_results["exports_performed"].append("document_inventory")
        export_results["file_locations"]["document_inventory"] = str(inventory_file)
        reporter.add_generated_file(str(inventory_file), "Complete document inventory")
        
        step8_results = export_results
        reporter.add_section("step_8_export_operations", step8_results)
        reporter.add_summary_stat("Files Exported", len(export_results["exports_performed"]))
        
        # Final summary statistics for reporter
        reporter.add_summary_stat("Analysis Steps Completed", 8)
        reporter.add_summary_stat("Total Analysis Time", str(datetime.now() - reporter.start_time))
        
        logger.info("Pre-chunking EDA completed successfully!")
        
        # Generate and display reports based on output format
        if output_format in ['console', 'all']:
            print(reporter.generate_console_report())
        
        if save_reports and output_format in ['markdown', 'json', 'all']:
            md_file, json_file = reporter.save_reports(reports_dir)
            logger.info(f"Reports saved: {md_file}, {json_file}")
            reporter.add_generated_file(md_file, "Markdown analysis report")
            reporter.add_generated_file(json_file, "JSON analysis summary")
        
        return reporter
        
    except Exception as e:
        logger.error(f"Error during pre-chunking EDA: {str(e)}")
        reporter.add_section("error", {
            "error_message": str(e),
            "error_type": type(e).__name__,
            "timestamp": datetime.now().isoformat()
        })
        raise


def main():
    """Command-line interface for pre-chunking EDA."""
    parser = argparse.ArgumentParser(
        description="Run comprehensive pre-chunking EDA with report generation"
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
        "--show-plots", 
        action='store_true',
        help="Display plots during analysis"
    )
    
    parser.add_argument(
        "--reports-dir", 
        default=None,
        help="Directory to save reports (default: reports/ in project root)"
    )
    
    args = parser.parse_args()
    
    try:
        reporter = run_prechunking_eda(
            data_dir=args.data_dir,
            output_format=args.output_format,
            save_reports=not args.no_save,
            show_plots=args.show_plots,
            reports_dir=args.reports_dir
        )
        
        print(f"\n✅ Analysis completed successfully!")
        print(f"📊 Total files generated: {len(reporter.files_generated)}")
        
        if reporter.files_generated:
            print("\n📁 Generated files:")
            for file_info in reporter.files_generated:
                print(f"  • {file_info['description']}: {file_info['filepath']}")
        
    except Exception as e:
        print(f"❌ Analysis failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()

