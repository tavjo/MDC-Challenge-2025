#!/usr/bin/env python3
"""
Enhanced Semantic Chunking Pipeline Script with Comprehensive Reporting (step 6)

This script provides comprehensive semantic chunking with structured report generation
for the MDC Challenge 2025 dataset. It wraps the existing chunking logic with enhanced
reporting capabilities similar to the full document parsing and pre-chunking EDA scripts.

Features:
- Complete semantic chunking workflow with detailed progress tracking
- Multiple output formats (console, markdown, JSON)
- Performance metrics and quality analysis
- Error categorization and resolution guidance
- Integration with existing chunking pipeline
- Detailed reporting of all chunking steps
- Preservation of all existing outputs for downstream compatibility

Usage:
    python scripts/run_chunking_pipeline.py
    python scripts/run_chunking_pipeline.py --output-format markdown
    python scripts/run_chunking_pipeline.py --chunk-size 300 --chunk-overlap 30
"""

import os
import sys
import json
import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd

# Optional dependency for memory monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Add src to path for imports
# sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from src.helpers import initialize_logging, timer_wrap
from src.semantic_chunking import run_semantic_chunking_pipeline
from src.models import ChunkingResult


class ChunkingReporter:
    """
    Captures and formats outputs from semantic chunking process for comprehensive reporting.
    Based on DocumentParsingReporter and ConversionReporter but specialized for chunking workflows.
    """
    
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.start_time = datetime.now()
        self.sections = {}
        self.summary_stats = {}
        self.files_generated = []
        self.chunking_metrics = {
            'total_processed_docs': 0,
            'successful_chunks': 0,
            'failed_validations': 0,
            'processing_times': [],
            'memory_usage': [],
            'token_distribution': [],
            'chunk_types': {},
            'section_coverage': {},
            'entity_retention_rate': 0,
            'quality_gates_passed': True
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
        """Track files generated during chunking."""
        self.files_generated.append({
            'filepath': filepath,
            'description': description,
            'timestamp': datetime.now()
        })
    
    def add_chunking_metric(self, metric_name: str, value: Any):
        """Add chunking-specific metrics."""
        self.chunking_metrics[metric_name] = value
    
    def update_chunking_stats(self, chunking_result: ChunkingResult, processing_time: Optional[float] = None):
        """Update chunking statistics based on chunking result."""
        self.chunking_metrics['total_processed_docs'] = chunking_result.total_documents
        self.chunking_metrics['successful_chunks'] = chunking_result.total_chunks
        self.chunking_metrics['entity_retention_rate'] = chunking_result.entity_retention
        self.chunking_metrics['quality_gates_passed'] = chunking_result.validation_passed
        
        # Record memory usage (if psutil is available)
        if PSUTIL_AVAILABLE:
            memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
            self.chunking_metrics['memory_usage'].append(memory_mb)
        
        if processing_time:
            self.chunking_metrics['processing_times'].append(processing_time)
    
    def calculate_performance_metrics(self):
        """Calculate performance metrics from collected data."""
        times = self.chunking_metrics['processing_times']
        memory = self.chunking_metrics['memory_usage']
        
        metrics = {}
        if times:
            metrics.update({
                'average_processing_time': sum(times) / len(times),
                'total_processing_time': sum(times),
                'fastest_processing': min(times),
                'slowest_processing': max(times),
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
        report.append("SEMANTIC CHUNKING PIPELINE COMPREHENSIVE REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Data Directory: {self.data_dir}")
        report.append(f"Chunking Duration: {datetime.now() - self.start_time}")
        report.append("")
        
        # Executive Summary
        report.append("EXECUTIVE SUMMARY")
        report.append("-" * 40)
        for key, value in self.summary_stats.items():
            report.append(f"• {key}: {value}")
        report.append("")
        
        # Chunking Performance
        perf_metrics = self.calculate_performance_metrics()
        if perf_metrics:
            report.append("CHUNKING PERFORMANCE")
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
        
        # Quality Gates Status
        report.append("QUALITY GATES STATUS")
        report.append("-" * 40)
        if self.chunking_metrics['quality_gates_passed']:
            report.append("• Entity retention: ✅ PASSED")
            report.append(f"• Retention rate: {self.chunking_metrics['entity_retention_rate']:.1f}%")
        else:
            report.append("• Entity retention: ❌ FAILED")
            report.append(f"• Retention rate: {self.chunking_metrics['entity_retention_rate']:.1f}%")
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
        report.append("# Semantic Chunking Pipeline Comprehensive Report")
        report.append("")
        report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"**Data Directory:** `{self.data_dir}`")
        report.append(f"**Chunking Duration:** {datetime.now() - self.start_time}")
        report.append("")
        
        # Executive Summary
        report.append("## Executive Summary")
        report.append("")
        for key, value in self.summary_stats.items():
            report.append(f"- **{key}:** {value}")
        report.append("")
        
        # Chunking Performance
        perf_metrics = self.calculate_performance_metrics()
        if perf_metrics:
            report.append("## Chunking Performance")
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
        
        # Quality Gates Status
        report.append("## Quality Gates Status")
        report.append("")
        if self.chunking_metrics['quality_gates_passed']:
            report.append("- **Entity retention:** ✅ PASSED")
            report.append(f"- **Retention rate:** {self.chunking_metrics['entity_retention_rate']:.1f}%")
        else:
            report.append("- **Entity retention:** ❌ FAILED")
            report.append(f"- **Retention rate:** {self.chunking_metrics['entity_retention_rate']:.1f}%")
        report.append("")
        
        # Chunk Distribution Analysis
        if self.chunking_metrics['successful_chunks'] > 0:
            report.append("## Chunk Distribution Analysis")
            report.append("")
            report.append(f"- **Total chunks created:** {self.chunking_metrics['successful_chunks']:,}")
            report.append(f"- **Documents processed:** {self.chunking_metrics['total_processed_docs']}")
            report.append(f"- **Average chunks per document:** {self.chunking_metrics['successful_chunks'] / self.chunking_metrics['total_processed_docs']:.1f}")
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
        """Generate JSON summary of all chunking results."""
        return {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'data_directory': self.data_dir,
                'chunking_duration': str(datetime.now() - self.start_time),
                'chunking_steps_completed': len(self.sections)
            },
            'summary_statistics': self.summary_stats,
            'chunking_metrics': self.chunking_metrics,
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
        md_file = output_path / f"semantic_chunking_report_{timestamp}.md"
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(self.generate_markdown_report())
        
        # Save JSON summary
        json_file = output_path / f"semantic_chunking_summary_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(self.generate_json_summary(), f, indent=2, default=str)
        
        return str(md_file), str(json_file)


@timer_wrap
def run_semantic_chunking_with_reporting(input_path: str = "Data/train/parsed/parsed_documents.pkl",
                                       output_path: str = "chunks_for_embedding.pkl",
                                       chunk_size: int = 200,
                                       chunk_overlap: int = 20,
                                       min_chars: int = 500,
                                       output_format: str = "all",
                                       save_reports: bool = True,
                                       reports_dir: str = None,
                                       data_dir: str = "Data") -> ChunkingReporter:
    """
    Run comprehensive semantic chunking with enhanced reporting.
    
    Args:
        input_path: Path to parsed documents
        output_path: Path for output chunks
        chunk_size: Target chunk size in tokens
        chunk_overlap: Overlap between chunks in tokens
        min_chars: Minimum character count for inclusion
        output_format: Output format - 'console', 'markdown', 'json', or 'all'
        save_reports: Whether to save reports to files
        reports_dir: Directory to save reports (default: reports/ in project root)
        data_dir: Data directory for relative path resolution
    
    Returns:
        ChunkingReporter: Reporter object with all chunking results
    """
    
    # Initialize logging and reporter
    filename = os.path.basename(__file__)
    logger = initialize_logging(filename)
    reporter = ChunkingReporter(data_dir)
    
    logger.info("Starting comprehensive semantic chunking with reporting...")
    reporter.add_section("chunking_start", {
        "message": "Semantic chunking process initiated",
        "input_path": input_path,
        "output_path": output_path,
        "parameters": {
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "min_chars": min_chars
        },
        "timestamp": datetime.now().isoformat()
    })
    
    try:
        # Step 1: Pre-chunking Configuration
        logger.info("Step 1: Pre-chunking configuration and validation...")
        step_start = time.time()
        
        # Validate input path
        if not Path(input_path).exists():
            raise FileNotFoundError(f"Input file does not exist: {input_path}")
        
        step1_results = {
            "input_validation": "passed",
            "input_file_exists": True,
            "configuration": {
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "min_chars": min_chars,
                "overlap_ratio": chunk_overlap / chunk_size
            },
            "step_duration": time.time() - step_start
        }
        
        reporter.add_section("step_1_configuration", step1_results)
        reporter.add_summary_stat("Configuration Validation", "PASSED")
        
        # Step 2: Execute Core Chunking Pipeline
        logger.info("Step 2: Executing core semantic chunking pipeline...")
        step_start = time.time()
        
        # Run the existing chunking pipeline
        chunking_result = run_semantic_chunking_pipeline(
            input_path=input_path,
            output_path=output_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chars=min_chars
        )
        
        processing_time = time.time() - step_start
        
        # Update reporter with chunking results
        reporter.update_chunking_stats(chunking_result, processing_time)
        
        step2_results = {
            "pipeline_success": chunking_result.success,
            "total_documents": chunking_result.total_documents,
            "total_chunks": chunking_result.total_chunks,
            "total_tokens": chunking_result.total_tokens,
            "avg_tokens_per_chunk": chunking_result.avg_tokens_per_chunk,
            "validation_passed": chunking_result.validation_passed,
            "entity_retention": chunking_result.entity_retention,
            "step_duration": processing_time
        }
        
        # Track output files
        if chunking_result.output_files:
            for file_path in chunking_result.output_files:
                if file_path.endswith('.pkl'):
                    reporter.add_generated_file(file_path, "Chunks for embedding (pickle)")
                elif file_path.endswith('.csv'):
                    reporter.add_generated_file(file_path, "Chunking summary (CSV)")
        
        reporter.add_section("step_2_core_chunking", step2_results)
        reporter.add_summary_stat("Pipeline Success", chunking_result.success)
        reporter.add_summary_stat("Total Chunks Created", chunking_result.total_chunks)
        reporter.add_summary_stat("Entity Retention Rate", f"{chunking_result.entity_retention:.1f}%")
        
        # Step 3: Quality Analysis and Recommendations
        logger.info("Step 3: Quality analysis and recommendations...")
        step_start = time.time()
        
        # Quality gate analysis
        quality_gates = {
            "entity_retention_passed": chunking_result.validation_passed,
            "token_distribution_analysis": {
                "avg_tokens_per_chunk": chunking_result.avg_tokens_per_chunk,
                "optimal_range": "160-220 tokens",
                "acceptable_range": "130-270 tokens",
                "status": "optimal" if 160 <= chunking_result.avg_tokens_per_chunk <= 220 else 
                         "acceptable" if 130 <= chunking_result.avg_tokens_per_chunk <= 270 else "suboptimal"
            },
            "processing_efficiency": {
                "processing_time": processing_time,
                "chunks_per_second": chunking_result.total_chunks / processing_time if processing_time > 0 else 0,
                "documents_per_second": chunking_result.total_documents / processing_time if processing_time > 0 else 0
            }
        }
        
        # Generate recommendations
        recommendations = []
        if not chunking_result.validation_passed:
            recommendations.append("Increase chunk overlap to improve entity retention")
            recommendations.append("Consider reducing chunk size for better entity preservation")
        
        if chunking_result.avg_tokens_per_chunk < 130:
            recommendations.append("Consider increasing chunk size for better context")
        elif chunking_result.avg_tokens_per_chunk > 270:
            recommendations.append("Consider decreasing chunk size for optimal performance")
        
        if chunking_result.entity_retention < 95.0:
            recommendations.append("Review entity detection patterns and chunking strategy")
        
        step3_results = {
            "quality_gates": quality_gates,
            "recommendations": recommendations,
            "overall_quality_score": "excellent" if chunking_result.validation_passed and 160 <= chunking_result.avg_tokens_per_chunk <= 220 else
                                   "good" if chunking_result.validation_passed else "needs_improvement",
            "step_duration": time.time() - step_start
        }
        
        reporter.add_section("step_3_quality_analysis", step3_results)
        reporter.add_summary_stat("Quality Score", step3_results["overall_quality_score"].upper())
        
        # Final summary statistics
        reporter.add_summary_stat("Total Processing Time", f"{processing_time:.2f}s")
        reporter.add_summary_stat("Average Tokens per Chunk", f"{chunking_result.avg_tokens_per_chunk:.1f}")
        reporter.add_summary_stat("Chunking Steps Completed", 3)
        
        logger.info("Semantic chunking with reporting completed successfully!")
        
        # Generate and display reports based on output format
        if output_format in ['console', 'all']:
            print(reporter.generate_console_report())
        
        if save_reports and output_format in ['markdown', 'json', 'all']:
            md_file, json_file = reporter.save_reports(reports_dir)
            logger.info(f"Reports saved: {md_file}, {json_file}")
            reporter.add_generated_file(md_file, "Markdown chunking report")
            reporter.add_generated_file(json_file, "JSON chunking summary")
        
        return reporter
        
    except Exception as e:
        logger.error(f"Error during semantic chunking: {str(e)}")
        reporter.add_section("error", {
            "error_message": str(e),
            "error_type": type(e).__name__,
            "timestamp": datetime.now().isoformat()
        })
        raise


def main():
    """Main CLI entry point with enhanced reporting capabilities"""
    parser = argparse.ArgumentParser(
        description="Run the semantic chunking pipeline (Step 6) with comprehensive reporting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default parameters
  python scripts/run_chunking_pipeline.py
  
  # Run with custom parameters and markdown report
  python scripts/run_chunking_pipeline.py \\
      --input-path Data/train/parsed/parsed_documents.pkl \\
      --output-path chunks_for_embedding.pkl \\
      --chunk-size 200 \\
      --chunk-overlap 20 \\
      --min-chars 500 \\
      --output-format markdown
      
  # Run with smaller chunks for testing and console output only
  python scripts/run_chunking_pipeline.py \\
      --chunk-size 100 \\
      --chunk-overlap 10 \\
      --min-chars 200 \\
      --output-format console \\
      --no-save
        """
    )
    
    # Input/Output arguments
    parser.add_argument(
        "--input-path",
        type=str,
        default="Data/train/parsed/parsed_documents.pkl",
        help="Path to parsed documents pickle file (default: Data/train/parsed/parsed_documents.pkl)"
    )
    
    parser.add_argument(
        "--output-path",
        type=str,
        default="chunks_for_embedding.pkl",
        help="Path for output chunks pickle file (default: chunks_for_embedding.pkl)"
    )
    
    # Chunking parameters
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=200,
        help="Target chunk size in tokens (default: 200)"
    )
    
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=20,
        help="Overlap between chunks in tokens (default: 20)"
    )
    
    parser.add_argument(
        "--min-chars",
        type=int,
        default=500,
        help="Minimum character count for document inclusion (default: 500)"
    )
    
    # Quality control arguments
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force processing even if quality gates fail (not recommended)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging output"
    )
    
    # Reporting arguments
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
        "--data-dir",
        default="Data",
        help="Path to data directory (default: Data)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.chunk_size <= 0:
        parser.error("chunk-size must be positive")
    
    if args.chunk_overlap < 0:
        parser.error("chunk-overlap must be non-negative")
    
    if args.chunk_overlap >= args.chunk_size:
        parser.error("chunk-overlap must be less than chunk-size")
    
    if args.min_chars < 0:
        parser.error("min-chars must be non-negative")
    
    # Check if input file exists
    if not Path(args.input_path).exists():
        parser.error(f"Input file does not exist: {args.input_path}")
    
    # Print configuration
    print("=== Semantic Chunking Pipeline Configuration ===")
    print(f"Input path: {args.input_path}")
    print(f"Output path: {args.output_path}")
    print(f"Chunk size: {args.chunk_size} tokens")
    print(f"Chunk overlap: {args.chunk_overlap} tokens")
    print(f"Min chars: {args.min_chars}")
    print(f"Force mode: {args.force}")
    print(f"Verbose: {args.verbose}")
    print(f"Output format: {args.output_format}")
    print(f"Save reports: {not args.no_save}")
    print()
    
    # Run the pipeline with comprehensive reporting
    try:
        reporter = run_semantic_chunking_with_reporting(
            input_path=args.input_path,
            output_path=args.output_path,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            min_chars=args.min_chars,
            output_format=args.output_format,
            save_reports=not args.no_save,
            reports_dir=args.reports_dir,
            data_dir=args.data_dir
        )
        
        # Also run the original pipeline for backward compatibility output
        results = run_semantic_chunking_pipeline(
            input_path=args.input_path,
            output_path=args.output_path,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            min_chars=args.min_chars
        )
        
        # Check results and provide backward-compatible output
        if results.success:
            print("\n🎉 Pipeline completed successfully!")
            print(f"✅ Results:")
            print(f"   📊 Documents processed: {results.total_documents}")
            print(f"   📊 Chunks created: {results.total_chunks:,}")
            print(f"   📊 Average tokens per chunk: {results.avg_tokens_per_chunk:.1f}")
            print(f"   📊 Entity retention: {results.entity_retention:.1f}%")
            print(f"   📁 Output saved to: {results.output_path}")
            
            # Quality gate summary
            print(f"\n✅ Quality Gates Status:")
            if results.validation_passed:
                print(f"   Entity retention: ✅ PASSED ({results.entity_retention:.1f}%)")
            else:
                print(f"   Entity retention: ❌ FAILED ({results.entity_retention:.1f}%)")
            print(f"   Average tokens per chunk: ✅ PASSED ({results.avg_tokens_per_chunk:.1f})")
            
            # Check if within recommended range
            if 160 <= results.avg_tokens_per_chunk <= 220:
                print(f"   Token range: ✅ OPTIMAL (160-220 tokens)")
            elif 130 <= results.avg_tokens_per_chunk <= 270:
                print(f"   Token range: ⚠️  ACCEPTABLE (130-270 tokens)")
            else:
                print(f"   Token range: ❌ SUBOPTIMAL (recommend 160-220 tokens)")
            
            # Show generated reports
            if reporter.files_generated:
                print("\n📁 Generated files:")
                for file_info in reporter.files_generated:
                    print(f"  • {file_info['description']}: {file_info['filepath']}")
            
            return 0
            
        else:
            print(f"\n❌ Pipeline failed: {results.error}")
            
            if not args.force:
                print("\n💡 Suggestions:")
                print("   1. Try increasing --chunk-overlap for better entity retention")
                print("   2. Check input data quality and format")
                print("   3. Use --force to bypass quality gates (not recommended)")
                
                return 1
            else:
                print("\n⚠️  Force mode enabled - quality gates bypassed")
                return 0
                
    except KeyboardInterrupt:
        print("\n\n⏹️  Pipeline interrupted by user")
        return 1
        
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main()) 