# Enhanced Full Document Parsing Script Guide

## Overview

The enhanced `run_full_doc_parsing.py` script provides comprehensive document parsing and section extraction for the MDC Challenge 2025 dataset with structured report generation. It wraps the existing parsing logic from `src/run_full_doc_parsing.py` with enhanced reporting capabilities while preserving all existing outputs for downstream compatibility.

## Features

### 🔍 **Comprehensive Document Processing**
- **Document Inventory Loading**: Load and validate document inventory from Step 4 conversion
- **Multi-format Parsing**: Support for TEI (from Grobid) and JATS (existing) XML formats
- **Section Extraction**: Intelligent section type detection and content extraction
- **Validation Pipeline**: Comprehensive validation of parsed documents and sections
- **Output Generation**: Preserve all existing outputs (pickle, CSV, JSON) for downstream steps
- **Performance Monitoring**: Track parsing times, memory usage, and success rates

### 📊 **Enhanced Reporting**
- **Multiple Output Formats**: Console, Markdown, JSON, or all formats simultaneously
- **Performance Metrics**: Processing times, memory usage, parsing rates, and success statistics
- **Format Analysis**: TEI vs JATS distribution, format-specific success rates
- **Section Analytics**: Section type distribution, content quality metrics
- **Error Categorization**: Detailed error analysis with resolution guidance
- **Validation Tracking**: Real-time validation success rates and failure patterns

### 🛠 **Downstream Compatibility**
- **Zero Breaking Changes**: All existing outputs preserved exactly as before
- **Pydantic v2 Integration**: Uses latest model validation throughout processing
- **File Structure Preservation**: Maintains `Data/train/parsed/` directory structure
- **Pickle Compatibility**: Ensures `parsed_documents.pkl` format compatibility
- **CSV Format Preservation**: Maintains exact `parsed_documents_summary.csv` format

## Installation & Setup

### Prerequisites
```bash
# Ensure you have the required dependencies
pip install pandas numpy pydantic psutil lxml tqdm
```

### Project Structure
```
MDC-Challenge-2025/
├── Data/
│   ├── document_inventory.csv              # Input from Step 4
│   └── train/
│       ├── XML/                            # Input XML files
│       └── parsed/                         # Generated parsing outputs
│           ├── parsed_documents.pkl        # Main parsed corpus
│           ├── parsed_documents_summary.csv # Summary statistics
│           └── validation_stats.json       # Validation metrics
├── reports/                                # Analysis reports (NEW)
│   ├── full_doc_parsing_report_*.md        # Generated reports
│   └── full_doc_parsing_summary_*.json     # Generated summaries
├── scripts/
│   ├── run_full_doc_parsing.py             # Enhanced script
│   └── run_pdf_to_xml_conversion.py        # Previous step
├── src/
│   ├── run_full_doc_parsing.py             # Core parsing logic
│   ├── document_parser.py                  # Document parsing functions
│   ├── models.py                           # Pydantic models
│   └── helpers.py                          # Utility functions
└── docs/
    └── full_doc_parsing_script_guide.md    # This guide
```

## Usage

### Command Line Interface

#### Basic Usage
```bash
# Run with default settings (reports to reports/, parsed data to Data/train/parsed/)
python scripts/run_full_doc_parsing.py

# Custom data directory
python scripts/run_full_doc_parsing.py --data-dir /path/to/data

# Console output only (no file saving)
python scripts/run_full_doc_parsing.py --output-format console --no-save

# Generate only JSON summary
python scripts/run_full_doc_parsing.py --output-format json

# Custom reports directory
python scripts/run_full_doc_parsing.py --reports-dir /path/to/custom/reports
```

#### Command Line Options
- `--data-dir`: Path to data directory (default: "Data")
- `--output-format`: Output format - 'console', 'markdown', 'json', or 'all' (default: 'all')
- `--no-save`: Don't save reports to files
- `--reports-dir`: Directory to save reports (default: "reports/" in project root)

### Programmatic Usage

```python
from scripts.run_full_doc_parsing import run_full_document_parsing

# Basic usage (reports go to reports/, parsed data to Data/train/parsed/)
reporter = run_full_document_parsing(
    data_dir="Data",
    output_format="all",
    save_reports=True
)

# Custom reports directory
reporter = run_full_document_parsing(
    data_dir="Data",
    reports_dir="custom_reports",
    output_format="all"
)

# Access specific statistics
total_parsed = reporter.summary_stats.get('Successfully Parsed Documents')
success_rate = reporter.summary_stats.get('Parsing Success Rate')

# Get detailed section data
parsing_results = reporter.sections['step_2_document_processing']['content']
validation_results = reporter.sections['step_3_corpus_validation']['content']

# Generate custom reports
markdown_report = reporter.generate_markdown_report()
json_summary = reporter.generate_json_summary()
```

## File Organization

### Generated Files by Location

#### **Data Directory** (Required Outputs)
These files are essential for downstream processing and remain in the Data directory:

- **`Data/train/parsed/parsed_documents.pkl`**: Complete parsed document corpus
  - Used by chunking pipeline (Step 6)
  - Contains Document objects with sections and metadata
  - Preserved in exact format for compatibility

- **`Data/train/parsed/parsed_documents_summary.csv`**: Document parsing summary
  - Used for quality analysis and reporting
  - Contains validation statistics and document metadata
  - Exact format preserved for downstream tools

- **`Data/train/parsed/validation_stats.json`**: Validation statistics
  - Used for pipeline quality monitoring
  - Contains aggregate validation metrics
  - Structured for programmatic access

#### **Reports Directory** (Analysis Reports)
These files are for documentation and reporting, saved to `reports/`:

- **`full_doc_parsing_report_YYYYMMDD_HHMMSS.md`**: Comprehensive markdown report
  - For documentation and team sharing
  - Includes performance metrics and quality analysis

- **`full_doc_parsing_summary_YYYYMMDD_HHMMSS.json`**: Structured data summary
  - For programmatic access and automation
  - Contains all metrics and detailed analysis results

## Processing Steps

The script follows a 5-step workflow that wraps the existing parsing logic:

### Step 1: Document Inventory Loading
- **Purpose**: Load and validate document inventory from Step 4
- **Inputs**: `Data/document_inventory.csv`
- **Processing**: Validates XML file paths and format types
- **Outputs**: Inventory statistics, format distribution analysis
- **Key Metrics**: Total documents, TEI vs JATS distribution, file availability

### Step 2: Document Processing
- **Purpose**: Parse all XML documents using existing parsing logic
- **Processing**: 
  - Routes documents to appropriate parsers (TEI/JATS)
  - Extracts sections with intelligent type detection
  - Validates Document objects using Pydantic models
  - Tracks performance metrics and error handling
- **Outputs**: Parsed Document objects with validation metadata
- **Performance Tracking**: Individual parsing times, memory usage, success rates

### Step 3: Corpus Validation
- **Purpose**: Validate parsed corpus according to checklist criteria
- **Processing**: Uses existing validation logic with enhanced reporting
- **Quality Checks**:
  - Section presence validation (methods, results, data availability)
  - Content sufficiency checks (>1000 characters clean text)
  - Format-specific quality metrics
- **Outputs**: Comprehensive validation statistics and quality metrics

### Step 4: Corpus Storage
- **Purpose**: Save parsed corpus in required formats for downstream steps
- **Processing**: Uses existing save logic to preserve compatibility
- **Generated Files**:
  - `parsed_documents.pkl`: Complete corpus for chunking
  - `parsed_documents_summary.csv`: Metadata and statistics
  - `validation_stats.json`: Aggregate validation metrics
- **Compatibility**: Ensures exact format preservation for Step 6

### Step 5: Performance Analysis
- **Purpose**: Analyze parsing performance and generate insights
- **Processing**: Calculate comprehensive performance metrics
- **Analysis Areas**:
  - Format-specific success rates
  - Section type distribution and quality
  - Performance benchmarks and recommendations
  - Error pattern analysis and resolution guidance

## Generated Reports

### Executive Summary (Example)
```
• Total Documents in Inventory: 523
• TEI Files Available: 124
• JATS Files Available: 399
• Successfully Parsed Documents: 518
• Failed Documents: 5
• Parsing Success Rate: 99.0%
• Validation Success Rate: 95.2%
• Key Sections Coverage: 87.4%
• Section Type Diversity: 12
• Total Processing Time: 45.30s
• Average Processing Time: 0.09s
```

### Performance Metrics
- **Processing Speed**: Average time per document, total processing time
- **Memory Usage**: Peak memory, average memory consumption during processing
- **Success Rates**: Overall parsing success, format-specific success rates
- **Quality Metrics**: Validation success rates, section extraction quality

### Format Analysis
- **TEI Documents**: Count, success rate, common section types
- **JATS Documents**: Count, success rate, parsing performance
- **Unknown Formats**: Fallback parser usage and success rates

### Section Analytics
- **Section Type Distribution**: Most common section types across corpus
- **Content Quality**: Average section lengths, text quality metrics
- **Validation Patterns**: Common validation failures and patterns

## Report Formats

### 1. Console Output
- **Purpose**: Real-time monitoring during script execution
- **Format**: Structured text with progress updates and final summary
- **Best For**: Interactive use, debugging, immediate feedback

### 2. Markdown Report (reports/)
- **Filename**: `reports/full_doc_parsing_report_YYYYMMDD_HHMMSS.md`
- **Purpose**: Documentation, sharing, integration with documentation systems
- **Content**: Comprehensive analysis with performance metrics, format distribution, and quality insights
- **Best For**: Documentation, team sharing, consolidated reporting

### 3. JSON Summary (reports/)
- **Filename**: `reports/full_doc_parsing_summary_YYYYMMDD_HHMMSS.json`
- **Purpose**: Programmatic access, API integration, automation
- **Format**: Structured JSON with detailed metrics and analysis results
- **Best For**: Automation, data pipeline integration, programmatic analysis

### JSON Structure
```json
{
  "metadata": {
    "generated_at": "2024-01-15T14:30:00",
    "data_directory": "Data",
    "parsing_duration": "0:00:45",
    "parsing_steps_completed": 5
  },
  "summary_statistics": {
    "Successfully Parsed Documents": 518,
    "Parsing Success Rate": "99.0%",
    "Validation Success Rate": "95.2%",
    // ... all key metrics
  },
  "parsing_metrics": {
    "total_processed": 523,
    "successful_parses": 518,
    "failed_parses": 5,
    "tei_successes": 122,
    "jats_successes": 396,
    "processing_times": [0.08, 0.12, 0.09, ...],
    "sections_by_type": {
      "abstract": 487,
      "methods": 412,
      "results": 445,
      // ... section counts
    }
  },
  "performance_metrics": {
    "average_processing_time": 0.087,
    "total_processing_time": 45.30,
    "average_memory_mb": 234.5,
    "peak_memory_mb": 287.2
  },
  "generated_files": [
    {
      "filepath": "Data/train/parsed/parsed_documents.pkl",
      "description": "Parsed documents pickle file",
      "timestamp": "2024-01-15T14:30:45"
    }
    // ... all generated files
  ]
}
```

## Integration with Pipeline

### Prerequisites (Step 4)
- Completed PDF→XML conversion
- Generated `Data/document_inventory.csv`
- XML files available in `Data/train/XML/`

### Next Steps (Step 6)
1. **Chunking Pipeline**: Use `Data/train/parsed/parsed_documents.pkl`
2. **Quality Review**: Use summary CSV for quality analysis
3. **Performance Monitoring**: Use validation stats for pipeline health
4. **Reporting**: Use `reports/` files for consolidated documentation

### Pipeline Integration
```bash
#!/bin/bash
# Example pipeline integration

echo "Starting document parsing (Step 5)..."
python scripts/run_full_doc_parsing.py --data-dir $DATA_DIR --output-format json

# Check parsing success rate
SUCCESS_RATE=$(python -c "
import json
with open('reports/full_doc_parsing_summary_*.json') as f:
    data = json.load(f)
    print(data['parsing_metrics']['successful_parses'] / data['parsing_metrics']['total_processed'])
")

if (( $(echo "$SUCCESS_RATE > 0.95" | bc -l) )); then
    echo "Parsing successful (${SUCCESS_RATE}), proceeding to chunking..."
    python scripts/run_chunking_pipeline.py --input Data/train/parsed/parsed_documents.pkl
else
    echo "Parsing success rate too low (${SUCCESS_RATE}), reviewing errors..."
    # Error analysis and manual review
fi
```

## Quality Assurance

### Validation Criteria
The script validates parsed documents according to MDC Challenge requirements:

1. **Section Presence**: Documents must have key sections (methods AND (results OR data_availability))
2. **Content Sufficiency**: Clean text length must exceed 1000 characters
3. **Format Recognition**: XML format must be successfully detected and parsed
4. **Structure Validation**: Sections must have proper order and hierarchy

### Success Metrics
- **Parsing Success Rate**: Target ≥95% of documents successfully parsed
- **Validation Success Rate**: Target ≥90% of parsed documents pass validation
- **Key Sections Coverage**: Target ≥85% of documents have required sections
- **Processing Performance**: Target <0.1s average processing time per document

### Error Handling
```python
# Common error categories tracked:
- "File not found": XML file missing or inaccessible
- "Parse error": XML structure issues or format problems
- "No sections extracted": Document parsing succeeded but no content found
- "Validation failed": Document parsed but failed quality criteria
- "Format unknown": XML format not recognized (TEI/JATS/unknown)
```

## Performance Optimization

### Typical Performance
- **Runtime**: 30-60 seconds for full dataset (500+ documents)
- **Memory Usage**: <300MB peak for typical document sizes
- **Processing Rate**: 10-15 documents per second
- **Disk Space**: 
  - Parsed data: <50MB in Data/train/parsed/
  - Report files: <2MB in reports/

### Performance Tips
- **Memory Management**: Monitor peak memory usage during large batches
- **Processing Speed**: Use progress tracking to identify slow documents
- **Error Recovery**: Check validation failures for systematic issues
- **Resource Monitoring**: Track CPU and memory usage for optimization

## Troubleshooting

### Common Issues

#### Missing Document Inventory
```bash
# Ensure Step 4 (PDF→XML conversion) completed successfully
ls -la Data/document_inventory.csv
python scripts/run_pdf_to_xml_conversion.py --data-dir Data
```

#### XML Parsing Errors
```bash
# Check XML file integrity
xmllint --noout Data/train/XML/*.xml

# Validate namespace handling
python -c "
from src.xml_format_detector import detect_xml_format
from pathlib import Path
for xml_file in Path('Data/train/XML').glob('*.xml'):
    print(f'{xml_file}: {detect_xml_format(xml_file)}')
"
```

#### Pydantic Validation Errors
```bash
# Check model compatibility
python -c "
from src.models import Document, Section
print(f'Document model fields: {list(Document.model_fields.keys())}')
print(f'Section model fields: {list(Section.model_fields.keys())}')
"
```

#### Low Success Rates
```python
# Analyze parsing failures
reporter = run_full_document_parsing(data_dir="Data")
failures = reporter.parsing_metrics['validation_failures']

# Group by failure reason
from collections import Counter
failure_reasons = Counter(f['reason'] for f in failures)
print("Failure distribution:", failure_reasons)

# Check specific documents
for failure in failures[:5]:  # First 5 failures
    print(f"Document {failure['doi']}: {failure['reason']}")
```

### Debug Mode
```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Run with enhanced error reporting
reporter = run_full_document_parsing(
    data_dir="Data",
    output_format="console"  # Immediate feedback
)

# Check specific parsing steps
for step_name, step_data in reporter.sections.items():
    print(f"{step_name}: {step_data['content']}")
```

## Compatibility & Migration

### Pydantic v2 Migration
The script uses modern Pydantic v2 methods:

```python
# Modern Pydantic usage (July 2025 compatible)
document = Document.model_validate(doc_data)  # Instead of Document(**doc_data)
doc_dict = document.model_dump()              # Instead of document.dict()
json_str = document.model_dump_json()         # Instead of document.json()

# Validation with error handling
try:
    validated_doc = Document.model_validate(doc_data)
except ValidationError as e:
    logger.error(f"Validation failed: {e}")
```

### Backward Compatibility
- **Pickle Format**: Maintains compatibility with existing chunking pipeline
- **CSV Schema**: Preserves exact column structure for downstream tools
- **JSON Format**: Maintains structured validation statistics format
- **Directory Structure**: Uses established `Data/train/parsed/` convention

## Customization

### Adding Custom Validation
```python
# Extend validation logic
def custom_document_validation(doc: Document, validation: Dict[str, Any]) -> Dict[str, Any]:
    # Add custom validation criteria
    validation['has_custom_section'] = any(
        section.section_type == 'custom_type' for section in doc.sections
    )
    return validation

# Integration point in processing loop
entry, validation = create_document_entry(article_id, sections, xml_path, source_type)
validation = custom_document_validation(entry, validation)
```

### Custom Performance Metrics
```python
# Extend DocumentParsingReporter
class CustomDocumentParsingReporter(DocumentParsingReporter):
    def __init__(self, data_dir: str):
        super().__init__(data_dir)
        self.custom_metrics = {
            'custom_section_counts': {},
            'custom_quality_scores': []
        }
    
    def add_custom_metric(self, doc: Document):
        # Custom metric calculation
        custom_score = calculate_custom_quality_score(doc)
        self.custom_metrics['custom_quality_scores'].append(custom_score)
```

### Custom Report Sections
```python
# Add custom analysis step
def add_custom_analysis_step(reporter: DocumentParsingReporter, parsed_documents):
    logger.info("Step 6: Custom document analysis...")
    
    custom_results = {
        "custom_metric": calculate_custom_analysis(parsed_documents),
        "custom_insights": generate_custom_insights(parsed_documents)
    }
    
    reporter.add_section("step_6_custom_analysis", custom_results)
    reporter.add_summary_stat("Custom Quality Score", custom_results["custom_metric"])
```

## Best Practices

### For Development
1. **Test with Subset**: Use small document sets for rapid iteration
2. **Monitor Performance**: Watch processing times and memory usage
3. **Validate Outputs**: Verify parsed documents meet quality criteria
4. **Check Compatibility**: Ensure downstream tools can consume outputs

### For Production
1. **Monitor Success Rates**: Track parsing and validation success rates
2. **Archive Reports**: Save reports for audit trails and debugging
3. **Resource Management**: Monitor memory and processing time limits
4. **Error Recovery**: Implement retry logic for transient failures

### For Pipeline Integration
1. **Success Validation**: Check success rates before proceeding to next step
2. **Output Verification**: Validate all required files are generated
3. **Quality Gates**: Set minimum thresholds for validation success
4. **Performance Monitoring**: Track processing times for capacity planning

## Advanced Usage

### Batch Processing
```python
# Process multiple data directories
data_dirs = ["Data_batch1", "Data_batch2", "Data_batch3"]

for data_dir in data_dirs:
    reporter = run_full_document_parsing(
        data_dir=data_dir,
        reports_dir=f"reports/{data_dir}_parsing",
        output_format="json"
    )
    
    # Aggregate results
    success_rate = reporter.parsing_metrics['successful_parses'] / reporter.parsing_metrics['total_processed']
    print(f"{data_dir}: {success_rate:.1%} success rate")
```

### Quality Analysis
```python
# Detailed quality analysis
def analyze_parsing_quality(reporter: DocumentParsingReporter):
    metrics = reporter.parsing_metrics
    
    # Format-specific success rates
    tei_success_rate = metrics['tei_successes'] / max(1, metrics['tei_successes'] + metrics['failed_parses'])
    jats_success_rate = metrics['jats_successes'] / max(1, metrics['jats_successes'] + metrics['failed_parses'])
    
    # Section quality analysis
    section_diversity = len(metrics['sections_by_type'])
    avg_sections_per_doc = sum(metrics['sections_by_type'].values()) / metrics['successful_parses']
    
    # Performance analysis
    processing_times = metrics['processing_times']
    median_time = sorted(processing_times)[len(processing_times)//2]
    
    return {
        'format_success_rates': {'TEI': tei_success_rate, 'JATS': jats_success_rate},
        'section_quality': {'diversity': section_diversity, 'avg_per_doc': avg_sections_per_doc},
        'performance': {'median_time': median_time, 'total_docs': len(processing_times)}
    }
```

## Conclusion

The enhanced full document parsing script provides a comprehensive, production-ready solution for Step 5 of the MDC Challenge 2025 pipeline. Key benefits include:

- **Complete Compatibility**: All existing outputs preserved for downstream processing
- **Enhanced Monitoring**: Comprehensive performance and quality tracking
- **Modern Standards**: Uses latest Pydantic v2 methods and best practices
- **Robust Reporting**: Multi-format reports for documentation and automation
- **Production Ready**: Error handling, performance optimization, and quality assurance

The script successfully bridges the gap between raw XML documents and the structured parsed corpus needed for chunking, while providing the visibility and quality assurance required for a production ML pipeline.

### Integration Summary
- **Input**: `Data/document_inventory.csv` (from Step 4)
- **Processing**: Parse XML → Extract sections → Validate quality → Generate reports
- **Output**: `Data/train/parsed/` (for Step 6) + `reports/` (for documentation)
- **Quality**: ≥95% parsing success, ≥90% validation success, comprehensive error tracking

This script serves as a crucial component in the preprocessing pipeline, ensuring high-quality document parsing with full visibility into the process and results. 