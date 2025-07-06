# Enhanced Pre-Chunking EDA Script Guide

## Overview

The enhanced `run_prechunking_eda.py` script provides comprehensive exploratory data analysis (EDA) for the MDC Challenge 2025 dataset with structured report generation. It mirrors the analysis performed in the `label_doc_mapping.ipynb` notebook but captures all outputs into structured reports for documentation and workflow integration.

## Features

### 🔍 **Comprehensive Analysis**
- **Label Loading & Validation**: Load and validate training labels with detailed statistics
- **Unique Articles Analysis**: Distinguish between label entries (1,028) and unique articles (523)
- **Document Inventory**: Scan PDF/XML file availability and categorize for conversion workflow
- **Quality Checks**: Duplicate detection, null value analysis, class distribution validation
- **Conversion Planning**: Identify articles needing PDF→XML conversion (124 articles)
- **Export Operations**: Generate workflow files for next processing steps

### 📊 **Report Generation**
- **Multiple Output Formats**: Console, Markdown, JSON, or all formats simultaneously
- **Structured Data Capture**: All analysis outputs captured in organized sections
- **Executive Summary**: Key statistics and findings at a glance
- **File Tracking**: Complete record of generated files and their purposes
- **Timestamped Results**: All analysis sections include timestamps for tracking
- **Organized File Structure**: Workflow files in Data/, reports in reports/

### 🛠 **Workflow Integration**
- **Command-Line Interface**: Easy integration into automation pipelines
- **Programmatic Access**: Use as Python module for custom workflows
- **Error Handling**: Comprehensive error reporting and graceful failure handling
- **File Export**: Automatic generation of workflow files (conversion candidates, inventories)

## Installation & Setup

### Prerequisites
```bash
# Ensure you have the required dependencies
pip install pandas numpy matplotlib pathlib
```

### Project Structure
```
MDC-Challenge-2025/
├── Data/                               # Data and workflow files
│   ├── train_labels.csv
│   ├── conversion_candidates.csv       # Generated workflow file
│   ├── document_inventory.csv          # Generated workflow file
│   ├── problematic_articles.txt        # Generated workflow file
│   └── train/
│       ├── PDF/
│       └── XML/
├── reports/                            # Analysis reports (NEW)
│   ├── prechunking_eda_report_*.md     # Generated reports
│   └── prechunking_eda_summary_*.json  # Generated summaries
├── scripts/
│   ├── run_prechunking_eda.py          # Main script
│   └── demo_prechunking_eda.py         # Demonstration script
├── src/
│   ├── label_mapper.py                 # Core analysis classes
│   └── helpers.py                      # Utility functions
└── docs/
    └── prechunking_eda_script_guide.md # This guide
```

## Usage

### Command Line Interface

#### Basic Usage
```bash
# Run with default settings (reports to reports/, workflow files to Data/)
python scripts/run_prechunking_eda.py

# Custom data directory
python scripts/run_prechunking_eda.py --data-dir /path/to/data

# Console output only (no file saving)
python scripts/run_prechunking_eda.py --output-format console --no-save

# Generate only JSON summary
python scripts/run_prechunking_eda.py --output-format json

# Include plots (for interactive use)
python scripts/run_prechunking_eda.py --show-plots

# Custom reports directory
python scripts/run_prechunking_eda.py --reports-dir /path/to/custom/reports
```

#### Command Line Options
- `--data-dir`: Path to data directory (default: "Data")
- `--output-format`: Output format - 'console', 'markdown', 'json', or 'all' (default: 'all')
- `--no-save`: Don't save reports to files
- `--show-plots`: Display plots during analysis
- `--reports-dir`: Directory to save reports (default: "reports/" in project root)

### Programmatic Usage

```python
from scripts.run_prechunking_eda import run_prechunking_eda

# Basic usage (reports go to reports/, workflow files to Data/)
reporter = run_prechunking_eda(
    data_dir="Data",
    output_format="all",
    save_reports=True,
    show_plots=False
)

# Custom reports directory
reporter = run_prechunking_eda(
    data_dir="Data",
    reports_dir="custom_reports",
    output_format="all"
)

# Access specific statistics
total_articles = reporter.summary_stats.get('Unique Articles')
conversion_needed = reporter.summary_stats.get('Articles Needing Conversion')

# Get detailed section data
inventory_results = reporter.sections['step_3_document_inventory']['content']

# Generate custom reports
markdown_report = reporter.generate_markdown_report()
json_summary = reporter.generate_json_summary()
```

## File Organization

### Generated Files by Location

#### **Data Directory** (Workflow Files)
These files are essential for downstream processing and remain in the Data directory:

- **`conversion_candidates.csv`**: 124 articles needing PDF→XML conversion
  - Used by conversion pipeline
  - Contains article IDs, paths, and priority information

- **`document_inventory.csv`**: Complete document availability inventory
  - Used for file tracking and validation
  - Contains all metadata about file availability

- **`problematic_articles.txt`**: Articles missing both formats (0 articles)
  - Used for error handling and manual review
  - Lists articles that cannot be processed

#### **Reports Directory** (Analysis Reports)
These files are for documentation and reporting, saved to `reports/`:

- **`prechunking_eda_report_YYYYMMDD_HHMMSS.md`**: Comprehensive markdown report
  - For documentation and team sharing
  - Includes all analysis sections and findings

- **`prechunking_eda_summary_YYYYMMDD_HHMMSS.json`**: Structured data summary
  - For programmatic access and automation
  - Contains all metrics and detailed analysis results

## Analysis Steps

The script follows the exact workflow from the notebook, performing 8 comprehensive analysis steps:

### Step 1: Label Loading & Basic Stats
- **Purpose**: Load training labels and create unique articles summary
- **Outputs**: Total entries, unique articles, columns, validation results
- **Files**: Validates `train_labels.csv` structure

### Step 2: Unique Articles Analysis
- **Purpose**: Analyze multi-label distribution and unique article statistics
- **Outputs**: Multi-label counts, label type distribution, comparison metrics
- **Key Insight**: 505 articles have multiple labels (1,028 entries → 523 unique articles)

### Step 3: Document Inventory
- **Purpose**: Scan PDF/XML file availability and calculate coverage
- **Outputs**: File counts, availability percentages, full-text coverage
- **Key Findings**: 
  - 523 PDF files (100% coverage)
  - 399 XML files (76.3% coverage)
  - 523 articles with full-text (100% coverage)

### Step 4: Conversion Workflow Analysis
- **Purpose**: Categorize articles for PDF→XML conversion workflow
- **Outputs**: Conversion candidates, priority analysis, workflow categorization
- **Key Results**:
  - 399 articles ready for processing (both PDF & XML)
  - 124 articles need PDF→XML conversion
  - 0 articles cannot be processed

### Step 5: Quality Checks
- **Purpose**: Validate data quality and check expected distributions
- **Outputs**: Duplicate analysis, null checks, class distribution validation
- **Quality Metrics**: No duplicates, no nulls, class distribution matches expectations

### Step 6: Summary Statistics
- **Purpose**: Generate comprehensive statistics for all aspects
- **Outputs**: Complete metrics dictionary with all key numbers
- **Integration**: Provides structured data for downstream processes

### Step 7: Enhanced Inventory Analysis
- **Purpose**: Detailed analysis of conversion priorities and label types
- **Outputs**: Cross-tabulated analysis, priority matrices, detailed breakdowns
- **Strategic Insights**: Distribution of conversion needs by label type

### Step 8: Export Operations
- **Purpose**: Generate workflow files for next processing steps
- **Generated Files**:
  - `Data/conversion_candidates.csv`: 124 articles needing PDF→XML conversion
  - `Data/problematic_articles.txt`: Articles missing both formats (0 articles)
  - `Data/document_inventory.csv`: Complete inventory with all metadata
  - `reports/prechunking_eda_report_*.md`: Analysis documentation
  - `reports/prechunking_eda_summary_*.json`: Structured data summary

## Generated Reports

### Executive Summary (Example)
```
• Total Label Entries: 1028
• Unique Articles: 523
• PDF Files Available: 523
• XML Files Available: 399
• Full-text Coverage: 100.0%
• Articles Needing Conversion: 124
• Ready for Processing: 399
• Primary Labels: 270
• Secondary Labels: 449
• Missing Labels: 309
• Analysis Steps Completed: 8
```

### Report Formats

#### 1. Console Output
- **Purpose**: Immediate viewing during script execution
- **Format**: Structured text with sections and statistics
- **Best For**: Quick review, debugging, interactive use

#### 2. Markdown Report (reports/)
- **Filename**: `reports/prechunking_eda_report_YYYYMMDD_HHMMSS.md`
- **Purpose**: Documentation, sharing, integration with documentation systems
- **Format**: Well-formatted markdown with headers, lists, and code blocks
- **Best For**: Documentation, GitHub integration, team sharing, consolidated preprocessing reports

#### 3. JSON Summary (reports/)
- **Filename**: `reports/prechunking_eda_summary_YYYYMMDD_HHMMSS.json`
- **Purpose**: Programmatic access, API integration, data processing
- **Format**: Structured JSON with metadata, statistics, and detailed sections
- **Best For**: Automation, API consumption, data pipeline integration

### JSON Structure
```json
{
  "metadata": {
    "generated_at": "2024-01-15T10:30:00",
    "data_directory": "Data",
    "analysis_duration": "0:02:15",
    "analysis_steps_completed": 8
  },
  "summary_statistics": {
    "Total Label Entries": 1028,
    "Unique Articles": 523,
    // ... all key metrics
  },
  "detailed_sections": {
    "step_1_label_loading": { /* detailed results */ },
    "step_2_unique_articles_analysis": { /* detailed results */ },
    // ... all 8 analysis steps
  },
  "generated_files": [
    {
      "filepath": "Data/conversion_candidates.csv",
      "description": "Articles needing PDF→XML conversion",
      "timestamp": "2024-01-15T10:32:00"
    },
    {
      "filepath": "reports/prechunking_eda_report_20240115_103200.md",
      "description": "Markdown analysis report",
      "timestamp": "2024-01-15T10:32:00"
    }
    // ... all generated files
  ]
}
```

## Integration with Workflow

### Next Steps After EDA
1. **PDF→XML Conversion**: Use `Data/conversion_candidates.csv` for selective conversion
2. **Document Processing**: Process the 399 ready articles immediately
3. **Quality Monitoring**: Use generated statistics for progress tracking
4. **Reporting**: Use `reports/` files for consolidated preprocessing documentation

### Automation Integration
```bash
#!/bin/bash
# Example automation script

echo "Starting pre-chunking EDA..."
python scripts/run_prechunking_eda.py --data-dir $DATA_DIR --output-format json

# Check if conversion candidates exist
if [ -f "$DATA_DIR/conversion_candidates.csv" ]; then
    echo "Starting PDF→XML conversion for $(wc -l < $DATA_DIR/conversion_candidates.csv) articles..."
    # Run conversion pipeline
    python scripts/run_doc_conversion.py --input $DATA_DIR/conversion_candidates.csv
fi

# Process reports for consolidated documentation
echo "Generating consolidated preprocessing report..."
python scripts/consolidate_reports.py --reports-dir reports/

echo "EDA and conversion workflow completed!"
```

## Consolidated Preprocessing Reports

The separated reports structure enables easy creation of consolidated preprocessing documentation:

### Using Markdown Reports
```bash
# Combine multiple preprocessing step reports
cat reports/prechunking_eda_report_*.md > reports/consolidated_preprocessing_report.md
cat reports/conversion_report_*.md >> reports/consolidated_preprocessing_report.md
cat reports/chunking_report_*.md >> reports/consolidated_preprocessing_report.md
```

### Using JSON Summaries
```python
import json
import glob

# Combine JSON summaries programmatically
all_reports = []
for json_file in glob.glob("reports/*_summary_*.json"):
    with open(json_file) as f:
        all_reports.append(json.load(f))

# Create consolidated summary
consolidated = {
    "preprocessing_pipeline": {
        "steps": all_reports,
        "total_duration": sum_durations(all_reports),
        "combined_statistics": combine_stats(all_reports)
    }
}
```

## Error Handling

The script includes comprehensive error handling:

### Common Issues
- **Missing Data Directory**: Clear error message with path validation
- **Invalid Label File**: Validation of CSV structure and required columns
- **Missing Reports Directory**: Automatically creates reports/ if it doesn't exist
- **File Permission Issues**: Graceful handling of read/write permissions
- **Import Errors**: Clear messages for missing dependencies

### Error Reporting
```python
# Errors are captured in the reporter
reporter.sections['error'] = {
    'error_message': 'Detailed error description',
    'error_type': 'FileNotFoundError',
    'timestamp': '2024-01-15T10:30:00'
}
```

## Performance Considerations

### Typical Performance
- **Runtime**: 2-5 minutes for full dataset (523 articles)
- **Memory Usage**: < 100MB for typical dataset sizes
- **Disk Space**: 
  - Workflow files: < 5MB in Data/
  - Report files: < 1MB in reports/

### Optimization Tips
- Use `--no-save` for repeated runs during development
- Use `--output-format console` for fastest execution
- Disable plots (`show_plots=False`) for automated runs

## Troubleshooting

### Common Issues

#### Import Errors
```bash
# Ensure you're running from project root
cd /path/to/MDC-Challenge-2025
python scripts/run_prechunking_eda.py
```

#### Missing Dependencies
```bash
# Install required packages
pip install pandas numpy matplotlib
```

#### Permission Issues
```bash
# Ensure write permissions for Data and reports directories
chmod 755 Data/ reports/
```

#### Missing Reports Directory
```bash
# Create reports directory if it doesn't exist (script does this automatically)
mkdir -p reports/
```

### Debug Mode
```python
# For detailed debugging
import logging
logging.basicConfig(level=logging.DEBUG)

reporter = run_prechunking_eda(data_dir="Data")
```

## Best Practices

### For Development
1. Use console output for quick iterations
2. Save reports for important milestones
3. Version control the generated reports for reproducibility
4. Use the reports/ directory for documentation builds

### For Production
1. Always save reports for audit trail
2. Use JSON output for programmatic integration
3. Include in automated testing pipeline
4. Archive reports with timestamps for historical tracking

### For Documentation
1. Use markdown reports from reports/ for consolidated documentation
2. Include reports/ in your documentation build process
3. Share reports with team for transparency
4. Use timestamped files for version tracking

## Customization

### Adding New Analysis Steps
```python
# In run_prechunking_eda.py
# Step 9: Custom Analysis
logger.info("Step 9: Performing custom analysis...")

custom_results = {
    "custom_metric": calculate_custom_metric(),
    "custom_analysis": perform_custom_analysis()
}

reporter.add_section("step_9_custom_analysis", custom_results)
reporter.add_summary_stat("Custom Metric", custom_results["custom_metric"])
```

### Custom Report Formats
```python
# Extend EDAReporter class
class CustomEDAReporter(EDAReporter):
    def generate_html_report(self) -> str:
        # Custom HTML generation logic
        pass
    
    def generate_csv_summary(self) -> str:
        # Custom CSV generation logic
        pass
```

### Custom File Organization
```python
# Custom reports directory
reporter = run_prechunking_eda(
    data_dir="Data",
    reports_dir="custom_reports/preprocessing",
    output_format="all"
)
```

## Conclusion

The enhanced pre-chunking EDA script provides a comprehensive, automated solution for analyzing the MDC Challenge 2025 dataset. The organized file structure ensures:

- **Workflow files** remain in `Data/` for downstream processing
- **Analysis reports** are saved to `reports/` for documentation and consolidation
- **Separation of concerns** between operational files and documentation
- **Easy integration** with consolidated preprocessing reporting

The script serves as a crucial first step in the preprocessing pipeline, providing the data understanding and file categorization needed for effective downstream processing while maintaining clean file organization for documentation purposes. 