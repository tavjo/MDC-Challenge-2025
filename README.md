# Make Data Count - Finding Data References Challenge 2025

A comprehensive solution for identifying and classifying data citations in scientific literature using advanced NLP techniques.

## 🎯 Challenge Overview

This project addresses the **Make Data Count - Finding Data References** Kaggle Challenge, which aims to identify all data citations from scientific literature and classify them by citation type:

- **Primary**: Raw or processed data generated as part of the paper, specifically for the study
- **Secondary**: Raw or processed data derived or reused from existing records or published data

The [challenge](CHALLENGE_OVERVIEW.md) involves processing scientific papers from the Europe PMC open access subset, available in both PDF and XML formats, to extract research data references and classify them accurately.

## 📁 Project Structure

```
MDC-Challenge-2025/
├── Data/                               # Dataset and generated workflow files
│   ├── train_labels.csv               # Training labels
│   ├── train/                         # Training documents
│   │   ├── PDF/                       # PDF files (524 files)
│   │   └── XML/                       # XML files (400 files, ~75% coverage)
│   ├── test/                          # Test documents
│   │   ├── PDF/                       # Test PDF files
│   │   └── XML/                       # Test XML files
│   ├── conversion_candidates.csv      # Generated: Articles needing PDF→XML conversion
│   ├── document_inventory.csv         # Generated: Complete document inventory
│   └── problematic_articles.txt       # Generated: Articles missing both formats
├── src/                               # Core modules
│   ├── label_mapper.py                # Label analysis and document mapping
│   ├── document_parser.py             # Document parsing and extraction
│   ├── semantic_chunking.py           # Text chunking for processing
│   ├── pdf_to_xml_conversion.py       # PDF to XML conversion utilities
│   ├── xml_format_detector.py         # XML format detection and validation
│   ├── section_mapping.py             # Document section mapping
│   ├── pipeline_visualization.py      # Pipeline visualization and progress tracking
│   ├── pipeline_config.py             # Configuration management and validation
│   ├── helpers.py                     # Utility functions and logging
│   └── models.py                      # Data models and schemas
├── scripts/                           # Executable scripts
│   ├── run_prechunking_eda.py         # Pre-chunking exploratory data analysis
│   ├── run_full_doc_parsing.py        # Full document parsing pipeline
│   ├── run_chunking_pipeline.py       # Semantic chunking pipeline
│   ├── run_doc_conversion.py          # PDF→XML conversion script
│   └── demo_prechunking_eda.py        # EDA demonstration script
├── notebooks/                         # Jupyter notebooks
│   └── label_doc_mapping.ipynb        # Label and document mapping analysis
├── docs/                              # Documentation
│   ├── prechunking_eda_script_guide.md # EDA script usage guide
│   ├── pdf_to_xml_guide.md            # PDF to XML conversion guide
│   └── troubleshooting_guide.md       # Troubleshooting and debugging guide
├── reports/                           # Generated analysis reports
│   ├── prechunking_eda_report_*.md    # EDA analysis reports
│   └── prechunking_eda_summary_*.json # EDA summary data
├── models/                            # Trained models and artifacts
├── tests/                             # Test files
│   ├── test_preprocessing_pipeline.py # Unit tests for preprocessing pipeline
│   ├── test_pipeline_integration.py   # Integration tests for full pipeline
│   ├── test_semantic_chunking.py      # Tests for semantic chunking
│   └── test_xml_formats.py            # Tests for XML format detection
├── configs/                           # Configuration files
│   ├── development_config.yaml        # Development environment configuration
│   ├── production_config.yaml         # Production environment configuration
│   └── fast_config.yaml               # Fast processing configuration
├── logs/                              # Application logs
├── guides/                            # Additional guides and documentation
├── main.py                            # Main entry point
├── preprocessing.py                   # Preprocessing pipeline coordinator
├── pyproject.toml                     # Project configuration and dependencies
└── README.md                          # This file
└── preprocessing_pipeline_guide.md # Comprehensive preprocessing pipeline guide
```

## 🚀 Getting Started

### Prerequisites

- Python 3.12+
- UV package manager (recommended) or pip
- Sufficient disk space (10GB+ for full dataset)
- Optional: Rich library for enhanced visualization

### Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/MDC-Challenge-2025.git
cd MDC-Challenge-2025
```

2. Install dependencies:
```bash
# Using UV (recommended)
uv sync

# Or using pip
pip install -e .
```

### Quick Start

#### Option 1: Complete Preprocessing Pipeline (Recommended)

```bash
# Run the complete preprocessing pipeline
python preprocessing.py --data-dir Data --progress

# Run with custom configuration
python preprocessing.py --config configs/development_config.yaml

# Run specific steps only
python preprocessing.py --steps pre_chunking_eda,doc_conversion --verbose
```

#### Option 2: Individual Steps

```bash
# Run individual preprocessing steps
python scripts/run_prechunking_eda.py --data-dir Data
python scripts/run_pdf_to_xml_conversion.py --data-dir Data
python scripts/run_full_doc_parsing.py --data-dir Data
python scripts/run_chunking_pipeline.py --input-path Data/train/parsed/parsed_documents.pkl
```

#### Option 3: Resume After Interruption

```bash
# Resume from last successful step
python preprocessing.py --resume --verbose

# Resume with force retry of failed steps
python preprocessing.py --resume --force
```

## 🔧 Core Components

### Preprocessing Pipeline (`preprocessing.py`)
- **Central orchestration script** for the entire preprocessing workflow
- **Step-by-step execution** with dependency management and validation
- **Flexible execution modes**: run all, specific steps, up to step, from step, resume
- **CLI interface** with comprehensive argument parsing and configuration support
- **Progress tracking** with real-time monitoring and visualization
- **Error handling** with automatic retry, rollback, and recovery mechanisms
- **Consolidated reporting** with JSON and Markdown outputs
- **Resource monitoring** with memory, CPU, and disk usage tracking
- **Configuration management** with YAML/JSON support and templates

### Data Analysis (`src/label_mapper.py`)
- Comprehensive label and document analysis
- PDF↔XML file mapping and availability tracking
- Quality checks and validation
- Conversion workflow planning

### Document Processing (`src/document_parser.py`)
- Multi-format document parsing (PDF, XML)
- Text extraction and preprocessing
- Section identification and mapping
- Metadata extraction

### Semantic Chunking (`src/semantic_chunking.py`)
- Intelligent text segmentation
- Context-aware chunking strategies
- Optimized for citation detection
- Configurable chunk sizes and overlap

### PDF to XML Conversion (`src/pdf_to_xml_conversion.py`)
- Automated PDF to XML conversion
- Format validation and quality checks
- Batch processing capabilities
- Error handling and logging

### Pipeline Visualization (`src/pipeline_visualization.py`)
- **Console progress bars** with Rich library integration
- **Real-time status dashboard** for monitoring pipeline execution
- **Dependency graph generation** with matplotlib and networkx
- **Mermaid diagram export** for documentation and visualization
- **Performance metrics visualization** and resource usage tracking

### Configuration Management (`src/pipeline_config.py`)
- **YAML/JSON configuration files** with validation and schema support
- **Configuration templates** for different environments (development, production, fast)
- **CLI argument to configuration** conversion and persistence
- **Parameter validation** and conflict resolution
- **Configuration inheritance** and override capabilities

## 📊 Analysis and Reporting

The project includes comprehensive analysis tools:

- **Pre-chunking EDA**: Detailed analysis of labels and document availability
- **Document Inventory**: Complete mapping of available files
- **Conversion Workflow**: Automated PDF→XML conversion planning
- **Quality Checks**: Data validation and consistency checks
- **Progress Tracking**: Detailed logging and reporting

## 🏆 Challenge Details

- **Competition**: Make Data Count - Finding Data References
- **Platform**: Kaggle
- **Evaluation Metric**: F1-Score
- **Dataset**: Europe PMC open access subset
- **Timeline**: June 11, 2025 - September 9, 2025
- **Total Prize Pool**: $100,000

## 👥 Contributors

- [**Douaa Mugahid**](https://www.linkedin.com/in/doaa-megahed-185150100/) - Lead
- [**Elliott Risch**](https://www.linkedin.com/in/modusponens/) - Solutions Architect
- [**Taïsha Joseph-Risch**](http://www.linkedin.com/in/taïsha-joseph-0974229b) - ML Specialist (taishajo@mit.edu)

## 📈 Development Workflow

### Recommended Approach (Using Preprocessing Pipeline)

1. **Setup and Configuration**: Configure the preprocessing pipeline for your environment
   ```bash
   python preprocessing.py --template development --save-config my_config.yaml
   ```

2. **Complete Preprocessing**: Run the full preprocessing pipeline with monitoring
   ```bash
   python preprocessing.py --config my_config.yaml --progress --monitor-resources
   ```

3. **Review Results**: Examine consolidated reports and pipeline metrics
   ```bash
   # Reports are generated automatically in reports/
   ls reports/preprocessing_pipeline_*.md
   ```

4. **Iterate and Refine**: Adjust parameters and re-run specific steps as needed
   ```bash
   python preprocessing.py --steps semantic_chunking --chunk-size 300 --force
   ```

5. **Model Development**: Use processed data for model training and evaluation

### Alternative Approach (Step-by-Step)

1. **Data Analysis**: Run EDA scripts to understand dataset characteristics
   ```bash
   python preprocessing.py --steps pre_chunking_eda --show-plots --detailed-analysis
   ```

2. **Document Processing**: Convert PDFs to XML where needed
   ```bash
   python preprocessing.py --steps doc_conversion --verbose
   ```

3. **Text Extraction**: Parse documents and extract relevant sections
   ```bash
   python preprocessing.py --steps document_parsing --extract-sections
   ```

4. **Chunking**: Apply semantic chunking for optimal processing
   ```bash
   python preprocessing.py --steps semantic_chunking --chunk-size 200 --chunk-overlap 20
   ```

5. **Advanced Processing**: Vector embeddings, QC, and artifact export
   ```bash
   python preprocessing.py --from vector_embeddings
   ```

6. **Feature Engineering**: Extract features relevant to citation detection
7. **Model Training**: Train models for citation identification and classification
8. **Evaluation**: Validate model performance using F1-score
9. **Deployment**: Prepare final submission

### Testing and Validation

```bash
# Run unit tests
python -m pytest tests/test_preprocessing_pipeline.py -v

# Run integration tests
python -m pytest tests/test_pipeline_integration.py -v

# Validate pipeline with test data
python preprocessing.py --test-mode --sample-size 10 --validate-only
```

## 🔍 Key Features

- **Multi-format Support**: Handles both PDF and XML documents
- **Intelligent Preprocessing**: Automated workflow planning and execution
- **Comprehensive Analysis**: Detailed EDA and quality assessment
- **Scalable Processing**: Efficient batch processing capabilities
- **Robust Logging**: Detailed progress tracking and error handling
- **Flexible Configuration**: Configurable parameters for different use cases

## 🚀 Preprocessing Pipeline

The project includes a comprehensive preprocessing pipeline that orchestrates all data processing steps with advanced features:

### Key Features

- **Centralized Orchestration**: Single entry point for all preprocessing steps
- **Flexible Execution**: Run all steps, specific steps, or resume from interruption
- **Progress Monitoring**: Real-time progress bars and status dashboard
- **Error Recovery**: Automatic retry, rollback, and recovery mechanisms
- **Resource Monitoring**: Memory, CPU, and disk usage tracking with alerts
- **Configuration Management**: YAML/JSON configuration files with templates
- **Comprehensive Reporting**: Consolidated JSON and Markdown reports
- **Visualization**: Pipeline flow diagrams and dependency graphs

### CLI Command Examples

```bash
# Run complete pipeline with progress monitoring
python preprocessing.py --data-dir Data --progress --verbose

# Run specific steps only
python preprocessing.py --steps pre_chunking_eda,doc_conversion,document_parsing

# Run up to semantic chunking
python preprocessing.py --up-to semantic_chunking --chunk-size 200

# Resume after interruption
python preprocessing.py --resume --force

# Use configuration file
python preprocessing.py --config configs/production_config.yaml

# Generate pipeline diagrams
python preprocessing.py --generate-diagrams --dashboard

# Monitor resources with custom thresholds
python preprocessing.py --monitor-resources --memory-threshold 80 --cpu-threshold 90
```

### Configuration Templates

```bash
# Development environment (debug logging, visualization enabled)
python preprocessing.py --template development --save-config dev_config.yaml

# Production environment (optimized for performance)
python preprocessing.py --template production --save-config prod_config.yaml

# Fast processing (reduced quality for testing)
python preprocessing.py --template fast --save-config fast_config.yaml
```

### Error Handling and Recovery

```bash
# Automatic retry with custom parameters
python preprocessing.py --retry-count 5 --retry-delay 10

# Rollback failed steps and restart
python preprocessing.py --rollback-failed --force

# Skip problematic files and continue
python preprocessing.py --skip-corrupted --skip-missing --continue-on-error
```

## 📚 Documentation

- [**Preprocessing Pipeline Guide**](docs/preprocessing_pipeline_guide.md) - Comprehensive guide for the preprocessing pipeline
- [**Troubleshooting Guide**](docs/troubleshooting_guide.md) - Common issues and solutions
- [Pre-chunking EDA Guide](docs/prechunking_eda_script_guide.md) - EDA script usage guide
- [PDF to XML Conversion Guide](docs/pdf_to_xml_guide.md) - PDF conversion documentation
- [Label Document Mapping Notebook](notebooks/label_doc_mapping.ipynb) - Analysis notebook

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Make Data Count initiative and DataCite for organizing the challenge
- The Navigation Fund and Chan Zuckerberg Initiative for sponsoring the prizes
- Europe PMC for providing the open access corpus
- The scientific community for their valuable research contributions

---

*Competition sponsored by DataCite International Data Citation Initiative e.V, with prize funds from The Navigation Fund and Chan Zuckerberg Initiative.*