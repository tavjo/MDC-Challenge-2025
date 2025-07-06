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
│   └── pdf_to_xml_guide.md            # PDF to XML conversion guide
├── reports/                           # Generated analysis reports
│   ├── prechunking_eda_report_*.md    # EDA analysis reports
│   └── prechunking_eda_summary_*.json # EDA summary data
├── models/                            # Trained models and artifacts
├── tests/                             # Test files
├── logs/                              # Application logs
├── guides/                            # Additional guides and documentation
├── main.py                            # Main entry point
├── preprocessing.py                   # Preprocessing pipeline coordinator
├── pyproject.toml                     # Project configuration and dependencies
└── README.md                          # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.12+
- UV package manager (recommended) or pip

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

1. **Run Pre-chunking EDA**:
```bash
python scripts/run_prechunking_eda.py --data-dir Data
```

## 🔧 Core Components

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

1. **Data Analysis**: Run EDA scripts to understand dataset characteristics
2. **Document Processing**: Convert PDFs to XML where needed
3. **Text Extraction**: Parse documents and extract relevant sections
4. **Chunking**: Apply semantic chunking for optimal processing
5. **Feature Engineering**: Extract features relevant to citation detection
6. **Model Training**: Train models for citation identification and classification
7. **Evaluation**: Validate model performance using F1-score
8. **Deployment**: Prepare final submission

## 🔍 Key Features

- **Multi-format Support**: Handles both PDF and XML documents
- **Intelligent Preprocessing**: Automated workflow planning and execution
- **Comprehensive Analysis**: Detailed EDA and quality assessment
- **Scalable Processing**: Efficient batch processing capabilities
- **Robust Logging**: Detailed progress tracking and error handling
- **Flexible Configuration**: Configurable parameters for different use cases

## 📚 Documentation

- [Pre-chunking EDA Guide](docs/prechunking_eda_script_guide.md)
- [PDF to XML Conversion Guide](docs/pdf_to_xml_guide.md)
- [Label Document Mapping Notebook](notebooks/label_doc_mapping.ipynb)

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