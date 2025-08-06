# Make Data Count - Finding Data References Challenge 2025

A comprehensive solution for identifying and classifying data citations in scientific literature using advanced NLP techniques. The logical architecture of the implementation can be found [here](logical_arch.svg).

## 🎯 Challenge Overview

This project addresses the **Make Data Count - Finding Data References** Kaggle Challenge, which aims to identify all data citations from scientific literature and classify them by citation type:

- **Primary**: Raw or processed data generated as part of the paper, specifically for the study
- **Secondary**: Raw or processed data derived or reused from existing records or published data

The [challenge](CHALLENGE_OVERVIEW.md) involves processing scientific papers from the Europe PMC open access subset, available in both PDF and XML formats, to extract research data references and classify them accurately.

## 📁 Project Structure

```
MDC-Challenge-2025/
├── api/                               # Microservices APIs
│   ├── parse_doc_api.py              # Document parsing microservice
│   ├── chunk_and_embed_api.py        # Chunking & embedding microservice
│   ├── services/                     # API service modules
│   ├── utils/                        # API utilities
│   └── database/                     # Database configurations
├── src/                              # Core processing modules
│   ├── models.py                     # Pydantic data models
│   ├── document_parser.py            # PDF/XML document parsing
│   ├── semantic_chunking.py          # Text chunking algorithms
│   ├── get_citation_entities.py      # Citation entity extraction
│   ├── get_document_objects.py       # Document object creation
│   ├── retriever.py                  # Document retrieval utilities
│   ├── helpers.py                    # Utility functions and logging
│   ├── patterns.py                   # Text pattern matching
│   ├── pdf_to_xml_conversion.py      # PDF to XML conversion
│   ├── xml_format_detector.py        # XML format detection
│   ├── section_mapping.py            # Document section mapping
│   ├── pipeline_config.py            # Configuration management
│   ├── pipeline_visualization.py     # Progress visualization
│   ├── run_semantic_chunking.py      # Semantic chunking pipeline
│   ├── run_full_doc_parsing.py       # Full document parsing
│   ├── baml_client/                  # BAML client integration
│   └── baml_src/                     # BAML source files
├── Data/                             # Dataset and workflow files
│   ├── train_labels.csv               # Training labels
│   ├── train/                         # Training documents
│   │   ├── PDF/                       # PDF files (524 files)
│   │   └── XML/                       # XML files (400 files, ~75% 
coverage)
│   └── test/                         # Test documents (PDF/XML)
├── configs/                          # Configuration files
│   ├── development_config.json       # Development configuration
│   ├── production_config.json        # Production configuration
│   ├── fast_config.json              # Fast processing configuration
│   ├── chunking.yaml                 # Chunking pipeline config
│   └── chunking_offline.yaml         # Offline chunking config
├── artifacts/                        # Generated artifacts and databases
│   ├── bioregistry/                  # Bioregistry data
│   ├── lookups/                      # Lookup tables
│   ├── bioregistry_data.json         # Bioregistry dataset
│   ├── citation_patterns.yaml        # Citation pattern definitions
│   └── entity_patterns.yaml          # Entity pattern definitions
├── scripts/                          # Standalone execution scripts
├── notebooks/                        # Jupyter notebooks for analysis
├── docs/                            # Documentation
├── tests/                           # Test files
├── reports/                         # Generated analysis reports
├── logs/                            # Application logs
├── models/                          # Trained models and artifacts
├── guides/                          # Additional guides
├── docker-compose.yml               # Docker service orchestration
├── Dockerfile                       # Main service Docker image
├── Dockerfile.api                   # API service Docker image
├── Makefile                         # Docker management automation
├── pyproject.toml                   # Project configuration
├── uv.lock                          # Dependency lock file
└── main.py                          # Main entry point
```

## 🚀 Quick Start

### Prerequisites

- **Docker** and **Docker Compose** installed
- **Python 3.12+** (if running locally)
- **uv** package manager (recommended) or pip
- Sufficient disk space (10GB+ for full dataset)

### Docker-based Setup (Recommended)

The project uses Docker containers for both microservices. Use the included Makefile for easy management:

#### 1. Build and Start Both APIs

```bash
# Build both services and start them
make build-up-no-cache

# Or for development (with existing cache)
make build-up
```

This will:
- Build the document parsing API (port 3000)
- Build the chunking & embedding API (port 8000)
- Start both services in detached mode

#### 2. Check Service Status

```bash
# Check if both services are running
make status

# View logs from both services
make logs

# View logs from specific service
make logs-main  # Document parsing API
make logs-api   # Chunking API
```

#### 3. Test API Health

```bash
# Test both APIs
make test

# Access API documentation
open http://localhost:3000/docs  # Document parsing API
open http://localhost:8000/docs  # Chunking & embedding API
```

#### 4. Stop Services

```bash
# Stop both services
make down

# Or just pause them (without removing containers)
make stop
```

### Available Makefile Commands

The Makefile provides convenient commands for managing Docker services:

| Command | Description |
|---------|-------------|
| `make help` | Show all available commands |
| `make build` | Build both services |
| `make build-no-cache` | Build both services without Docker cache |
| `make up` | Start both services |
| `make down` | Stop and remove both services |
| `make status` | Show running container status |
| `make logs` | Show logs from both services |
| `make logs-main` | Show logs from document parsing API |
| `make logs-api` | Show logs from chunking API |
| `make test` | Test both API health endpoints |
| `make restart` | Restart both services |
| `make clean` | Remove all containers and volumes |
| `make dev-setup` | Complete development setup |
| `make main-shell` | Open shell in document parsing container |
| `make api-shell` | Open shell in chunking API container |

### Local Development Setup

If you prefer to run services locally without Docker:

1. **Clone and install dependencies:**
```bash
git clone https://github.com/your-username/MDC-Challenge-2025.git
cd MDC-Challenge-2025

# Using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

2. **Start the APIs directly:**
```bash
# Terminal 1: Document parsing API
python api/parse_doc_api.py

# Terminal 2: Chunking & embedding API  
python api/chunk_and_embed_api.py
```

## 🔧 API Services

### Document Parsing API (Port 3000)

The document parsing microservice handles PDF document processing:

- **Endpoint**: `http://localhost:3000`
- **Documentation**: `http://localhost:3000/docs`
- **Health Check**: `http://localhost:3000/health`

**Key Endpoints:**
- `GET /parse_doc` - Parse a single document
- `GET /bulk_parse_docs` - Parse multiple documents
- `GET /health` - Service health status

### Chunking & Embedding API (Port 8000)

The chunking & embedding microservice handles text segmentation and embeddings:

- **Endpoint**: `http://localhost:8000`
- **Documentation**: `http://localhost:8000/docs`
- **Health Check**: `http://localhost:8000/health`

**Key Endpoints:**
- `POST /create_chunks` - Create text chunks from input
- `POST /run_semantic_chunking` - Run semantic chunking pipeline
- `POST /chunk/documents` - Process specific documents
- `GET /health` - Service health status with database connectivity

## 🏗️ Architecture

The project follows a microservices architecture with two main APIs:

1. **Document Parsing Service**: Handles PDF/XML processing and document object creation
2. **Chunking & Embedding Service**: Manages text chunking, embeddings, and ChromaDB integration

Both services use:
- **DuckDB** for structured data storage
- **ChromaDB** for vector embeddings
- **FastAPI** for REST API endpoints
- **Pydantic** for data validation
- **Docker** for containerization

## 🔍 Key Features

- **Microservices Architecture**: Separated concerns with dedicated APIs
- **Multi-format Support**: Handles both PDF and XML documents
- **Docker Integration**: Full containerization with Docker Compose
- **Health Monitoring**: Built-in health checks for all services
- **Comprehensive Logging**: Detailed progress tracking and error handling
- **Database Integration**: DuckDB for structured data, ChromaDB for embeddings
- **Flexible Configuration**: Configurable parameters via YAML/JSON files
- **Development Tools**: Makefile automation for easy management

## 🛠️ Development Workflow

### Starting Development

1. **Set up the environment:**
```bash
make dev-setup
```

2. **Monitor services:**
```bash
# Check status
make status

# Follow logs in real-time
make logs
```

3. **Access services:**
- Document parsing API: http://localhost:3000/docs
- Chunking & embedding API: http://localhost:8000/docs

### Testing Changes

1. **Rebuild with changes:**
```bash
make build-up-no-cache
```

2. **Test API endpoints:**
```bash
make test
```

3. **Debug issues:**
```bash
# Access container shell
make main-shell  # or make api-shell

# Check specific service logs
make logs-main   # or make logs-api
```

### Cleanup

```bash
# Stop services and clean up
make clean

# Full cleanup including images
make clean-all
```

## 🤖 Machine Learning Training

The project includes a comprehensive Random Forest training pipeline with enhanced error handling and documentation.

### Training Scripts

Two bash scripts are provided for running the Random Forest training:

#### Quick Training Script
For basic training with default parameters:

```bash
# Simple training with defaults
./quick_train.sh
```

This script:
- Uses `Data/train/train_data.csv` as input
- Outputs models to `artifacts/models/`
- Uses default parameters (seed=42, 100 iterations)
- Provides basic status updates

#### Full Training Script
For advanced training with custom parameters:

```bash
# Basic usage
./run_training.sh

# Custom parameters
./run_training.sh --input my_data.csv --output my_models --seed 123 --n-iter 200

# With balanced Random Forest
./run_training.sh --balanced

# With grouping column for StratifiedGroupKFold
./run_training.sh --group article_id

# Show help
./run_training.sh --help
```

**Available Options:**
- `-i, --input`: Input CSV file (default: Data/train/train_data.csv)
- `-o, --output`: Output directory (default: artifacts/models)
- `-d, --duckdb`: DuckDB database path (default: artifacts/mdc-challenge.db)
- `-t, --target`: Target column name (default: target)
- `-g, --group`: Optional group column for StratifiedGroupKFold
- `-s, --seed`: Random seed (default: 42)
- `-n, --n-iter`: Number of RandomizedSearch iterations (default: 100)
- `-b, --balanced`: Use BalancedRandomForestClassifier from imblearn
- `-h, --help`: Show help message

### Training Features

The training pipeline includes:

- **Comprehensive Error Handling**: Validates data, handles missing files, and provides clear error messages
- **Dependency Checking**: Automatically verifies required Python packages
- **Data Validation**: Checks data dimensions, target column existence, and data quality
- **Progress Monitoring**: Detailed logging throughout the training process
- **Model Artifacts**: Saves trained model, feature importance, cross-validation results, and metadata
- **Database Integration**: Optional DuckDB storage for model metadata
- **Flexible Configuration**: Support for various Random Forest configurations

### Training Output

The training process generates several artifacts in the output directory:

- `rf_model.pkl`: Trained Random Forest model
- `best_params.json`: Optimal hyperparameters found during search
- `feature_names.json`: List of feature names used in training
- `cv_results.csv`: Cross-validation performance metrics
- `feature_importance_permutation.csv`: Feature importance rankings
- `train_log.txt`: Detailed training log

### Requirements

For training, ensure you have:
- Python 3.8+
- Required packages: pandas, scikit-learn, numpy, joblib, scipy
- Optional: imbalanced-learn (for balanced Random Forest)
- Optional: duckdb (for metadata storage)

## 🧪 Testing

The project includes comprehensive testing:

```bash
# Run all tests (when implemented)
python -m pytest tests/

# Test API health
make test

# Test training pipeline
./quick_train.sh

# Manual API testing via documentation
open http://localhost:3000/docs
open http://localhost:8000/docs
```

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

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make changes and test with `make build-up-no-cache`
4. Commit your changes (`git commit -m 'Add amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Make Data Count initiative and DataCite for organizing the challenge
- The Navigation Fund and Chan Zuckerberg Initiative for sponsoring the prizes
- Europe PMC for providing the open access corpus
- The scientific community for their valuable research contributions

---

*Competition sponsored by DataCite International Data Citation Initiative e.V, with prize funds from The Navigation Fund and Chan Zuckerberg Initiative.*