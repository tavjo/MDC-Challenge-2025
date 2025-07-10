# Preprocessing Pipeline Guide

## Overview

The preprocessing pipeline is a comprehensive system for processing scientific documents in the Make Data Count challenge. It provides step-by-step execution with dependency management, flexible execution modes, CLI interface, and consolidated reporting.

## Table of Contents

1. [Installation and Setup](#installation-and-setup)
2. [Basic Usage](#basic-usage)
3. [CLI Command Reference](#cli-command-reference)
4. [Configuration Management](#configuration-management)
5. [Step-by-Step Guide](#step-by-step-guide)
6. [Advanced Features](#advanced-features)
7. [Troubleshooting](#troubleshooting)
8. [API Reference](#api-reference)

## Installation and Setup

### Prerequisites

- Python 3.12+
- UV package manager or pip
- Sufficient disk space for processing documents

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/MDC-Challenge-2025.git
cd MDC-Challenge-2025

# Install dependencies
uv sync
```

### Verify Installation

```bash
python preprocessing.py --help
```

## Basic Usage

### Running the Complete Pipeline

```bash
# Run all preprocessing steps
python preprocessing.py --data-dir Data

# Run with progress bars and verbose output
python preprocessing.py --data-dir Data --verbose --progress

# Run with custom parameters
python preprocessing.py --data-dir Data --chunk-size 300 --chunk-overlap 30
```

### Running Specific Steps

```bash
# Run only pre-chunking EDA and document conversion
python preprocessing.py --steps pre_chunking_eda,doc_conversion

# Run from document parsing onwards
python preprocessing.py --from document_parsing

# Run up to semantic chunking
python preprocessing.py --up-to semantic_chunking
```

### Resume After Interruption

```bash
# Resume from the last successful step
python preprocessing.py --resume

# Resume and force re-run the last failed step
python preprocessing.py --resume --force
```

## CLI Command Reference

### Core Commands

| Command | Description | Example |
|---------|-------------|---------|
| `--data-dir` | Specify data directory | `--data-dir /path/to/data` |
| `--steps` | Run specific steps | `--steps pre_chunking_eda,doc_conversion` |
| `--up-to` | Run up to specific step | `--up-to semantic_chunking` |
| `--from` | Run from specific step | `--from document_parsing` |
| `--resume` | Resume from last successful step | `--resume` |
| `--force` | Force re-run completed steps | `--force` |

### Output and Reporting

| Command | Description | Example |
|---------|-------------|---------|
| `--output-format` | Report formats | `--output-format json,markdown` |
| `--save-reports` | Save reports to disk | `--save-reports` |
| `--reports-dir` | Reports directory | `--reports-dir /path/to/reports` |

### Visualization and Monitoring

| Command | Description | Example |
|---------|-------------|---------|
| `--progress` | Show progress bars | `--progress` |
| `--dashboard` | Enable real-time dashboard | `--dashboard` |
| `--generate-diagrams` | Generate pipeline diagrams | `--generate-diagrams` |

### Step-Specific Parameters

| Command | Description | Example |
|---------|-------------|---------|
| `--chunk-size` | Semantic chunking size | `--chunk-size 200` |
| `--chunk-overlap` | Semantic chunking overlap | `--chunk-overlap 20` |
| `--show-plots` | Show plots in EDA | `--show-plots` |

### Configuration

| Command | Description | Example |
|---------|-------------|---------|
| `--config` | Use configuration file | `--config config.yaml` |
| `--save-config` | Save configuration | `--save-config my_config.json` |
| `--template` | Use config template | `--template production` |

### Logging and Debugging

| Command | Description | Example |
|---------|-------------|---------|
| `--verbose` | Verbose logging | `--verbose` |
| `--log-level` | Set log level | `--log-level DEBUG` |
| `--log-file` | Log to file | `--log-file pipeline.log` |

## Configuration Management

### Configuration Files

The pipeline supports both JSON and YAML configuration files:

```yaml
# config.yaml
name: "My Preprocessing Pipeline"
description: "Custom configuration for data processing"
version: "1.0.0"

data_directory: "Data"
log_level: "INFO"
enable_progress_bars: true
enable_dashboard: false
generate_diagrams: true

steps:
  semantic_chunking:
    enabled: true
    parameters:
      chunk_size: 200
      chunk_overlap: 20
    timeout: 3600
    retry_count: 3

resources:
  memory_alert_threshold: 85.0
  cpu_alert_threshold: 90.0
  disk_alert_threshold: 90.0

reporting:
  output_formats: ["json", "markdown"]
  save_reports: true
  include_performance_metrics: true
```

### Configuration Templates

Pre-defined templates are available:

```bash
# Development configuration (debug logging, visualization enabled)
python preprocessing.py --template development

# Production configuration (optimized for performance)
python preprocessing.py --template production

# Fast configuration (reduced quality for quick testing)
python preprocessing.py --template fast
```

### Creating Custom Configurations

```bash
# Generate configuration from current CLI arguments
python preprocessing.py --chunk-size 300 --verbose --save-config my_config.json

# Load and use custom configuration
python preprocessing.py --config my_config.json
```

## Step-by-Step Guide

### Step 1: Pre-Chunking EDA

Performs exploratory data analysis before chunking.

```bash
python preprocessing.py --steps pre_chunking_eda --show-plots
```

**Parameters:**
- `--show-plots`: Display visualization plots
- `--detailed-analysis`: Enable detailed analysis mode

### Step 2: Document Conversion

Converts PDF documents to XML format.

```bash
python preprocessing.py --steps doc_conversion
```

**Parameters:**
- `--timeout`: Conversion timeout in seconds
- `--batch-size`: Number of documents to process in parallel

### Step 3: Document Parsing

Extracts structured information from documents.

```bash
python preprocessing.py --steps document_parsing
```

**Parameters:**
- `--extract-sections`: Extract specific document sections
- `--preserve-formatting`: Preserve original formatting

### Step 4: Semantic Chunking

Breaks documents into semantic chunks.

```bash
python preprocessing.py --steps semantic_chunking --chunk-size 200 --chunk-overlap 20
```

**Parameters:**
- `--chunk-size`: Target chunk size in tokens
- `--chunk-overlap`: Overlap between chunks
- `--strategy`: Chunking strategy (sentence, paragraph, semantic)

### Step 5-8: Advanced Processing

Vector embeddings, chunk-level EDA, quality control, and artifact export.

```bash
python preprocessing.py --from vector_embeddings
```

## Advanced Features

### Progress Visualization

```bash
# Enable progress bars
python preprocessing.py --progress

# Enable real-time dashboard
python preprocessing.py --dashboard

# Generate pipeline diagrams
python preprocessing.py --generate-diagrams
```

### Resource Monitoring

```bash
# Monitor resource usage with alerts
python preprocessing.py --monitor-resources --memory-threshold 80
```

### Parallel Processing

```bash
# Enable parallel processing where supported
python preprocessing.py --parallel --workers 4
```

### Error Recovery

```bash
# Automatic retry with exponential backoff
python preprocessing.py --retry-count 5 --retry-delay 10

# Rollback failed steps
python preprocessing.py --rollback-on-failure
```

## Troubleshooting

### Common Issues

#### 1. Out of Memory Errors

```bash
# Solution: Reduce chunk size or enable batch processing
python preprocessing.py --chunk-size 100 --batch-size 10
```

#### 2. File Permission Errors

```bash
# Solution: Check file permissions and data directory access
chmod -R 755 Data/
```

#### 3. Missing Dependencies

```bash
# Solution: Reinstall dependencies
uv sync --force
```

#### 4. Corrupted State

```bash
# Solution: Clear pipeline state and restart
rm Data/pipeline_state.pkl
python preprocessing.py --force
```

### Debug Mode

```bash
# Enable debug logging
python preprocessing.py --log-level DEBUG --verbose

# Save debug logs to file
python preprocessing.py --log-level DEBUG --log-file debug.log
```

### Performance Issues

```bash
# Profile performance
python preprocessing.py --profile --performance-report

# Monitor resource usage
python preprocessing.py --monitor-resources --resource-alerts
```

### Validation Issues

```bash
# Validate data directory structure
python preprocessing.py --validate-only

# Check file dependencies
python preprocessing.py --check-dependencies
```

## API Reference

### PreprocessingPipeline Class

```python
from preprocessing import PreprocessingPipeline

# Initialize pipeline
pipeline = PreprocessingPipeline(data_dir="Data")

# Run all steps
success = pipeline.run_all()

# Run specific steps
success = pipeline.run_specific_steps(["pre_chunking_eda", "doc_conversion"])

# Run up to a step
success = pipeline.run_up_to_step("semantic_chunking")

# Resume execution
success = pipeline.resume_pipeline()

# Generate reports
pipeline.generate_consolidated_reports()
```

### Configuration Management

```python
from src.pipeline_config import ConfigurationManager, PipelineConfig

# Load configuration
config_manager = ConfigurationManager()
config = config_manager.load_config("config.yaml")

# Create pipeline with configuration
pipeline = PreprocessingPipeline(data_dir="Data", config_file="config.yaml")

# Update step parameters
pipeline.update_step_parameters("semantic_chunking", chunk_size=300)
```

### Visualization

```python
from src.pipeline_visualization import PipelineVisualizer

# Create visualizer
visualizer = PipelineVisualizer(pipeline.STEP_DEFINITIONS)

# Generate diagrams
visualizer.generate_all_diagrams()

# Start progress tracking
visualizer.start_progress_tracking(total_steps=8)
```

### Error Handling

```python
try:
    success = pipeline.run_all()
except Exception as e:
    error_type = pipeline.categorize_error(e)
    if error_type == "RECOVERABLE":
        success = pipeline.rollback_step("failed_step")
```

## Examples

### Example 1: Basic Processing

```python
#!/usr/bin/env python3
"""
Basic preprocessing pipeline example.
"""
from preprocessing import PreprocessingPipeline

def main():
    # Initialize pipeline
    pipeline = PreprocessingPipeline(data_dir="Data")
    
    # Run all steps
    success = pipeline.run_all()
    
    if success:
        print("Pipeline completed successfully!")
        # Generate reports
        pipeline.generate_consolidated_reports()
    else:
        print("Pipeline failed!")
        # Check failed steps
        print(f"Failed steps: {pipeline.state.failed_steps}")

if __name__ == "__main__":
    main()
```

### Example 2: Custom Configuration

```python
#!/usr/bin/env python3
"""
Custom configuration example.
"""
from preprocessing import PreprocessingPipeline
from src.pipeline_config import ConfigurationManager, PipelineConfig

def main():
    # Create custom configuration
    config = PipelineConfig(
        data_directory="Data",
        enable_progress_bars=True,
        enable_dashboard=True,
        log_level="DEBUG"
    )
    
    # Save configuration
    config_manager = ConfigurationManager()
    config_manager.save_config("custom_config.yaml", config)
    
    # Initialize pipeline with configuration
    pipeline = PreprocessingPipeline(
        data_dir="Data", 
        config_file="custom_config.yaml"
    )
    
    # Run pipeline
    success = pipeline.run_all()
    
    if success:
        print("Pipeline completed with custom configuration!")

if __name__ == "__main__":
    main()
```

### Example 3: Step-by-Step Processing

```python
#!/usr/bin/env python3
"""
Step-by-step processing example.
"""
from preprocessing import PreprocessingPipeline

def main():
    pipeline = PreprocessingPipeline(data_dir="Data")
    
    # Run steps individually with custom parameters
    steps = [
        ("pre_chunking_eda", {"show_plots": True}),
        ("doc_conversion", {"timeout": 1800}),
        ("document_parsing", {"extract_sections": True}),
        ("semantic_chunking", {"chunk_size": 300, "chunk_overlap": 30})
    ]
    
    for step_id, params in steps:
        print(f"Running step: {step_id}")
        
        # Update parameters
        pipeline.update_step_parameters(step_id, params)
        
        # Run step
        success = pipeline.run_single_step(step_id)
        
        if success:
            print(f"✅ {step_id} completed successfully")
        else:
            print(f"❌ {step_id} failed")
            break
    
    # Generate final report
    pipeline.generate_consolidated_reports()

if __name__ == "__main__":
    main()
```

### Example 4: Error Handling and Recovery

```python
#!/usr/bin/env python3
"""
Error handling and recovery example.
"""
from preprocessing import PreprocessingPipeline, ErrorType

def main():
    pipeline = PreprocessingPipeline(data_dir="Data")
    
    try:
        # Attempt to run pipeline
        success = pipeline.run_all()
        
        if not success:
            # Handle failures
            handle_pipeline_failures(pipeline)
            
    except Exception as e:
        print(f"Unexpected error: {e}")
        # Categorize error
        error_type = pipeline.categorize_error(e)
        print(f"Error type: {error_type}")
        
        if error_type == ErrorType.RECOVERABLE:
            # Attempt recovery
            print("Attempting recovery...")
            recovery_success = attempt_recovery(pipeline)
            if recovery_success:
                print("Recovery successful!")
            else:
                print("Recovery failed!")

def handle_pipeline_failures(pipeline):
    """Handle pipeline failures with appropriate recovery strategies."""
    failed_steps = pipeline.state.failed_steps
    
    for step_id in failed_steps:
        metadata = pipeline.step_metadata[step_id]
        print(f"Step {step_id} failed: {metadata.error_message}")
        
        # Check error type
        if metadata.error_type == ErrorType.TRANSIENT:
            # Retry transient errors
            print(f"Retrying {step_id}...")
            success = pipeline.run_single_step(step_id, force=True)
            if success:
                print(f"✅ {step_id} succeeded on retry")
        
        elif metadata.error_type == ErrorType.RECOVERABLE:
            # Rollback and retry recoverable errors
            print(f"Rolling back {step_id}...")
            pipeline.rollback_step(step_id)
            success = pipeline.run_single_step(step_id, force=True)
            if success:
                print(f"✅ {step_id} succeeded after rollback")
        
        else:
            # Fatal errors require manual intervention
            print(f"❌ {step_id} has fatal error - manual intervention required")

def attempt_recovery(pipeline):
    """Attempt to recover from pipeline failure."""
    # Clear any corrupted state
    pipeline.state.current_step = None
    pipeline._save_state()
    
    # Resume from last successful step
    return pipeline.resume_pipeline(force=True)

if __name__ == "__main__":
    main()
```

## Best Practices

1. **Always validate your data directory structure before running**
2. **Use configuration files for complex setups**
3. **Monitor resource usage for large datasets**
4. **Enable progress bars for long-running processes**
5. **Use specific steps for debugging issues**
6. **Keep backups of successful pipeline states**
7. **Review consolidated reports after completion**

## Support

For issues and questions:
- Check the troubleshooting section
- Review the logs in the `logs/` directory
- Examine the consolidated reports in `reports/`
- Open an issue on the GitHub repository

---

*This guide covers the comprehensive preprocessing pipeline system. For more specific information about individual steps, refer to the dedicated documentation files in the `docs/` directory.* 