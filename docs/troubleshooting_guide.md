# Preprocessing Pipeline Troubleshooting Guide

## Table of Contents

1. [Common Issues and Solutions](#common-issues-and-solutions)
2. [Error Categories](#error-categories)
3. [Debug Techniques](#debug-techniques)
4. [Performance Issues](#performance-issues)
5. [Data Issues](#data-issues)
6. [Configuration Problems](#configuration-problems)
7. [System Resource Issues](#system-resource-issues)
8. [Recovery Procedures](#recovery-procedures)
9. [Diagnostic Tools](#diagnostic-tools)
10. [FAQ](#faq)

## Common Issues and Solutions

### 1. Pipeline Fails to Start

**Symptoms:**
- Pipeline exits immediately
- "No such file or directory" errors
- Permission denied errors

**Solutions:**

```bash
# Check data directory exists and is accessible
ls -la Data/
chmod -R 755 Data/

# Verify Python environment
python --version
python -c "import preprocessing; print('Import successful')"

# Check dependencies
pip list | grep -E "(pydantic|psutil|rich)"
```

### 2. Out of Memory Errors

**Symptoms:**
- Process killed by system
- "MemoryError" exceptions
- System becomes unresponsive

**Solutions:**

```bash
# Reduce memory usage
python preprocessing.py --chunk-size 100 --batch-size 5

# Monitor memory usage
python preprocessing.py --monitor-resources --memory-threshold 80

# Use streaming processing
python preprocessing.py --stream-processing --max-memory 4GB
```

### 3. Step Hangs or Takes Too Long

**Symptoms:**
- Step appears stuck
- No progress for extended periods
- High CPU usage without progress

**Solutions:**

```bash
# Set timeouts
python preprocessing.py --timeout 3600 --step-timeout 1800

# Enable progress bars
python preprocessing.py --progress --verbose

# Run with smaller batch sizes
python preprocessing.py --batch-size 1 --parallel false
```

### 4. File Not Found Errors

**Symptoms:**
- "FileNotFoundError" exceptions
- "No such file or directory" messages
- Missing input files

**Solutions:**

```bash
# Validate directory structure
python preprocessing.py --validate-only

# Check file dependencies
python preprocessing.py --check-dependencies --verbose

# List missing files
find Data/ -name "*.pkl" -o -name "*.xml" -o -name "*.pdf" | head -20
```

### 5. Corrupted State

**Symptoms:**
- Pipeline shows completed steps that weren't run
- Inconsistent state between runs
- "pickle.UnpicklingError" exceptions

**Solutions:**

```bash
# Clear pipeline state
rm Data/pipeline_state.pkl

# Force restart from beginning
python preprocessing.py --force --reset-state

# Validate state consistency
python preprocessing.py --validate-state
```

## Error Categories

### Fatal Errors
These require manual intervention:

| Error Type | Description | Solution |
|------------|-------------|----------|
| File Permission | Cannot read/write files | `chmod -R 755 Data/` |
| Missing Dependencies | Required packages not installed | `pip install -r requirements.txt` |
| Corrupted Data | Data files are corrupted | Restore from backup |
| Disk Full | No space left on device | Free disk space |
| Invalid Configuration | Config file syntax errors | Fix configuration file |

### Recoverable Errors
These can be automatically retried:

| Error Type | Description | Solution |
|------------|-------------|----------|
| Temporary File Lock | File temporarily locked | Wait and retry |
| Network Timeout | Network connection issues | Retry with backoff |
| Resource Exhaustion | Temporary resource unavailable | Retry after delay |
| Processing Exception | Processing step failure | Rollback and retry |

### Transient Errors
These should be retried immediately:

| Error Type | Description | Solution |
|------------|-------------|----------|
| Connection Timeout | Network connection timeout | Immediate retry |
| Temporary Unavailable | Resource temporarily busy | Short delay and retry |
| Lock Contention | File lock contention | Brief retry |

## Debug Techniques

### 1. Enable Debug Logging

```bash
# Full debug logging
python preprocessing.py --log-level DEBUG --verbose

# Debug specific step
python preprocessing.py --steps pre_chunking_eda --log-level DEBUG

# Save debug logs to file
python preprocessing.py --log-level DEBUG --log-file debug.log
```

### 2. Validate Before Running

```bash
# Validate data directory structure
python preprocessing.py --validate-only

# Check file dependencies
python preprocessing.py --check-dependencies

# Validate configuration
python preprocessing.py --config config.yaml --validate-config
```

### 3. Run Single Steps

```bash
# Test individual steps
python preprocessing.py --steps pre_chunking_eda --verbose
python preprocessing.py --steps doc_conversion --verbose
python preprocessing.py --steps document_parsing --verbose
```

### 4. Monitor Resource Usage

```bash
# Enable resource monitoring
python preprocessing.py --monitor-resources --resource-alerts

# Set custom thresholds
python preprocessing.py --memory-threshold 75 --cpu-threshold 85
```

### 5. Use Test Mode

```bash
# Run with minimal data
python preprocessing.py --test-mode --sample-size 10

# Dry run (validate without execution)
python preprocessing.py --dry-run --verbose
```

## Performance Issues

### Slow Processing

**Diagnosis:**
```bash
# Profile performance
python preprocessing.py --profile --performance-report

# Monitor system resources
htop
iostat -x 1

# Check disk usage
df -h
du -sh Data/*
```

**Solutions:**

1. **Reduce Data Size:**
```bash
# Process subset of data
python preprocessing.py --limit 100 --test-mode

# Use smaller chunks
python preprocessing.py --chunk-size 100 --batch-size 5
```

2. **Optimize Parameters:**
```bash
# Disable visualization
python preprocessing.py --no-plots --no-dashboard

# Use faster processing
python preprocessing.py --template fast
```

3. **Parallel Processing:**
```bash
# Enable parallel processing
python preprocessing.py --parallel --workers 4

# Process in batches
python preprocessing.py --batch-size 10 --parallel-batches
```

### High Memory Usage

**Diagnosis:**
```bash
# Monitor memory usage
python preprocessing.py --monitor-resources --memory-alerts

# Check memory leaks
python -m memory_profiler preprocessing.py
```

**Solutions:**

1. **Reduce Memory Footprint:**
```bash
# Smaller batch sizes
python preprocessing.py --batch-size 1 --stream-processing

# Disable caching
python preprocessing.py --no-cache --clear-cache
```

2. **Memory Limits:**
```bash
# Set memory limits
python preprocessing.py --max-memory 4GB --memory-threshold 80

# Use swap if available
python preprocessing.py --use-swap --swap-threshold 90
```

## Data Issues

### Missing Files

**Diagnosis:**
```bash
# Check file inventory
python preprocessing.py --inventory --verbose

# List missing files
python preprocessing.py --check-dependencies --list-missing
```

**Solutions:**

1. **Skip Missing Files:**
```bash
# Continue with available files
python preprocessing.py --skip-missing --warn-missing

# Ignore specific file types
python preprocessing.py --ignore-missing-xml --ignore-missing-pdf
```

2. **Generate Missing Files:**
```bash
# Attempt to generate missing files
python preprocessing.py --generate-missing --force-conversion
```

### Corrupted Data

**Diagnosis:**
```bash
# Validate data integrity
python preprocessing.py --validate-data --verbose

# Check file corruption
python preprocessing.py --check-corruption --report-corrupt
```

**Solutions:**

1. **Skip Corrupted Files:**
```bash
# Skip corrupted files
python preprocessing.py --skip-corrupted --report-skipped

# Quarantine corrupted files
python preprocessing.py --quarantine-corrupted --quarantine-dir quarantine/
```

2. **Attempt Recovery:**
```bash
# Try to recover corrupted files
python preprocessing.py --recover-corrupted --backup-originals
```

## Configuration Problems

### Invalid Configuration

**Diagnosis:**
```bash
# Validate configuration syntax
python preprocessing.py --config config.yaml --validate-config

# Check configuration values
python preprocessing.py --config config.yaml --show-config
```

**Solutions:**

1. **Fix Configuration:**
```bash
# Use configuration template
python preprocessing.py --template development --save-config fixed_config.yaml

# Validate and fix
python preprocessing.py --config config.yaml --fix-config --save-config
```

2. **Reset Configuration:**
```bash
# Use default configuration
python preprocessing.py --reset-config --save-config default_config.yaml
```

### Parameter Conflicts

**Diagnosis:**
```bash
# Check parameter conflicts
python preprocessing.py --check-params --verbose

# Show effective parameters
python preprocessing.py --show-effective-params
```

**Solutions:**

1. **Resolve Conflicts:**
```bash
# Use explicit parameters
python preprocessing.py --chunk-size 200 --force-params

# Override with configuration
python preprocessing.py --config config.yaml --override-cli
```

## System Resource Issues

### Disk Space

**Diagnosis:**
```bash
# Check disk usage
df -h
du -sh Data/* | sort -hr

# Monitor disk usage during processing
python preprocessing.py --monitor-disk --disk-alerts
```

**Solutions:**

1. **Free Space:**
```bash
# Clean temporary files
python preprocessing.py --clean-temp --clean-cache

# Compress old reports
python preprocessing.py --compress-reports --archive-old
```

2. **Use Different Location:**
```bash
# Use different temp directory
export TMPDIR=/path/to/large/temp
python preprocessing.py --temp-dir /path/to/large/temp

# Move data to different location
python preprocessing.py --data-dir /path/to/large/data
```

### CPU/Memory Limits

**Diagnosis:**
```bash
# Monitor resource usage
python preprocessing.py --monitor-resources --resource-report

# Check system limits
ulimit -a
```

**Solutions:**

1. **Adjust Limits:**
```bash
# Increase limits
ulimit -v 8388608  # 8GB virtual memory
ulimit -m 8388608  # 8GB resident memory

# Run with nice priority
nice -n 10 python preprocessing.py
```

2. **Reduce Usage:**
```bash
# Limit CPU usage
python preprocessing.py --cpu-limit 80 --throttle-cpu

# Limit memory usage
python preprocessing.py --memory-limit 4GB --stream-processing
```

## Recovery Procedures

### 1. Complete Recovery

```bash
# Stop current pipeline
pkill -f preprocessing.py

# Backup current state
cp Data/pipeline_state.pkl Data/pipeline_state_backup.pkl

# Clear state and restart
rm Data/pipeline_state.pkl
python preprocessing.py --force --verbose
```

### 2. Partial Recovery

```bash
# Resume from last successful step
python preprocessing.py --resume --verbose

# Resume with force retry of failed step
python preprocessing.py --resume --force --retry-failed
```

### 3. Step-Specific Recovery

```bash
# Rollback specific step
python preprocessing.py --rollback-step semantic_chunking

# Retry specific step
python preprocessing.py --retry-step semantic_chunking --force
```

### 4. Data Recovery

```bash
# Restore from backup
cp backup/Data/* Data/

# Regenerate missing files
python preprocessing.py --regenerate-missing --force
```

## Diagnostic Tools

### 1. Built-in Diagnostics

```bash
# System check
python preprocessing.py --system-check --verbose

# Health check
python preprocessing.py --health-check --report-health

# Performance benchmark
python preprocessing.py --benchmark --benchmark-report
```

### 2. External Tools

```bash
# Monitor processes
htop
ps aux | grep preprocessing

# Monitor I/O
iostat -x 1
iotop

# Monitor memory
free -h
vmstat 1

# Monitor disk
df -h
iotop -o
```

### 3. Log Analysis

```bash
# Search for errors
grep -i error logs/preprocessing.log

# Show recent activity
tail -f logs/preprocessing.log

# Analyze performance
grep -i "duration\|time\|performance" logs/preprocessing.log
```

## FAQ

### Q: Pipeline runs but produces no output
**A:** Check file permissions and ensure the `reports/` directory is writable:
```bash
mkdir -p reports/
chmod 755 reports/
python preprocessing.py --save-reports --verbose
```

### Q: Getting "module not found" errors
**A:** Ensure all dependencies are installed and Python path is correct:
```bash
pip install -r requirements.txt
export PYTHONPATH=$PYTHONPATH:$(pwd)
python preprocessing.py
```

### Q: Pipeline runs out of memory on large datasets
**A:** Use streaming processing and smaller batch sizes:
```bash
python preprocessing.py --stream-processing --batch-size 1 --chunk-size 100
```

### Q: Steps are failing with timeout errors
**A:** Increase timeouts or disable them:
```bash
python preprocessing.py --timeout 0 --step-timeout 7200
```

### Q: Cannot resume after interruption
**A:** Check if state file is corrupted and rebuild if necessary:
```bash
python preprocessing.py --validate-state --fix-state
```

### Q: Getting permission denied errors
**A:** Ensure proper file permissions:
```bash
chmod -R 755 Data/
chown -R $USER:$USER Data/
```

### Q: Pipeline produces inconsistent results
**A:** Clear caches and run with deterministic settings:
```bash
python preprocessing.py --clear-cache --deterministic --seed 42
```

### Q: How to run only failed steps
**A:** Use the retry failed steps option:
```bash
python preprocessing.py --retry-failed --force
```

### Q: How to skip problematic documents
**A:** Enable skip options:
```bash
python preprocessing.py --skip-corrupted --skip-missing --continue-on-error
```

### Q: How to get detailed error information
**A:** Enable debug logging and verbose output:
```bash
python preprocessing.py --log-level DEBUG --verbose --traceback
```

---

## Getting Help

If you encounter issues not covered in this guide:

1. **Check the logs** in `logs/preprocessing.log`
2. **Run diagnostics** with `--system-check --health-check`
3. **Validate your setup** with `--validate-only`
4. **Use debug mode** with `--log-level DEBUG --verbose`
5. **Check GitHub issues** for similar problems
6. **Create a new issue** with detailed error information

Remember to include the following information when reporting issues:
- Operating system and Python version
- Full command line used
- Complete error messages and stack traces
- Contents of log files
- System resource usage (memory, disk, CPU)
- Configuration files used

---

*This troubleshooting guide covers the most common issues and solutions. For specific problems, refer to the detailed documentation and log files.* 