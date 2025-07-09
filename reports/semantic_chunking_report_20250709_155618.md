# Semantic Chunking Pipeline Comprehensive Report

**Generated:** 2025-07-09 15:56:18
**Data Directory:** `Data`
**Chunking Duration:** 0:00:00.455101

## Executive Summary

- **Configuration Validation:** PASSED
- **Pipeline Success:** True
- **Total Chunks Created:** 3
- **Entity Retention Rate:** 100.0%
- **Quality Score:** GOOD
- **Total Processing Time:** 0.43s
- **Average Tokens per Chunk:** 261.0
- **Chunking Steps Completed:** 3

## Chunking Performance

- **average_processing_time:** 0.43s
- **total_processing_time:** 0.43s
- **fastest_processing:** 0.4302248954772949
- **slowest_processing:** 0.4302248954772949
- **processing_rate:** 2.32 docs/sec
- **average_memory_mb:** 171.0MB
- **peak_memory_mb:** 171.0MB
- **min_memory_mb:** 171.0MB

## Quality Gates Status

- **Entity retention:** ✅ PASSED
- **Retention rate:** 100.0%

## Chunk Distribution Analysis

- **Total chunks created:** 3
- **Documents processed:** 3
- **Average chunks per document:** 1.0

## Chunking Start

**message:** Semantic chunking process initiated
**input_path:** /Users/taishajoseph/Documents/Projects/MDC-Challenge-2025/Data/train/parsed/parsed_documents.pkl
**output_path:** /Users/taishajoseph/Documents/Projects/MDC-Challenge-2025/Data/chunks_for_embedding.pkl
**parameters:**
  - chunk_size: 300
  - chunk_overlap: 30
  - min_chars: 500
**timestamp:** 2025-07-09T15:56:18.479689

## Step 1 Configuration

**input_validation:** passed
**input_file_exists:** True
**configuration:**
  - chunk_size: 300
  - chunk_overlap: 30
  - min_chars: 500
  - overlap_ratio: 0.1
**step_duration:** 0.0002269744873046875

## Step 2 Core Chunking

**pipeline_success:** True
**total_documents:** 3
**total_chunks:** 3
**total_tokens:** 783
**avg_tokens_per_chunk:** 261.0
**validation_passed:** True
**entity_retention:** 100.0
**step_duration:** 0.4302248954772949

## Step 3 Quality Analysis

**quality_gates:**
  - entity_retention_passed: True
  - token_distribution_analysis: {'avg_tokens_per_chunk': 261.0, 'optimal_range': '160-220 tokens', 'acceptable_range': '130-270 tokens', 'status': 'acceptable'}
  - processing_efficiency: {'processing_time': 0.4302248954772949, 'chunks_per_second': 6.973097167405378, 'documents_per_second': 6.973097167405378}
**recommendations:**
**overall_quality_score:** good
**step_duration:** 1.6689300537109375e-05

## Generated Files

- **Chunks for embedding (pickle):** `/Users/taishajoseph/Documents/Projects/MDC-Challenge-2025/Data/chunks_for_embedding.pkl`
- **Chunking summary (CSV):** `/Users/taishajoseph/Documents/Projects/MDC-Challenge-2025/Data/chunks_for_embedding_summary.csv`
