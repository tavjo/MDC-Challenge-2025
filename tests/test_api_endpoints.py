"""
Test script for API endpoints in the MDC Challenge 2025 project.

This script tests all three API endpoints:
1. GET /health - Health check
2. POST /run_semantic_chunking - Main pipeline trigger
3. POST /chunk/documents - Process specific documents

Prerequisites:
- DuckDB database must be populated (run migrate_data.py first)
- OPENAI_API_KEY environment variable must be set
- Dependencies: fastapi, uvicorn, httpx, pytest
"""

import os
import sys
import json
import asyncio
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

import pytest
import httpx
from fastapi.testclient import TestClient

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from api.chunk_and_embed_api import app
from api.utils.duckdb_utils import DuckDBHelper, test_connection
from src.models import Document, ChunkingResult
from src.helpers import initialize_logging

# Initialize logging
logger = initialize_logging("test_api_endpoints")

# Test client
client = TestClient(app)

class TestAPIEndpoints:
    """Test suite for API endpoints."""
    
    @classmethod
    def setup_class(cls):
        """Setup test environment."""
        cls.db_path = "artifacts/mdc_challenge.db"
        
        # Ensure database exists and is accessible
        if not test_connection(cls.db_path):
            pytest.skip("Database not accessible - run migrate_data.py first")
        
        logger.info("Starting API endpoint tests")
    
    def test_health_endpoint(self):
        """Test the health check endpoint."""
        logger.info("Testing health endpoint")
        
        response = client.get("/health")
        
        assert response.status_code == 200
        health_data = response.json()
        
        # Check required fields
        assert "status" in health_data
        assert "duckdb_connected" in health_data
        assert "chromadb_connected" in health_data
        assert "embedding_model" in health_data
        assert "db_path" in health_data
        
        # Check that DuckDB is connected
        assert health_data["duckdb_connected"] is True
        assert health_data["status"] == "healthy"
        
        logger.info(f"✅ Health check passed: {health_data}")
    
    def test_health_endpoint_with_custom_db_path(self):
        """Test health endpoint with custom database path."""
        logger.info("Testing health endpoint with custom db path")
        
        response = client.get(f"/health?db_path={self.db_path}")
        
        assert response.status_code == 200
        health_data = response.json()
        
        assert health_data["db_path"] == self.db_path
        assert health_data["duckdb_connected"] is True
        
        logger.info("✅ Health check with custom path passed")
    
    def test_run_semantic_chunking_endpoint(self):
        """Test the main pipeline endpoint."""
        logger.info("Testing run_semantic_chunking endpoint")
        
        # Test with subset to make it faster
        response = client.post(
            "/run_semantic_chunking",
            params={
                "subset": True,
                "subset_size": 2,
                "chunk_size": 150,
                "chunk_overlap": 10,
                "collection_name": "test_chunks"
            }
        )
        
        assert response.status_code == 200
        result_data = response.json()
        
        # Validate ChunkingResult structure
        assert "success" in result_data
        assert "total_documents" in result_data
        assert "total_chunks" in result_data
        assert "total_tokens" in result_data
        assert "avg_tokens_per_chunk" in result_data
        assert "validation_passed" in result_data
        assert "pipeline_completed_at" in result_data
        assert "entity_retention" in result_data
        
        # Check that pipeline succeeded
        assert result_data["success"] is True
        assert result_data["total_documents"] > 0
        assert result_data["total_chunks"] > 0
        assert result_data["validation_passed"] is True
        
        logger.info(f"✅ Pipeline completed successfully:")
        logger.info(f"   Documents: {result_data['total_documents']}")
        logger.info(f"   Chunks: {result_data['total_chunks']}")
        logger.info(f"   Avg tokens per chunk: {result_data['avg_tokens_per_chunk']:.1f}")
        logger.info(f"   Entity retention: {result_data['entity_retention']:.1f}%")
    
    def test_chunk_documents_endpoint(self):
        """Test the chunk specific documents endpoint."""
        logger.info("Testing chunk_documents endpoint")
        
        # First, get a sample document from the database
        helper = DuckDBHelper(self.db_path)
        sample_docs = helper.get_documents_by_query("SELECT * FROM documents", limit=1)
        helper.close()
        
        if not sample_docs:
            pytest.skip("No documents in database for testing")
        
        # Convert to JSON-serializable format
        sample_doc = sample_docs[0]
        doc_data = sample_doc.model_dump()
        
        # Test the endpoint
        response = client.post(
            "/chunk/documents",
            json=[doc_data],
            params={
                "chunk_size": 100,
                "chunk_overlap": 10,
                "collection_name": "test_specific_chunks"
            }
        )
        
        assert response.status_code == 200
        result_data = response.json()
        
        # Validate result structure
        assert "success" in result_data
        assert "total_documents" in result_data
        assert "total_chunks" in result_data
        
        # Check that processing succeeded
        assert result_data["success"] is True
        assert result_data["total_documents"] >= 1
        
        logger.info(f"✅ Document processing completed:")
        logger.info(f"   Documents processed: {result_data['total_documents']}")
        logger.info(f"   Chunks created: {result_data['total_chunks']}")
    
    def test_api_error_handling(self):
        """Test API error handling."""
        logger.info("Testing API error handling")
        
        # Test with invalid database path
        response = client.get("/health?db_path=/nonexistent/path/db.db")
        
        assert response.status_code == 200
        health_data = response.json()
        assert health_data["status"] == "unhealthy"
        assert health_data["duckdb_connected"] is False
        
        # Test with invalid document data
        response = client.post(
            "/chunk/documents",
            json=[{"invalid": "data"}]
        )
        
        # Should return 422 for validation error
        assert response.status_code == 422
        
        logger.info("✅ Error handling tests passed")
    
    def test_database_integration(self):
        """Test database integration through API."""
        logger.info("Testing database integration")
        
        # Get database stats through helper
        helper = DuckDBHelper(self.db_path)
        stats = helper.get_database_stats()
        helper.close()
        
        # Verify we have data
        assert stats["total_documents"] > 0
        assert stats["total_citations"] > 0
        
        logger.info(f"✅ Database integration verified:")
        logger.info(f"   Documents: {stats['total_documents']}")
        logger.info(f"   Citations: {stats['total_citations']}")
        logger.info(f"   Chunks: {stats['total_chunks']}")

def run_manual_tests():
    """Run manual tests for development/debugging."""
    logger.info("Running manual API tests")
    
    # Test 1: Health check
    print("\n=== Testing Health Endpoint ===")
    try:
        response = client.get("/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error: {str(e)}")
    
    # Test 2: Run pipeline with subset
    print("\n=== Testing Pipeline Endpoint ===")
    try:
        response = client.post(
            "/run_semantic_chunking",
            params={
                "subset": True,
                "subset_size": 1,
                "chunk_size": 100,
                "collection_name": "manual_test"
            }
        )
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Success: {result['success']}")
            print(f"Documents: {result['total_documents']}")
            print(f"Chunks: {result['total_chunks']}")
        else:
            print(f"Error: {response.json()}")
    except Exception as e:
        print(f"Error: {str(e)}")
    
    # Test 3: Database helper
    print("\n=== Testing Database Helper ===")
    try:
        helper = DuckDBHelper()
        stats = helper.get_database_stats()
        print(f"Database stats: {json.dumps(stats, indent=2)}")
        helper.close()
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    # Check if we're running in test mode or manual mode
    if len(sys.argv) > 1 and sys.argv[1] == "manual":
        run_manual_tests()
    else:
        # Run pytest
        pytest.main([__file__, "-v", "-s"]) 