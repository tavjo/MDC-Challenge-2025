"""
Integration tests for the preprocessing pipeline.

This module contains comprehensive integration tests for:
- Full pipeline execution from start to finish
- Partial pipeline execution (specific steps)
- Resume functionality after interruption
- Various parameter combinations
- End-to-end workflow validation
"""

import unittest
import tempfile
import shutil
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import time

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from preprocessing import PreprocessingPipeline
from src.tools.pipeline_config import ConfigurationManager, PipelineConfig
from src.tools.pipeline_visualization import PipelineVisualizer


class TestFullPipelineExecution(unittest.TestCase):
    """Integration tests for full pipeline execution."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = os.path.join(self.temp_dir, "test_data")
        
        # Create comprehensive test data structure
        self.setup_test_data()
        
        self.pipeline = PreprocessingPipeline(data_dir=self.data_dir)
    
    def tearDown(self):
        """Clean up after tests."""
        shutil.rmtree(self.temp_dir)
    
    def setup_test_data(self):
        """Set up realistic test data structure."""
        # Create directories
        paths = [
            "train/documents",
            "train/xml",
            "train/parsed",
            "test/documents",
            "test/xml",
            "test/parsed",
            "reports"
        ]
        
        for path in paths:
            os.makedirs(os.path.join(self.data_dir, path), exist_ok=True)
        
        # Create sample documents
        train_docs = os.path.join(self.data_dir, "train", "documents")
        test_docs = os.path.join(self.data_dir, "test", "documents")
        
        for i in range(3):
            with open(os.path.join(train_docs, f"doc_{i}.txt"), 'w') as f:
                f.write(f"This is training document {i} content.")
            
            with open(os.path.join(test_docs, f"doc_{i}.txt"), 'w') as f:
                f.write(f"This is test document {i} content.")
        
        # Create sample XML files
        train_xml = os.path.join(self.data_dir, "train", "xml")
        test_xml = os.path.join(self.data_dir, "test", "xml")
        
        for i in range(2):
            xml_content = f"""<?xml version="1.0"?>
            <document>
                <title>Document {i}</title>
                <content>This is XML content for document {i}</content>
            </document>"""
            
            with open(os.path.join(train_xml, f"doc_{i}.xml"), 'w') as f:
                f.write(xml_content)
            
            with open(os.path.join(test_xml, f"doc_{i}.xml"), 'w') as f:
                f.write(xml_content)
        
        # Create sample parsed documents
        parsed_data = {
            "doc_0": {"title": "Document 0", "content": "Parsed content 0"},
            "doc_1": {"title": "Document 1", "content": "Parsed content 1"}
        }
        
        import pickle
        with open(os.path.join(self.data_dir, "train", "parsed", "parsed_documents.pkl"), 'wb') as f:
            pickle.dump(parsed_data, f)
    
    def test_full_pipeline_execution_mocked(self):
        """Test full pipeline execution with mocked step methods."""
        # Mock all step methods to avoid actual processing
        mock_methods = [
            'pre_chunking_eda', 'doc_conversion', 'document_parsing', 
            'semantic_chunking', 'create_vector_embeddings', 'chunk_level_eda',
            'qc', 'export_artifacts'
        ]
        
        patches = []
        for method in mock_methods:
            patcher = patch.object(self.pipeline, method, return_value={"status": "success", "items_processed": 10})
            patches.append(patcher)
        
        # Start all patches
        for patcher in patches:
            patcher.start()
        
        try:
            # Run full pipeline
            result = self.pipeline.run_all()
            
            # Verify execution
            self.assertTrue(result)
            
            # Check that all steps were completed
            expected_steps = len(self.pipeline.STEP_DEFINITIONS)
            self.assertEqual(len(self.pipeline.state.completed_steps), expected_steps)
            
            # Verify no steps failed
            self.assertEqual(len(self.pipeline.state.failed_steps), 0)
            
            # Check that reports were generated
            self.assertIsNotNone(self.pipeline.generate_consolidated_json_summary())
            
        finally:
            # Stop all patches
            for patcher in patches:
                patcher.stop()
    
    def test_pipeline_execution_with_failures(self):
        """Test pipeline execution with step failures."""
        # Mock some steps to fail
        def mock_pre_chunking_eda(*args, **kwargs):
            return {"status": "success", "items_processed": 10}
        
        def mock_doc_conversion(*args, **kwargs):
            raise Exception("Mock conversion failure")
        
        with patch.object(self.pipeline, 'pre_chunking_eda', side_effect=mock_pre_chunking_eda), \
             patch.object(self.pipeline, 'doc_conversion', side_effect=mock_doc_conversion):
            
            # Run pipeline - should fail at doc_conversion
            result = self.pipeline.run_all()
            
            # Verify failure
            self.assertFalse(result)
            
            # Check that pre_chunking_eda completed but doc_conversion failed
            self.assertIn("pre_chunking_eda", self.pipeline.state.completed_steps)
            self.assertIn("doc_conversion", self.pipeline.state.failed_steps)
    
    def test_pipeline_state_persistence_across_restarts(self):
        """Test that pipeline state persists across restarts."""
        # Mock first step to succeed
        with patch.object(self.pipeline, 'pre_chunking_eda', return_value={"status": "success"}):
            # Run first step
            result = self.pipeline.run_single_step("pre_chunking_eda")
            self.assertTrue(result)
        
        # Create new pipeline instance (simulating restart)
        new_pipeline = PreprocessingPipeline(data_dir=self.data_dir)
        
        # Verify state was loaded
        self.assertIn("pre_chunking_eda", new_pipeline.state.completed_steps)
        self.assertEqual(new_pipeline.step_metadata["pre_chunking_eda"].status.value, "completed")


class TestPartialPipelineExecution(unittest.TestCase):
    """Integration tests for partial pipeline execution."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = os.path.join(self.temp_dir, "test_data")
        
        # Create basic test data structure
        os.makedirs(os.path.join(self.data_dir, "train", "documents"), exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, "test", "documents"), exist_ok=True)
        
        self.pipeline = PreprocessingPipeline(data_dir=self.data_dir)
    
    def tearDown(self):
        """Clean up after tests."""
        shutil.rmtree(self.temp_dir)
    
    def test_run_specific_steps(self):
        """Test running specific steps."""
        # Mock step methods
        mock_methods = ['pre_chunking_eda', 'doc_conversion', 'document_parsing']
        
        patches = []
        for method in mock_methods:
            patcher = patch.object(self.pipeline, method, return_value={"status": "success"})
            patches.append(patcher)
        
        # Start all patches
        for patcher in patches:
            patcher.start()
        
        try:
            # Run specific steps
            steps_to_run = ["pre_chunking_eda", "doc_conversion"]
            result = self.pipeline.run_specific_steps(steps_to_run)
            
            # Verify execution
            self.assertTrue(result)
            
            # Check that only specified steps were completed
            for step_id in steps_to_run:
                self.assertIn(step_id, self.pipeline.state.completed_steps)
            
            # Check that other steps were not run
            self.assertNotIn("document_parsing", self.pipeline.state.completed_steps)
            
        finally:
            # Stop all patches
            for patcher in patches:
                patcher.stop()
    
    def test_run_up_to_step(self):
        """Test running up to a specific step."""
        # Mock step methods
        mock_methods = ['pre_chunking_eda', 'doc_conversion', 'document_parsing']
        
        patches = []
        for method in mock_methods:
            patcher = patch.object(self.pipeline, method, return_value={"status": "success"})
            patches.append(patcher)
        
        # Start all patches
        for patcher in patches:
            patcher.start()
        
        try:
            # Run up to document_parsing
            result = self.pipeline.run_up_to_step("document_parsing")
            
            # Verify execution
            self.assertTrue(result)
            
            # Check that all steps up to document_parsing were completed
            expected_steps = ["pre_chunking_eda", "doc_conversion", "document_parsing"]
            for step_id in expected_steps:
                self.assertIn(step_id, self.pipeline.state.completed_steps)
            
            # Check that later steps were not run
            self.assertNotIn("semantic_chunking", self.pipeline.state.completed_steps)
            
        finally:
            # Stop all patches
            for patcher in patches:
                patcher.stop()
    
    def test_run_from_step(self):
        """Test running from a specific step."""
        # Pre-populate some completed steps
        self.pipeline.state.completed_steps = ["pre_chunking_eda", "doc_conversion"]
        
        # Mock remaining step methods
        mock_methods = ['document_parsing', 'semantic_chunking']
        
        patches = []
        for method in mock_methods:
            patcher = patch.object(self.pipeline, method, return_value={"status": "success"})
            patches.append(patcher)
        
        # Start all patches
        for patcher in patches:
            patcher.start()
        
        try:
            # Run from document_parsing
            result = self.pipeline.run_from_step("document_parsing")
            
            # Verify execution
            self.assertTrue(result)
            
            # Check that steps from document_parsing onwards were completed
            expected_new_steps = ["document_parsing", "semantic_chunking"]
            for step_id in expected_new_steps:
                self.assertIn(step_id, self.pipeline.state.completed_steps)
            
        finally:
            # Stop all patches
            for patcher in patches:
                patcher.stop()


class TestResumeAfterInterruption(unittest.TestCase):
    """Integration tests for resume functionality after interruption."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = os.path.join(self.temp_dir, "test_data")
        
        # Create basic test data structure
        os.makedirs(os.path.join(self.data_dir, "train", "documents"), exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, "test", "documents"), exist_ok=True)
        
        self.pipeline = PreprocessingPipeline(data_dir=self.data_dir)
    
    def tearDown(self):
        """Clean up after tests."""
        shutil.rmtree(self.temp_dir)
    
    def test_resume_from_last_successful_step(self):
        """Test resuming from the last successful step."""
        # Simulate partial completion
        self.pipeline.state.completed_steps = ["pre_chunking_eda", "doc_conversion"]
        self.pipeline._save_state()
        
        # Mock remaining step methods
        mock_methods = ['document_parsing', 'semantic_chunking']
        
        patches = []
        for method in mock_methods:
            patcher = patch.object(self.pipeline, method, return_value={"status": "success"})
            patches.append(patcher)
        
        # Start all patches
        for patcher in patches:
            patcher.start()
        
        try:
            # Resume pipeline
            result = self.pipeline.resume_pipeline()
            
            # Verify execution
            self.assertTrue(result)
            
            # Check that remaining steps were completed
            expected_total_steps = len(self.pipeline.STEP_DEFINITIONS)
            self.assertEqual(len(self.pipeline.state.completed_steps), expected_total_steps)
            
        finally:
            # Stop all patches
            for patcher in patches:
                patcher.stop()
    
    def test_resume_with_failed_step(self):
        """Test resuming when the last step failed."""
        # Simulate failure at document_parsing
        self.pipeline.state.completed_steps = ["pre_chunking_eda", "doc_conversion"]
        self.pipeline.state.failed_steps = ["document_parsing"]
        self.pipeline._save_state()
        
        # Mock step methods - document_parsing should succeed on retry
        with patch.object(self.pipeline, 'document_parsing', return_value={"status": "success"}):
            # Resume pipeline with force=True to retry failed step
            result = self.pipeline.resume_pipeline(force=True)
            
            # Verify execution
            self.assertTrue(result)
            
            # Check that failed step was retried and completed
            self.assertIn("document_parsing", self.pipeline.state.completed_steps)
            self.assertNotIn("document_parsing", self.pipeline.state.failed_steps)
    
    def test_resume_when_all_steps_completed(self):
        """Test resume when all steps are already completed."""
        # Mark all steps as completed
        all_steps = list(self.pipeline.STEP_DEFINITIONS.keys())
        self.pipeline.state.completed_steps = all_steps
        self.pipeline._save_state()
        
        # Resume pipeline
        result = self.pipeline.resume_pipeline()
        
        # Should return True (success) but not execute any steps
        self.assertTrue(result)


class TestParameterCombinations(unittest.TestCase):
    """Integration tests for various parameter combinations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = os.path.join(self.temp_dir, "test_data")
        
        # Create basic test data structure
        os.makedirs(os.path.join(self.data_dir, "train", "documents"), exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, "test", "documents"), exist_ok=True)
        
        self.pipeline = PreprocessingPipeline(data_dir=self.data_dir)
    
    def tearDown(self):
        """Clean up after tests."""
        shutil.rmtree(self.temp_dir)
    
    def test_different_chunk_sizes(self):
        """Test pipeline with different chunk sizes."""
        chunk_sizes = [100, 200, 500]
        
        for chunk_size in chunk_sizes:
            # Update chunking parameters
            self.pipeline.update_step_parameters(
                "semantic_chunking", 
                {"chunk_size": chunk_size}
            )
            
            # Verify parameters were updated
            self.assertEqual(
                self.pipeline.step_parameters["semantic_chunking"]["chunk_size"],
                chunk_size
            )
    
    def test_different_output_formats(self):
        """Test pipeline with different output formats."""
        output_formats = ["json", "markdown", "all"]
        
        for output_format in output_formats:
            # Update output format for all steps
            for step_id in self.pipeline.step_parameters:
                if "output_format" in self.pipeline.step_parameters[step_id]:
                    self.pipeline.update_step_parameters(
                        step_id, 
                        {"output_format": output_format}
                    )
    
    def test_retry_configurations(self):
        """Test different retry configurations."""
        # Test with high retry count
        self.pipeline.max_retries = 5
        self.pipeline.retry_delay = 1  # Short delay for testing
        
        # Mock a step that fails multiple times then succeeds
        failure_count = 0
        def mock_failing_step(*args, **kwargs):
            nonlocal failure_count
            failure_count += 1
            if failure_count < 3:
                raise Exception("Transient error")
            return {"status": "success"}
        
        with patch.object(self.pipeline, 'pre_chunking_eda', side_effect=mock_failing_step):
            # Run step - should succeed after retries
            result = self.pipeline.run_single_step("pre_chunking_eda")
            self.assertTrue(result)
            self.assertEqual(failure_count, 3)


class TestConfigurationIntegration(unittest.TestCase):
    """Integration tests for configuration management."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = os.path.join(self.temp_dir, "test_data")
        
        # Create basic test data structure
        os.makedirs(os.path.join(self.data_dir, "train", "documents"), exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, "test", "documents"), exist_ok=True)
        
        self.config_manager = ConfigurationManager()
    
    def tearDown(self):
        """Clean up after tests."""
        shutil.rmtree(self.temp_dir)
    
    def test_pipeline_with_configuration_file(self):
        """Test pipeline execution with configuration file."""
        # Create test configuration
        config = PipelineConfig(
            data_directory=self.data_dir,
            enable_progress_bars=False,
            enable_dashboard=False,
            log_level="WARNING"
        )
        
        # Save configuration
        config_path = os.path.join(self.temp_dir, "test_config.json")
        self.config_manager.save_config(config_path, config)
        
        # Create pipeline with configuration
        pipeline = PreprocessingPipeline(data_dir=self.data_dir, config_file=config_path)
        
        # Verify configuration was loaded
        self.assertEqual(pipeline.data_dir, self.data_dir)
    
    def test_configuration_validation(self):
        """Test configuration validation."""
        # Create configuration with potential issues
        config = PipelineConfig(
            data_directory="/nonexistent/path",
            reporting={"reports_directory": "/nonexistent/reports"}
        )
        
        # Validate configuration
        warnings = self.config_manager.validate_config(config)
        
        # Should have warnings about missing directories
        self.assertGreater(len(warnings), 0)
        warning_text = " ".join(warnings)
        self.assertIn("does not exist", warning_text)


class TestVisualizationIntegration(unittest.TestCase):
    """Integration tests for visualization features."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = os.path.join(self.temp_dir, "test_data")
        
        # Create basic test data structure
        os.makedirs(os.path.join(self.data_dir, "train", "documents"), exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, "test", "documents"), exist_ok=True)
        
        self.pipeline = PreprocessingPipeline(data_dir=self.data_dir)
        self.visualizer = PipelineVisualizer(self.pipeline.STEP_DEFINITIONS)
    
    def tearDown(self):
        """Clean up after tests."""
        shutil.rmtree(self.temp_dir)
    
    def test_diagram_generation(self):
        """Test pipeline diagram generation."""
        # Change to temp directory for output
        original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
        
        try:
            # Generate diagrams
            results = self.visualizer.generate_all_diagrams()
            
            # Check that diagrams were generated
            self.assertGreater(len(results), 0)
            
            # Check for Mermaid diagram file
            mermaid_file = Path("reports/pipeline_flow.mmd")
            if mermaid_file.exists():
                with open(mermaid_file, 'r') as f:
                    content = f.read()
                    self.assertIn("graph TD", content)
                    self.assertIn("pre_chunking_eda", content)
        
        finally:
            os.chdir(original_cwd)
    
    def test_progress_tracking_integration(self):
        """Test progress tracking integration."""
        # Mock pipeline execution with progress tracking
        with patch.object(self.pipeline, 'pre_chunking_eda', return_value={"status": "success"}):
            # Start progress tracking
            self.visualizer.start_progress_tracking(1)
            
            # Run a step
            result = self.pipeline.run_single_step("pre_chunking_eda")
            self.assertTrue(result)
            
            # Stop progress tracking
            self.visualizer.stop_progress_tracking()


class TestEndToEndWorkflow(unittest.TestCase):
    """End-to-end workflow integration tests."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = os.path.join(self.temp_dir, "test_data")
        
        # Create comprehensive test data
        self.setup_realistic_test_data()
        
        self.pipeline = PreprocessingPipeline(data_dir=self.data_dir)
    
    def tearDown(self):
        """Clean up after tests."""
        shutil.rmtree(self.temp_dir)
    
    def setup_realistic_test_data(self):
        """Set up realistic test data structure."""
        # Create directory structure
        paths = [
            "train/documents",
            "train/xml", 
            "train/parsed",
            "test/documents",
            "test/xml",
            "test/parsed",
            "reports"
        ]
        
        for path in paths:
            os.makedirs(os.path.join(self.data_dir, path), exist_ok=True)
        
        # Create sample documents with realistic content
        sample_content = """
        This is a sample scientific document that contains data citations.
        We used dataset XYZ from the European Space Agency.
        The methodology is described in detail in the methods section.
        Results show significant improvements over baseline approaches.
        """
        
        for subset in ["train", "test"]:
            docs_dir = os.path.join(self.data_dir, subset, "documents")
            for i in range(2):
                with open(os.path.join(docs_dir, f"paper_{i}.txt"), 'w') as f:
                    f.write(f"Paper {i}: {sample_content}")
    
    def test_complete_workflow_simulation(self):
        """Test complete workflow simulation with mocked external dependencies."""
        # Mock all external dependencies and step methods
        mock_methods = [
            'pre_chunking_eda', 'doc_conversion', 'document_parsing',
            'semantic_chunking', 'create_vector_embeddings', 'chunk_level_eda',
            'qc', 'export_artifacts'
        ]
        
        # Create realistic return values for each step
        mock_returns = {
            'pre_chunking_eda': {
                "status": "success",
                "items_processed": 4,
                "documents_analyzed": 4,
                "insights": ["Good document coverage", "Balanced train/test split"]
            },
            'doc_conversion': {
                "status": "success", 
                "items_processed": 2,
                "converted_files": 2,
                "conversion_rate": 50.0
            },
            'document_parsing': {
                "status": "success",
                "items_processed": 2,
                "parsed_documents": 2,
                "extraction_success_rate": 100.0
            },
            'semantic_chunking': {
                "status": "success",
                "items_processed": 10,
                "total_chunks": 10,
                "avg_chunk_size": 200
            }
        }
        
        patches = []
        for method in mock_methods:
            return_value = mock_returns.get(method, {"status": "success", "items_processed": 1})
            patcher = patch.object(self.pipeline, method, return_value=return_value)
            patches.append(patcher)
        
        # Start all patches
        for patcher in patches:
            patcher.start()
        
        try:
            # Run complete pipeline
            start_time = time.time()
            result = self.pipeline.run_all()
            end_time = time.time()
            
            # Verify successful completion
            self.assertTrue(result)
            
            # Verify all steps completed
            self.assertEqual(len(self.pipeline.state.completed_steps), len(self.pipeline.STEP_DEFINITIONS))
            self.assertEqual(len(self.pipeline.state.failed_steps), 0)
            
            # Verify timing
            self.assertLess(end_time - start_time, 10.0)  # Should complete quickly with mocks
            
            # Verify reports were generated
            json_summary = self.pipeline.generate_consolidated_json_summary()
            self.assertIn("pipeline_summary", json_summary)
            self.assertEqual(json_summary["pipeline_summary"]["success_rate"], 100.0)
            
            markdown_report = self.pipeline.generate_consolidated_markdown_report()
            self.assertIn("# Preprocessing Pipeline Report", markdown_report)
            self.assertIn("100.0%", markdown_report)  # Success rate
            
        finally:
            # Stop all patches
            for patcher in patches:
                patcher.stop()
    
    def test_workflow_with_realistic_failures(self):
        """Test workflow with realistic failure scenarios."""
        # Mock steps with realistic failure pattern
        def mock_pre_chunking_eda(*args, **kwargs):
            return {"status": "success", "items_processed": 4}
        
        def mock_doc_conversion(*args, **kwargs):
            return {"status": "success", "items_processed": 2}
        
        def mock_document_parsing(*args, **kwargs):
            # Simulate parsing failure
            raise Exception("XML parsing failed for malformed document")
        
        with patch.object(self.pipeline, 'pre_chunking_eda', side_effect=mock_pre_chunking_eda), \
             patch.object(self.pipeline, 'doc_conversion', side_effect=mock_doc_conversion), \
             patch.object(self.pipeline, 'document_parsing', side_effect=mock_document_parsing):
            
            # Run pipeline - should fail at document_parsing
            result = self.pipeline.run_all()
            
            # Verify failure handling
            self.assertFalse(result)
            
            # Check partial completion
            self.assertIn("pre_chunking_eda", self.pipeline.state.completed_steps)
            self.assertIn("doc_conversion", self.pipeline.state.completed_steps)
            self.assertIn("document_parsing", self.pipeline.state.failed_steps)
            
            # Verify error was properly categorized and logged
            parsing_metadata = self.pipeline.step_metadata["document_parsing"]
            self.assertIsNotNone(parsing_metadata.error_message)
            self.assertIn("parsing failed", parsing_metadata.error_message)


if __name__ == '__main__':
    # Run integration tests
    unittest.main(verbosity=2) 