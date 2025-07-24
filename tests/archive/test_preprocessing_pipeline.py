"""
Unit tests for the preprocessing pipeline.

This module contains comprehensive tests for:
- Step management and execution
- Dependency validation
- Error handling and recovery
- Report generation functionality
- Resource monitoring
- Configuration management
"""

import unittest
import tempfile
import shutil
import json
import pickle
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from preprocessing import PreprocessingPipeline, StepStatus, ErrorType
from src.pipeline_config import ConfigurationManager, PipelineConfig, StepConfig
from src.pipeline_visualization import PipelineVisualizer


class TestPreprocessingPipeline(unittest.TestCase):
    """Test cases for the PreprocessingPipeline class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = os.path.join(self.temp_dir, "test_data")
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Create basic directory structure
        os.makedirs(os.path.join(self.data_dir, "train", "documents"), exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, "test", "documents"), exist_ok=True)
        
        self.pipeline = PreprocessingPipeline(data_dir=self.data_dir)
    
    def tearDown(self):
        """Clean up after each test method."""
        shutil.rmtree(self.temp_dir)
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        self.assertIsNotNone(self.pipeline)
        self.assertEqual(self.pipeline.data_dir, self.data_dir)
        self.assertIsInstance(self.pipeline.step_metadata, dict)
        self.assertIsInstance(self.pipeline.state, type(self.pipeline.state))
        
        # Check that all steps are initialized
        expected_steps = [
            "pre_chunking_eda", "doc_conversion", "document_parsing", 
            "semantic_chunking", "vector_embeddings", "chunk_level_eda", 
            "qc", "export_artifacts"
        ]
        for step_id in expected_steps:
            self.assertIn(step_id, self.pipeline.STEP_DEFINITIONS)
            self.assertIn(step_id, self.pipeline.step_metadata)
    
    def test_step_order_generation(self):
        """Test that step order is generated correctly based on dependencies."""
        order = self.pipeline.get_step_order()
        
        # Check that dependencies are respected
        for i, step_id in enumerate(order):
            step_def = self.pipeline.STEP_DEFINITIONS[step_id]
            for dep in step_def["dependencies"]:
                dep_index = order.index(dep)
                self.assertLess(dep_index, i, 
                               f"Dependency {dep} should come before {step_id}")
    
    def test_step_prerequisite_validation(self):
        """Test step prerequisite validation."""
        # Test that first step (no dependencies) passes validation
        self.assertTrue(self.pipeline.validate_step_prerequisites("pre_chunking_eda"))
        
        # Test that dependent step fails validation when dependency not completed
        self.assertFalse(self.pipeline.validate_step_prerequisites("doc_conversion"))
        
        # Test that dependent step passes validation when dependency completed
        self.pipeline.state.completed_steps.append("pre_chunking_eda")
        self.assertTrue(self.pipeline.validate_step_prerequisites("doc_conversion"))
    
    def test_data_directory_validation(self):
        """Test data directory structure validation."""
        # Should pass with basic structure
        self.assertTrue(self.pipeline.validate_data_directory_structure())
        
        # Test with missing data directory
        pipeline_bad = PreprocessingPipeline(data_dir="/nonexistent/path")
        self.assertFalse(pipeline_bad.validate_data_directory_structure())
    
    def test_file_dependency_validation(self):
        """Test file dependency validation for steps."""
        # Create test files for pre_chunking_eda
        with open(os.path.join(self.data_dir, "train", "documents", "test.txt"), 'w') as f:
            f.write("test content")
        
        # Should pass for pre_chunking_eda with basic files
        self.assertTrue(self.pipeline.validate_file_dependencies("pre_chunking_eda"))
        
        # Should fail for document_parsing without XML files
        self.assertFalse(self.pipeline.validate_file_dependencies("document_parsing"))
    
    def test_error_categorization(self):
        """Test error categorization functionality."""
        # Test transient errors
        network_error = Exception("Connection timeout")
        self.assertEqual(self.pipeline.categorize_error(network_error), ErrorType.TRANSIENT)
        
        # Test fatal errors
        file_error = Exception("No such file or directory")
        self.assertEqual(self.pipeline.categorize_error(file_error), ErrorType.FATAL)
        
        # Test recoverable errors (default case)
        generic_error = Exception("Some other error")
        self.assertEqual(self.pipeline.categorize_error(generic_error), ErrorType.RECOVERABLE)
    
    def test_checkpoint_creation(self):
        """Test checkpoint creation and state persistence."""
        step_id = "pre_chunking_eda"
        
        # Create checkpoint
        self.pipeline.create_checkpoint(step_id)
        
        # Verify checkpoint was created
        self.assertIn(step_id, self.pipeline.state.checkpoints)
        self.assertIsInstance(self.pipeline.state.checkpoints[step_id], datetime)
    
    def test_step_rollback(self):
        """Test step rollback functionality."""
        step_id = "pre_chunking_eda"
        
        # Mark step as completed
        self.pipeline.state.completed_steps.append(step_id)
        self.pipeline.step_metadata[step_id].status = StepStatus.COMPLETED
        
        # Rollback the step
        result = self.pipeline.rollback_step(step_id)
        
        # Verify rollback
        self.assertTrue(result)
        self.assertNotIn(step_id, self.pipeline.state.completed_steps)
        self.assertEqual(self.pipeline.step_metadata[step_id].status, StepStatus.NOT_STARTED)
    
    def test_resource_monitoring(self):
        """Test resource monitoring functionality."""
        resources = self.pipeline.monitor_resources()
        
        # Check that resource metrics are returned
        self.assertIsInstance(resources, dict)
        expected_keys = ['cpu_percent', 'memory_percent', 'disk_percent']
        for key in expected_keys:
            self.assertIn(key, resources)
    
    def test_file_size_tracking(self):
        """Test file size tracking functionality."""
        # Create test file
        test_file = os.path.join(self.temp_dir, "test.txt")
        with open(test_file, 'w') as f:
            f.write("test content")
        
        # Test file size tracking
        file_sizes = self.pipeline.get_file_sizes([test_file])
        self.assertIn(test_file, file_sizes)
        self.assertGreater(file_sizes[test_file], 0)
    
    def test_throughput_calculation(self):
        """Test throughput calculation."""
        # Test normal case
        throughput = self.pipeline.calculate_throughput(100, 10.0)
        self.assertEqual(throughput, 10.0)
        
        # Test edge case (zero duration)
        throughput = self.pipeline.calculate_throughput(100, 0)
        self.assertEqual(throughput, 0.0)
    
    def test_parameter_management(self):
        """Test step parameter management."""
        step_id = "semantic_chunking"
        
        # Test parameter update
        new_params = {"chunk_size": 500, "chunk_overlap": 50}
        self.pipeline.update_step_parameters(step_id, new_params)
        
        # Verify parameters were updated
        self.assertEqual(self.pipeline.step_parameters[step_id]["chunk_size"], 500)
        self.assertEqual(self.pipeline.step_parameters[step_id]["chunk_overlap"], 50)
    
    def test_step_execution_order(self):
        """Test that steps are executed in correct order."""
        # Mock step execution to avoid actual processing
        with patch.object(self.pipeline, '_execute_step_method') as mock_execute:
            mock_execute.return_value = {"status": "success"}
            
            # Mock validation methods to return True
            with patch.object(self.pipeline, 'validate_data_directory_structure', return_value=True), \
                 patch.object(self.pipeline, 'validate_file_dependencies', return_value=True), \
                 patch.object(self.pipeline, 'validate_step_input_format', return_value=True), \
                 patch.object(self.pipeline, 'monitor_resources', return_value={}):
                
                # Run first two steps
                result1 = self.pipeline.run_single_step("pre_chunking_eda")
                self.assertTrue(result1)
                
                result2 = self.pipeline.run_single_step("doc_conversion")
                self.assertTrue(result2)
                
                # Verify execution order
                self.assertIn("pre_chunking_eda", self.pipeline.state.completed_steps)
                self.assertIn("doc_conversion", self.pipeline.state.completed_steps)
    
    def test_json_summary_generation(self):
        """Test JSON summary generation."""
        # Mark some steps as completed
        self.pipeline.state.completed_steps = ["pre_chunking_eda", "doc_conversion"]
        self.pipeline.step_metadata["pre_chunking_eda"].status = StepStatus.COMPLETED
        self.pipeline.step_metadata["pre_chunking_eda"].duration = 10.5
        
        # Generate JSON summary
        summary = self.pipeline.generate_consolidated_json_summary()
        
        # Verify summary structure
        self.assertIn("pipeline_summary", summary)
        self.assertIn("step_details", summary)
        self.assertIn("performance_summary", summary)
        self.assertEqual(summary["pipeline_summary"]["completed_steps"], 2)
    
    def test_markdown_report_generation(self):
        """Test markdown report generation."""
        # Mark some steps as completed
        self.pipeline.state.completed_steps = ["pre_chunking_eda"]
        self.pipeline.step_metadata["pre_chunking_eda"].status = StepStatus.COMPLETED
        self.pipeline.step_metadata["pre_chunking_eda"].duration = 10.5
        
        # Generate markdown report
        report = self.pipeline.generate_consolidated_markdown_report()
        
        # Verify report content
        self.assertIn("# Preprocessing Pipeline Report", report)
        self.assertIn("Executive Summary", report)
        self.assertIn("Step Details", report)
        self.assertIn("✅", report)  # Check for completed step emoji
    
    def test_state_persistence(self):
        """Test pipeline state persistence."""
        # Add some state data
        self.pipeline.state.completed_steps = ["pre_chunking_eda"]
        self.pipeline.state.current_step = "doc_conversion"
        
        # Save state
        self.pipeline._save_state()
        
        # Create new pipeline instance
        new_pipeline = PreprocessingPipeline(data_dir=self.data_dir)
        
        # Verify state was loaded
        self.assertEqual(new_pipeline.state.completed_steps, ["pre_chunking_eda"])
        self.assertEqual(new_pipeline.state.current_step, "doc_conversion")


class TestStepDependencyValidation(unittest.TestCase):
    """Test cases for step dependency validation logic."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.pipeline = PreprocessingPipeline(data_dir=self.temp_dir)
    
    def tearDown(self):
        """Clean up after tests."""
        shutil.rmtree(self.temp_dir)
    
    def test_circular_dependency_detection(self):
        """Test that circular dependencies are handled properly."""
        # Create a mock step definition with circular dependency
        original_definitions = self.pipeline.STEP_DEFINITIONS.copy()
        
        # Add circular dependency (for testing purposes)
        test_definitions = {
            "step_a": {"dependencies": ["step_b"], "name": "Step A"},
            "step_b": {"dependencies": ["step_a"], "name": "Step B"}
        }
        
        # Temporarily replace step definitions
        self.pipeline.STEP_DEFINITIONS = test_definitions
        
        # This should not crash but should return empty list or handle gracefully
        order = self.pipeline.get_step_order()
        
        # Restore original definitions
        self.pipeline.STEP_DEFINITIONS = original_definitions
    
    def test_complex_dependency_chain(self):
        """Test complex dependency chains are resolved correctly."""
        order = self.pipeline.get_step_order()
        
        # Verify specific known dependencies
        pre_chunking_idx = order.index("pre_chunking_eda")
        doc_conversion_idx = order.index("doc_conversion")
        doc_parsing_idx = order.index("document_parsing")
        semantic_chunking_idx = order.index("semantic_chunking")
        
        # Check order is correct
        self.assertLess(pre_chunking_idx, doc_conversion_idx)
        self.assertLess(doc_conversion_idx, doc_parsing_idx)
        self.assertLess(doc_parsing_idx, semantic_chunking_idx)
    
    def test_missing_dependency_handling(self):
        """Test handling of missing dependencies."""
        # Test with a step that depends on a non-existent step
        result = self.pipeline.validate_step_prerequisites("nonexistent_step")
        self.assertFalse(result)


class TestErrorHandlingAndRecovery(unittest.TestCase):
    """Test cases for error handling and recovery mechanisms."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.pipeline = PreprocessingPipeline(data_dir=self.temp_dir)
    
    def tearDown(self):
        """Clean up after tests."""
        shutil.rmtree(self.temp_dir)
    
    def test_retry_logic(self):
        """Test retry logic for failed steps."""
        step_id = "pre_chunking_eda"
        
        # Mock step method to fail first time, succeed second time
        call_count = 0
        def mock_execute(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Transient error")
            return {"status": "success"}
        
        with patch.object(self.pipeline, '_execute_step_method', side_effect=mock_execute), \
             patch.object(self.pipeline, 'validate_data_directory_structure', return_value=True), \
             patch.object(self.pipeline, 'validate_file_dependencies', return_value=True), \
             patch.object(self.pipeline, 'validate_step_input_format', return_value=True), \
             patch.object(self.pipeline, 'monitor_resources', return_value={}):
            
            # Run step - should succeed on retry
            result = self.pipeline.run_single_step(step_id)
            self.assertTrue(result)
            self.assertEqual(call_count, 2)  # Should have been called twice
    
    def test_fatal_error_no_retry(self):
        """Test that fatal errors don't trigger retries."""
        step_id = "pre_chunking_eda"
        
        # Mock step method to raise fatal error
        def mock_execute(*args, **kwargs):
            raise Exception("No such file or directory")
        
        with patch.object(self.pipeline, '_execute_step_method', side_effect=mock_execute), \
             patch.object(self.pipeline, 'validate_data_directory_structure', return_value=True), \
             patch.object(self.pipeline, 'validate_file_dependencies', return_value=True), \
             patch.object(self.pipeline, 'validate_step_input_format', return_value=True), \
             patch.object(self.pipeline, 'monitor_resources', return_value={}):
            
            # Run step - should fail without retry
            result = self.pipeline.run_single_step(step_id)
            self.assertFalse(result)
            self.assertEqual(self.pipeline.step_metadata[step_id].error_type, ErrorType.FATAL)
    
    def test_resource_alert_generation(self):
        """Test resource alert generation."""
        # Mock high resource usage
        high_usage_resources = {
            'memory_percent': 90.0,
            'cpu_percent': 95.0,
            'disk_percent': 88.0
        }
        
        # Check resource thresholds
        self.pipeline._check_resource_thresholds(high_usage_resources)
        
        # Verify alerts were generated
        self.assertGreater(len(self.pipeline.state.resource_alerts), 0)
        
        # Check alert types
        alert_types = [alert.alert_type for alert in self.pipeline.state.resource_alerts]
        self.assertIn('memory', alert_types)
        self.assertIn('cpu', alert_types)


class TestReportGeneration(unittest.TestCase):
    """Test cases for report generation functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.pipeline = PreprocessingPipeline(data_dir=self.temp_dir)
        
        # Set up some mock data
        self.pipeline.state.completed_steps = ["pre_chunking_eda", "doc_conversion"]
        self.pipeline.step_metadata["pre_chunking_eda"].status = StepStatus.COMPLETED
        self.pipeline.step_metadata["pre_chunking_eda"].duration = 15.7
        self.pipeline.step_metadata["doc_conversion"].status = StepStatus.COMPLETED
        self.pipeline.step_metadata["doc_conversion"].duration = 42.3
    
    def tearDown(self):
        """Clean up after tests."""
        shutil.rmtree(self.temp_dir)
    
    def test_json_report_structure(self):
        """Test JSON report structure and content."""
        summary = self.pipeline.generate_consolidated_json_summary()
        
        # Check required sections
        required_sections = [
            "pipeline_summary", "step_details", "performance_summary",
            "resource_alerts", "cross_step_analysis", "generated_at"
        ]
        
        for section in required_sections:
            self.assertIn(section, summary)
        
        # Check pipeline summary content
        pipeline_summary = summary["pipeline_summary"]
        self.assertEqual(pipeline_summary["completed_steps"], 2)
        self.assertEqual(pipeline_summary["total_steps"], len(self.pipeline.STEP_DEFINITIONS))
        self.assertGreater(pipeline_summary["success_rate"], 0)
    
    def test_markdown_report_formatting(self):
        """Test markdown report formatting."""
        report = self.pipeline.generate_consolidated_markdown_report()
        
        # Check markdown structure
        self.assertIn("# Preprocessing Pipeline Report", report)
        self.assertIn("## Executive Summary", report)
        self.assertIn("## Step Details", report)
        
        # Check that step status is properly formatted
        self.assertIn("✅", report)  # Completed step emoji
        
        # Check that metrics are included
        self.assertIn("15.7", report)  # Duration for pre_chunking_eda
        self.assertIn("42.3", report)  # Duration for doc_conversion
    
    def test_report_file_generation(self):
        """Test report file generation and saving."""
        # Create reports directory
        reports_dir = os.path.join(self.temp_dir, "reports")
        os.makedirs(reports_dir, exist_ok=True)
        
        # Change to temp directory for testing
        original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
        
        try:
            # Generate consolidated reports
            self.pipeline.generate_consolidated_reports()
            
            # Check that files were created
            reports_path = Path("reports")
            json_files = list(reports_path.glob("preprocessing_pipeline_*.json"))
            markdown_files = list(reports_path.glob("preprocessing_pipeline_*.md"))
            
            self.assertGreater(len(json_files), 0)
            self.assertGreater(len(markdown_files), 0)
            
            # Check report index file
            index_file = reports_path / "report_index.md"
            self.assertTrue(index_file.exists())
            
        finally:
            os.chdir(original_cwd)
    
    def test_performance_recommendations(self):
        """Test performance recommendations generation."""
        # Create performance metrics with issues
        from preprocessing import PerformanceMetrics
        
        metrics = PerformanceMetrics(
            execution_time=5000,  # High execution time
            memory_usage={"memory_percent": 90.0},  # High memory usage
            cpu_usage=85.0,  # High CPU usage
            disk_usage={"disk_percent": 88.0},  # High disk usage
            throughput=0.5  # Low throughput
        )
        
        recommendations = self.pipeline.generate_performance_recommendations(metrics)
        
        # Check that recommendations were generated
        self.assertGreater(len(recommendations), 0)
        
        # Check specific recommendations
        rec_text = " ".join(recommendations).lower()
        self.assertIn("memory", rec_text)
        self.assertIn("cpu", rec_text)
        self.assertIn("throughput", rec_text)
    
    def test_cross_step_analysis(self):
        """Test cross-step analysis generation."""
        analysis = self.pipeline._generate_cross_step_analysis()
        
        # Check that analysis contains expected sections
        self.assertIn("step_flow", analysis)
        self.assertIn("execution_order", analysis["step_flow"])
        self.assertIn("completed_steps", analysis["step_flow"])
        
        # If we have completed steps with duration, check duration analysis
        if any(meta.duration for meta in self.pipeline.step_metadata.values()):
            self.assertIn("total_pipeline_duration", analysis)
            self.assertIn("average_step_duration", analysis)


if __name__ == '__main__':
    unittest.main() 