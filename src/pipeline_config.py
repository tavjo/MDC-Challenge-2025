"""
Pipeline Configuration Management Module

This module provides configuration management for the preprocessing pipeline including:
- YAML/JSON configuration file support
- Configuration validation and schema
- Configuration templates for different use cases
- Configuration file generation from CLI arguments
"""

import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, ValidationError, field_validator
from enum import StrEnum
import argparse
from datetime import datetime


class ConfigFormat(StrEnum):
    """Configuration file formats."""
    JSON = "json"
    YAML = "yaml"
    YML = "yml"


class StepConfig(BaseModel):
    """Configuration for a single pipeline step."""
    enabled: bool = Field(default=True, description="Whether this step is enabled")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Step-specific parameters")
    timeout: Optional[int] = Field(default=None, description="Timeout in seconds for step execution")
    retry_count: int = Field(default=3, description="Number of retry attempts")
    retry_delay: int = Field(default=5, description="Delay between retries in seconds")
    
    @field_validator('timeout')
    @classmethod
    def validate_timeout(cls, v):
        if v is not None and v <= 0:
            raise ValueError("Timeout must be positive")
        return v
    
    @field_validator('retry_count')
    @classmethod
    def validate_retry_count(cls, v):
        if v < 0:
            raise ValueError("Retry count must be non-negative")
        return v


class ResourceConfig(BaseModel):
    """Resource monitoring and limits configuration."""
    memory_limit_gb: Optional[float] = Field(default=None, description="Memory limit in GB")
    cpu_limit_percent: Optional[float] = Field(default=None, description="CPU usage limit percentage")
    disk_limit_gb: Optional[float] = Field(default=None, description="Disk usage limit in GB")
    
    # Alert thresholds
    memory_alert_threshold: float = Field(default=85.0, description="Memory usage alert threshold percentage")
    cpu_alert_threshold: float = Field(default=90.0, description="CPU usage alert threshold percentage")
    disk_alert_threshold: float = Field(default=90.0, description="Disk usage alert threshold percentage")
    
    @field_validator('memory_alert_threshold', 'cpu_alert_threshold', 'disk_alert_threshold')
    @classmethod
    def validate_thresholds(cls, v):
        if v < 0 or v > 100:
            raise ValueError("Threshold must be between 0 and 100")
        return v


class ReportingConfig(BaseModel):
    """Reporting configuration."""
    output_formats: List[str] = Field(default=["json", "markdown"], description="Output formats for reports")
    save_reports: bool = Field(default=True, description="Whether to save reports to disk")
    reports_directory: str = Field(default="reports", description="Directory for saving reports")
    include_performance_metrics: bool = Field(default=True, description="Include performance metrics in reports")
    include_resource_usage: bool = Field(default=True, description="Include resource usage in reports")
    
    @field_validator('output_formats')
    @classmethod
    def validate_output_formats(cls, v):
        allowed_formats = ["json", "markdown", "html", "csv"]
        for format_name in v:
            if format_name not in allowed_formats:
                raise ValueError(f"Invalid output format: {format_name}. Allowed: {allowed_formats}")
        return v


class PipelineConfig(BaseModel):
    """Complete pipeline configuration."""
    # Basic settings
    name: str = Field(default="Preprocessing Pipeline", description="Pipeline name")
    description: str = Field(default="Data preprocessing pipeline", description="Pipeline description")
    version: str = Field(default="1.0.0", description="Configuration version")
    
    # Data configuration
    data_directory: str = Field(default="Data", description="Base data directory")
    
    # Step configurations
    steps: Dict[str, StepConfig] = Field(default_factory=dict, description="Step-specific configurations")
    
    # Resource configuration
    resources: ResourceConfig = Field(default_factory=ResourceConfig, description="Resource monitoring configuration")
    
    # Reporting configuration
    reporting: ReportingConfig = Field(default_factory=ReportingConfig, description="Reporting configuration")
    
    # Visualization settings
    enable_progress_bars: bool = Field(default=True, description="Enable progress bars during execution")
    enable_dashboard: bool = Field(default=False, description="Enable real-time status dashboard")
    generate_diagrams: bool = Field(default=True, description="Generate pipeline diagrams")
    
    # Logging configuration
    log_level: str = Field(default="INFO", description="Logging level")
    log_to_file: bool = Field(default=True, description="Save logs to file")
    log_directory: str = Field(default="logs", description="Directory for log files")
    
    # Metadata
    created_at: Optional[datetime] = Field(default=None, description="Configuration creation timestamp")
    updated_at: Optional[datetime] = Field(default=None, description="Configuration last update timestamp")
    
    @field_validator('log_level')
    @classmethod
    def validate_log_level(cls, v):
        allowed_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in allowed_levels:
            raise ValueError(f"Invalid log level: {v}. Allowed: {allowed_levels}")
        return v.upper()


class ConfigurationManager:
    """Manages pipeline configuration loading, validation, and saving."""
    
    def __init__(self):
        self.config: Optional[PipelineConfig] = None
        self.config_file_path: Optional[Path] = None
    
    def load_config(self, config_path: Union[str, Path]) -> PipelineConfig:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            Loaded and validated configuration
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValidationError: If config validation fails
            ValueError: If config format is not supported
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Determine file format
        file_format = self._detect_format(config_path)
        
        # Load configuration data
        with open(config_path, 'r', encoding='utf-8') as f:
            if file_format == ConfigFormat.JSON:
                config_data = json.load(f)
            elif file_format in [ConfigFormat.YAML, ConfigFormat.YML]:
                config_data = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported configuration format: {file_format}")
        
        # Validate and create configuration object
        try:
            self.config = PipelineConfig(**config_data)
            self.config_file_path = config_path
            
            # Update timestamps
            self.config.updated_at = datetime.now()
            if self.config.created_at is None:
                self.config.created_at = datetime.now()
            
            return self.config
            
        except ValidationError as e:
            raise ValidationError(f"Configuration validation failed: {e}")
    
    def save_config(self, config_path: Union[str, Path], config: Optional[PipelineConfig] = None) -> None:
        """
        Save configuration to file.
        
        Args:
            config_path: Path to save the configuration file
            config: Configuration to save (uses current config if None)
        """
        config_path = Path(config_path)
        config_to_save = config or self.config
        
        if config_to_save is None:
            raise ValueError("No configuration to save")
        
        # Update timestamp
        config_to_save.updated_at = datetime.now()
        
        # Determine file format
        file_format = self._detect_format(config_path)
        
        # Convert to dictionary
        config_dict = config_to_save.model_dump()
        
        # Handle datetime serialization
        for key, value in config_dict.items():
            if isinstance(value, datetime):
                config_dict[key] = value.isoformat()
        
        # Create parent directory if it doesn't exist
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        with open(config_path, 'w', encoding='utf-8') as f:
            if file_format == ConfigFormat.JSON:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
            elif file_format in [ConfigFormat.YAML, ConfigFormat.YML]:
                yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
            else:
                raise ValueError(f"Unsupported configuration format: {file_format}")
        
        self.config_file_path = config_path
    
    def _detect_format(self, config_path: Path) -> ConfigFormat:
        """Detect configuration file format from extension."""
        suffix = config_path.suffix.lower()
        
        if suffix == '.json':
            return ConfigFormat.JSON
        elif suffix in ['.yaml', '.yml']:
            return ConfigFormat.YAML
        else:
            raise ValueError(f"Unsupported configuration file extension: {suffix}")
    
    def create_default_config(self) -> PipelineConfig:
        """Create a default configuration."""
        self.config = PipelineConfig(
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        return self.config
    
    def validate_config(self, config: PipelineConfig) -> List[str]:
        """
        Validate configuration and return list of warnings.
        
        Args:
            config: Configuration to validate
            
        Returns:
            List of validation warnings
        """
        warnings = []
        
        # Check data directory existence
        data_path = Path(config.data_directory)
        if not data_path.exists():
            warnings.append(f"Data directory does not exist: {config.data_directory}")
        
        # Check reports directory
        reports_path = Path(config.reporting.reports_directory)
        if not reports_path.exists():
            warnings.append(f"Reports directory does not exist: {config.reporting.reports_directory}")
        
        # Check logs directory
        logs_path = Path(config.log_directory)
        if not logs_path.exists():
            warnings.append(f"Logs directory does not exist: {config.log_directory}")
        
        # Validate step configurations
        for step_id, step_config in config.steps.items():
            if step_config.timeout and step_config.timeout < 60:
                warnings.append(f"Step '{step_id}' has very short timeout: {step_config.timeout}s")
            
            if step_config.retry_count > 10:
                warnings.append(f"Step '{step_id}' has high retry count: {step_config.retry_count}")
        
        # Resource configuration warnings
        if config.resources.memory_alert_threshold > 95:
            warnings.append("Memory alert threshold is very high (>95%)")
        
        if config.resources.cpu_alert_threshold > 95:
            warnings.append("CPU alert threshold is very high (>95%)")
        
        return warnings
    
    def get_step_config(self, step_id: str) -> StepConfig:
        """Get configuration for a specific step."""
        if self.config is None:
            raise ValueError("No configuration loaded")
        
        if step_id not in self.config.steps:
            # Return default configuration for unknown steps
            return StepConfig()
        
        return self.config.steps[step_id]
    
    def update_step_config(self, step_id: str, **kwargs) -> None:
        """Update configuration for a specific step."""
        if self.config is None:
            raise ValueError("No configuration loaded")
        
        if step_id not in self.config.steps:
            self.config.steps[step_id] = StepConfig()
        
        # Update step configuration
        step_config = self.config.steps[step_id]
        for key, value in kwargs.items():
            if hasattr(step_config, key):
                setattr(step_config, key, value)
            else:
                # Add to parameters if not a direct attribute
                step_config.parameters[key] = value
        
        self.config.updated_at = datetime.now()


class ConfigurationTemplates:
    """Provides pre-defined configuration templates."""
    
    @staticmethod
    def get_development_config() -> PipelineConfig:
        """Get development configuration template."""
        return PipelineConfig(
            name="Development Pipeline",
            description="Configuration for development environment",
            log_level="DEBUG",
            enable_progress_bars=True,
            enable_dashboard=True,
            generate_diagrams=True,
            resources=ResourceConfig(
                memory_alert_threshold=70.0,
                cpu_alert_threshold=80.0,
                disk_alert_threshold=85.0
            ),
            reporting=ReportingConfig(
                output_formats=["json", "markdown"],
                include_performance_metrics=True,
                include_resource_usage=True
            ),
            steps={
                "pre_chunking_eda": StepConfig(
                    enabled=True,
                    parameters={"show_plots": True, "detailed_analysis": True}
                ),
                "doc_conversion": StepConfig(
                    enabled=True,
                    timeout=1800,  # 30 minutes
                    retry_count=2
                ),
                "document_parsing": StepConfig(
                    enabled=True,
                    timeout=3600,  # 1 hour
                    retry_count=3
                ),
                "semantic_chunking": StepConfig(
                    enabled=True,
                    parameters={"chunk_size": 200, "chunk_overlap": 20}
                )
            },
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
    
    @staticmethod
    def get_production_config() -> PipelineConfig:
        """Get production configuration template."""
        return PipelineConfig(
            name="Production Pipeline",
            description="Configuration for production environment",
            log_level="INFO",
            enable_progress_bars=False,
            enable_dashboard=False,
            generate_diagrams=False,
            resources=ResourceConfig(
                memory_alert_threshold=85.0,
                cpu_alert_threshold=90.0,
                disk_alert_threshold=90.0
            ),
            reporting=ReportingConfig(
                output_formats=["json"],
                include_performance_metrics=True,
                include_resource_usage=False
            ),
            steps={
                "pre_chunking_eda": StepConfig(
                    enabled=True,
                    parameters={"show_plots": False, "detailed_analysis": False}
                ),
                "doc_conversion": StepConfig(
                    enabled=True,
                    timeout=3600,  # 1 hour
                    retry_count=3
                ),
                "document_parsing": StepConfig(
                    enabled=True,
                    timeout=7200,  # 2 hours
                    retry_count=3
                ),
                "semantic_chunking": StepConfig(
                    enabled=True,
                    parameters={"chunk_size": 200, "chunk_overlap": 20}
                )
            },
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
    
    @staticmethod
    def get_fast_config() -> PipelineConfig:
        """Get fast processing configuration template."""
        return PipelineConfig(
            name="Fast Processing Pipeline",
            description="Configuration for fast processing (reduced quality)",
            log_level="WARNING",
            enable_progress_bars=True,
            enable_dashboard=False,
            generate_diagrams=False,
            resources=ResourceConfig(
                memory_alert_threshold=90.0,
                cpu_alert_threshold=95.0,
                disk_alert_threshold=95.0
            ),
            reporting=ReportingConfig(
                output_formats=["json"],
                include_performance_metrics=False,
                include_resource_usage=False
            ),
            steps={
                "pre_chunking_eda": StepConfig(
                    enabled=True,
                    parameters={"show_plots": False, "detailed_analysis": False, "sample_size": 100}
                ),
                "doc_conversion": StepConfig(
                    enabled=True,
                    timeout=900,  # 15 minutes
                    retry_count=1
                ),
                "document_parsing": StepConfig(
                    enabled=True,
                    timeout=1800,  # 30 minutes
                    retry_count=1
                ),
                "semantic_chunking": StepConfig(
                    enabled=True,
                    parameters={"chunk_size": 100, "chunk_overlap": 10}
                )
            },
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
    
    @staticmethod
    def get_available_templates() -> List[str]:
        """Get list of available configuration templates."""
        return ["development", "production", "fast"]
    
    @staticmethod
    def get_template(template_name: str) -> PipelineConfig:
        """Get a specific configuration template."""
        templates = {
            "development": ConfigurationTemplates.get_development_config,
            "production": ConfigurationTemplates.get_production_config,
            "fast": ConfigurationTemplates.get_fast_config
        }
        
        if template_name not in templates:
            raise ValueError(f"Unknown template: {template_name}. Available: {list(templates.keys())}")
        
        return templates[template_name]()


class CLIConfigGenerator:
    """Generates configuration from CLI arguments."""
    
    @staticmethod
    def generate_config_from_args(args: argparse.Namespace) -> PipelineConfig:
        """Generate configuration from CLI arguments."""
        config = PipelineConfig()
        
        # Basic settings
        if hasattr(args, 'data_dir'):
            config.data_directory = args.data_dir
        
        if hasattr(args, 'log_level'):
            config.log_level = args.log_level
        
        if hasattr(args, 'verbose') and args.verbose:
            config.log_level = "DEBUG"
        
        # Progress and visualization
        if hasattr(args, 'no_progress'):
            config.enable_progress_bars = not args.no_progress
        
        if hasattr(args, 'dashboard'):
            config.enable_dashboard = args.dashboard
        
        if hasattr(args, 'generate_diagrams'):
            config.generate_diagrams = args.generate_diagrams
        
        # Output format
        if hasattr(args, 'output_format'):
            config.reporting.output_formats = args.output_format
        
        # Step-specific parameters
        step_params = {}
        
        # Chunking parameters
        if hasattr(args, 'chunk_size'):
            step_params["semantic_chunking"] = StepConfig(
                parameters={"chunk_size": args.chunk_size}
            )
        
        if hasattr(args, 'chunk_overlap'):
            if "semantic_chunking" in step_params:
                step_params["semantic_chunking"].parameters["chunk_overlap"] = args.chunk_overlap
            else:
                step_params["semantic_chunking"] = StepConfig(
                    parameters={"chunk_overlap": args.chunk_overlap}
                )
        
        config.steps = step_params
        
        # Set timestamps
        config.created_at = datetime.now()
        config.updated_at = datetime.now()
        
        return config
    
    @staticmethod
    def save_config_from_args(args: argparse.Namespace, output_path: str) -> None:
        """Generate and save configuration from CLI arguments."""
        config = CLIConfigGenerator.generate_config_from_args(args)
        
        config_manager = ConfigurationManager()
        config_manager.save_config(output_path, config)
        
        print(f"Configuration saved to: {output_path}")


def create_sample_configs():
    """Create sample configuration files for all templates."""
    templates = ConfigurationTemplates.get_available_templates()
    config_manager = ConfigurationManager()
    
    output_dir = Path("configs")
    output_dir.mkdir(exist_ok=True)
    
    for template_name in templates:
        config = ConfigurationTemplates.get_template(template_name)
        
        # Save as both JSON and YAML
        json_path = output_dir / f"{template_name}_config.json"
        yaml_path = output_dir / f"{template_name}_config.yaml"
        
        config_manager.save_config(json_path, config)
        config_manager.save_config(yaml_path, config)
        
        print(f"Created {template_name} configuration templates:")
        print(f"  - {json_path}")
        print(f"  - {yaml_path}")


if __name__ == "__main__":
    # Create sample configuration files
    create_sample_configs() 