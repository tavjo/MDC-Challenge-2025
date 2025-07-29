"""
Pipeline Visualization Module

This module provides visualization tools for the preprocessing pipeline including:
- Console progress bars
- Step dependency visualization
- Pipeline flow diagrams
- Real-time status dashboard
"""

import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import threading
from enum import StrEnum

try:
    from rich.console import Console
    from rich.progress import Progress, TaskID, BarColumn, TextColumn, TimeRemainingColumn
    from rich.live import Live
    from rich.layout import Layout
    from rich.panel import Panel
    from rich.table import Table
    from rich.tree import Tree
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.patches import FancyBboxPatch
    import networkx as nx
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class ProgressBarManager:
    """Manages console progress bars for pipeline steps."""
    
    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console()
        self.progress = None
        self.tasks: Dict[str, TaskID] = {}
        self.active = False
        
    def start(self, total_steps: int):
        """Start the progress bar system."""
        if not RICH_AVAILABLE:
            return
            
        self.progress = Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeRemainingColumn(),
            console=self.console
        )
        self.progress.start()
        self.active = True
        
        # Add main pipeline task
        self.tasks["main"] = self.progress.add_task(
            "Pipeline Progress", total=total_steps
        )
    
    def add_step(self, step_id: str, description: str, total: int = 100):
        """Add a step to the progress tracking."""
        if not self.active or not RICH_AVAILABLE:
            return
            
        self.tasks[step_id] = self.progress.add_task(
            description, total=total
        )
    
    def update_step(self, step_id: str, advance: int = 1):
        """Update progress for a specific step."""
        if not self.active or not RICH_AVAILABLE:
            return
            
        if step_id in self.tasks:
            self.progress.update(self.tasks[step_id], advance=advance)
    
    def complete_step(self, step_id: str):
        """Mark a step as completed."""
        if not self.active or not RICH_AVAILABLE:
            return
            
        if step_id in self.tasks:
            self.progress.update(self.tasks[step_id], completed=True)
            # Update main pipeline progress
            self.progress.update(self.tasks["main"], advance=1)
    
    def fail_step(self, step_id: str, error_msg: str):
        """Mark a step as failed."""
        if not self.active or not RICH_AVAILABLE:
            return
            
        if step_id in self.tasks:
            self.progress.update(
                self.tasks[step_id], 
                description=f"[red]FAILED: {error_msg}"
            )
    
    def stop(self):
        """Stop the progress bar system."""
        if self.progress and self.active:
            self.progress.stop()
            self.active = False


class PipelineFlowDiagram:
    """Generates pipeline flow diagrams."""
    
    def __init__(self, step_definitions: Dict[str, Dict[str, Any]]):
        self.step_definitions = step_definitions
    
    def generate_dependency_graph(self, output_path: str = "pipeline_dependencies.png"):
        """Generate a dependency graph visualization."""
        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib not available. Cannot generate dependency graph.")
            return False
            
        # Create directed graph
        G = nx.DiGraph()
        
        # Add nodes
        for step_id, step_def in self.step_definitions.items():
            G.add_node(step_id, label=step_def["name"])
        
        # Add edges based on dependencies
        for step_id, step_def in self.step_definitions.items():
            for dep in step_def["dependencies"]:
                if dep in self.step_definitions:
                    G.add_edge(dep, step_id)
        
        # Create layout
        plt.figure(figsize=(14, 10))
        pos = nx.spring_layout(G, k=3, iterations=50)
        
        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos, 
            node_color='lightblue', 
            node_size=3000,
            alpha=0.7
        )
        
        # Draw edges
        nx.draw_networkx_edges(
            G, pos, 
            edge_color='gray', 
            arrows=True, 
            arrowsize=20,
            arrowstyle='->'
        )
        
        # Add labels
        labels = {step_id: step_def["name"] for step_id, step_def in self.step_definitions.items()}
        nx.draw_networkx_labels(
            G, pos, 
            labels, 
            font_size=8,
            font_weight='bold'
        )
        
        plt.title("Pipeline Step Dependencies", fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Dependency graph saved to: {output_path}")
        return True
    
    def generate_mermaid_diagram(self) -> str:
        """Generate a Mermaid diagram of the pipeline flow."""
        mermaid_code = "graph TD\n"
        
        # Add nodes
        for step_id, step_def in self.step_definitions.items():
            safe_name = step_def["name"].replace(" ", "_")
            mermaid_code += f'    {step_id}["{step_def["name"]}"]\n'
        
        # Add dependencies
        for step_id, step_def in self.step_definitions.items():
            for dep in step_def["dependencies"]:
                if dep in self.step_definitions:
                    mermaid_code += f"    {dep} --> {step_id}\n"
        
        return mermaid_code
    
    def save_mermaid_diagram(self, output_path: str = "pipeline_flow.mmd"):
        """Save Mermaid diagram to file."""
        mermaid_code = self.generate_mermaid_diagram()
        
        with open(output_path, 'w') as f:
            f.write(mermaid_code)
        
        print(f"Mermaid diagram saved to: {output_path}")
        return True


class RealTimeStatusDashboard:
    """Real-time status dashboard for pipeline monitoring."""
    
    def __init__(self, step_definitions: Dict[str, Dict[str, Any]]):
        self.step_definitions = step_definitions
        self.console = Console()
        self.running = False
        self.step_status = {}
        self.resource_metrics = {}
        self.last_update = None
        
    def start(self):
        """Start the real-time dashboard."""
        if not RICH_AVAILABLE:
            print("Rich not available. Cannot start real-time dashboard.")
            return
            
        self.running = True
        self.last_update = datetime.now()
        
        # Initialize step status
        for step_id in self.step_definitions.keys():
            self.step_status[step_id] = {
                "status": "pending",
                "progress": 0,
                "start_time": None,
                "duration": 0,
                "error": None
            }
        
        # Start dashboard in separate thread
        dashboard_thread = threading.Thread(target=self._run_dashboard)
        dashboard_thread.daemon = True
        dashboard_thread.start()
    
    def update_step_status(self, step_id: str, status: str, progress: int = 0, error: str = None):
        """Update the status of a step."""
        if step_id in self.step_status:
            self.step_status[step_id]["status"] = status
            self.step_status[step_id]["progress"] = progress
            self.step_status[step_id]["error"] = error
            
            if status == "running" and not self.step_status[step_id]["start_time"]:
                self.step_status[step_id]["start_time"] = datetime.now()
            elif status in ["completed", "failed"]:
                if self.step_status[step_id]["start_time"]:
                    duration = (datetime.now() - self.step_status[step_id]["start_time"]).total_seconds()
                    self.step_status[step_id]["duration"] = duration
    
    def update_resource_metrics(self, metrics: Dict[str, Any]):
        """Update resource metrics."""
        self.resource_metrics = metrics
        self.last_update = datetime.now()
    
    def _run_dashboard(self):
        """Run the dashboard display loop."""
        layout = Layout()
        
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3)
        )
        
        layout["body"].split_row(
            Layout(name="steps"),
            Layout(name="resources")
        )
        
        with Live(layout, console=self.console, refresh_per_second=2) as live:
            while self.running:
                # Update header
                layout["header"].update(
                    Panel(
                        f"Pipeline Status Dashboard - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                        style="bold blue"
                    )
                )
                
                # Update steps panel
                steps_table = Table(title="Step Status", show_header=True)
                steps_table.add_column("Step")
                steps_table.add_column("Status")
                steps_table.add_column("Progress")
                steps_table.add_column("Duration")
                
                for step_id, step_def in self.step_definitions.items():
                    status_info = self.step_status[step_id]
                    
                    # Status with color coding
                    status_color = {
                        "pending": "yellow",
                        "running": "blue",
                        "completed": "green",
                        "failed": "red"
                    }.get(status_info["status"], "white")
                    
                    status_text = f"[{status_color}]{status_info['status']}[/{status_color}]"
                    
                    # Progress bar
                    progress = status_info["progress"]
                    progress_bar = "█" * (progress // 10) + "░" * (10 - progress // 10)
                    progress_text = f"{progress_bar} {progress}%"
                    
                    # Duration
                    duration = status_info["duration"]
                    duration_text = f"{duration:.1f}s" if duration > 0 else "-"
                    
                    steps_table.add_row(
                        step_def["name"],
                        status_text,
                        progress_text,
                        duration_text
                    )
                
                layout["steps"].update(Panel(steps_table, title="Pipeline Steps"))
                
                # Update resources panel
                if self.resource_metrics:
                    resources_table = Table(title="Resource Usage", show_header=True)
                    resources_table.add_column("Metric")
                    resources_table.add_column("Value")
                    
                    for key, value in self.resource_metrics.items():
                        if isinstance(value, (int, float)) and key.endswith('_percent'):
                            # Color code percentages
                            color = "green" if value < 70 else "yellow" if value < 85 else "red"
                            value_text = f"[{color}]{value:.1f}%[/{color}]"
                        else:
                            value_text = str(value)
                        
                        resources_table.add_row(key.replace('_', ' ').title(), value_text)
                    
                    layout["resources"].update(Panel(resources_table, title="System Resources"))
                else:
                    layout["resources"].update(Panel("No resource data available", title="System Resources"))
                
                # Update footer
                footer_text = f"Last Updated: {self.last_update.strftime('%H:%M:%S') if self.last_update else 'Never'}"
                layout["footer"].update(Panel(footer_text, style="dim"))
                
                time.sleep(0.5)
    
    def stop(self):
        """Stop the dashboard."""
        self.running = False


class PipelineVisualizer:
    """Main pipeline visualization coordinator."""
    
    def __init__(self, step_definitions: Dict[str, Dict[str, Any]]):
        self.step_definitions = step_definitions
        self.progress_manager = ProgressBarManager()
        self.flow_diagram = PipelineFlowDiagram(step_definitions)
        self.dashboard = RealTimeStatusDashboard(step_definitions)
    
    def start_progress_tracking(self, total_steps: int):
        """Start progress tracking for the pipeline."""
        self.progress_manager.start(total_steps)
    
    def stop_progress_tracking(self):
        """Stop progress tracking."""
        self.progress_manager.stop()
    
    def generate_all_diagrams(self, output_dir: str = "reports"):
        """Generate all available diagrams."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        results = []
        
        # Generate dependency graph
        if self.flow_diagram.generate_dependency_graph(
            str(output_path / "pipeline_dependencies.png")
        ):
            results.append("Dependency graph generated")
        
        # Generate Mermaid diagram
        if self.flow_diagram.save_mermaid_diagram(
            str(output_path / "pipeline_flow.mmd")
        ):
            results.append("Mermaid diagram generated")
        
        return results
    
    def start_dashboard(self):
        """Start the real-time dashboard."""
        self.dashboard.start()
    
    def stop_dashboard(self):
        """Stop the real-time dashboard."""
        self.dashboard.stop()


# Console-based fallback for when Rich is not available
class SimpleProgressBar:
    """Simple console progress bar fallback."""
    
    def __init__(self, total: int, description: str = "Progress"):
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = time.time()
    
    def update(self, advance: int = 1):
        """Update progress."""
        self.current += advance
        self.current = min(self.current, self.total)
        self._display()
    
    def _display(self):
        """Display the progress bar."""
        percent = (self.current / self.total) * 100
        bar_length = 50
        filled_length = int(bar_length * self.current // self.total)
        
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        elapsed = time.time() - self.start_time
        
        sys.stdout.write(f'\r{self.description}: |{bar}| {percent:.1f}% ({self.current}/{self.total}) - {elapsed:.1f}s')
        sys.stdout.flush()
        
        if self.current >= self.total:
            print()  # New line when complete 