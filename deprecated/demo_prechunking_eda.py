# scripts/demo_prechunking_eda.py

"""
Demonstration script for the enhanced pre-chunking EDA with report generation.

This script shows different ways to use the run_prechunking_eda.py script
both programmatically and via command line.
"""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from run_prechunking_eda import run_prechunking_eda, EDAReporter

def demo_programmatic_usage():
    """Demonstrate programmatic usage of the EDA script."""
    print("=" * 60)
    print("DEMO: Programmatic Usage")
    print("=" * 60)
    
    # Basic usage with all outputs
    print("\n1. Running comprehensive EDA with all outputs...")
    try:
        reporter = run_prechunking_eda(
            data_dir="Data",
            output_format="all",
            save_reports=True,
            show_plots=False
        )
        
        print(f"✅ Analysis completed!")
        print(f"📊 Summary stats collected: {len(reporter.summary_stats)}")
        print(f"📝 Analysis sections: {len(reporter.sections)}")
        print(f"📁 Files generated: {len(reporter.files_generated)}")
        
        # Access specific statistics
        print(f"\n📋 Key Statistics:")
        print(f"   • Total Articles: {reporter.summary_stats.get('Unique Articles', 'N/A')}")
        print(f"   • PDF Files: {reporter.summary_stats.get('PDF Files Available', 'N/A')}")
        print(f"   • XML Files: {reporter.summary_stats.get('XML Files Available', 'N/A')}")
        print(f"   • Conversion Needed: {reporter.summary_stats.get('Articles Needing Conversion', 'N/A')}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

def demo_console_only():
    """Demonstrate console-only output (no file saving)."""
    print("\n" + "=" * 60)
    print("DEMO: Console-Only Output")
    print("=" * 60)
    
    try:
        reporter = run_prechunking_eda(
            data_dir="Data",
            output_format="console",
            save_reports=False,
            show_plots=False
        )
        
        print(f"✅ Console-only analysis completed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")

def demo_json_summary():
    """Demonstrate JSON summary generation."""
    print("\n" + "=" * 60)
    print("DEMO: JSON Summary Generation")
    print("=" * 60)
    
    try:
        reporter = run_prechunking_eda(
            data_dir="Data",
            output_format="json",
            save_reports=True,
            show_plots=False
        )
        
        # Get JSON summary
        json_summary = reporter.generate_json_summary()
        
        print(f"✅ JSON summary generated!")
        print(f"📊 Metadata keys: {list(json_summary['metadata'].keys())}")
        print(f"📈 Summary statistics: {len(json_summary['summary_statistics'])} items")
        print(f"📝 Analysis sections: {len(json_summary['detailed_sections'])} sections")
        
        # Show a few key statistics
        stats = json_summary['summary_statistics']
        print(f"\n🔢 Key Numbers:")
        for key, value in list(stats.items())[:5]:
            print(f"   • {key}: {value}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

def demo_error_handling():
    """Demonstrate error handling with invalid data directory."""
    print("\n" + "=" * 60)
    print("DEMO: Error Handling")
    print("=" * 60)
    
    try:
        reporter = run_prechunking_eda(
            data_dir="NonexistentDirectory",
            output_format="console",
            save_reports=False,
            show_plots=False
        )
        
    except Exception as e:
        print(f"✅ Error properly caught: {type(e).__name__}")
        print(f"   Message: {str(e)[:100]}...")

def show_command_line_examples():
    """Show command line usage examples."""
    print("\n" + "=" * 60)
    print("COMMAND LINE USAGE EXAMPLES")
    print("=" * 60)
    
    examples = [
        {
            "description": "Basic usage with default settings",
            "command": "python scripts/run_prechunking_eda.py"
        },
        {
            "description": "Custom data directory",
            "command": "python scripts/run_prechunking_eda.py --data-dir /path/to/data"
        },
        {
            "description": "Console output only (no file saving)",
            "command": "python scripts/run_prechunking_eda.py --output-format console --no-save"
        },
        {
            "description": "JSON output only",
            "command": "python scripts/run_prechunking_eda.py --output-format json"
        },
        {
            "description": "With plots displayed (for interactive use)",
            "command": "python scripts/run_prechunking_eda.py --show-plots"
        },
        {
            "description": "Markdown report only",
            "command": "python scripts/run_prechunking_eda.py --output-format markdown"
        },
        {
            "description": "Custom reports directory",
            "command": "python scripts/run_prechunking_eda.py --reports-dir /path/to/custom/reports"
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n{i}. {example['description']}:")
        print(f"   {example['command']}")

def main():
    """Run all demonstrations."""
    print("🚀 PRE-CHUNKING EDA SCRIPT DEMONSTRATION")
    print("This demo shows various ways to use the enhanced EDA script")
    
    # Check if Data directory exists
    if not Path("Data").exists():
        print("\n⚠️  Warning: 'Data' directory not found.")
        print("   Some demos may fail. Please ensure you're running from project root.")
        print("   Or modify the data_dir parameter in the demos.\n")
    
    # Run demonstrations
    demo_programmatic_usage()
    demo_console_only() 
    demo_json_summary()
    demo_error_handling()
    show_command_line_examples()
    
    print("\n" + "=" * 60)
    print("✅ DEMONSTRATION COMPLETE")
    print("=" * 60)
    print("Next steps:")
    print("1. Run the actual script: python scripts/run_prechunking_eda.py")
    print("2. Check generated reports in your Data directory")
    print("3. Integrate into your workflow as needed")

if __name__ == "__main__":
    main() 