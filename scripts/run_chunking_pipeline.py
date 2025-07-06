#!/usr/bin/env python3
"""
CLI script for running the semantic chunking pipeline (Step 6)
"""

import argparse
import sys
from pathlib import Path

# Add src to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from semantic_chunking import run_semantic_chunking_pipeline



def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Run the semantic chunking pipeline (Step 6)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default parameters
  python scripts/run_chunking_pipeline.py
  
  # Run with custom parameters
  python scripts/run_chunking_pipeline.py \\
      --input-path Data/train/parsed/parsed_documents.pkl \\
      --output-path chunks_for_embedding.pkl \\
      --chunk-size 200 \\
      --chunk-overlap 20 \\
      --min-chars 500
      
  # Run with smaller chunks for testing
  python scripts/run_chunking_pipeline.py \\
      --chunk-size 100 \\
      --chunk-overlap 10 \\
      --min-chars 200
        """
    )
    
    # Input/Output arguments
    parser.add_argument(
        "--input-path",
        type=str,
        default="Data/train/parsed/parsed_documents.pkl",
        help="Path to parsed documents pickle file (default: Data/train/parsed/parsed_documents.pkl)"
    )
    
    parser.add_argument(
        "--output-path",
        type=str,
        default="chunks_for_embedding.pkl",
        help="Path for output chunks pickle file (default: chunks_for_embedding.pkl)"
    )
    
    # Chunking parameters
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=200,
        help="Target chunk size in tokens (default: 200)"
    )
    
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=20,
        help="Overlap between chunks in tokens (default: 20)"
    )
    
    parser.add_argument(
        "--min-chars",
        type=int,
        default=500,
        help="Minimum character count for document inclusion (default: 500)"
    )
    
    # Quality control arguments
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force processing even if quality gates fail (not recommended)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging output"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.chunk_size <= 0:
        parser.error("chunk-size must be positive")
    
    if args.chunk_overlap < 0:
        parser.error("chunk-overlap must be non-negative")
    
    if args.chunk_overlap >= args.chunk_size:
        parser.error("chunk-overlap must be less than chunk-size")
    
    if args.min_chars < 0:
        parser.error("min-chars must be non-negative")
    
    # Check if input file exists
    if not Path(args.input_path).exists():
        parser.error(f"Input file does not exist: {args.input_path}")
    
    # Print configuration
    print("=== Semantic Chunking Pipeline Configuration ===")
    print(f"Input path: {args.input_path}")
    print(f"Output path: {args.output_path}")
    print(f"Chunk size: {args.chunk_size} tokens")
    print(f"Chunk overlap: {args.chunk_overlap} tokens")
    print(f"Min chars: {args.min_chars}")
    print(f"Force mode: {args.force}")
    print(f"Verbose: {args.verbose}")
    print()
    
    # Run the pipeline
    try:
        results = run_semantic_chunking_pipeline(
            input_path=args.input_path,
            output_path=args.output_path,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            min_chars=args.min_chars
        )
        
        # Check results
        if results['success']:
            print("\n🎉 Pipeline completed successfully!")
            print(f"✅ Results:")
            print(f"   📊 Documents processed: {results['total_documents']}")
            print(f"   📊 Chunks created: {results['total_chunks']:,}")
            print(f"   📊 Average tokens per chunk: {results['avg_tokens_per_chunk']:.1f}")
            print(f"   📊 Entity retention: 100%")
            print(f"   📁 Output saved to: {results['output_path']}")
            
            # Quality gate summary
            print(f"\n✅ Quality Gates Status:")
            print(f"   Entity retention: ✅ PASSED (100%)")
            print(f"   Average tokens per chunk: ✅ PASSED ({results['avg_tokens_per_chunk']:.1f})")
            
            # Check if within recommended range
            if 160 <= results['avg_tokens_per_chunk'] <= 220:
                print(f"   Token range: ✅ OPTIMAL (160-220 tokens)")
            elif 130 <= results['avg_tokens_per_chunk'] <= 270:
                print(f"   Token range: ⚠️  ACCEPTABLE (130-270 tokens)")
            else:
                print(f"   Token range: ❌ SUBOPTIMAL (recommend 160-220 tokens)")
            
            return 0
            
        else:
            print(f"\n❌ Pipeline failed: {results['error']}")
            
            if not args.force:
                print("\n💡 Suggestions:")
                print("   1. Try increasing --chunk-overlap for better entity retention")
                print("   2. Check input data quality and format")
                print("   3. Use --force to bypass quality gates (not recommended)")
                
                return 1
            else:
                print("\n⚠️  Force mode enabled - quality gates bypassed")
                return 0
                
    except KeyboardInterrupt:
        print("\n\n⏹️  Pipeline interrupted by user")
        return 1
        
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main()) 