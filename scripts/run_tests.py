#!/usr/bin/env python3
"""
Test runner script for the enhanced entity extraction system.
This script can be run inside the Docker container to validate the implementation.
"""
import sys
import os
import subprocess
from pathlib import Path

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

def run_tests():
    """Run all tests for the entity extraction system."""
    print("🧪 Running Entity Extraction Tests...")
    print("=" * 50)
    
    # Test 1: Basic lexical filtering
    print("\n1. Testing lexical filtering...")
    try:
        from src.get_citation_entities import _passes_lexical_filters
        
        # Test year filtering
        assert not _passes_lexical_filters("2023"), "Years should be filtered out"
        assert _passes_lexical_filters("SRR123456"), "Valid entities should pass"
        print("✅ Lexical filtering tests passed")
    except Exception as e:
        print(f"❌ Lexical filtering tests failed: {e}")
        return False
    
    # Test 2: Pattern loading
    print("\n2. Testing pattern loading...")
    try:
        from src.patterns import PATTERNS
        assert len(PATTERNS) > 0, "Patterns should be loaded"
        print(f"✅ Pattern loading tests passed ({len(PATTERNS)} patterns loaded)")
    except Exception as e:
        print(f"❌ Pattern loading tests failed: {e}")
        return False
    
    # Test 3: Extractor initialization
    print("\n3. Testing extractor initialization...")
    try:
        from src.get_citation_entities import CitationEntityExtractor
        
        # Test without NER
        extractor = CitationEntityExtractor(known_entities=False)
        assert not extractor.use_ner, "NER should be disabled by default"
        assert len(extractor.patterns) > 0, "Patterns should be loaded"
        print("✅ Extractor initialization tests passed")
    except Exception as e:
        print(f"❌ Extractor initialization tests failed: {e}")
        return False
    
    # Test 4: Basic entity extraction
    print("\n4. Testing basic entity extraction...")
    try:
        extractor = CitationEntityExtractor(known_entities=False)
        test_text = "The dataset SRR123456 was deposited in the repository"
        entities = extractor._get_unknown_entities([test_text], "test_article")
        
        # Should extract SRR123456
        entity_texts = [entity.data_citation for entity in entities]
        assert "SRR123456" in entity_texts, "Should extract SRR123456"
        print("✅ Basic entity extraction tests passed")
    except Exception as e:
        print(f"❌ Basic entity extraction tests failed: {e}")
        return False
    
    # Test 5: Precision filtering
    print("\n5. Testing precision filtering...")
    try:
        extractor = CitationEntityExtractor(known_entities=False)
        test_text = "From 2018 we deposited SRR123456 in the repository"
        entities = extractor._get_unknown_entities([test_text], "test_article")
        
        entity_texts = [entity.data_citation for entity in entities]
        assert "SRR123456" in entity_texts, "Should extract SRR123456"
        assert "2018" not in entity_texts, "Should not extract year 2018"
        print("✅ Precision filtering tests passed")
    except Exception as e:
        print(f"❌ Precision filtering tests failed: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("🎉 All tests passed! The enhanced entity extraction system is working correctly.")
    return True

def run_pytest():
    """Run pytest for more comprehensive testing."""
    print("\n🧪 Running pytest for comprehensive testing...")
    try:
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            "tests/test_extractor.py", 
            "-v", "--tb=short"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Pytest tests passed")
            print(result.stdout)
            return True
        else:
            print("❌ Pytest tests failed")
            print(result.stdout)
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ Pytest execution failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Entity Extraction System Test Runner")
    print("=" * 50)
    
    # Run basic tests
    basic_success = run_tests()
    
    # Run pytest if basic tests pass
    if basic_success:
        pytest_success = run_pytest()
        if pytest_success:
            print("\n🎉 All tests completed successfully!")
            sys.exit(0)
        else:
            print("\n⚠️  Basic tests passed but pytest failed. Check for environment issues.")
            sys.exit(1)
    else:
        print("\n❌ Basic tests failed. Fix issues before running pytest.")
        sys.exit(1) 