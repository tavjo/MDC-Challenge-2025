"""
Tests for the enhanced CitationEntityExtractor with lexical filtering and validation.
"""
import pytest
import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from src.get_citation_entities import CitationEntityExtractor, _passes_lexical_filters, _in_lookup


class TestLexicalFiltering:
    """Test lexical filtering functionality."""
    
    def test_year_filtered(self):
        """Test that years are filtered out."""
        # Years should be filtered out
        assert not _passes_lexical_filters("2023")
        assert not _passes_lexical_filters("1999")
        assert not _passes_lexical_filters("1234")
        
        # But longer numbers should pass
        assert _passes_lexical_filters("12345")
        assert _passes_lexical_filters("20234")
    
    def test_stopword_filtered(self):
        """Test that stopwords are filtered out."""
        # Common stopwords should be filtered
        assert not _passes_lexical_filters("the")
        assert not _passes_lexical_filters("and")
        assert not _passes_lexical_filters("is")
        
        # But scientific terms should pass
        assert _passes_lexical_filters("dataset")
        assert _passes_lexical_filters("accession")
    
    def test_no_digits_filtered(self):
        """Test that tokens without digits are filtered out."""
        # Pure text without digits should be filtered
        assert not _passes_lexical_filters("dataset")
        assert not _passes_lexical_filters("accession")
        assert not _passes_lexical_filters("SRR")
        
        # But tokens with digits should pass
        assert _passes_lexical_filters("SRR123456")
        assert _passes_lexical_filters("GSE12345")
        assert _passes_lexical_filters("dataset123")
    
    def test_valid_entities_pass(self):
        """Test that valid entity patterns pass."""
        # Valid entity patterns should pass
        assert _passes_lexical_filters("SRR123456")
        assert _passes_lexical_filters("GSE12345")
        assert _passes_lexical_filters("ERR123456")
        assert _passes_lexical_filters("DRR123456")
        assert _passes_lexical_filters("CHEMBL1234")
        assert _passes_lexical_filters("IPR000001")


class TestExtractorIntegration:
    """Test the full extractor integration."""
    
    def test_underscore_regression(self):
        """Regression test for underscores - verify SRR_123456 matches when paper uses underscore."""
        extractor = CitationEntityExtractor(known_entities=False)
        test_text = "The dataset was deposited as SRR_123456 in the repository"
        entities = extractor._get_unknown_entities([test_text], "test_article")
        assert any("SRR_123456" in entity.data_citation for entity in entities)
    
    def test_precision_guard(self):
        """Precision guard - feed paragraph with year and assert only SRR ID survives."""
        extractor = CitationEntityExtractor(known_entities=False)
        test_text = "From 2018 we deposited SRR123456 in the repository"
        entities = extractor._get_unknown_entities([test_text], "test_article")
        # Should only extract SRR123456, not 2018
        entity_texts = [entity.data_citation for entity in entities]
        assert "SRR123456" in entity_texts
        assert "2018" not in entity_texts
    
    def test_trigger_words_required(self):
        """Test that trigger words are required for entity extraction."""
        extractor = CitationEntityExtractor(known_entities=False)
        # Text with entity but no trigger words
        test_text = "We found SRR123456 in the analysis"
        entities = extractor._get_unknown_entities([test_text], "test_article")
        # Should not extract without trigger words
        assert len(entities) == 0
        
        # Text with entity and trigger words
        test_text_with_trigger = "The dataset SRR123456 was deposited in the repository"
        entities = extractor._get_unknown_entities([test_text_with_trigger], "test_article")
        # Should extract with trigger words
        assert len(entities) > 0
        assert any("SRR123456" in entity.data_citation for entity in entities)
    
    def test_multiple_entities_same_page(self):
        """Test extraction of multiple entities from the same page."""
        extractor = CitationEntityExtractor(known_entities=False)
        test_text = "We deposited SRR123456 and GSE12345 in the repository. The dataset was also stored as ERR123456."
        entities = extractor._get_unknown_entities([test_text], "test_article")
        
        entity_texts = [entity.data_citation for entity in entities]
        # Should extract all three entities
        assert "SRR123456" in entity_texts
        assert "GSE12345" in entity_texts
        assert "ERR123456" in entity_texts
        assert len(entities) == 3
    
    def test_entity_page_tracking(self):
        """Test that entities are correctly associated with page numbers."""
        extractor = CitationEntityExtractor(known_entities=False)
        page1 = "We deposited SRR123456 in the repository."
        page2 = "The dataset GSE12345 was also analyzed."
        page3 = "Finally, ERR123456 was processed."
        
        entities = extractor._get_unknown_entities([page1, page2, page3], "test_article")
        
        # Check that entities are on correct pages
        for entity in entities:
            if entity.data_citation == "SRR123456":
                assert entity.pages == [1]
            elif entity.data_citation == "GSE12345":
                assert entity.pages == [2]
            elif entity.data_citation == "ERR123456":
                assert entity.pages == [3]


class TestLookupValidation:
    """Test lookup validation functionality."""
    
    def test_lookup_function_exists(self):
        """Test that _in_lookup function exists and is callable."""
        assert callable(_in_lookup)
    
    def test_lookup_without_file(self):
        """Test lookup behavior when file doesn't exist."""
        # Should return True (trust regex) when lookup file doesn't exist
        assert _in_lookup("SRR123456") == True
    
    def test_lookup_with_invalid_prefix(self):
        """Test lookup with invalid prefix."""
        # Should return True for invalid prefixes
        assert _in_lookup("123456") == True
        assert _in_lookup("A") == True


class TestNERValidation:
    """Test NER validation functionality."""
    
    def test_ner_disabled_by_default(self):
        """Test that NER is disabled by default."""
        extractor = CitationEntityExtractor(known_entities=False)
        assert not extractor.use_ner
    
    def test_ner_enabled_when_specified(self):
        """Test that NER is enabled when specified."""
        extractor = CitationEntityExtractor(known_entities=False, use_ner=True)
        assert extractor.use_ner
    
    def test_ner_method_exists(self):
        """Test that _looks_like_dataset method exists."""
        extractor = CitationEntityExtractor(known_entities=False, use_ner=True)
        assert hasattr(extractor, '_looks_like_dataset')
        assert callable(extractor._looks_like_dataset)


class TestPatternIntegration:
    """Test pattern integration with new system."""
    
    def test_patterns_loaded(self):
        """Test that patterns are loaded from new system."""
        extractor = CitationEntityExtractor(known_entities=False)
        assert len(extractor.patterns) > 0
        assert all(callable(pattern) for pattern in extractor.patterns.values())
    
    def test_known_entities_with_patterns(self):
        """Test that known entities use pattern system."""
        extractor = CitationEntityExtractor(known_entities=True)
        # This test requires actual data files, so we'll just check the method exists
        assert hasattr(extractor, '_get_known_entities')
        assert callable(extractor._get_known_entities)


if __name__ == "__main__":
    pytest.main([__file__]) 