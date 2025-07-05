#!/usr/bin/env python3
"""
Section Mapping Module for Step 5
Handles mapping of various section types to canonical labels
"""

import re
from typing import Dict, Set


# TEI Section Mapping (for Step 4 Grobid-generated files)
TEI_SECTION_MAPPING = {
    'abstract': 'abstract',
    'introduction': 'introduction', 
    'background': 'introduction',
    'method': 'methods',
    'methods': 'methods',
    'methodology': 'methods',
    'materials': 'methods',
    'materials and methods': 'methods',
    'experimental': 'methods',
    'procedure': 'methods',
    'result': 'results',
    'results': 'results',
    'findings': 'results',
    'discussion': 'discussion',
    'conclusion': 'conclusion',
    'conclusions': 'conclusion',
    'summary': 'conclusion',
    'availability': 'data_availability',
    'data availability': 'data_availability',
    'data_availability': 'data_availability',
    'code availability': 'data_availability',
    'acknowledgment': 'acknowledgments',
    'acknowledgments': 'acknowledgments',
    'acknowledgement': 'acknowledgments',
    'acknowledgements': 'acknowledgments',
    'references': 'references',
    'bibliography': 'references',
    'funding': 'funding',
    'author': 'author_info',
    'authors': 'author_info',
    'conflict': 'conflicts',
    'conflicts': 'conflicts',
    'competing interests': 'conflicts',
    'ethics': 'ethics',
    'supplementary': 'supplementary',
    'appendix': 'appendix',
    'appendices': 'appendix'
}


# Enhanced JATS Title Patterns with case-insensitive regex
JATS_TITLE_PATTERNS = {
    'abstract': r'abstract',
    'introduction': r'(background|introduction)',
    'methods': r'(methods?|methodology|materials?\s+and\s+methods?|experimental\s+(procedure|design|methods?)|procedure)',
    'results': r'(results?|findings)',
    'discussion': r'discussion',
    'conclusion': r'(conclus(ion|ions)|summary)',
    'data_availability': r'(data\s+availability|data\s+&\s+code\s+availability|code\s+availability|availability\s+of\s+data)',
    'acknowledgments': r'acknowledgm(ent|ents)',
    'references': r'(references?|bibliography)',
    'funding': r'funding',
    'author_info': r'(author|authors)(\s+(information|details|contributions?))?',
    'conflicts': r'(conflict|conflicts|competing\s+interests)',
    'ethics': r'(ethics|ethical\s+(approval|statement))',
    'supplementary': r'(supplementary|supplement)',
    'appendix': r'(appendix|appendices)'
}


def normalize_section_type(section_type: str) -> str:
    """Normalize section type string for mapping."""
    if not section_type:
        return ""
    return section_type.lower().strip().replace('_', ' ').replace('-', ' ')


def map_tei_section_type(section_type: str) -> str:
    """
    Map TEI section type to canonical type.
    
    Args:
        section_type: The @type attribute value from TEI <div> element
        
    Returns:
        Canonical section type
    """
    normalized = normalize_section_type(section_type)
    return TEI_SECTION_MAPPING.get(normalized, 'other')


def map_jats_section_type(sec_type: str, title: str) -> str:
    """
    Map JATS section using sec-type attribute and title patterns.
    
    This implements enhanced mapping with case-insensitive title matching
    as specified in the checklist.
    
    Args:
        sec_type: The @sec-type attribute value from JATS <sec> element
        title: The text content of the <title> element
        
    Returns:
        Canonical section type
    """
    # First try sec-type attribute
    if sec_type:
        normalized_sec_type = normalize_section_type(sec_type)
        for canonical, pattern in JATS_TITLE_PATTERNS.items():
            if re.search(pattern, normalized_sec_type, re.IGNORECASE):
                return canonical
    
    # Then try title text
    if title:
        normalized_title = normalize_section_type(title)
        for canonical, pattern in JATS_TITLE_PATTERNS.items():
            if re.search(pattern, normalized_title, re.IGNORECASE):
                return canonical
    
    return 'other'


def get_canonical_section_types() -> Set[str]:
    """Get set of all canonical section types."""
    canonical_types = set(TEI_SECTION_MAPPING.values())
    canonical_types.update(JATS_TITLE_PATTERNS.keys())
    canonical_types.add('other')
    return canonical_types


def is_key_section(section_type: str) -> bool:
    """
    Check if section is a key section for validation purposes.
    
    According to the checklist, we check for methods AND (results OR data_availability).
    
    Args:
        section_type: Canonical section type
        
    Returns:
        True if this is a key section type
    """
    key_sections = {'methods', 'results', 'data_availability'}
    return section_type in key_sections 