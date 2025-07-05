#!/usr/bin/env python3
"""
Document Parser Module for Step 5
Implements namespace-aware parsing of TEI and JATS XML documents
Following the MDC-Challenge-2025 Step 5 checklist
"""

import pandas as pd
import pickle
from pathlib import Path
from lxml import etree
from typing import Dict, List, Optional, Any, Tuple
import hashlib
import re
from datetime import datetime

from .xml_format_detector import detect_xml_format, setup_namespaces
from .section_mapping import map_tei_section_type, map_jats_section_type, is_key_section


def extract_text_content(element) -> str:
    """
    Extract clean text content from XML element.
    
    This implements clean text extraction without XML artifacts
    as specified in the checklist.
    """
    if element is None:
        return ""
    
    # Get all text content and clean it
    text = etree.tostring(element, method="text", encoding="unicode")
    
    # Clean up whitespace and normalize
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text


def extract_section_text(sec_element, namespaces: Dict[str, str]) -> str:
    """
    Extract text content from a section element, excluding nested sections.
    
    This is important for recursive section handling.
    """
    if sec_element is None:
        return ""
    
    # Create a copy to avoid modifying original
    sec_copy = etree.fromstring(etree.tostring(sec_element))
    
    # Remove nested section elements
    if namespaces:
        # JATS with namespaces
        ns_prefix = list(namespaces.keys())[0] if namespaces else None
        if ns_prefix:
            nested_xpath = f'.//{{{namespaces[ns_prefix]}}}sec'
        else:
            nested_xpath = './/sec'
    else:
        # TEI or JATS without explicit namespaces
        nested_xpath = './/sec | .//div[@type]'
    
    nested_sections = sec_copy.xpath(nested_xpath, namespaces=namespaces)
    for nested_sec in nested_sections:
        if nested_sec != sec_copy:  # Don't remove the root element
            parent = nested_sec.getparent()
            if parent is not None:
                parent.remove(nested_sec)
    
    return extract_text_content(sec_copy)


def parse_tei_xml(xml_path: Path) -> List[Dict[str, Any]]:
    """
    Parse TEI XML format (from Step 4 Grobid conversion).
    
    Args:
        xml_path: Path to TEI XML file
        
    Returns:
        List of section dictionaries
    """
    try:
        tree = etree.parse(str(xml_path))
        root = tree.getroot()
        
        # Set up TEI namespace
        ns = {"tei": "http://www.tei-c.org/ns/1.0"}
        
        sections = []
        order = 0
        
        # Extract abstract first (if present)
        abstracts = root.xpath('.//tei:abstract', namespaces=ns)
        for abstract in abstracts:
            text = extract_text_content(abstract)
            if text.strip():
                sections.append({
                    'type': 'abstract',
                    'text': text,
                    'order': order,
                    'char_length': len(text),
                    'sec_level': 0,
                    'original_type': 'abstract',
                    'title': 'Abstract'
                })
                order += 1
        
        # Process body sections
        body_divs = root.xpath('.//tei:body//tei:div[@type]', namespaces=ns)
        for div in body_divs:
            div_type = div.get('type', '').lower()
            
            # Get section title
            head_elem = div.find('.//tei:head', namespaces=ns)
            title = head_elem.text if head_elem is not None else ''
            
            # Map to canonical type
            canonical_type = map_tei_section_type(div_type)
            
            # Extract text content
            text = extract_section_text(div, ns)
            
            if text.strip():
                sections.append({
                    'type': canonical_type,
                    'text': text,
                    'order': order,
                    'char_length': len(text),
                    'sec_level': 1,  # TEI sections are generally flat
                    'original_type': div_type,
                    'title': title
                })
                order += 1
        
        return sections
        
    except Exception as e:
        print(f"TEI parsing error for {xml_path}: {e}")
        return []


def process_section_recursive(sec, sections: List[Dict], ns: Dict[str, str], level: int = 1, order_counter: List[int] = None):
    """
    Recursively process nested JATS sections.
    
    This implements the recursive section parsing required for JATS format.
    """
    if order_counter is None:
        order_counter = [len(sections)]
    
    # Extract section type and title
    sec_type = sec.get('sec-type', '').lower()
    
    # Get title element with namespace awareness
    if ns and 'j' in ns:
        title_elem = sec.find(f'{{{ns["j"]}}}title')
    else:
        title_elem = sec.find('title')
    
    title = title_elem.text if title_elem is not None else ''
    
    # Map to canonical type
    canonical_type = map_jats_section_type(sec_type, title)
    
    # Extract text content (excluding nested sections)
    text = extract_section_text(sec, ns)
    
    if text.strip():
        sections.append({
            'type': canonical_type,
            'text': text,
            'order': order_counter[0],
            'char_length': len(text),
            'sec_level': level,
            'original_type': sec_type,
            'title': title
        })
        order_counter[0] += 1
    
    # Process nested sections
    if ns and 'j' in ns:
        nested_xpath = f'{{{ns["j"]}}}sec'
    else:
        nested_xpath = 'sec'
    
    for nested_sec in sec.findall(nested_xpath):
        process_section_recursive(nested_sec, sections, ns, level + 1, order_counter)


def parse_jats_xml(xml_path: Path) -> List[Dict[str, Any]]:
    """
    Parse JATS XML with proper namespace handling.
    
    This implements the CRITICAL namespace-aware JATS parsing from the checklist.
    """
    try:
        tree = etree.parse(str(xml_path))
        root = tree.getroot()
        
        # Register namespaces dynamically
        nsmap = root.nsmap or {}
        jats_ns = None
        for prefix, uri in nsmap.items():
            if uri and ('jats.nlm.nih.gov' in uri or '/JATS' in uri):
                jats_ns = uri
                break
        
        # Set up namespace dictionary
        if jats_ns:
            ns = {'j': jats_ns}
            sec_xpath = './/j:sec'
            abstract_xpath = './/j:article-meta/j:abstract'
        else:
            ns = {}
            sec_xpath = './/sec'
            abstract_xpath = './/article-meta/abstract'
        
        sections = []
        
        # Extract abstract from article-meta first
        abstracts = root.xpath(abstract_xpath, namespaces=ns)
        for abstract in abstracts:
            text = extract_text_content(abstract)
            if text.strip():
                sections.append({
                    'type': 'abstract',
                    'text': text,
                    'order': 0,
                    'char_length': len(text),
                    'sec_level': 0,
                    'original_type': 'abstract',
                    'title': 'Abstract'
                })
        
        # Process body sections recursively
        body_sections = root.xpath(sec_xpath, namespaces=ns)
        order_counter = [len(sections)]
        
        for sec in body_sections:
            process_section_recursive(sec, sections, ns, level=1, order_counter=order_counter)
            
        return sections
        
    except Exception as e:
        print(f"JATS parsing error for {xml_path}: {e}")
        try:
            tree = etree.parse(str(xml_path))
            root = tree.getroot()
            print(f"Root nsmap: {root.nsmap}")
            print(f"Root tag: {root.tag}")
        except:
            pass
        return []


def parse_document(file_path: Path, source_type: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Main document parsing dispatcher.
    
    Routes to appropriate parser based on detected format.
    """
    format_type = detect_xml_format(file_path)
    
    if format_type == 'TEI':
        return parse_tei_xml(file_path)
    elif format_type == 'JATS':
        return parse_jats_xml(file_path)
    else:
        # Log diagnostic info for unknown formats
        try:
            tree = etree.parse(str(file_path))
            root = tree.getroot()
            print(f"Unknown format - nsmap: {root.nsmap}, root tag: {root.tag}")
        except Exception as e:
            print(f"Parse error for {file_path}: {e}")
        return parse_fallback(file_path)


def parse_fallback(file_path: Path) -> List[Dict[str, Any]]:
    """
    Fallback parser for unknown formats.
    
    Attempts basic text extraction without specific structure.
    """
    try:
        tree = etree.parse(str(file_path))
        root = tree.getroot()
        
        # Extract all text as a single section
        text = extract_text_content(root)
        
        if text.strip():
            return [{
                'type': 'other',
                'text': text,
                'order': 0,
                'char_length': len(text),
                'sec_level': 0,
                'original_type': 'unknown',
                'title': 'Full Document'
            }]
        
        return []
        
    except Exception as e:
        print(f"Fallback parsing error for {file_path}: {e}")
        return []


def validate_document(sections: List[Dict[str, Any]], doi: str) -> Dict[str, Any]:
    """
    Validate parsed document according to checklist criteria.
    
    Returns validation results with flags and metrics.
    """
    validation = {
        'doi': doi,
        'section_count': len(sections),
        'has_sections': len(sections) > 0,
        'has_methods': False,
        'has_results': False,
        'has_data_availability': False,
        'key_sections_present': False,
        'total_char_length': 0,
        'clean_text_length': 0,
        'has_sufficient_content': False,
        'section_types': [],
        'validation_passed': False
    }
    
    # Analyze sections
    for section in sections:
        section_type = section['type']
        validation['section_types'].append(section_type)
        validation['total_char_length'] += section.get('char_length', 0)
        
        # Check for key sections
        if section_type == 'methods':
            validation['has_methods'] = True
        elif section_type == 'results':
            validation['has_results'] = True
        elif section_type == 'data_availability':
            validation['has_data_availability'] = True
    
    # Clean text length (strip XML tags)
    clean_text = ' '.join(section['text'] for section in sections)
    validation['clean_text_length'] = len(clean_text)
    
    # Key sections check: methods AND (results OR data_availability)
    validation['key_sections_present'] = (
        validation['has_methods'] and 
        (validation['has_results'] or validation['has_data_availability'])
    )
    
    # Sufficient content check (>1000 characters of clean text)
    validation['has_sufficient_content'] = validation['clean_text_length'] > 1000
    
    # Overall validation
    validation['validation_passed'] = (
        validation['has_sections'] and
        validation['key_sections_present'] and
        validation['has_sufficient_content']
    )
    
    return validation


def compute_file_hash(file_path: Path) -> str:
    """Compute MD5 hash of XML file for debugging reference."""
    try:
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception:
        return ""


def create_document_entry(doi: str, sections: List[Dict[str, Any]], 
                         file_path: Path, source_type: Optional[str] = None) -> Dict[str, Any]:
    """
    Create complete document entry for the parsed corpus.
    
    Returns dictionary in the format expected by Step 6.
    """
    # Validation
    validation = validate_document(sections, doi)
    
    # Full text assembly
    full_text = '\n\n'.join(f"## {section.get('title', section['type'].title())}\n{section['text']}" 
                           for section in sections if section['text'].strip())
    
    # Section labels (for compatibility)
    section_labels = [section['type'] for section in sections]
    
    # Create entry
    entry = {
        'doi': doi,
        'full_text': full_text,
        'sections': sections,
        'section_labels': section_labels,
        'section_count': len(sections),
        'total_char_length': validation['total_char_length'],
        'clean_text_length': validation['clean_text_length'],
        'format_type': detect_xml_format(file_path),
        'source_type': source_type,
        'file_path': str(file_path),
        'xml_hash': compute_file_hash(file_path),
        'parsed_timestamp': datetime.now().isoformat(),
        'validation': validation
    }
    
    return entry 