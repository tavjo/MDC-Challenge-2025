#!/usr/bin/env python3
"""
XML Format Detection Module for Step 5
Implements namespace-aware detection of TEI vs JATS formats
"""

from lxml import etree
from pathlib import Path
from typing import Dict, Optional, Tuple


def detect_xml_format(xml_path: Path) -> str:
    """
    Detect XML format by parsing root element and inspecting namespaces.
    
    This is CRITICAL for proper parsing since TEI and JATS formats require
    different XPath queries and namespace handling.
    
    Args:
        xml_path: Path to XML file
        
    Returns:
        Format type: 'TEI', 'JATS', 'UNKNOWN', or 'ERROR'
    """
    try:
        parser = etree.XMLParser(ns_clean=True, recover=True)
        tree = etree.parse(str(xml_path), parser)
        root = tree.getroot()
        
        # Check namespace map for JATS patterns
        nsmap = root.nsmap or {}
        for uri in nsmap.values():
            if uri and ('jats.nlm.nih.gov' in uri or '/JATS' in uri or uri.startswith('http://jats.')):
                return 'JATS'
        
        # Check for TEI namespace
        if any('tei-c.org' in str(uri) for uri in nsmap.values() if uri):
            return 'TEI'
            
        # Fallback to root tag inspection
        if root.tag.endswith('TEI'):
            return 'TEI'
        elif root.tag.endswith('article'):
            return 'JATS'
            
        return 'UNKNOWN'
        
    except Exception as e:
        print(f"Error detecting format for {xml_path}: {e}")
        return 'ERROR'


def get_namespace_info(xml_path: Path) -> Tuple[Dict, Optional[str]]:
    """
    Get detailed namespace information for debugging purposes.
    
    Args:
        xml_path: Path to XML file
        
    Returns:
        Tuple of (nsmap, default_namespace)
    """
    try:
        parser = etree.XMLParser(ns_clean=True, recover=True)
        tree = etree.parse(str(xml_path), parser)
        root = tree.getroot()
        return root.nsmap or {}, root.nsmap.get(None)
    except Exception as e:
        print(f"Error getting namespace info for {xml_path}: {e}")
        return {}, None


def setup_namespaces(xml_path: Path, format_type: str) -> Dict[str, str]:
    """
    Setup appropriate namespaces for XPath queries based on format type.
    
    Args:
        xml_path: Path to XML file
        format_type: Detected format ('TEI' or 'JATS')
        
    Returns:
        Dictionary mapping namespace prefixes to URIs
    """
    if format_type == 'TEI':
        return {"tei": "http://www.tei-c.org/ns/1.0"}
    
    elif format_type == 'JATS':
        # For JATS, we need to dynamically register namespaces
        try:
            parser = etree.XMLParser(ns_clean=True, recover=True)
            tree = etree.parse(str(xml_path), parser)
            root = tree.getroot()
            
            nsmap = root.nsmap or {}
            jats_ns = None
            
            # Look for JATS namespace
            for prefix, uri in nsmap.items():
                if uri and ('jats.nlm.nih.gov' in uri or '/JATS' in uri):
                    jats_ns = uri
                    break
            
            if jats_ns:
                return {'j': jats_ns}
            else:
                # No explicit JATS namespace, use default
                return {}
                
        except Exception as e:
            print(f"Error setting up JATS namespaces for {xml_path}: {e}")
            return {}
    
    return {} 