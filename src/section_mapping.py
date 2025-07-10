#!/usr/bin/env python3
"""
Section Mapping Module (Step 5)
Maps heterogeneous section names in TEI or JATS XML to a
canonical, compact label set needed by downstream NER.
"""

import re
from typing import Dict, Set

# ------------------------------------------------------------------
# 1. Canonical label → direct-match table for TEI  @type attributes
# ------------------------------------------------------------------
TEI_SECTION_MAPPING: Dict[str, str] = {
    # IMRaD core
    "abstract": "abstract",
    "introduction": "introduction",
    "background": "introduction",
    "method": "methods",
    "methods": "methods",
    "methodology": "methods",
    "materials": "methods",
    "materials and methods": "methods",
    "experimental": "methods",
    "procedure": "methods",
    "data and methods": "methods",
    "results": "results",
    "result": "results",
    "findings": "results",
    "results and discussion": "results",          # give priority to results
    "discussion": "discussion",
    "conclusion": "conclusion",
    "conclusions": "conclusion",
    "summary": "conclusion",
    # Data / code availability
    "availability": "data_availability",
    "data availability": "data_availability",
    "data_availability": "data_availability",
    "code availability": "data_availability",
    # Rear-matter
    "acknowledgment": "acknowledgments",
    "acknowledgment": "acknowledgments",
    "acknowledgement": "acknowledgments",
    "acknowledgements": "acknowledgments",
    "references": "references",
    "bibliography": "references",
    "funding": "funding",
    "author": "author_info",
    "authors": "author_info",
    "conflict": "conflicts",
    "conflicts": "conflicts",
    "competing interests": "conflicts",
    "ethics": "ethics",
    # Supplements
    "supplementary": "supplementary",
    "supplement": "supplementary",
    "appendix": "appendix",
    "appendices": "appendix",
}

# ------------------------------------------------------------------
# 2. Regex pattern tables for title-based inference
# ------------------------------------------------------------------
TEI_TITLE_PATTERNS: Dict[str, str] = {
    #   NB: keep patterns permissive, case-insensitive
    "abstract": r"\babstract\b",
    "introduction": r"\b(background|introduction)\b",
    "methods": (
        r"(data\s+and\s+methods|materials?\s+(and\s+)?methods?|"
        r"methods?|methodology|experimental\s+(procedure|design|methods?)|"
        r"statistical\s+(analysis|methods?)|procedure)"
    ),
    "results": r"(results?)(\s*&?\s*discussion)?|findings|observations",
    "discussion": r"\bdiscussion\b(?!\s*&)",
    "conclusion": r"\bconclus(?:ion|ions)\b|summary",
    "data_availability": r"(data\s+availability|code\s+availability|availability\s+of\s+data)",
    "acknowledgments": r"\backnowledgm?(?:ent|ents)\b",
    "references": r"\b(references?|bibliography)\b",
    "funding": r"\bfunding\b",
    "author_info": r"\b(authors?|author information|author details|author contributions?)\b",
    "conflicts": r"\b(conflicts?|competing\s+interests?)\b",
    "ethics": r"\b(ethics|ethical\s+(approval|statement))\b",
    "supplementary": r"\b(supplementary|supplement)\b",
    "appendix": r"\b(appendix|appendices)\b",
}

JATS_TITLE_PATTERNS: Dict[str, str] = {
    # unchanged from your original but kept here for completeness
    "abstract": r"abstract",
    "introduction": r"(background|introduction)",
    "methods": (
        r"(methods?|methodology|materials?\s+and\s+methods?|"
        r"experimental\s+(procedure|design|methods?)|procedure)"
    ),
    "results": r"(results?|findings)",
    "discussion": r"discussion",
    "conclusion": r"(conclus(?:ion|ions)|summary)",
    "data_availability": r"(data\s+availability|data\s+&\s+code\s+availability|"
    r"code\s+availability|availability\s+of\s+data)",
    "acknowledgments": r"acknowledgm(?:ent|ents)",
    "references": r"(references?|bibliography)",
    "funding": r"funding",
    "author_info": r"(author|authors)(\s+(information|details|contributions?))?",
    "conflicts": r"(conflict|conflicts|competing\s+interests)",
    "ethics": r"(ethics|ethical\s+(approval|statement))",
    "supplementary": r"(supplementary|supplement)",
    "appendix": r"(appendix|appendices)",
}

# ------------------------------------------------------------------
# 3. Helper functions
# ------------------------------------------------------------------
def normalize_section_type(text: str) -> str:
    """
    Case-fold, strip, replace separators and HTML entities so that
    'Results & Discussion', 'results-and-discussion', etc. normalise
    to the same string.
    """
    if not text:
        return ""
    text = (
        text.lower()
        .strip()
        .replace("_", " ")
        .replace("-", " ")
        .replace("&", " and ")
    )
    # squash multiple spaces
    return re.sub(r"\s+", " ", text)


# ---------- Mapping by TEI  @type  ----------
def map_tei_section_type(section_type: str) -> str:
    """Direct lookup against the TEI_SECTION_MAPPING table."""
    return TEI_SECTION_MAPPING.get(normalize_section_type(section_type), "other")


# ---------- Mapping by TEI  <head> title ----------
def map_tei_title(title: str) -> str:
    """Regex-match a heading string against TEI_TITLE_PATTERNS."""
    norm = normalize_section_type(title)
    for canonical, pattern in TEI_TITLE_PATTERNS.items():
        if re.search(pattern, norm, flags=re.IGNORECASE):
            return canonical
    return "other"


# ---------- Mapping for JATS ----------
def map_jats_section_type(sec_type: str, title: str) -> str:
    """
    Resolve a JATS <sec> either by @sec-type or by its <title>.
    """
    # First: @sec-type attribute
    if sec_type:
        norm = normalize_section_type(sec_type)
        for canonical, pattern in JATS_TITLE_PATTERNS.items():
            if re.search(pattern, norm, flags=re.IGNORECASE):
                return canonical

    # Second: visible <title> text
    if title:
        norm = normalize_section_type(title)
        for canonical, pattern in JATS_TITLE_PATTERNS.items():
            if re.search(pattern, norm, flags=re.IGNORECASE):
                return canonical

    return "other"


# ------------------------------------------------------------------
# 4. Utility helpers used elsewhere in the pipeline
# ------------------------------------------------------------------
def get_canonical_section_types() -> Set[str]:
    """
    Union of every canonical label recognised anywhere in the mapper.
    """
    canonical = set(TEI_SECTION_MAPPING.values())
    canonical.update(TEI_TITLE_PATTERNS.keys())
    canonical.update(JATS_TITLE_PATTERNS.keys())
    canonical.add("other")
    return canonical


def is_key_section(section_type: str) -> bool:
    """
    A “key” section is one that must be present for a paper to pass
    minimal validation:  METHODS  and  (RESULTS  or  DATA_AVAILABILITY).
    """
    return section_type in {"methods", "results", "data_availability"}
