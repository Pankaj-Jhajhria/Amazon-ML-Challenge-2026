"""Reusable suffix / abbreviation lookup tables for name & address normalization."""
from __future__ import annotations

from typing import Dict, List, Tuple

# Legal / entity-type suffixes -> canonical short form.
# Applied token-wise after tokenization, order-independent.
LEGAL_SUFFIXES: Dict[str, str] = {
    "inc": "inc", "incorporated": "inc",
    "corp": "corp", "corporation": "corp",
    "co": "co", "company": "co",
    "ltd": "ltd", "limited": "ltd",
    "llc": "llc",
    "llp": "llp",
    "lp": "lp",
    "pvt": "pvt", "private": "pvt",
    "plc": "plc",
    "gmbh": "gmbh",
    "sarl": "sarl",
    "sa": "sa",
    "sas": "sas",
    "bv": "bv",
    "ag": "ag",
    "kg": "kg",
    "pty": "pty",
    "group": "group", "grp": "group",
    "holdings": "holdings", "holding": "holdings",
    "enterprises": "ent", "enterprise": "ent", "ent": "ent",
    "industries": "ind", "industry": "ind",
    "solutions": "sol", "solution": "sol",
    "services": "svc", "service": "svc",
    "international": "intl", "intl": "intl",
    "trading": "trdg",
    "associates": "assoc", "assocs": "assoc",
    "brothers": "bros", "bros": "bros",
}

# Business-word abbreviations that can appear anywhere inside a name.
BUSINESS_ABBREV: Dict[str, str] = {
    "mfg": "manufacturing", "manufacturing": "manufacturing",
    "dept": "department", "department": "department",
    "assn": "association", "association": "association",
    "mgmt": "management", "management": "management",
    "svcs": "services", "svc": "service",
    "natl": "national", "national": "national",
    "govt": "government", "government": "government",
    "&": "and", "+": "and",
}

# Address component abbreviations -> expanded canonical form.
ADDRESS_ABBREV: Dict[str, str] = {
    "st": "street", "str": "street", "street": "street",
    "rd": "road", "road": "road",
    "ave": "avenue", "av": "avenue", "avenue": "avenue",
    "blvd": "boulevard", "boulevard": "boulevard",
    "dr": "drive", "drive": "drive",
    "ln": "lane", "lane": "lane",
    "ct": "court", "court": "court",
    "pl": "place", "place": "place",
    "sq": "square", "square": "square",
    "hwy": "highway", "highway": "highway",
    "pkwy": "parkway", "parkway": "parkway",
    "ter": "terrace", "terrace": "terrace",
    "cir": "circle", "circle": "circle",
    "apt": "apartment", "apartment": "apartment",
    "ste": "suite", "suite": "suite",
    "unit": "unit",
    "fl": "floor", "floor": "floor",
    "bldg": "building", "building": "building",
    "no": "number", "number": "number",
    "opp": "opposite", "opposite": "opposite",
    "nr": "near", "near": "near",
    "sec": "sector", "sector": "sector",
    "ph": "phase", "phase": "phase",
    "colony": "colony",
    "nagar": "nagar",
    "marg": "road",
    "rte": "route", "route": "route",
}

DIRECTIONALS: Dict[str, str] = {
    "n": "north", "north": "north",
    "s": "south", "south": "south",
    "e": "east", "east": "east",
    "w": "west", "west": "west",
    "ne": "northeast", "northeast": "northeast",
    "nw": "northwest", "northwest": "northwest",
    "se": "southeast", "southeast": "southeast",
    "sw": "southwest", "southwest": "southwest",
}


def merge_address_maps() -> Dict[str, str]:
    merged = dict(ADDRESS_ABBREV)
    merged.update(DIRECTIONALS)
    return merged


def apply_token_map(tokens: List[str], mapping: Dict[str, str]) -> List[str]:
    return [mapping.get(tok, tok) for tok in tokens]


def strip_legal_suffix(tokens: List[str]) -> Tuple[List[str], bool]:
    """Remove a trailing legal-suffix token (after abbreviation expansion).
    Returns (remaining_tokens, had_suffix)."""
    if not tokens:
        return tokens, False
    canonical_suffixes = set(LEGAL_SUFFIXES.values())
    if tokens[-1] in canonical_suffixes:
        return tokens[:-1], True
    return tokens, False
