"""Deterministic normalization for business names and addresses.

Produces several parallel representations (character-normalized, token
representation, "core" canonical form with legal suffixes stripped) instead
of collapsing everything into one string, so downstream blocking/feature code
can pick whichever representation suits it.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd

try:
    from unidecode import unidecode
except ImportError:  # pragma: no cover - unidecode is a pinned dependency
    def unidecode(text: str) -> str:
        return text

from .aliases import BUSINESS_ABBREV, LEGAL_SUFFIXES, apply_token_map, merge_address_maps, strip_legal_suffix

_ADDRESS_MAP = merge_address_maps()

_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS_RE = re.compile(r"\s+")
_NUMBER_RE = re.compile(r"\d+")
# Common postal-code shapes: US ZIP (5 or 5+4), Indian PIN (6 digits), a
# generic Canada/UK-style alphanumeric code, and a generic 4-5 digit European
# code (covers the France test split without hard-coding "France" anywhere).
_POSTAL_RE = re.compile(
    r"\b(\d{5}-\d{4}|\d{6}|\d{5}|[A-Za-z]\d[A-Za-z]\s?\d[A-Za-z]\d|\d{4,5}[A-Za-z]{0,2})\b"
)
_UNIT_RE = re.compile(
    r"\b(?:apt|apartment|ste|suite|unit|fl|floor|bldg|building|#)\s*[:.\-]?\s*([a-zA-Z0-9\-]+)",
    flags=re.IGNORECASE,
)


def _to_ascii_lower(text: str) -> str:
    text = unicodedata.normalize("NFKD", text or "")
    text = unidecode(text)
    return text.lower()


def _tokenize(text: str) -> List[str]:
    text = _PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return [t for t in text.split(" ") if t]


@dataclass
class NormalizedName:
    raw: str
    char_norm: str
    tokens: List[str] = field(default_factory=list)
    core: str = ""
    had_legal_suffix: bool = False

    @property
    def token_str(self) -> str:
        return " ".join(self.tokens)

    @property
    def core_sorted(self) -> str:
        return " ".join(sorted(self.tokens))


@dataclass
class NormalizedAddress:
    raw: str
    char_norm: str
    tokens: List[str] = field(default_factory=list)
    numbers: List[str] = field(default_factory=list)
    postal: Optional[str] = None
    unit: Optional[str] = None

    @property
    def token_str(self) -> str:
        return " ".join(self.tokens)


def normalize_name(name: str) -> NormalizedName:
    raw = name or ""
    ascii_lower = _to_ascii_lower(raw)
    char_norm = _WS_RE.sub(" ", _PUNCT_RE.sub(" ", ascii_lower)).strip()
    tokens = _tokenize(ascii_lower)
    tokens = apply_token_map(tokens, BUSINESS_ABBREV)
    canon_tokens = [LEGAL_SUFFIXES.get(t, t) for t in tokens]
    core_tokens, had_suffix = strip_legal_suffix(canon_tokens)
    core = " ".join(core_tokens) if core_tokens else char_norm
    return NormalizedName(raw=raw, char_norm=char_norm, tokens=tokens, core=core, had_legal_suffix=had_suffix)


def normalize_address(address: str) -> NormalizedAddress:
    raw = address or ""
    ascii_lower = _to_ascii_lower(raw)
    char_norm = _WS_RE.sub(" ", _PUNCT_RE.sub(" ", ascii_lower)).strip()
    tokens = _tokenize(ascii_lower)
    tokens = apply_token_map(tokens, _ADDRESS_MAP)
    numbers = _NUMBER_RE.findall(char_norm)
    postal_match = _POSTAL_RE.search(char_norm)
    postal = postal_match.group(0).replace(" ", "") if postal_match else None
    unit_match = _UNIT_RE.search(char_norm)
    unit = unit_match.group(1) if unit_match else None
    return NormalizedAddress(raw=raw, char_norm=char_norm, tokens=tokens, numbers=numbers, postal=postal, unit=unit)


def normalize_dataframe(df: pd.DataFrame, name_col: str = "business_name",
                         addr_col: str = "business_address") -> pd.DataFrame:
    """Return a copy of df with normalized columns added. Original columns are kept."""
    out = df.copy()
    names = [normalize_name(v) for v in out[name_col].tolist()]
    addrs = [normalize_address(v) for v in out[addr_col].tolist()]

    out["name_char"] = [n.char_norm for n in names]
    out["name_tokens"] = [n.token_str for n in names]
    out["name_core"] = [n.core for n in names]
    out["name_core_sorted"] = [n.core_sorted for n in names]

    out["addr_char"] = [a.char_norm for a in addrs]
    out["addr_tokens"] = [a.token_str for a in addrs]
    out["addr_numbers"] = [",".join(a.numbers) for a in addrs]
    out["addr_postal"] = [a.postal or "" for a in addrs]
    out["addr_unit"] = [a.unit or "" for a in addrs]

    out["country_norm"] = out["country"].astype(str).str.strip().str.lower()
    return out
