import re
import pandas as pd
import yaml
from decimal import Decimal, InvalidOperation

_OZ_TO_ML = 29.5735

ML_PATTERNS = [re.compile(r"(\d{1,4})\s*ML\b", re.I)]
OZ_PATTERNS = [re.compile(r"(\d+(?:\.\d+)?)\s*(?:FL\.?\s*OZ|FL\s*OZ|OZ)\b", re.I)]

def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def to_float_aed(x):
    if pd.isna(x): return None
    s = str(x)
    s = s.replace(",", "").replace("AED", "").replace("د.إ", "").replace("د", "")
    s = re.sub(r"[^\d\.\-]", "", s)
    try:
        return float(s) if s else None
    except Exception:
        return None

def parse_size_ml(name: str):
    if not isinstance(name, str): return None
    u = name.upper()
    for pat in ML_PATTERNS:
        m = pat.search(u)
        if m:
            try:
                ml = int(m.group(1))
                if 2 <= ml <= 250: return ml
            except: pass
    for pat in OZ_PATTERNS:
        m = pat.search(u)
        if m:
            try:
                oz = float(m.group(1))
                ml = oz * _OZ_TO_ML
                ml_round = int(5 * round(ml / 5.0))
                if 2 <= ml_round <= 250: return ml_round
            except: pass
    return None

def normalize_barcode(val):
    if val is None or (isinstance(val, float) and pd.isna(val)): return None
    s = str(val).strip()
    if s == "": return None
    try:
        if re.fullmatch(r"[+-]?\d+(\.\d+)?([eE][+-]?\d+)?", s):
            s = format(Decimal(s).quantize(Decimal("1")), "f")
    except (InvalidOperation, Exception):
        pass
    runs = re.findall(r"\d{8,}", s)
    return max(runs, key=len) if runs else None

def barcode_equivalent_keys(bc: str):
    keys = set()
    if not bc: return keys
    keys.add(bc)
    if len(bc) == 13 and bc.startswith("0"): keys.add(bc[1:])   # EAN13 -> UPC12
    if len(bc) == 12: keys.add("0"+bc)                         # UPC12 -> EAN13
    if len(bc) == 14 and bc.startswith("0"): keys.add(bc[1:])  # GTIN14 -> EAN13
    return keys

def numeric_tokens(s: str):
    """Return set of standalone digit tokens ('46', '100', '212')."""
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return set()
    return set(re.findall(r"\b\d+\b", str(s).upper()))


def is_tester_text(s: str, tokens=None) -> bool:
    """Detect tester SKUs by tokens; whole-word, case-insensitive."""
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return False
    s2 = str(s).upper()
    toks = tokens or ["TESTER", "tester", "Tester"]
    pattern = r'(?<!\w)(' + "|".join(map(re.escape, toks)) + r')(?!\w)'
    return bool(re.search(pattern, s2))


def normalize_name(name: str, strip_tokens=None):
    if not isinstance(name, str): return ""
    s = name.upper()
    s = re.sub(r"[^A-Z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if strip_tokens:
        for t in strip_tokens:
            s = re.sub(rf"\b{re.escape(t)}\b", "", s)
        s = re.sub(r"\s+", " ", s).strip()
    return s

def safe_read_table(file, sheet=None):
    name = getattr(file, "name", str(file))
    if name.lower().endswith(".csv"):
        return pd.read_csv(file, dtype="object")
    else:
        if sheet:
            return pd.read_excel(file, sheet_name=sheet, dtype="object", engine="openpyxl")
        return pd.read_excel(file, dtype="object", engine="openpyxl")

def detect_aed_header(df: pd.DataFrame, default_col: str = None):
    # Scan first ~40 rows for a cell equal to "AED" and use that column, then data starts next row
    aed_cell = None
    max_scan = min(40, len(df))
    for r in range(max_scan):
        row = df.iloc[r]
        for c in df.columns:
            val = row[c]
            if isinstance(val, str) and val.strip().upper() == "AED":
                aed_cell = (r, c); break
        if aed_cell: break
    if aed_cell:
        start_row, aed_col = aed_cell[0] + 1, aed_cell[1]
        return start_row, aed_col
    if default_col and default_col in df.columns:
        return 1, default_col
    return None, None
