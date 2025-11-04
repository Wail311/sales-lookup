# etl.py
import pandas as pd
import io
import re
from utils import to_float_aed, parse_size_ml, normalize_name, safe_read_table, load_yaml, detect_aed_header, normalize_barcode, is_tester_text 

def _clean_text_col(s: pd.Series) -> pd.Series:
    return (
        s.astype("string")
         .fillna("")
         .str.replace(r"\s+", " ", regex=True)
         .str.strip()
    )

def load_stock(stock_file) -> pd.DataFrame:
    df = safe_read_table(stock_file)

    # Minimal requirement: just 'name'
    if "name" not in df.columns:
        raise ValueError("Missing column 'name' in stock file. Provide at least: name (and optionally quantity).")

    # Optional: product_id (auto-generate if missing)
    if "product_id" not in df.columns:
        df["product_id"] = [f"SKU-{i+1:06d}" for i in range(len(df))]

    # Optional: quantity
    if "quantity" not in df.columns:
        df["quantity"] = 0
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0).astype("Int64")

    # Optional: brand / concentration / cost_aed
    if "brand" not in df.columns: df["brand"] = ""
    if "concentration" not in df.columns: df["concentration"] = ""
    if "cost_aed" not in df.columns:
        df["cost_aed"] = pd.NA
    else:
        df["cost_aed"] = pd.to_numeric(
            df["cost_aed"].astype(str).str.replace(r"[^\d\.\-]", "", regex=True),
            errors="coerce"
        )

    # size_ml optional → parse from 'name' if missing/blank
    if "size_ml" not in df.columns:
        df["size_ml"] = df["name"].apply(parse_size_ml)
    else:
        s_ml = pd.to_numeric(df["size_ml"], errors="coerce")
        fill = df["name"].apply(parse_size_ml)
        df["size_ml"] = s_ml.fillna(fill)
    df["size_ml"] = pd.to_numeric(df["size_ml"], errors="coerce").astype("Int64")

    # Clean text fields
    for c in ["product_id", "brand", "name", "concentration"]:
        if c in df.columns:
            df[c] = _clean_text_col(df[c])

    # Optional: barcode
    if "barcode" in df.columns:
        df["barcode"] = df["barcode"].apply(normalize_barcode)
    else:
        df["barcode"] = pd.NA


    try:
        syn = load_yaml("config/synonyms.yml") or {}
        tester_tokens = syn.get("tester_tokens", None)
    except Exception:
        tester_tokens = None
    df["is_tester"] = df["name"].apply(lambda s: is_tester_text(s, tokens=tester_tokens))

    # Drop empty names
    df = df[df["name"] != ""].reset_index(drop=True)

    # Auto-detect barcode column on stock
    barcode_candidates = [c for c in df.columns
                          if str(c).strip().lower() in {"barcode","bar code","ean","ean13","upc","upc-a","gtin","gtin-13","gtin14","gtin-14"}]
    if barcode_candidates:
        bc_col = barcode_candidates[0]
        df["barcode"] = df[bc_col].apply(normalize_barcode)
    else:
        # if already present as 'barcode', still normalize
        if "barcode" in df.columns:
            df["barcode"] = df["barcode"].apply(normalize_barcode)
        else:
            df["barcode"] = pd.NA

    return df


def load_competitor_from_config(file, comp_cfg: dict, synonyms=None) -> pd.DataFrame:
    # Handle CSV/XLSX, optional sheet, optional 'detect_aed_header_in_sheet'
    df = safe_read_table(file, sheet=(comp_cfg.get("sheet") or None))
    name_col = comp_cfg.get("name_column")
    price_col = comp_cfg.get("price_column")
    barcode_col = comp_cfg.get("barcode_column")
    detect_in_sheet = comp_cfg.get("detect_aed_header_in_sheet", False)
    if name_col not in df.columns:
        raise ValueError(f"{comp_cfg.get('competitor_id')}: name column '{name_col}' not found. Found: {list(df.columns)}")

    if detect_in_sheet:
        start_row, detected_price_col = detect_aed_header(df, default_col=price_col)
        if not detected_price_col:
            raise ValueError(f"{comp_cfg.get('competitor_id')}: could not detect 'AED' price column in first rows.")
        work = df.iloc[start_row:].copy()
        price_series = work[detected_price_col]
        name_series = work[name_col]
    else:
        price_series = df[price_col] if price_col in df.columns else df.iloc[:, 0] * 0  # fail fast later
        name_series = df[name_col]

    src = df if not comp_cfg.get("detect_aed_header_in_sheet", False) else work

    if not barcode_col:
        candidates = [c for c in src.columns
                      if str(c).strip().lower() in {"barcode","bar code","ean","ean13","upc","upc-a","gtin","gtin-13","gtin14","gtin-14"}]
        barcode_col = candidates[0] if candidates else None

    if barcode_col and barcode_col in src.columns:
        barcode_series = src[barcode_col].apply(normalize_barcode)
    else:
        barcode_series = pd.Series([None] * len(src))



    # Tester flag using synonyms tokens if provided
    tester_tokens = (synonyms or {}).get("tester_tokens") if isinstance(synonyms, dict) else None

    out = pd.DataFrame({
        "competitor_id": comp_cfg.get("competitor_id"),
        "competitor_product_name_raw": name_series.astype(str),
        "price_aed": price_series.apply(to_float_aed),
        "barcode": barcode_series,
    })
    out = out[out["competitor_product_name_raw"].str.strip()!=""]
    out = out[out["price_aed"].notna()]

    out["size_ml"] = out["competitor_product_name_raw"].apply(parse_size_ml)
    out["name_clean"] = out["competitor_product_name_raw"].apply(lambda s: normalize_name(s, strip_tokens=(synonyms or {}).get("strip_tokens") if isinstance(synonyms, dict) else None))
    out["is_tester"] = out["competitor_product_name_raw"].apply(lambda s: is_tester_text(s, tokens=tester_tokens))
    return out.reset_index(drop=True)


# etl.py (append)

import io
import pandas as pd
import re

def _digits_only(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    s = re.sub(r"\D", "", str(x))
    return s or None

def _read_excel_or_csv(upload):
    # Try Excel first
    raw = upload.read()
    b = io.BytesIO(raw)
    name = (getattr(upload, "name", "") or "").lower()

    looks_like_xlsx = name.endswith((".xlsx", ".xls")) or (len(raw) >= 4 and raw[:2] == b"PK")
    if looks_like_xlsx:
        try:
            b.seek(0)
            return pd.read_excel(b, engine="openpyxl")
        except Exception:
            pass

    # CSV encodings
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin1", "cp1256"):
        try:
            b.seek(0)
            return pd.read_csv(io.BytesIO(raw), encoding=enc)
        except Exception:
            continue

    # Fallback: decode text with replacement
    b.seek(0)
    txt = raw.decode("utf-8", errors="replace")
    return pd.read_csv(io.StringIO(txt))

def load_sales_history(upload, mapping: dict) -> pd.DataFrame:
    """
    mapping keys (lowercase, case-insensitive):
      - name  (optional if barcode present)
      - barcode (optional if name present)
      - qty          (avg monthly units sold)
      - avg_price    (avg selling price, AED)
      - qty_total    (optional)
      - months_covered (optional)
    """
    df = _read_excel_or_csv(upload)
    if df is None or df.empty:
        raise ValueError("Empty or unreadable sales file.")

    # Build case-insensitive map
    cols = {c.lower(): c for c in df.columns}

    def pick(key, default=None):
        k = key.lower()
        if k in mapping and mapping[k]:
            return mapping[k]
        return cols.get(k, default)

    c_name   = pick("name")
    c_bar    = pick("barcode")
    c_qty    = pick("qty")
    c_avg    = pick("avg_price")
    c_qtot   = pick("qty_total")
    c_months = pick("months_covered")

    if c_avg is None or (c_name is None and c_bar is None):
        raise ValueError("Need at least avg_price and one of (name or barcode).")

    out = pd.DataFrame()
    if c_name is not None:   out["name"] = df[c_name].astype(str).str.strip().str.upper()
    else:                    out["name"] = pd.NA
    if c_bar is not None:    out["barcode"] = df[c_bar].apply(_digits_only)
    else:                    out["barcode"] = pd.NA

    # qty (avg monthly)
    if c_qty is not None and c_qty in df.columns:
        out["qty"] = pd.to_numeric(df[c_qty], errors="coerce")
    else:
        out["qty"] = pd.NA

    # derive qty if missing but have qty_total / months_covered
    if out["qty"].isna().all() and (c_qtot in df.columns if c_qtot else False) and (c_months in df.columns if c_months else False):
        qtot   = pd.to_numeric(df[c_qtot], errors="coerce")
        months = pd.to_numeric(df[c_months], errors="coerce")
        out["qty"] = (qtot / months).where((qtot.notna()) & (months.notna()) & (months > 0))

    out["avg_price"] = pd.to_numeric(df[c_avg], errors="coerce")

    # remove rows without either name or barcode and without avg_price
    out = out[(out["avg_price"].notna()) & ((out["barcode"].notna()) | (out["name"].notna()))]

    # drop full-dupe lines
    out = out.drop_duplicates(subset=["barcode","name","qty","avg_price"], keep="first").reset_index(drop=True)
    return out
