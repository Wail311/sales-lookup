# match.py
import pandas as pd
from utils import normalize_barcode, barcode_equivalent_keys

# --- Robust scorer: rapidfuzz if available; difflib fallback ---
try:
    from rapidfuzz import fuzz
    def _score(a, b): return int(fuzz.ratio(a or "", b or ""))
except Exception:
    from difflib import SequenceMatcher
    def _score(a, b): return int(SequenceMatcher(None, a or "", b or "").ratio() * 100)

# --- Numeric token helper (import or fallback) ---
try:
    from utils import numeric_tokens
except Exception:
    import re
    def numeric_tokens(s: str):
        if s is None or (isinstance(s, float) and pd.isna(s)):
            return set()
        return set(re.findall(r"\b\d+\b", str(s).upper()))

def _name_key(row: pd.Series) -> str:
    parts = []
    for c in ["brand", "name", "concentration"]:
        if c in row and pd.notna(row[c]):
            parts.append(str(row[c]).upper())
    return " ".join(parts).strip()

def build_stock_index(stock: pd.DataFrame) -> pd.DataFrame:
    s = stock.copy()
    s["name_key"] = s.apply(_name_key, axis=1)
    if "barcode" not in s.columns:   s["barcode"] = pd.NA
    if "is_tester" not in s.columns: s["is_tester"] = False
    # (optional) normalize again defensively
    s["barcode"] = s["barcode"].apply(normalize_barcode)
    return s

def _pick_best_barcode_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    If multiple stock rows share the same barcode for a single offer_row, pick:
      1) same size_ml when both sides have size_ml; else
      2) the first occurrence.
    """
    def pick(group):
        g = group
        if g["size_ml"].notna().any() and g["size_ml_stock"].notna().any():
            same = g[g["size_ml"] == g["size_ml_stock"]]
            if not same.empty:
                return same.iloc[0]
        return g.iloc[0]
    return df.groupby("offer_row", as_index=False, group_keys=False).apply(pick)

def _expand_barcode_keys(df, col="barcode", id_col=None):
    recs = []
    for idx, row in df.iterrows():
        bc = row.get(col)
        if pd.isna(bc) or not bc: continue
        for k in barcode_equivalent_keys(str(bc)):
            r = row.to_dict(); r["barcode_key"] = k
            if id_col and id_col not in r: r[id_col] = idx
            recs.append(r)
    return pd.DataFrame(recs) if recs else pd.DataFrame(columns=list(df.columns)+["barcode_key"])

def match_offers(
    stock_idx: pd.DataFrame,
    offers: pd.DataFrame,
    accept=90,
    borderline_low=86,
    enforce_ml=True,
    barcode_size_check=True,      # kept for diagnostics only
    enforce_tester=True,          # kept for fuzzy stage
    strict_barcode_for_offers=True,
    require_numeric_overlap=True,
):
    # Ensure required columns exist on offers
    for col in ["barcode","size_ml","name_clean","is_tester"]:
        if col not in offers.columns:
            offers[col] = (False if col == "is_tester" else pd.NA)

    # Defensive normalization (in case ETL missed anything)
    offers = offers.copy()
    offers["barcode"] = offers["barcode"].apply(normalize_barcode)
    stock_idx["barcode"] = stock_idx["barcode"].apply(normalize_barcode)

    stock_idx = stock_idx.copy()
    if "barcode" not in stock_idx.columns:
        stock_idx["barcode"] = pd.NA
    stock_idx["barcode"] = stock_idx["barcode"].apply(normalize_barcode)

    # Stable row id for offers
    offers = offers.reset_index(drop=False).rename(columns={"index": "offer_row"})

    # ---------- 1) BARCODE PASS (absolute priority; no tester/size filters) ----------
    bc_offers = offers[offers["barcode"].notna()].copy()
    bc_stock  = stock_idx[stock_idx["barcode"].notna()][["product_id","size_ml","name_key","barcode","is_tester"]] \
                .rename(columns={"product_id":"product_id_stock","size_ml":"size_ml_stock","name_key":"name_key_stock","is_tester":"is_tester_stock"})

    accepted_barcode = pd.DataFrame()
    if not bc_offers.empty and not bc_stock.empty:
        o_keys = _expand_barcode_keys(bc_offers, col="barcode", id_col="offer_row")
        s_keys = _expand_barcode_keys(bc_stock,   col="barcode", id_col=None)
        if not o_keys.empty and not s_keys.empty:
            bc_join = o_keys.merge(s_keys.drop_duplicates(subset=["barcode_key"]),
                                   on="barcode_key", how="inner", suffixes=("","_stock"))
            if not bc_join.empty:
                # pick best (prefer same size)
                def _pick(g):
                    if g["size_ml"].notna().any() and g["size_ml_stock"].notna().any():
                        same = g[g["size_ml"] == g["size_ml_stock"]]
                        return same.iloc[0] if not same.empty else g.iloc[0]
                    return g.iloc[0]
                bc_join = bc_join.groupby("offer_row", as_index=False, group_keys=False).apply(_pick)
                accepted_barcode = bc_join.copy()
                accepted_barcode["score"] = 100
                accepted_barcode["match_method"] = "barcode"
                accepted_barcode = accepted_barcode.rename(columns={
                    "competitor_id":"competitor",
                    "competitor_product_name_raw":"offer_name",
                    "price_aed":"offer_price_aed",
                    "product_id_stock":"matched_product_id",
                    "name_key_stock":"matched_name_key",
                })[
                    ["offer_row","competitor","offer_name","offer_price_aed",
                     "matched_product_id","matched_name_key","size_ml","size_ml_stock",
                     "barcode","barcode_key","score","match_method","is_tester","is_tester_stock"]
                ]

    matched_rows = set(accepted_barcode["offer_row"].tolist()) if not accepted_barcode.empty else set()

    # If an offer HAS a barcode but didn't match, do NOT fuzz it (prevents wrong pairings)
    if strict_barcode_for_offers:
        offers_rem = offers[(~offers["offer_row"].isin(matched_rows)) & (offers["barcode"].isna())].copy()
    else:
        offers_rem = offers[~offers["offer_row"].isin(matched_rows)].copy()

    # --- DIAGNOSTIC: why did barcoded offers NOT match? ---
    debug_unmatched = pd.DataFrame()
    if not bc_offers.empty:
        probe = bc_offers.merge(
            bc_stock.rename(columns={
                "product_id_stock":"_probe_pid",   # harmless if names differ
                "product_id":"_probe_pid",
                "size_ml_stock":"_probe_size",
                "size_ml":"_probe_size",
                "is_tester_stock":"_probe_tester",
                "is_tester":"_probe_tester",
                "name_key_stock":"_probe_name",
                "name_key":"_probe_name",
            }, errors="ignore"),
            on="barcode", how="left"
        )
        probe["barcode_reason"] = probe.get("_probe_pid").isna().map(
            {True: "no_stock_barcode_match", False: "has_stock_barcode"}
        )
        # Tester / size notes (diagnostic only)
        if barcode_size_check and "_probe_size" in probe.columns:
            def _size_ok(r):
                if pd.isna(r.get("_probe_pid")): return pd.NA
                if pd.isna(r.get("size_ml")) or pd.isna(r.get("_probe_size")): return True
                return r.get("size_ml") == r.get("_probe_size")
            probe["size_ok"] = probe.apply(_size_ok, axis=1)
            probe.loc[(probe["size_ok"] == False), "barcode_reason"] = "size_mismatch"
        if "is_tester" in probe.columns and "_probe_tester" in probe.columns:
            probe["tester_ok"] = probe.apply(
                lambda r: (r.get("is_tester") == r.get("_probe_tester")) if pd.notna(r.get("_probe_pid")) else pd.NA,
                axis=1
            )
            probe.loc[(probe["tester_ok"] == False), "barcode_reason"] = "tester_mismatch"

        accepted_rows = set(accepted_barcode["offer_row"]) if not accepted_barcode.empty else set()
        debug_unmatched = probe[~probe["offer_row"].isin(accepted_rows)][[
            "offer_row","competitor_id","competitor_product_name_raw","barcode","price_aed","barcode_reason"
        ]].copy()

    # ---------- 2) FUZZY NAME (only for non-barcoded offers) ----------
    stock_fuzzy = stock_idx.copy()
    if enforce_ml:
        offers_rem = offers_rem[offers_rem["size_ml"].notna()]
        stock_fuzzy = stock_fuzzy[stock_fuzzy["size_ml"].notna()]

    stock_cols = ["product_id","size_ml","name_key","is_tester"]

    if enforce_ml:
        cand = offers_rem.merge(stock_fuzzy[stock_cols], on="size_ml", how="left", suffixes=("","_stock"))
    else:
        cand = offers_rem.assign(_k=1).merge(
            stock_fuzzy[stock_cols].assign(_k=1), on="_k", how="left", suffixes=("","_stock")
        ).drop(columns=["_k"])

    if enforce_tester:
        cand = cand[cand["is_tester"].fillna(False) == cand["is_tester_stock"].fillna(False)]

    # Numeric token overlap guard (e.g., block "Guidance 46" → "Guidance" unless '46' also in stock)
    if require_numeric_overlap and not cand.empty:
        cand["offer_digits"] = cand["name_clean"].apply(numeric_tokens)
        cand["stock_digits"] = cand["name_key"].apply(numeric_tokens)

        def _has_overlap(row):
            od = row["offer_digits"]; sd = row["stock_digits"]
            if not isinstance(od, set) or not isinstance(sd, set): return True
            if len(od) == 0 or len(sd) == 0: return True
            return len(od.intersection(sd)) > 0

        cand = cand[cand.apply(_has_overlap, axis=1)]

    if cand.empty:
        accepted_fuzzy = borderline_fuzzy = rejected_fuzzy = pd.DataFrame(columns=[
            "competitor","offer_name","offer_price_aed","matched_product_id","matched_name_key","score","match_method"
        ])
    else:
        cand["score"] = [_score(a, b) for a, b in zip(cand["name_clean"].astype(str), cand["name_key"].astype(str))]
        best = cand.sort_values("score", ascending=False).groupby("offer_row").head(1)

        accepted_fuzzy   = best[best["score"] >= accept].copy()
        borderline_fuzzy = best[(best["score"] >= borderline_low) & (best["score"] < accept)].copy()
        rejected_fuzzy   = best[best["score"] < borderline_low].copy()

        for df in (accepted_fuzzy, borderline_fuzzy, rejected_fuzzy):
            df["match_method"] = "fuzzy"
            df.rename(columns={
                "competitor_id":"competitor",
                "competitor_product_name_raw":"offer_name",
                "price_aed":"offer_price_aed",
                "product_id":"matched_product_id",
                "name_key":"matched_name_key",
            }, inplace=True)

    accepted   = pd.concat([accepted_barcode, accepted_fuzzy], ignore_index=True) if not accepted_barcode.empty else accepted_fuzzy
    borderline = borderline_fuzzy
    rejected   = rejected_fuzzy

    if not accepted.empty:
        accepted = accepted.sort_values(["match_method","score"], ascending=[True, False]) \
                           .drop_duplicates(subset=["offer_row"], keep="first")

    return accepted, borderline, rejected, debug_unmatched


