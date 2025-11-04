import io, time
import pandas as pd
import numpy as np
import streamlit as st

from utils import load_yaml
from etl import load_stock, load_competitor_from_config
from match import build_stock_index, match_offers
from price import aggregate_market, apply_pricing, consistency_pairs

# ---- Safe init of session DataFrames ----
def _ensure_df_key(key: str):
    if key not in st.session_state or not isinstance(st.session_state[key], pd.DataFrame):
        st.session_state[key] = pd.DataFrame()

for _k in ["pairs_table", "aggregated_table", "priced_table", "below_table", "costgap_table",
           "accepted_all", "borderline_all", "sales_history", "sales_join"]:
    _ensure_df_key(_k)


def warn_duplicate_barcodes(df: pd.DataFrame, label: str):
    """Show a Streamlit warning if the DataFrame has duplicate barcodes."""
    if df is None or "barcode" not in df.columns:
        return
    # treat barcodes as strings; ignore blanks/NaN
    bc = df["barcode"].astype(str).str.strip()
    bc = bc[bc.notna() & (bc != "")]
    dupes = bc.value_counts()
    dupes = dupes[dupes > 1]
    if not dupes.empty:
        st.warning(
            f"Duplicate barcodes in **{label}**: {len(dupes)} unique barcodes repeated. "
            f"Examples: {dupes.head(5).to_dict()}"
        )
        with st.expander(f"Show duplicate {label} rows"):
            st.dataframe(df[df["barcode"].astype(str).str.strip().isin(dupes.index)], use_container_width=True)


# -----------------------------------------------------------------------------
# App chrome
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Perfume Pricing Engine — Pro", layout="wide")
st.title("🧪 Perfume Pricing Engine — Pro")

syn = load_yaml("config/synonyms.yml")
pol = load_yaml("config/policy.yml")
comp_cfg = load_yaml("config/competitors.yml").get("competitors", [])

# -----------------------------------------------------------------------------
# 1) STOCK
# -----------------------------------------------------------------------------
with st.expander("1) Upload your STOCK list (canonical products)", expanded=True):
    st.markdown("You can upload **simple stock** (name, quantity) or **stock with costs** (name, quantity, optional cost).")
    stock_file = st.file_uploader("Upload stock CSV or XLSX", type=["csv","xlsx"], key="stock")

    cta1, cta2 = st.columns(2)
    with cta1:
        st.download_button("Download simple stock template",
                           data=open("templates/stock_template.csv","rb").read(),
                           file_name="stock_template.csv", mime="text/csv")
    with cta2:
        st.download_button("Download stock-with-costs template",
                           data=open("templates/stock_with_costs_template.csv","rb").read(),
                           file_name="stock_with_costs_template.csv", mime="text/csv")

    # Load & persist stock
    if stock_file:
        try:
            stock_df_loaded = load_stock(stock_file)
            st.session_state["stock_df"] = stock_df_loaded
            st.success(f"Loaded stock: {stock_df_loaded.shape[0]} rows")
            st.dataframe(stock_df_loaded.head(20), use_container_width=True)
            warn_duplicate_barcodes(stock_df_loaded, "STOCK")

            # --- Barcode sanity ---
            bc = stock_df_loaded.get("barcode")
            if bc is not None:
                nonnull = bc.notna().sum()
                lengths = bc.dropna().astype(str).str.len().value_counts().to_dict()
                st.caption(f"Stock barcodes present: **{nonnull}** rows. Lengths: {lengths}")
        except Exception as e:
            st.error(f"Stock load error: {e}")
    elif "stock_df" in st.session_state:
        stock_df_loaded = st.session_state["stock_df"]
        st.info(f"Using previously loaded stock: {stock_df_loaded.shape[0]} rows")
        st.dataframe(stock_df_loaded.head(20), use_container_width=True)
        warn_duplicate_barcodes(stock_df_loaded, "STOCK")

# -----------------------------------------------------------------------------
# 2) COMPETITORS
# -----------------------------------------------------------------------------
st.divider()
st.header("2) Add competitor price lists")

if "competitors" not in st.session_state:
    st.session_state["competitors"] = []

with st.form("add_comp"):
    c1, c2, c3 = st.columns([2,2,2])
    with c1:
        known_ids = ["(custom mapping)"] + [c["competitor_id"] for c in comp_cfg]
        pick = st.selectbox("Choose competitor (from config) or use custom", known_ids, index=0)
        comp_name = st.text_input("Competitor name / ID", value="" if pick != "(custom mapping)" else "COMP_NEW")
    with c2:
        file = st.file_uploader("Competitor file (CSV/XLSX)", type=["csv","xlsx"], key="comp_file")
        sheet = st.text_input("Sheet name (optional)", value="")
    with c3:
        if pick != "(custom mapping)":
            chosen = next((c for c in comp_cfg if c["competitor_id"] == pick), None)
            name_col    = st.text_input("Product Name column", value=chosen.get("name_column","") if chosen else "")
            price_col   = st.text_input("Price (AED) column", value=chosen.get("price_column","") if chosen else "")
            barcode_col = st.text_input("Barcode column (optional)", value=chosen.get("barcode_column","") if chosen else "")
            detect      = st.checkbox("Detect AED header in-sheet", value=chosen.get("detect_aed_header_in_sheet", False))
            comp_name   = chosen["competitor_id"] if chosen else comp_name
        else:
            name_col    = st.text_input("Product Name column", value="")
            price_col   = st.text_input("Price (AED) column", value="")
            barcode_col = st.text_input("Barcode column (optional)", value="")
            detect      = st.checkbox("Detect AED header in-sheet", value=False)
    add = st.form_submit_button("Add competitor")

if add:
    if not (comp_name and file and name_col and (price_col or detect)):
        st.warning("Please provide name, file, Name column and either a Price column or enable 'Detect AED header'.")
    else:
        st.session_state["competitors"].append({
            "competitor_id": comp_name,
            "file": file,
            "sheet": sheet.strip() or None,
            "name_column": name_col.strip(),
            "price_column": price_col.strip(),
            "detect_aed_header_in_sheet": detect,
            "barcode_column": (barcode_col.strip() or None),
        })

if st.session_state["competitors"]:
    st.subheader("Queued competitors")
    st.dataframe(pd.DataFrame([{
        "competitor": c["competitor_id"],
        "sheet": c["sheet"],
        "name_col": c["name_column"],
        "price_col": c["price_column"],
        "barcode_col": c["barcode_column"],
        "detect_aed_header": c["detect_aed_header_in_sheet"],
    } for c in st.session_state["competitors"]]), use_container_width=True)

# -----------------------------------------------------------------------------
# 3) MATCH
# -----------------------------------------------------------------------------
st.divider()
st.header("3) Run matching (barcode-first; fuzzy only for non-barcoded offers)")

# small helper: check a specific barcode presence in stock/offers
with st.expander("🔎 Quick barcode check (optional)"):
    q_bc = st.text_input("Check if a specific barcode exists (enter digits only)", value="")
    if q_bc:
        stock_df_dbg = st.session_state.get("stock_df")
        if stock_df_dbg is not None and "barcode" in stock_df_dbg.columns:
            in_stock = any(stock_df_dbg["barcode"].astype(str) == q_bc)
        else:
            in_stock = False
        st.write("In STOCK:", in_stock)
        for c in st.session_state.get("competitors", []):
            try:
                offers_dbg = load_competitor_from_config(c["file"], c, synonyms=syn)
                warn_duplicate_barcodes(offers_dbg, f"OFFERS — {c['competitor_id']}")
                in_offers = any(offers_dbg.get("barcode", pd.Series(dtype=str)).astype(str) == q_bc)
            except Exception:
                in_offers = False
            st.write(f"In OFFERS ({c['competitor_id']}):", in_offers)

run_match = st.button("⚙️ Match now", key="match_now")

if run_match:
    stock_df = st.session_state.get("stock_df")
    if stock_df is None:
        st.error("Please upload stock first.")
    elif not st.session_state.get("competitors"):
        st.error("Please add at least one competitor file.")
    else:
        stock_idx = build_stock_index(stock_df)
        acc_all, bor_all, rej_all = [], [], []
        debug_tables = []

        for c in st.session_state["competitors"]:
            try:
                offers = load_competitor_from_config(c["file"], c, synonyms=syn)
            except Exception as e:
                st.error(f"{c['competitor_id']}: load error — {e}")
                continue

            # ML gate
            if pol.get("ml_gate", {}).get("enforce", True):
                offers = offers[offers["size_ml"].notna()]

            accept = pol.get("match_thresholds", {}).get("accept", 90)
            borderline_low = pol.get("match_thresholds", {}).get("borderline_low", 86)

            a, b, r, debug = match_offers(
                stock_idx, offers,
                accept=accept, borderline_low=borderline_low,
                enforce_ml=pol.get("ml_gate", {}).get("enforce", True),
                barcode_size_check=pol.get("barcode", {}).get("require_same_size_on_barcode", True),
                enforce_tester=pol.get("tester", {}).get("enforce_separation", True),
                strict_barcode_for_offers=pol.get("barcode", {}).get("strict_for_offers", True),
                require_numeric_overlap=pol.get("fuzzy", {}).get("require_numeric_token_overlap", True),
            )

            if debug is not None and not debug.empty:
                debug["competitor"] = c["competitor_id"]
                debug_tables.append(debug)

            a["competitor"] = c["competitor_id"]
            b["competitor"] = c["competitor_id"]
            r["competitor"] = c["competitor_id"]

            if not a.empty: acc_all.append(a)
            if not b.empty: bor_all.append(b)
            if not r.empty: rej_all.append(r)

        if acc_all:
            accepted_all = pd.concat(acc_all, ignore_index=True)
            st.session_state["accepted_all"] = accepted_all
            st.success(f"Accepted matches: {accepted_all.shape[0]}")
            st.dataframe(accepted_all.head(200), use_container_width=True)
        else:
            st.warning("No accepted matches found.")

        if bor_all:
            borderline_all = pd.concat(bor_all, ignore_index=True)
            st.session_state["borderline_all"] = borderline_all
            st.info(f"Borderline matches: {borderline_all.shape[0]}")
            st.dataframe(borderline_all.head(200), use_container_width=True)

        if rej_all:
            st.session_state["rejected_all"] = pd.concat(rej_all, ignore_index=True)

        if debug_tables:
            with st.expander("🔎 Barcoded offers that didn’t match — reasons"):
                st.dataframe(pd.concat(debug_tables, ignore_index=True), use_container_width=True)

# ---------- 3.5) (Optional) Sales History ----------
st.divider()
st.header("3.5) (Optional) Upload Sales History")

with st.expander("Upload sales history (qty & avg_price)", expanded=False):
    st.caption("Provide **qty** = avg monthly units, **avg_price** = avg selling price (AED). Map the columns below.")
    sales_file = st.file_uploader("Sales history (Excel/CSV)", type=["xlsx","xls","csv"], key="sales_hist")

    # Simple mapping UI
    col1, col2, col3 = st.columns(3)
    with col1:
        m_name   = st.text_input("Name column (optional if barcode)", value="name")
        m_bar    = st.text_input("Barcode column (optional if name)", value="barcode")
    with col2:
        m_qty    = st.text_input("qty (avg monthly) column", value="qty")
        m_avg    = st.text_input("avg_price column (AED)", value="avg_price")
    with col3:
        m_qtot   = st.text_input("qty_total column (optional)", value="")
        m_months = st.text_input("months_covered column (optional)", value="")

    load_sales_btn = st.button("Load Sales History")

    if load_sales_btn:
        try:
            from etl import load_sales_history
            mapping = {
                "name": m_name.strip() or None,
                "barcode": m_bar.strip() or None,
                "qty": m_qty.strip() or None,
                "avg_price": m_avg.strip() or None,
                "qty_total": m_qtot.strip() or None,
                "months_covered": m_months.strip() or None,
            }
            sales_df = load_sales_history(sales_file, mapping)
            if sales_df is None or sales_df.empty:
                st.warning("No usable rows found in sales file.")
            else:
                st.session_state["sales_history"] = sales_df
                st.success(f"Loaded sales history: {sales_df.shape[0]} rows")
                st.dataframe(sales_df.head(20), use_container_width=True)
        except Exception as e:
            st.error(f"Sales load error: {e}")


# -----------------------------------------------------------------------------
# 4) AGGREGATE & PRICE (consolidated, session-safe)
# -----------------------------------------------------------------------------
st.divider()
st.header("4) Aggregate market, check consistency, and price")

cfg_inc = pol.get("aggregation", {}).get("include_borderline", False)
include_borderline = st.checkbox(
    "Include borderline matches (86–90) in aggregation & pricing",
    value=cfg_inc, key="include_borderline"
)

if st.button("📊 Aggregate & export", key="aggregate_export"):
    # -------------------- 4.1 Aggregate & base pricing --------------------
    stock_df       = st.session_state.get("stock_df")
    accepted_all   = st.session_state.get("accepted_all")
    borderline_all = st.session_state.get("borderline_all")

    if not isinstance(stock_df, pd.DataFrame) or stock_df.empty:
        st.error("Please upload stock first."); st.stop()
    if not isinstance(accepted_all, pd.DataFrame) or accepted_all.empty:
        st.error("No accepted matches yet—click “⚙️ Match now” first."); st.stop()

    # Build offers used for aggregation
    offers_for_agg = accepted_all.assign(match_bucket="accepted")
    if include_borderline and isinstance(borderline_all, pd.DataFrame) and not borderline_all.empty:
        offers_for_agg = pd.concat(
            [offers_for_agg, borderline_all.assign(match_bucket="borderline")],
            ignore_index=True
        )

    # Consistency (tick) pairs
    tick_cfg = pol.get("tick_rule", {})
    pairs = consistency_pairs(offers_for_agg, tick_cfg)

    # Market aggregation (with outlier handling)
    outlier_cfg = pol.get("pricing", {}).get("outlier", {})
    method  = outlier_cfg.get("method", "iqr")
    iqr_k   = outlier_cfg.get("iqr_k", 1.5)
    stdev_k = outlier_cfg.get("stdev_k", 2.0)

    aggregated = aggregate_market(
        offers_for_agg, stock_df,
        outlier_method=method, iqr_k=iqr_k, stdev_k=stdev_k
    )

    # Base pricing (market undercut strategy)
    pr_cfg     = pol.get("pricing", {})
    strat      = pr_cfg.get("strategy", "undercut_min_1pct")
    min_margin = pr_cfg.get("default_min_margin_pct", 0.20)
    round_to   = pr_cfg.get("round_to", 1)
    st.session_state["round_to_step"] = round_to  # keep for later blocks

    priced = apply_pricing(aggregated, strategy=strat, min_margin=min_margin, round_to=round_to)

    # Slices
    below_df = priced[pd.to_numeric(priced.get("below_cost_flag"), errors="coerce") == 1].copy() \
               if "below_cost_flag" in priced.columns else pd.DataFrame()
    costgap_df = priced[priced.get("cost_gap_flag") == True].copy() \
                 if "cost_gap_flag" in priced.columns else pd.DataFrame()

    # Persist base results
    st.session_state.update({
        "pairs_table":       pairs,
        "aggregated_table":  aggregated,
        "priced_table":      priced,          # important
        "below_table":       below_df,
        "costgap_table":     costgap_df,
    })

# -----------------------------------------------------------------------------
# 5) SALES HISTORY (optional): name+size(+tester) fuzzy join → slow/fast prices
# -----------------------------------------------------------------------------
st.divider()
st.header("5) (Optional) Sales history")

sales_df   = st.session_state.get("sales_history")
priced_tbl = st.session_state.get("priced_table")

# Config: fuzzy threshold (≥91 recommended for safety)
FUZZY_THRESHOLD = int(st.session_state.get("FUZZY_THRESHOLD", 91))

# Always start with a valid frame
sales_join = pd.DataFrame(columns=["product_id", "qty", "avg_price"])

def _round_to(v, step):
    import pandas as _pd
    if _pd.isna(v): return _pd.NA
    try: return round(float(v) / step) * step
    except Exception: return _pd.NA

if isinstance(sales_df, pd.DataFrame) and not sales_df.empty \
   and isinstance(priced_tbl, pd.DataFrame) and not priced_tbl.empty:

    # --- Detect & normalize columns in SALES ---
    sd = sales_df.copy()
    sd.columns = [str(c).strip() for c in sd.columns]
    low = {c.lower(): c for c in sd.columns}

    def pick(names): 
        for n in names:
            if n in low: return low[n]
        return None

    col_name = pick(["name","product","product name","item"])
    col_qty  = pick(["qty","quantity","qty_month","qty per month"])
    col_avg  = pick(["avg_price","average_price","avg price","average price","price_avg"])
    col_sz   = pick(["size_ml","ml","size"])
    col_tst  = pick(["is_tester","tester","is tester"])

    if col_name is None:
        st.warning("Sales file needs a product name column (e.g., name/product).")
    else:
        sd.rename(columns={col_name: "name"}, inplace=True)
        if col_qty: sd.rename(columns={col_qty: "qty"}, inplace=True)
        else:       sd["qty"] = pd.NA
        if col_avg: sd.rename(columns={col_avg: "avg_price"}, inplace=True)
        else:       sd["avg_price"] = pd.NA
        if col_sz:  sd.rename(columns={col_sz: "size_ml"}, inplace=True)
        if col_tst: sd.rename(columns={col_tst: "is_tester"}, inplace=True)

        # Normalize fields
        sd["name_key"]  = sd["name"].astype(str).str.strip().str.upper()
        sd["qty"]       = pd.to_numeric(sd["qty"], errors="coerce")
        sd["avg_price"] = pd.to_numeric(sd["avg_price"], errors="coerce")
        if "size_ml" in sd.columns:
            sd["size_ml"] = pd.to_numeric(sd["size_ml"], errors="coerce")
        if "is_tester" in sd.columns:
            sd["is_tester"] = sd["is_tester"].astype(str).str.strip().str.lower().isin(["1","true","yes","y","t"])

        # Drop rows with neither qty nor avg_price (no contribution)
        sd = sd[(sd["qty"].notna()) | (sd["avg_price"].notna())].copy()

        # --- PRICEBOOK side ---
        pm = priced_tbl.copy()
        if "name" not in pm.columns:
            pm["name"] = pm["product"] if "product" in pm.columns else pd.NA
        pm["name_key"] = pm["name"].astype(str).str.strip().str.upper()
        if "size_ml" in pm.columns:
            pm["size_ml"] = pd.to_numeric(pm["size_ml"], errors="coerce")
        if "is_tester" not in pm.columns:
            pm["is_tester"] = False

        # --- Candidate narrowing by size/tester (greatly reduces fuzzy mistakes) ---
        # If sales has size_ml, match only rows with same size; else allow any.
        if "size_ml" in sd.columns and "size_ml" in pm.columns:
            sd_size_known = sd[sd["size_ml"].notna()].copy()
            sd_size_na    = sd[sd["size_ml"].isna()].copy()
        else:
            sd_size_known = pd.DataFrame(columns=sd.columns)
            sd_size_na    = sd.copy()

        # If sales has tester flag, align tester; else allow any.
        def _pm_slice_for(r):
            base = pm
            if "size_ml" in sd.columns and pd.notna(r.get("size_ml")) and "size_ml" in pm.columns:
                base = base[base["size_ml"] == float(r["size_ml"])]
            if "is_tester" in sd.columns and isinstance(r.get("is_tester"), (bool,)) and "is_tester" in pm.columns:
                base = base[base["is_tester"] == bool(r["is_tester"])]
            return base[["product_id","name_key"]].dropna()

        # --- Fuzzy match name within narrowed candidates ---
        rows = []
        try:
            from rapidfuzz import process, fuzz
            use_rf = True
        except Exception:
            import difflib
            use_rf = False

        for _, r in sd.iterrows():
            q = str(r["name_key"])
            base = _pm_slice_for(r)
            if base.empty:
                continue
            choices = base["name_key"].tolist()
            if use_rf:
                hit = process.extractOne(q, choices, scorer=fuzz.WRatio)
                if not hit:
                    continue
                choice, score, idx = hit
                if score < FUZZY_THRESHOLD:
                    continue
                pid = base.iloc[idx]["product_id"]
            else:
                hits = difflib.get_close_matches(q, choices, n=1, cutoff=0.0)
                if not hits:
                    continue
                choice = hits[0]
                pid = base.loc[base["name_key"] == choice, "product_id"].iloc[0]
                score = 100  # coarse fallback

            rows.append({"product_id": pid, "qty": r.get("qty"), "avg_price": r.get("avg_price")})

        if rows:
            tmp = pd.DataFrame(rows)
            tmp["qty"]       = pd.to_numeric(tmp["qty"], errors="coerce")
            tmp["avg_price"] = pd.to_numeric(tmp["avg_price"], errors="coerce")

            # Aggregate per SKU: qty=sum; avg_price=weighted avg by qty (fallback mean)
            def _agg(g):
                q  = g["qty"].fillna(0.0)
                ap = g["avg_price"]
                qsum = float(q.sum())
                if qsum > 0 and ap.notna().any():
                    wavg = float((ap.fillna(0) * q).sum() / qsum)
                else:
                    wavg = float(ap.mean(skipna=True)) if ap.notna().any() else float("nan")
                return pd.Series({"product_id": g.name, "qty": (qsum if qsum > 0 else pd.NA), "avg_price": wavg})

            sales_join = (
                tmp.groupby("product_id", as_index=False)
                   .apply(_agg)
                   .reset_index(drop=True)
            )

# Diagnostics
st.caption(
    f"Sales rows uploaded: {0 if sales_df is None else sales_df.shape[0]}  |  "
    f"SKUs matched from sales: {0 if sales_join is None else sales_join.shape[0]}"
)

# ---- Merge onto pricebook & compute Slow/Fast prices ----
if isinstance(priced_tbl, pd.DataFrame) and not priced_tbl.empty:
    updated_priced = priced_tbl.copy()
else:
    updated_priced = pd.DataFrame()

if not updated_priced.empty and isinstance(sales_join, pd.DataFrame) and not sales_join.empty:
    updated_priced = updated_priced.merge(sales_join, on="product_id", how="left", suffixes=("", "_sales"))
    # Coalesce (robust to duplicate column labels returning DataFrames)
    def _coalesce_column(df: pd.DataFrame, base_col: str, alt_col: str):
        if alt_col not in df.columns:
            return df
        base_obj = df[base_col]
        alt_obj  = df[alt_col]
        # If selecting by label returns a DataFrame (duplicate column names), take the first occurrence
        if isinstance(base_obj, pd.DataFrame):
            base_series = base_obj.iloc[:, 0]
        else:
            base_series = base_obj
        if isinstance(alt_obj, pd.DataFrame):
            alt_series = alt_obj.iloc[:, 0]
        else:
            alt_series = alt_obj
        df[base_col] = base_series.where(base_series.notna(), alt_series)
        return df

    for _base, _alt in [("qty","qty_sales"), ("avg_price","avg_price_sales")]:
        updated_priced = _coalesce_column(updated_priced, _base, _alt)
    # Drop duplicate columns that may arise from repeated merges in a rerun
    updated_priced = updated_priced.loc[:, ~updated_priced.columns.duplicated()]

# Ensure columns exist
for col in [
    "recommended_price", "avg_price", "qty", "cost_aed",
    "price_market_undercut", "price_slow_sales", "price_fast_premium",
    "slow_candidate_below_cost", "fast_candidate_below_cost"
]:
    if col not in updated_priced.columns:
        updated_priced[col] = pd.NA

# Numerics for rules
updated_priced["qty"]       = pd.to_numeric(updated_priced["qty"], errors="coerce")
updated_priced["avg_price"] = pd.to_numeric(updated_priced["avg_price"], errors="coerce")

# Mirrors current recommended
updated_priced["price_market_undercut"] = updated_priced["recommended_price"]

round_step = int(st.session_state.get("round_to_step", 1) or 1)
avg_series  = pd.to_numeric(updated_priced["avg_price"], errors="coerce")
qty_series  = pd.to_numeric(updated_priced["qty"], errors="coerce")
undercut    = pd.to_numeric(updated_priced["price_market_undercut"], errors="coerce")
cost_series = pd.to_numeric(updated_priced.get("cost_aed"), errors="coerce")
has_sales   = avg_series.notna()

# Slow movers (<10): sales-only
cond_slow = has_sales & (qty_series < 10)
updated_priced.loc[cond_slow, "price_slow_sales"] = avg_series[cond_slow].apply(lambda v: _round_to(v, round_step))

# Fast movers (>=10): 5% premium to max(avg_price, undercut)
cond_fast = has_sales & (qty_series >= 10)
fast_cand = pd.DataFrame({"sales_premium": avg_series * 1.05, "undercut": undercut}).max(axis=1)
updated_priced.loc[cond_fast, "price_fast_premium"] = fast_cand[cond_fast].apply(lambda v: _round_to(v, round_step))

# Flags BEFORE cost guard
def _lt_cost(p, c):
    try:    return float(p) < float(c)
    except: return False
updated_priced["slow_candidate_below_cost"] = [_lt_cost(p, c) for p, c in zip(updated_priced["price_slow_sales"],   cost_series)]
updated_priced["fast_candidate_below_cost"] = [_lt_cost(p, c) for p, c in zip(updated_priced["price_fast_premium"], cost_series)]

# Cost guard
def _guard(p, c):
    try:    return max(float(p), float(c))
    except: return p
updated_priced["price_slow_sales"]   = [_guard(p, c) for p, c in zip(updated_priced["price_slow_sales"],   cost_series)]
updated_priced["price_fast_premium"] = [_guard(p, c) for p, c in zip(updated_priced["price_fast_premium"], cost_series)]

# Persist
st.session_state["priced_table"] = updated_priced
st.session_state["sales_join"]   = sales_join

# --- Recompute Recommended Price using sales-aware rules ---
pt = st.session_state.get("priced_table")
if isinstance(pt, pd.DataFrame) and not pt.empty:
    pt = pt.copy()

    # Ensure columns exist
    for col in ["min_market", "price_slow_sales", "price_fast_premium", "qty", "cost_aed", "recommended_price"]:
        if col not in pt.columns:
            pt[col] = pd.NA

    # Coerce numerics
    mn   = pd.to_numeric(pt["min_market"], errors="coerce")
    slow = pd.to_numeric(pt["price_slow_sales"], errors="coerce")
    fast = pd.to_numeric(pt["price_fast_premium"], errors="coerce")
    qty  = pd.to_numeric(pt["qty"], errors="coerce")
    cost = pd.to_numeric(pt["cost_aed"], errors="coerce")

    # Initialize as float series with NaNs (NOT pd.NA to avoid dtype clash)
    rec = pd.Series(np.nan, index=pt.index, dtype="float64")

    # --- Implement new rules ---
    cond_qty_slow = (qty < 10)
    cond_qty_fast = (qty > 10)  # strictly greater than 10 per requirement

    # 1) Recommended = market min if sales qty < 10 AND market min > sales_slow
    c1 = cond_qty_slow & mn.notna() & slow.notna() & (mn > slow)
    rec.loc[c1] = mn.loc[c1]

    # 2) Recommended = sales_slow if sales_slow > market min OR (no market min AND qty < 10)
    c2 = slow.notna() & ((mn.notna() & (slow > mn)) | (mn.isna() & cond_qty_slow))
    rec.loc[c2] = slow.loc[c2]

    # 3) Recommended = sales_fast if sales_fast > market min OR (no market min AND qty > 10)
    c3 = fast.notna() & ((mn.notna() & (fast > mn)) | (mn.isna() & cond_qty_fast))
    rec.loc[c3] = fast.loc[c3]

    # 4) Recommended = cost + 10% margin if there is no market price or sales history
    # Interpret as: when min_market, slow, and fast are all NaN → use cost * 1.10
    no_market_or_sales = mn.isna() & slow.isna() & fast.isna()
    fallback_cost = cost * 1.10
    rec.loc[no_market_or_sales & rec.isna() & cost.notna()] = fallback_cost.loc[no_market_or_sales & rec.isna() & cost.notna()]

    # Additional gentle fallbacks if still NA: use min_market if available
    rec = rec.where(rec.notna(), mn)

    # --- Cost guard (robust to NaN/NA) ---
    def _guard_num(p, c):
        p_isna = pd.isna(p)
        c_isna = pd.isna(c)
        if p_isna and c_isna:
            return np.nan
        if p_isna:
            return float(c)
        if c_isna:
            return float(p)
        try:
            return max(float(p), float(c))
        except Exception:
            return float(p) if not pd.isna(p) else (float(c) if not pd.isna(c) else np.nan)

    rec = pd.Series([_guard_num(p, c) for p, c in zip(rec, cost)], index=pt.index, dtype="float64")

    # --- Rounding (tolerates NaN) ---
    round_step = int(st.session_state.get("round_to_step", 1) or 1)
    rec = rec.apply(lambda v: round(v / round_step) * round_step if pd.notna(v) else v)

    # Save back
    pt["recommended_price"] = rec
    st.session_state["priced_table"] = pt
    st.success("Updated recommended prices using sales-aware rules.")
else:
    st.info("No pricebook in session yet (priced_table is empty). Run Aggregate & Price first.")

# Show how many rows got slow/fast prices (safe even if columns missing)
pt = st.session_state.get("priced_table")
if isinstance(pt, pd.DataFrame) and not pt.empty:
    slow_n = 0
    fast_n = 0
    if "price_slow_sales" in pt.columns:
        slow_n = pd.to_numeric(pt["price_slow_sales"], errors="coerce").notna().sum()
    if "price_fast_premium" in pt.columns:
        fast_n = pd.to_numeric(pt["price_fast_premium"], errors="coerce").notna().sum()
    st.caption(f"Slow-priced rows: {int(slow_n)}  |  Fast-priced rows: {int(fast_n)}")

# -------------------- 4.3 Export workbook --------------------
def df_or_empty(x):
    return x if isinstance(x, pd.DataFrame) else pd.DataFrame()

excel_buf = io.BytesIO()
with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
    pd.DataFrame({"build": ["ok"]}).to_excel(writer, index=False, sheet_name="Summary")
    df_or_empty(st.session_state.get("stock_df")).to_excel(writer, index=False, sheet_name="Stock")
    df_or_empty(st.session_state.get("accepted_all")).to_excel(writer, index=False, sheet_name="Accepted")
    df_or_empty(st.session_state.get("borderline_all")).to_excel(writer, index=False, sheet_name="Borderline")
    df_or_empty(st.session_state.get("pairs_table")).to_excel(writer, index=False, sheet_name="ConsistencyPairs")
    df_or_empty(st.session_state.get("aggregated_table")).to_excel(writer, index=False, sheet_name="Aggregated")
    df_or_empty(st.session_state.get("priced_table")).to_excel(writer, index=False, sheet_name="Pricebook")
    df_or_empty(st.session_state.get("below_table")).to_excel(writer, index=False, sheet_name="BelowCost")
    df_or_empty(st.session_state.get("costgap_table")).to_excel(writer, index=False, sheet_name="CostGap")
    df_or_empty(st.session_state.get("sales_join")).to_excel(writer, index=False, sheet_name="SalesHistoryJoin")

st.download_button("Download Pricebook.xlsx", data=excel_buf.getvalue(), file_name="Pricebook.xlsx")

# --- Simple export (barcode, name, quantity, recommended) ---
simple_export_btn = st.button("Export Simple (barcode, name, quantity, recommended)", key="export_simple")
if simple_export_btn:
    pt_simple = st.session_state.get("priced_table")
    if not isinstance(pt_simple, pd.DataFrame) or pt_simple.empty:
        st.error("Please run Aggregate & Price first.")
    else:
        base = pt_simple.copy()

        # Ensure required columns exist
        for c in ["barcode", "name", "recommended_price", "product_id"]:
            if c not in base.columns:
                base[c] = pd.NA

        # Bring quantity from stock if available
        stock = st.session_state.get("stock_df")
        if isinstance(stock, pd.DataFrame) and not stock.empty:
            s = stock.copy()
            # try to detect a quantity column
            qty_col = next((c for c in ["quantity", "qty", "stock_qty"] if c in s.columns), None)
            if qty_col is None:
                s["__qty__"] = pd.NA
            else:
                if "__qty__" in s.columns and qty_col != "__qty__":
                    s.drop(columns=["__qty__"], errors="ignore", inplace=True)
                s.rename(columns={qty_col: "__qty__"}, inplace=True)

            # Prefer product_id merge; fallback to barcode
            if "product_id" in base.columns and "product_id" in s.columns:
                out = base.merge(s[["product_id", "__qty__"]], on="product_id", how="left")
            elif "barcode" in base.columns and "barcode" in s.columns:
                out = base.merge(s[["barcode", "__qty__"]], on="barcode", how="left")
            else:
                out = base.copy()
                out["__qty__"] = pd.NA
        else:
            out = base.copy()
            out["__qty__"] = pd.NA

        # Final simple view
        simple_df = pd.DataFrame({
            "barcode": out.get("barcode"),
            "name": out.get("name"),
            "quantity": pd.to_numeric(out.get("__qty__"), errors="coerce"),
            "recommended_price": pd.to_numeric(out.get("recommended_price"), errors="coerce").round(0),
        })

        # Downloads
        buf_simple = io.BytesIO()
        with pd.ExcelWriter(buf_simple, engine="openpyxl") as writer:
            simple_df.to_excel(writer, index=False, sheet_name="Simple")
        st.download_button("Download Simple_Pricebook.xlsx", data=buf_simple.getvalue(), file_name="Simple_Pricebook.xlsx")

        st.download_button(
            "Download Simple_Pricebook.csv",
            data=simple_df.to_csv(index=False).encode("utf-8"),
            file_name="Simple_Pricebook.csv",
            mime="text/csv"
        )

# -----------------------------------------------------------------------------
# Pricebook preview
# -----------------------------------------------------------------------------
st.subheader("Pricebook preview")
priced_tbl = st.session_state.get("priced_table")
below_tbl  = st.session_state.get("below_table")
costgap_tbl = st.session_state.get("costgap_table")

if priced_tbl is not None and not priced_tbl.empty and "cost_gap_flag" in priced_tbl.columns:
    costgap_df = priced_tbl[priced_tbl["cost_gap_flag"] == True].copy()
else:
    costgap_df = pd.DataFrame()

st.session_state["costgap_table"] = costgap_df

# Choose which price to surface
choice = st.radio(
    "Frontline price column",
    ["price_market_undercut", "price_slow_sales", "price_fast_premium"],
    index=0, horizontal=True
)


view = priced_tbl.copy()
view["frontline_price"] = view.get(choice)
# Deduplicate any repeated columns before display (can happen after reruns)
view = view.loc[:, ~view.columns.duplicated()]
st.dataframe(view.head(200), use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    show_below_cost = st.checkbox("Only below-cost", value=False, key="show_below_cost")
with col2:
    show_cost_gap = st.checkbox("Only large cost gaps", value=False, key="show_cost_gap")
    
costgap_tbl = st.session_state.get("costgap_table")
if show_cost_gap and costgap_tbl is not None and not costgap_tbl.empty:
    view_df = costgap_tbl
elif show_below_cost and below_tbl is not None and not below_tbl.empty:
    view_df = below_tbl
else:
    view_df = priced_tbl

# Also ensure the chosen view has unique column names before display
if isinstance(view_df, pd.DataFrame):
    view_df = view_df.loc[:, ~view_df.columns.duplicated()]
st.dataframe(view_df.head(200), use_container_width=True)

# -----------------------------------------------------------------------------
# 5) International price comparison
# -----------------------------------------------------------------------------
st.divider()
st.header("5) International price comparison (optional)")

pricebook = st.session_state.get("priced_table")
if pricebook is None or pricebook.empty:
    st.info("Run **4) Aggregate market, check consistency, and price** first (or use the Sales Lookup app to export a Pricebook).")
else:
    for col in ["name", "recommended_price", "min_market", "max_market", "barcode", "size_ml", "product_id"]:
        if col not in pricebook.columns:
            pricebook[col] = pd.NA

    st.write("Upload an international price list (CSV or Excel). Include either **name** (or **product**) or **barcode**, and **price** (in AED).")
    intl_file = st.file_uploader("International price list", type=["csv","xlsx"], key="intl_upload")

    def _fuzzy_top_1(query: str, choices: list[str]):
        try:
            from rapidfuzz import process, fuzz
            hit = process.extractOne(query, choices, scorer=fuzz.WRatio)
            if hit is None: return None, None
            choice, score, idx = hit
            return idx, score
        except Exception:
            import difflib
            hits = difflib.get_close_matches(query, choices, n=1, cutoff=0.0)
            if not hits: return None, None
            name = hits[0]
            try: return choices.index(name), 100
            except ValueError: return None, None

    def _normalize_digits(s):
        import re
        if s is None: return ""
        return re.sub(r"\D", "", str(s))

    if intl_file:
        try:
            df_in = pd.read_excel(intl_file)
        except Exception:
            intl_file.seek(0); df_in = pd.read_csv(intl_file)

        cols = {c.lower(): c for c in df_in.columns}
        name_col  = cols.get("name") or cols.get("product")
        price_col = cols.get("price") or cols.get("price (aed)") or cols.get("price_aed") or cols.get("aed")
        bc_col    = cols.get("barcode")

        if price_col is None or (name_col is None and bc_col is None):
            st.error("Couldn’t detect columns. Need **price** and either **name**/**product** or **barcode**.")
        else:
            pb = pricebook.copy()
            pb["name_str"] = pb["name"].fillna("").astype(str)
            pb["barcode_str"] = pb["barcode"].fillna("").astype(str)

            out_rows, nohit_rows = [], []
            names_list = pb["name_str"].tolist()

            for _, r in df_in.iterrows():
                raw_name = str(r.get(name_col, "")).strip() if name_col else ""
                raw_bc   = str(r.get(bc_col, "")).strip() if bc_col else ""
                intl_price = pd.to_numeric(r.get(price_col), errors="coerce")
                if pd.isna(intl_price):
                    nohit_rows.append({"input": raw_name or raw_bc, "reason": "Missing/invalid price"})
                    continue

                best = None
                digits = _normalize_digits(raw_bc)
                if digits and len(digits) >= 8:
                    m = pb[pb["barcode_str"].str.contains(digits, na=False)]
                    if not m.empty:
                        best = m.iloc[0]

                if best is None and raw_name:
                    idx, _ = _fuzzy_top_1(raw_name, names_list)
                    if idx is not None:
                        best = pb.iloc[idx]

                if best is None:
                    nohit_rows.append({"input": raw_name or raw_bc, "reason": "No match"})
                    continue

                rec = pd.to_numeric(best.get("recommended_price"), errors="coerce")
                margin_pct = pd.NA if pd.isna(rec) or float(rec) <= 0 else (float(intl_price) - float(rec)) / float(rec)

                out_rows.append({
                    "input": raw_name or raw_bc,
                    "intl_price_aed": float(intl_price),
                    "matched_name": best.get("name"),
                    "product_id": best.get("product_id"),
                    "size_ml": best.get("size_ml"),
                    "barcode": best.get("barcode"),
                    "recommended_price": float(rec) if pd.notna(rec) else pd.NA,
                    "min_market": best.get("min_market"),
                    "max_market": best.get("max_market"),
                    "margin_pct": margin_pct
                })

            result = pd.DataFrame(out_rows)
            misses = pd.DataFrame(nohit_rows)

            if not result.empty:
                result = result.sort_values("margin_pct", ascending=False, na_position="last").reset_index(drop=True)
                result["margin_rank"] = result.index + 1
                view = result.copy()
                view["margin_pct"] = (view["margin_pct"] * 100).round(1).astype(str) + "%"
                st.subheader("International comparison")
                st.dataframe(view, use_container_width=True)

                buf = io.BytesIO()
                with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                    result.to_excel(writer, index=False, sheet_name="IntlCompare")
                    (misses if not misses.empty else pd.DataFrame()).to_excel(writer, index=False, sheet_name="NoMatch")
                st.download_button("Download IntlCompare.xlsx", data=buf.getvalue(), file_name="IntlCompare.xlsx")
            else:
                st.warning("No comparable rows were produced. Check that your file has price + name/barcode columns.")
    else:
        st.caption("Accepted columns (case-insensitive): **name**/**product** or **barcode**, and **price**/**price (AED)**/**AED**.")

