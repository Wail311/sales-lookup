import io
import re
import pandas as pd
import streamlit as st

# -----------------------------------------------------------------------------
# App chrome
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Sales Lookup — Perfume Pricebook", layout="wide")
st.title("🧾 Sales Lookup — Perfume Pricebook")
st.caption("Build: sales-lookup v1.2")
# ---- Globals / policy ----
BC_MIN_LEN = 8
FUZZY_THRESHOLD_DEFAULT = 91

# --- constants / helpers ---
def normalize_digits(x):
    import re as _re
    import pandas as _pd
    if x is None or (isinstance(x, float) and _pd.isna(x)):
        return ""
    return _re.sub(r"\D", "", str(x))

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    wanted = [
        "name", "recommended_price", "cost_aed",
        "min_market", "max_market",                # keep if you still want it elsewhere
        "price_slow_sales", "price_fast_premium",  # <-- add these
        "below_cost_flag", "cost_gap_flag",
        "product_id", "size_ml", "barcode"
    ]
    out = df.copy()
    for c in wanted:
        if c not in out.columns:
            out[c] = pd.NA
    return out

def normalize_barcode(val):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    s = str(val)
    digits = re.sub(r"\D", "", s)
    return digits or None

# --- Barcode helpers (paste after normalize_barcode) ---
BC_MIN_LEN = 8  # treat 8+ digit strings as barcodes

def detect_barcode_col(cols: list[str]) -> str | None:
    """
    Return the first column name that looks like a barcode field.
    Match is case/space-insensitive.
    """
    candidates = [
        "barcode", "bar code", "ean", "ean13", "upc", "gtin",
        "sku_barcode", "sku barcode", "code", "product code"
    ]
    low = {str(c).lower().strip(): c for c in cols}
    for k in candidates:
        if k in low:
            return low[k]
    return None

def digits_only(x):
    import re
    return re.sub(r"\D", "", str(x)) if x is not None else ""

@st.cache_data(show_spinner=False)
def load_pricebook_upload(file) -> tuple[pd.DataFrame | None, str | None]:
    """
    Robustly load an uploaded pricebook that might be Excel or CSV.
    Returns (df, error_message). If df is None, show error_message.
    """
    if not file:
        return None, "No file provided"

    # Read bytes once
    try:
        # reset stream if possible (avoid empty reads on repeated calls)
        if hasattr(file, "seek"):
            try:
                file.seek(0)
            except Exception:
                pass
        raw = file.read()
    except Exception as e:
        return None, f"Could not read upload buffer: {e!r}"

    name = (getattr(file, "name", "") or "").lower()
    import io as _io
    b = _io.BytesIO(raw)

    preferred_sheets = ["pricebook", "sheet1", "export", "priced", "data"]

    # 1) Excel first (let pandas auto-select engine for .xlsx/.xls)
    excel_err = None
    looks_like_xlsx = name.endswith((".xlsx", ".xls")) or (len(raw) >= 4 and raw[:2] == b"PK")
    if looks_like_xlsx:
        try:
            b.seek(0)
            xls = pd.ExcelFile(b)  # let pandas pick engine (openpyxl/xlrd)
            lower_names = {s.lower(): s for s in xls.sheet_names}
            sheet = next((lower_names[cand] for cand in preferred_sheets if cand in lower_names), None)
            df = xls.parse(sheet_name=sheet or 0, dtype=str)
            return df, None
        except ImportError as e:
            # Missing engine (e.g., openpyxl) — give a clear, actionable error
            excel_err = "Missing optional dependency 'openpyxl'. Please install it: pip install openpyxl"
        except Exception as e:
            excel_err = f"Excel open failed: {e!r}"

    # 2) CSV encodings (only attempt if filename suggests CSV)
    csv_err = None
    if name.endswith(".csv") or not looks_like_xlsx:
        for enc in ("utf-8", "utf-8-sig", "cp1252", "latin1", "cp1256"):
            try:
                b.seek(0)
                # Try robust CSV read with python engine and sep inference
                df = pd.read_csv(_io.BytesIO(raw), encoding=enc, engine="python", sep=None, dtype=str)
                return df, None
            except Exception:
                try:
                    b.seek(0)
                    df = pd.read_csv(_io.BytesIO(raw), encoding=enc, dtype=str)
                    return df, None
                except Exception as e:
                    csv_err = f"CSV read failed: {e!r}"

    # 3) Fallback: decode bytes ourselves (so no errors= kwarg is needed)
    try:
        if name.endswith(".csv") or not looks_like_xlsx:
            txt = raw.decode("utf-8", errors="replace")
            df = pd.read_csv(_io.StringIO(txt), engine="python", sep=None, dtype=str)
            return df, None
    except Exception as e:
        if name.endswith(".csv") or not looks_like_xlsx:
            csv_err = f"CSV read failed: {e!r}"
    else:
        csv_err = None

    msg = "Could not read the file. Use Excel .xlsx/.xls or CSV."
    diag = " | ".join([m for m in [excel_err, csv_err] if m])
    if diag:
        msg += f" (Details: {diag})"
    return None, msg


def fuzzy_top_k(query: str, choices: list, k: int = 5):
    """rapidfuzz if available; fallback to difflib."""
    try:
        from rapidfuzz import process, fuzz
        hits = process.extract(query, choices, scorer=fuzz.WRatio, limit=k)
        return [(idx, score) for (_choice, score, idx) in hits]
    except Exception:
        import difflib
        matches = difflib.get_close_matches(query, choices, n=k, cutoff=0.0)
        out = []
        for j, m in enumerate(matches):
            try:
                i = choices.index(m)
                out.append((i, max(100 - 5*j, 0)))
            except ValueError:
                pass
        return out

# -----------------------------------------------------------------------------
# Sidebar: upload & load Pricebook
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("Load Pricebook")
    pb_up = st.file_uploader("Upload Pricebook (Excel or CSV)", type=["xlsx","xls","csv"])

    df_pb, err = load_pricebook_upload(pb_up) if pb_up else (None, None)
    if pb_up and (df_pb is None or isinstance(df_pb, pd.DataFrame) and df_pb.empty):
        st.error(err or "Could not read the file. Make sure it’s an Excel (.xlsx/.xls) or CSV.")
        st.stop()

    def _prep_pricebook(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if "name" not in out.columns:
            if "product" in out.columns:
                out.rename(columns={"product": "name"}, inplace=True)
            else:
                out["name"] = pd.NA
        # Map common barcode aliases to a canonical 'barcode' column
        low = {str(c).lower().strip(): c for c in out.columns}
        if "barcode" not in out.columns:
            alias = None
            for cand in ["barcode", "bar code", "ean", "ean13", "gtin", "upc"]:
                if cand in low:
                    alias = low[cand]
                    break
            if alias is not None:
                out["barcode"] = out[alias]
            else:
                out["barcode"] = ""
        # Build normalized barcode
        out["barcode_norm"] = out["barcode"].apply(normalize_digits)
        for c in [
            "recommended_price", "cost_aed", "min_market", "max_market",
            "price_slow_sales", "price_fast_premium",
            "below_cost_flag", "cost_gap_flag", "size_ml", "is_tester",
            "product_id"
        ]:
            if c not in out.columns:
                out[c] = pd.NA
        out["size_ml"] = pd.to_numeric(out["size_ml"], errors="coerce")
        out["is_tester"] = out["is_tester"].astype(str).str.lower().isin(["1","true","yes","y","t"])
        return out

    if pb_up and isinstance(df_pb, pd.DataFrame):
        prepped = _prep_pricebook(df_pb)
        st.session_state["pricebook"] = prepped
        # Normalize barcodes once on load and ensure name exists
        pb_norm = st.session_state.get("pricebook")
        if pb_norm is not None and not pb_norm.empty:
            pb_norm = pb_norm.copy()
            if "barcode_norm" not in pb_norm.columns:
                if "barcode" not in pb_norm.columns:
                    pb_norm["barcode"] = ""
                pb_norm["barcode_norm"] = pb_norm["barcode"].apply(normalize_digits)
            if "name" not in pb_norm.columns:
                pb_norm["name"] = pb_norm["product"] if "product" in pb_norm.columns else ""
            st.session_state["pricebook"] = pb_norm
        st.dataframe(st.session_state["pricebook"].head(20), use_container_width=True)

    col_save, col_clear = st.columns(2)
    with col_save:
        save_btn = st.button("Load")
    with col_clear:
        clear_btn = st.button("Clear")

    if save_btn:
        if not pb_up:
            st.error("Please choose a file first.")
        else:
            df_pb, err = load_pricebook_upload(pb_up)
            if df_pb is None or df_pb.empty:
                st.error(err or "Could not read the file. Use Excel .xlsx/.xls or CSV.")
                st.stop()
            prepped = _prep_pricebook(df_pb)
            st.session_state["pricebook"] = prepped
            # Normalize barcodes once on load and ensure name exists
            pb_norm = st.session_state.get("pricebook")
            if pb_norm is not None and not pb_norm.empty:
                pb_norm = pb_norm.copy()
                if "barcode_norm" not in pb_norm.columns:
                    if "barcode" not in pb_norm.columns:
                        pb_norm["barcode"] = ""
                    pb_norm["barcode_norm"] = pb_norm["barcode"].apply(normalize_digits)
                if "name" not in pb_norm.columns:
                    pb_norm["name"] = pb_norm["product"] if "product" in pb_norm.columns else ""
                st.session_state["pricebook"] = pb_norm
            st.success(f"Loaded Pricebook: {prepped.shape[0]} rows")

    if clear_btn:
        st.session_state.pop("pricebook", None)
        st.success("Cleared Pricebook from session.")

# -----------------------------------------------------------------------------
# Main: search UI
# -----------------------------------------------------------------------------
pricebook = st.session_state.get("pricebook")
if pricebook is None:
    st.info("Upload a **Pricebook.xlsx/CSV** in the sidebar and click **Load** to start.")
else:
    with st.expander("Preview (first 200 rows)", expanded=False):
        st.dataframe(pricebook.head(200), use_container_width=True)

    # Search params
    qcol, ncol, qtycol = st.columns([4,1,1])
    with qcol:
        query = st.text_input("Search by name or barcode", placeholder="e.g., BLEU DE CHANEL 100ML or scan barcode...", key="sales_query")
    with ncol:
        top_n = st.number_input("Results", min_value=1, max_value=50, value=5, step=1)
    with qtycol:
        qty = st.number_input("Qty", min_value=1, max_value=999, value=1, step=1)

    # Search
    if query:
        q_digits = re.sub(r"\D", "", query or "")
        results_df = None

        # 1) Barcode-first if 8+ digits present (strict: no fuzzy fallback if present but no match)
        if q_digits and len(q_digits) >= BC_MIN_LEN and "barcode_norm" in pricebook.columns:
            mask = pricebook["barcode_norm"].astype(str).str.contains(q_digits, na=False)
            bc_hits = pricebook[mask].copy()
            if not bc_hits.empty:
                results_df = bc_hits.head(int(top_n))
            else:
                results_df = pd.DataFrame()  # explicit no-fuzzy when barcode provided but fails

        # 2) Fuzzy name fallback ONLY if no usable barcode in query
        if (not q_digits or len(q_digits) < BC_MIN_LEN) and (results_df is None or results_df.empty):
            names = pricebook["name"].fillna("").astype(str).tolist()
            idx_scores = fuzzy_top_k(query, names, k=int(top_n))
            if idx_scores:
                sel_idx = [i for (i, _score) in idx_scores]
                results_df = pricebook.iloc[sel_idx].copy()
                results_df["match_score"] = [s for (_i, s) in idx_scores]

        if results_df is None or results_df.empty:
            st.warning("No matches found.")
        else:
            st.caption(f"Showing {results_df.shape[0]} result(s) — barcode first, then fuzzy by name.")
            for _, r in results_df.iterrows():
                with st.container(border=True):
                    title = str(r.get("name", "")) or "(Unnamed)"
                    meta = []
                    if pd.notna(r.get("product_id")): meta.append(f"ID: {r.get('product_id')}")
                    if pd.notna(r.get("size_ml")):    meta.append(f"Size: {r.get('size_ml')}ml")
                    if pd.notna(r.get("barcode")):    meta.append(f"Barcode: {r.get('barcode')}")
                    st.markdown(f"**{title}**  \n" + (" · ".join(meta) if meta else ""))

                    # Metrics in desired order: Recommended, Cost, Market min, Sales slow, Sales fast
                    c1, c2, c3, c4, c5 = st.columns(5)

                    recf = pd.to_numeric(r.get("recommended_price"), errors="coerce")
                    costf = pd.to_numeric(r.get("cost_aed"), errors="coerce")
                    mnf  = pd.to_numeric(r.get("min_market"), errors="coerce")
                    slow = pd.to_numeric(r.get("price_slow_sales"), errors="coerce")
                    fast = pd.to_numeric(r.get("price_fast_premium"), errors="coerce")

                    fmt = lambda x: ("—" if pd.isna(x) else f"{float(x):.0f} AED")

                    c1.metric("Recommended", fmt(recf))
                    c2.metric("Cost",        fmt(costf))
                    c3.metric("Market min",  fmt(mnf))
                    c4.metric("Sales slow",  fmt(slow))
                    c5.metric("Sales fast",  fmt(fast))

                    # --- NEW: compute total for the copy line (based on qty input above) ---
                    try:
                        qty_val = float(qty)
                    except Exception:
                        qty_val = 1.0
                    total = (float(recf) * qty_val) if pd.notna(recf) else None

                    # Flags (optional)
                    badges = []
                    if bool(r.get("below_cost_flag")) and pd.notna(recf) and pd.notna(costf) and float(costf) > float(recf):
                        badges.append("🟥 Below cost")
                    if bool(r.get("cost_gap_flag")):
                        badges.append("⚠️ Cost gap")
                    if badges:
                        st.markdown("**Flags:** " + " | ".join(badges))

                    # Copy-friendly line
                    title = str(r.get("name", "")) or "(Unnamed)"
                    msg = f"{title}: {fmt(recf)}"
                    if total is not None:
                        msg += f"   |   {int(qty_val)} pcs total: {total:.0f} AED"
                    st.text_area("Copy", value=msg, height=40, label_visibility="collapsed")
    else:
        st.info("Type a perfume name (or scan a barcode) to search.")

# -----------------------------------------------------------------------------
# ---------- Bulk quote ----------
st.divider()
st.header("Bulk quote")

if st.session_state.get("pricebook") is None:
    st.info("Upload a **Pricebook.xlsx** (sidebar) to start.")
else:
    pricebook = st.session_state["pricebook"].copy()

    # Helpers
    FUZZY_THRESHOLD = int(st.session_state.get("FUZZY_THRESHOLD", 91))

    def _digits(s):
        import re
        return re.sub(r"\D", "", str(s or ""))

    def _fuzzy_top1(query: str, candidates_df: pd.DataFrame, name_col="name"):
        """
        Returns the best-matching row (Series) or None.
        name_col is the column to score on. Robust to duplicate column names.
        """
        # Coerce to a single Series if duplicate column labels exist
        col_obj = candidates_df[name_col]
        if isinstance(col_obj, pd.DataFrame):
            col_obj = col_obj.iloc[:, 0]  # take the first column if there are duplicates

        name_series = col_obj.fillna("").astype(str)
        choices = name_series.tolist()

        try:
            from rapidfuzz import process, fuzz
            hit = process.extractOne(str(query), choices, scorer=fuzz.WRatio)
            if not hit:
                return None
            choice, score, idx = hit
            # Enforce your fuzzy threshold (defaults to 91 in your app’s state)
            threshold = int(st.session_state.get("FUZZY_THRESHOLD", 91))
            if score < threshold:
                return None
            return candidates_df.iloc[idx]
        except Exception:
            import difflib
            hits = difflib.get_close_matches(str(query), choices, n=1, cutoff=0.0)
            if not hits:
                return None
            # pick the first match’s index
            first = hits[0]
            try:
                i = choices.index(first)
                return candidates_df.iloc[i]
            except ValueError:
                return None

            # accept difflib top as 100 for simplicity
            return candidates_df.loc[candidates_df[name_col] == hits[0]].iloc[0]

    # Normalize pricebook fields used in matching
    for c in ["name","barcode","size_ml","is_tester","recommended_price","cost_aed","min_market","max_market"]:
        if c not in pricebook.columns:
            pricebook[c] = pd.NA
    pricebook["name_key"]   = pricebook["name"].fillna("").astype(str).str.strip().str.upper()
    pricebook["barcode_str"]= pricebook["barcode"].astype(str).str.replace(r"\D","", regex=True)
    if "size_ml" in pricebook.columns:
        pricebook["size_ml"] = pd.to_numeric(pricebook["size_ml"], errors="coerce")
    if "is_tester" in pricebook.columns:
        # already boolean in your pipeline; keep as-is
        pass

    # UI: two tabs — Detailed quote (existing) and Simple price fill
    tab1, tab2 = st.tabs(["🧾 Detailed quote", "🧷 Simple price fill (append recommended price)"])

    # ---------------- Tab 1: Detailed quote (your existing behavior) ----------------
    with tab1:
        def parse_qty_and_query(line: str):
            """
            Extract a trailing qty if present at the END (x2, *2, (2), - 2, 'qty 2').
            Does not confuse '100ML'.
            """
            import re
            s = (line or "").strip()
            if not s:
                return "", 1
            m = re.search(r"(.*?)(?:\s*(?:x|\*)\s*(\d+)|\s*\(\s*(\d+)\s*\)\s*|\s*-\s*(\d+)\s*|\s*qty\s+(\d+))\s*$",
                          s, flags=re.IGNORECASE)
            if not m:
                return s, 1
            name = m.group(1).strip()
            qty = next((int(g) for g in m.groups()[1:] if g), 1)
            return (name or s), max(qty, 1)

        st.write("**Option A — Paste a list** (one per line, optional qty at the end like `x2`, `(3)`, or `- 2`):")
        pasted = st.text_area("Paste list", height=180,
                              placeholder="BLEU DE CHANEL 100ML x2\nDIOR SAUVAGE EDP 100ML (3)\nYSL Y LE PARFUM 100ML - 5",
                              key="bulk_paste_detailed")

        st.write("**Option B — Upload a CSV/Excel** with columns like `name` (or `product`) and optional `qty` or `barcode`:")
        bulk_file = st.file_uploader("Upload customer list (CSV/XLSX)", type=["csv","xlsx"], key="bulk_upload_detailed")

        run_bulk = st.button("Create detailed quote")

        # strict matcher used in both tabs
        def match_one_strict(row: dict):
            """
            Barcode-first (exact/contains on normalized digits) with NO fuzzy fallback
            if a usable barcode was provided. Only when there's no usable barcode do we
            fuzzy match by name (threshold ≥ 91 by default).
            """
            pb = st.session_state.get("pricebook")
            if pb is None or pb.empty:
                return None

            # pull normalized barcode (prefer provided barcode_norm; else normalize barcode)
            bc = normalize_digits(row.get("barcode_norm", row.get("barcode", ""))).strip()
            name = str(row.get("name", "")).strip()

            # 1) Barcode path (no fuzzy fallback if barcode exists but doesn't match)
            if bc and len(bc) >= BC_MIN_LEN and "barcode_norm" in pb.columns:
                # Prefer exact match on normalized digits; else allow substring for partial scans
                exact_hits = pb[pb["barcode_norm"].astype(str) == bc]
                if not exact_hits.empty:
                    return exact_hits.iloc[0].to_dict()
                contains_hits = pb[pb["barcode_norm"].astype(str).str.contains(bc, na=False)]
                if not contains_hits.empty:
                    return contains_hits.iloc[0].to_dict()
                return None  # barcode present but no match → do NOT fuzzy

            # 2) Fuzzy fallback ONLY if no usable barcode was provided
            names = pb["name"].fillna("").astype(str).tolist()
            if name:
                try:
                    from rapidfuzz import process, fuzz
                    hit = process.extractOne(name, names, scorer=fuzz.WRatio)
                    if hit is not None:
                        choice, score, idx = hit
                        if score >= int(st.session_state.get("FUZZY_THRESHOLD", 91)):
                            return pb.iloc[idx].to_dict()
                except Exception:
                    import difflib
                    hits = difflib.get_close_matches(name, names, n=1, cutoff=0.0)
                    if hits:
                        return pb[pb["name"].astype(str) == hits[0]].iloc[0].to_dict()

            return None

        if run_bulk:
            input_rows = []

            # File path
            if bulk_file is not None:
                # Read customer list (preserve barcodes as text)
                try:
                    try:
                        df_in = pd.read_excel(bulk_file, dtype=str)
                    except Exception:
                        bulk_file.seek(0)
                        df_in = pd.read_csv(bulk_file, dtype=str)
                except Exception as e:
                    st.error(f"Could not read file: {e}")
                    st.stop()

                # Normalize headers
                cols_low = {str(c).strip().lower(): c for c in df_in.columns}
                name_col = cols_low.get("name") or cols_low.get("product") or list(df_in.columns)[0]
                bc_col   = cols_low.get("barcode") or cols_low.get("bar code") or cols_low.get("ean") or cols_low.get("upc")
                qty_col  = cols_low.get("qty") or cols_low.get("quantity")

                # Build input rows with normalized barcode
                input_rows = []
                for _, r in df_in.iterrows():
                    raw_name = str(r.get(name_col, "")).strip() if name_col else ""
                    raw_bc   = str(r.get(bc_col, "")).strip() if bc_col else ""
                    qty_val  = r.get(qty_col, "1")
                    try:
                        qty_num = int(float(qty_val)) if qty_val not in (None, "", "nan") else 1
                    except Exception:
                        qty_num = 1

                    input_rows.append({
                        "name": raw_name,
                        "barcode": raw_bc,
                        "barcode_norm": normalize_digits(raw_bc),
                        "qty": max(qty_num, 1),
                    })

            # Paste path
            if not input_rows and pasted.strip():
                for line in pasted.splitlines():
                    name, q = parse_qty_and_query(line)
                    if name:
                        input_rows.append({"name": name, "qty": q, "barcode_norm": "", "size_ml": None, "is_tester": None})

            if not input_rows:
                st.warning("No items provided. Paste a list or upload a file.")
            else:
                out_rows, misses = [], []
                for item in input_rows:
                    hit = match_one_strict(item)
                    if hit is None:
                        misses.append({"input": item.get("name") or item.get("barcode_norm"), "qty": item.get("qty")})
                        continue

                    recf = pd.to_numeric(hit.get("recommended_price"), errors="coerce")
                    costf= pd.to_numeric(hit.get("cost_aed"), errors="coerce")
                    mnf  = pd.to_numeric(hit.get("min_market"), errors="coerce")
                    mxf  = pd.to_numeric(hit.get("max_market"), errors="coerce")
                    total= float(recf) * float(item["qty"]) if pd.notna(recf) else None

                    out_rows.append({
                        "input": item.get("name") or item.get("barcode_norm"),
                        "qty": item["qty"],
                        "matched_name": hit.get("name"),
                        "product_id": hit.get("product_id"),
                        "size_ml": hit.get("size_ml"),
                        "barcode": hit.get("barcode"),
                        "recommended_price": float(recf) if pd.notna(recf) else pd.NA,
                        "cost_aed": float(costf) if pd.notna(costf) else pd.NA,
                        "min_market": float(mnf) if pd.notna(mnf) else pd.NA,
                        "max_market": float(mxf) if pd.notna(mxf) else pd.NA,
                        "line_total": total,
                    })

                quote_df = pd.DataFrame(out_rows)
                misses_df = pd.DataFrame(misses)

                st.subheader("Quote results")
                if not quote_df.empty:
                    grand_total = pd.to_numeric(quote_df["line_total"], errors="coerce").fillna(0).sum()
                    st.metric("Grand total", f"{grand_total:.0f} AED")
                    st.dataframe(quote_df, use_container_width=True)
                else:
                    st.warning("No matches found in Pricebook for the provided list.")

                if not misses_df.empty:
                    st.info("Items without a match:")
                    st.dataframe(misses_df, use_container_width=True)

                buf = io.BytesIO()
                with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                    (quote_df if not quote_df.empty else pd.DataFrame()).to_excel(writer, index=False, sheet_name="Quote")
                    (misses_df if not misses_df.empty else pd.DataFrame()).to_excel(writer, index=False, sheet_name="NoMatch")
                st.download_button("Download Quote.xlsx", data=buf.getvalue(), file_name="Quote.xlsx")

    # ---------------- Tab 2: Simple price fill ----------------
    with tab2:
        st.write("Upload a customer list (CSV/Excel). Output keeps **exact same columns** and adds two new columns: `recommended_price` and (if a qty column exists) `total_price_aed`.")
        simple_up = st.file_uploader("Upload customer list", type=["csv","xlsx"], key="simple_price_upload")

        # Column mapping
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1: name_hint = st.text_input("Name column", value="name")
        with col2: bc_hint   = st.text_input("Barcode column (optional)", value="barcode")
        with col3: size_hint = st.text_input("Size column (optional, e.g. size_ml/ml)", value="size_ml")
        with col4: tst_hint  = st.text_input("Tester column (optional, true/false)", value="is_tester")
        with col5: qty_hint  = st.text_input("Qty column (optional)", value="qty")

        run_simple = st.button("Create simple price list")

        if run_simple:
            if not simple_up:
                st.warning("Please upload a file.")
            else:
                # Read (preserve barcode digits)
                try:
                    try:
                        df_in = pd.read_excel(simple_up, dtype=str)
                    except Exception:
                        simple_up.seek(0)
                        df_in = pd.read_csv(simple_up, dtype=str)
                except Exception as e:
                    st.error(f"Could not read file: {e}")
                    st.stop()

                # Preserve original order & columns
                orig_cols = list(df_in.columns)
                out_df = df_in.copy()

                # Detect mapped columns (don’t alter originals)
                cols_low = {str(c).strip().lower(): c for c in df_in.columns}
                name_col = cols_low.get("name") or cols_low.get("product")
                bc_col   = cols_low.get("barcode") or cols_low.get("bar code") or cols_low.get("ean") or cols_low.get("upc")
                q_col    = cols_low.get("qty") or cols_low.get("quantity")

                # Working copy with barcode_norm
                df_work = df_in.copy()
                if bc_col:
                    df_work["barcode_norm"] = df_work[bc_col].apply(normalize_digits)
                else:
                    df_work["barcode_norm"] = ""

                # Compute recommended for each row using strict matcher
                rec_prices = []
                for _, r in df_work.iterrows():
                    row = {
                        "name": str(r.get(name_col, "") if name_col else ""),
                        "barcode": str(r.get(bc_col, "") if bc_col else ""),
                        "barcode_norm": str(r.get("barcode_norm", "")),
                    }
                    hit = match_one_strict(row)
                    if hit and ("recommended_price" in hit):
                        rec_prices.append(pd.to_numeric(hit["recommended_price"], errors="coerce"))
                    else:
                        rec_prices.append(pd.NA)

                # Add recommended_price
                out_df["recommended_price"] = rec_prices

                # Add total_price_aed ONLY if a qty column exists in input
                if q_col:
                    qty_vals  = pd.to_numeric(out_df[q_col], errors="coerce")
                    price_vals= pd.to_numeric(out_df["recommended_price"], errors="coerce")
                    out_df["total_price_aed"] = (qty_vals * price_vals).round(0)

                st.subheader("Simple price list")
                st.dataframe(out_df, use_container_width=True)

                # Downloads
                buf2 = io.BytesIO()
                with pd.ExcelWriter(buf2, engine="openpyxl") as writer:
                    out_df.to_excel(writer, index=False, sheet_name="PriceFill")
                st.download_button("Download Simple Price Fill.xlsx", data=buf2.getvalue(),
                                   file_name="Simple_Price_Fill.xlsx")

                st.download_button("Download Simple Price Fill.csv",
                                   data=out_df.to_csv(index=False).encode("utf-8"),
                                   file_name="Simple_Price_Fill.csv", mime="text/csv")
