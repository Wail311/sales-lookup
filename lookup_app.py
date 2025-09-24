
import io
import re
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Sales Lookup — Perfume Pricebook", layout="wide")
st.title("🧾 Sales Lookup — Perfume Pricebook")

# ---------- Utils ----------

def normalize_barcode(val):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    s = str(val)
    digits = re.sub(r"\D", "", s)
    return digits or None

@st.cache_data(show_spinner=False)
def load_pricebook_file(file) -> pd.DataFrame:
    """
    Load a Pricebook from Excel or CSV.
    Prefers the 'Pricebook' sheet if present; otherwise first sheet.
    CSV fallback if needed.
    """
    try:
        xls = pd.ExcelFile(file)
        sheet = "Pricebook" if "Pricebook" in xls.sheet_names else xls.sheet_names[0]
        df = pd.read_excel(xls, sheet_name=sheet)
        return df
    except Exception:
        try:
            file.seek(0)
            return pd.read_csv(file)
        except Exception as e2:
            raise ValueError(f"Could not read file as Excel or CSV: {e2}")

def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the columns we need exist, even if blank.
    """
    wanted = [
        "name", "recommended_price", "cost_aed", "min_market", "max_market",
        "below_cost_flag", "product_id", "size_ml", "barcode"
    ]
    out = df.copy()
    for c in wanted:
        if c not in out.columns:
            out[c] = pd.NA
    # Normalize barcode for easier search
    out["barcode"] = out["barcode"].apply(normalize_barcode) if "barcode" in out.columns else pd.NA
    return out

def fuzzy_top_k(query: str, choices: list, k: int = 5):
    """rapidfuzz if available; fallback to difflib."""
    try:
        from rapidfuzz import process, fuzz
        # Returns [(choice, score, index), ...]; map to index + score
        hits = process.extract(query, choices, scorer=fuzz.WRatio, limit=k)
        return [(idx, score) for (_, score, idx) in hits]
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

# ---------- Sidebar: Load / Manage Pricebook ----------
with st.sidebar:
    st.header("Load Pricebook")
    pb_up = st.file_uploader("Upload Pricebook (Excel/CSV)", type=["xlsx","csv"])
    col_save, col_clear = st.columns(2)
    with col_save:
        save_btn = st.button("Load")
    with col_clear:
        clear_btn = st.button("Clear")

    if save_btn and pb_up is not None:
        try:
            df_pb = load_pricebook_file(pb_up)
            df_pb = ensure_columns(df_pb)
            st.session_state["pricebook"] = df_pb
            st.success(f"Loaded Pricebook: {df_pb.shape[0]} rows")
        except Exception as e:
            st.error(f"Failed to load: {e}")

    if clear_btn:
        st.session_state.pop("pricebook", None)
        st.success("Cleared Pricebook from session.")

# ---------- Main: Search UI ----------

pricebook = st.session_state.get("pricebook")
if pricebook is None:
    st.info("Upload a **Pricebook.xlsx** (exported from the Pricing Engine) to start.")
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
        digits_only = re.sub(r"\D", "", query or "")
        results_df = None

        # 1) Barcode-first if 8+ digits present
        if digits_only and len(digits_only) >= 8 and "barcode" in pricebook.columns:
            mask = pricebook["barcode"].astype(str).str.contains(digits_only, na=False)
            bc_hits = pricebook[mask].copy()
            if not bc_hits.empty:
                results_df = bc_hits.head(int(top_n))

        # 2) Fuzzy name fallback
        if results_df is None or results_df.empty:
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

                    # Metrics
                    c1, c2, c3, c4, c5 = st.columns(5)
                    rec = r.get("recommended_price")
                    cost = r.get("cost_aed")
                    mn   = r.get("min_market")
                    mx   = r.get("max_market")

                    try:
                        recf = float(rec) if pd.notna(rec) else None
                    except Exception:
                        recf = None
                    try:
                        costf = float(cost) if pd.notna(cost) else None
                    except Exception:
                        costf = None
                    try:
                        mnf = float(mn) if pd.notna(mn) else None
                    except Exception:
                        mnf = None
                    try:
                        mxf = float(mx) if pd.notna(mx) else None
                    except Exception:
                        mxf = None

                    total = (recf * float(qty)) if (recf is not None) else None

                    c1.metric("Recommended", f"{recf:.0f} AED" if recf is not None else "—")
                    c2.metric("Cost", f"{costf:.0f} AED" if costf is not None else "—")
                    c3.metric("Market min", f"{mnf:.0f} AED" if mnf is not None else "—")
                    c4.metric("Market max", f"{mxf:.0f} AED" if mxf is not None else "—")
                    c5.metric("Total", f"{total:.0f} AED" if total is not None else "—")

                    if bool(r.get("below_cost_flag")) and (recf is not None) and (costf is not None) and (costf > recf):
                        st.error("⚠️ Recommended is below your cost")

                    # Copy-friendly line for messages
                    msg = f"{title}: {recf:.0f} AED" if recf is not None else f"{title}: (no price)"
                    if total is not None:
                        msg += f"   |   {qty} pcs total: {total:.0f} AED"
                    st.text_area("Copy", value=msg, height=40, label_visibility="collapsed")
    else:
        st.info("Type a perfume name (or scan a barcode) to search.")

# ---------- Bulk quote (paste list or upload CSV) ----------
st.divider()
st.header("Bulk quote")

if st.session_state.get("pricebook") is None:
    st.info("Upload a **Pricebook.xlsx** in the sidebar first.")
else:
    pricebook = st.session_state["pricebook"]

    def parse_qty_and_query(line: str):
        """
        Extract a trailing qty if present at the END of the line (not the 100ML size):
        accepts 'x2', '*2', '(2)', '- 2', 'qty 2'.
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

    def match_one_item(query: str):
        """
        Barcode-first (8+ digits), then fuzzy name (top-1).
        Returns {} if no hit.
        """
        import re
        q = (query or "").strip()
        if not q:
            return {}

        # Barcode-first
        digits = re.sub(r"\D", "", q)
        if digits and len(digits) >= 8 and "barcode" in pricebook.columns:
            mask = pricebook["barcode"].astype(str).str.contains(digits, na=False)
            if mask.any():
                row = pricebook[mask].iloc[0]
                return {"match_method":"barcode", **row.to_dict()}

        # Fuzzy fallback (uses fuzzy_top_k defined above)
        names = pricebook["name"].fillna("").astype(str).tolist()
        hits = fuzzy_top_k(q, names, k=1)
        if hits:
            idx, score = hits[0]
            row = pricebook.iloc[idx]
            return {"match_method":f"fuzzy({score})", **row.to_dict()}

        return {}

    st.write("**Option A — Paste a list** (one perfume per line, optional qty at the end like `x2`, `(3)`, or `- 2`):")
    pasted = st.text_area("Paste list", height=180, placeholder="BLEU DE CHANEL 100ML x2\nDIOR SAUVAGE EDP 100ML (3)\nYSL Y LE PARFUM 100ML - 5")

    st.write("**Option B — Upload a CSV/Excel** with columns like `name` (or `product`) and optional `qty` or `barcode`:")
    bulk_file = st.file_uploader("Upload customer list (CSV/XLSX)", type=["csv","xlsx"], key="bulk_upload")

    run_bulk = st.button("Create quote")

    if run_bulk:
        import io
        import pandas as pd

        # Build an input table from either paste or file
        input_rows = []

        if bulk_file is not None:
            try:
                try:
                    df_in = pd.read_excel(bulk_file)
                except Exception:
                    bulk_file.seek(0)
                    df_in = pd.read_csv(bulk_file)

                cols = {c.lower(): c for c in df_in.columns}
                name_col = cols.get("name") or cols.get("product") or list(df_in.columns)[0]
                qty_col  = cols.get("qty") or cols.get("quantity")
                bc_col   = cols.get("barcode")

                for _, r in df_in.iterrows():
                    if bc_col and pd.notna(r.get(bc_col)):
                        q = str(r.get(bc_col))
                    else:
                        q = str(r.get(name_col, "")).strip()
                    q_qty = int(r.get(qty_col, 1)) if qty_col in df_in.columns else 1
                    input_rows.append({"raw_query": q, "qty": max(q_qty, 1)})

            except Exception as e:
                st.error(f"Could not read file: {e}")

        if not input_rows and pasted.strip():
            for line in pasted.splitlines():
                name, qty = parse_qty_and_query(line)
                if name:
                    input_rows.append({"raw_query": name, "qty": qty})

        if not input_rows:
            st.warning("No items provided. Paste a list or upload a file.")
        else:
            out_rows, nohit_rows = [], []
            for row in input_rows:
                hit = match_one_item(row["raw_query"])
                if hit:
                    rec = hit.get("recommended_price")
                    try:
                        recf = float(rec) if pd.notna(rec) else None
                    except Exception:
                        recf = None
                    total = recf * row["qty"] if recf is not None else None

                    out_rows.append({
                        "input": row["raw_query"],
                        "qty": row["qty"],
                        "matched_name": hit.get("name"),
                        "product_id": hit.get("product_id"),
                        "size_ml": hit.get("size_ml"),
                        "barcode": hit.get("barcode"),
                        "recommended_price": recf,
                        "cost_aed": hit.get("cost_aed"),
                        "min_market": hit.get("min_market"),
                        "max_market": hit.get("max_market"),
                        "below_cost_flag": hit.get("below_cost_flag"),
                        "match_method": hit.get("match_method"),
                        "line_total": total,
                    })
                else:
                    nohit_rows.append({"input": row["raw_query"], "qty": row["qty"]})

            quote_df = pd.DataFrame(out_rows)
            misses_df = pd.DataFrame(nohit_rows)

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

            # Export to Excel
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                (quote_df if not quote_df.empty else pd.DataFrame()).to_excel(writer, index=False, sheet_name="Quote")
                (misses_df if not misses_df.empty else pd.DataFrame()).to_excel(writer, index=False, sheet_name="NoMatch")
            st.download_button("Download Quote.xlsx", data=buf.getvalue(), file_name="Quote.xlsx")
