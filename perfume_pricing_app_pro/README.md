# Perfume Pricing Engine — Pro

A scalable Streamlit app to ingest **10+ competitor** price lists, match to your stock (names-only with **strict ML gate**), compute market stats, check **consistency** (20%/>300 or ≤50 AED), and export a **Pricebook.xlsx**.

## Features
- Config-driven competitor mappings (see `config/competitors.yml`)
- **ML gate** (bottle size match) to avoid 50ml↔100ml mistakes
- Fuzzy matching via **RapidFuzz** (defaults: accept ≥90, borderline 86–90)
- **Consistency rule** across competitor pairs: >300 → within 20%, else within 50 AED
- Outlier filtering (IQR or stdev) before computing min/median/mean/max
- Pricing strategies: `market_match`, `margin_first`, `premium`
- Excel export with sheets: **Stock**, **Accepted**, **Borderline**, **ConsistencyPairs**, **Aggregated**, **Pricebook**

## Install & run
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Configure competitors
Edit `config/competitors.yml`:
```yaml
competitors:
  - competitor_id: "PRICE LIST 08-01-2025"
    file_hint: "PRICE LIST 08-01-2025.csv"
    sheet: ""
    name_column: "Unnamed: 1"
    price_column: "AED"
    detect_aed_header_in_sheet: true
  # add more competitors here...
```
- If a competitor’s price column is not in the header, set `detect_aed_header_in_sheet: true` to detect an in-sheet `AED` header cell.

## Workflow
1. Upload **Stock** (or use the template).
2. Add competitor files – pick from configured competitors or enter a **custom mapping**.
3. Click **Match now** to produce **Accepted** and **Borderline** matches (ML-enforced).
4. Click **Aggregate & export** to generate **ConsistencyPairs**, **Aggregated**, and **Pricebook.xlsx**.

## Notes
- Extend normalization tokens & brand aliases in `config/synonyms.yml`.
- Adjust thresholds & pricing strategy in `config/policy.yml`.
- For persistent review/overrides, connect a database (SQLite/Postgres) and add a review page.
