import itertools
import pandas as pd

def _tick_within(p1, p2, high_avg_threshold=300, high_avg_within_pct=0.20, low_avg_within_abs=50):
    if pd.isna(p1) or pd.isna(p2): return None
    avg = (p1 + p2) / 2.0
    diff = abs(p1 - p2)
    if avg > high_avg_threshold:
        return diff <= high_avg_within_pct * avg
    return diff <= low_avg_within_abs

def consistency_pairs(accepted: pd.DataFrame, tick_cfg: dict) -> pd.DataFrame:
    # Build all pairwise competitor comparisons per product
    rows = []
    for pid, grp in accepted.groupby("matched_product_id"):
        offers = list(grp[["competitor","offer_price_aed"]].itertuples(index=False, name=None))
        for (c1, p1), (c2, p2) in itertools.combinations(offers, 2):
            ok = _tick_within(
                p1, p2,
                high_avg_threshold=tick_cfg.get("high_avg_threshold",300),
                high_avg_within_pct=tick_cfg.get("high_avg_within_pct",0.20),
                low_avg_within_abs=tick_cfg.get("low_avg_within_abs",50),
            )
            avg = None if (pd.isna(p1) or pd.isna(p2)) else (p1+p2)/2.0
            diff = None if (pd.isna(p1) or pd.isna(p2)) else abs(p1-p2)
            rows.append({
                "product_id": pid, "competitor_a": c1, "price_a": p1,
                "competitor_b": c2, "price_b": p2, "avg_ab": avg, "abs_diff": diff,
                "within_rule": ok
            })
    return pd.DataFrame(rows)

def aggregate_market(accepted: pd.DataFrame, stock: pd.DataFrame, outlier_method="iqr", iqr_k=1.5, stdev_k=2.0):
    g = accepted.groupby("matched_product_id")
    rows = []
    for pid, grp in g:
        prices = pd.to_numeric(grp["offer_price_aed"], errors="coerce").dropna()
        if prices.empty:
            stats = dict(competitor_count=0, min_market=None, median_market=None, mean_market=None, max_market=None)
        else:
            q1 = prices.quantile(0.25); q3 = prices.quantile(0.75); iqr = q3 - q1
            if outlier_method == "iqr" and iqr > 0:
                low = q1 - iqr_k * iqr; high = q3 + iqr_k * iqr
                keep = prices[(prices >= low) & (prices <= high)]
            elif outlier_method == "stdev":
                mu = prices.mean(); sd = prices.std(ddof=0)
                keep = prices[(prices >= mu - stdev_k*sd) & (prices <= mu + stdev_k*sd)]
            else:
                keep = prices
            stats = {
                "competitor_count": int(keep.shape[0]),
                "min_market": float(keep.min()) if not keep.empty else None,
                "median_market": float(keep.median()) if not keep.empty else None,
                "mean_market": float(keep.mean()) if not keep.empty else None,
                "max_market": float(keep.max()) if not keep.empty else None,
            }
        rows.append({"product_id": pid, **stats})
    return stock.merge(pd.DataFrame(rows), on="product_id", how="left")

def _nan_to_none(x):
    try:
        return None if pd.isna(x) else float(x)
    except Exception:
        return None

def recommend_price(row, strategy="undercut_min_1pct", min_margin=0.20, round_to=1,cost_floor_enabled=True):
    cost         = _nan_to_none(row.get("cost_aed"))
    min_market   = _nan_to_none(row.get("min_market"))
    median_market= _nan_to_none(row.get("median_market"))

    rec = None
    used_cost_floor = False

    if strategy == "undercut_min_1pct":
        # ---- COST FLOOR: if the market minimum is below your cost, use cost (exact) ----
        if cost_floor_enabled and (cost is not None) and (min_market is not None) and (min_market < cost):
            rec = cost
            used_cost_floor = True
        else:
            base = min_market if min_market is not None else median_market
            rec = None if base is None else base * 0.99
    elif strategy == "market_match":
        rec = median_market
    elif strategy == "premium":
        rec = (median_market + 5) if median_market is not None else None
    else:  # margin_first fallback
        if cost is not None and cost > 0:
            target = cost * (1 + (min_margin or 0.0))
            rec = max(target, median_market or 0.0) if median_market is not None else target
        else:
            rec = median_market

    # If still no price, leave blank
    if rec is None or (isinstance(rec, float) and pd.isna(rec)):
        return None

    # Keep cost EXACT when cost floor triggered (don't round it)
    if round_to and not used_cost_floor:
        try:
            rec = round(rec / float(round_to)) * float(round_to)
        except Exception:
            pass

    return rec

def apply_pricing(aggregated: pd.DataFrame, strategy="undercut_min_1pct", min_margin=0.20, round_to=1, cost_floor_enabled=True):
    df = aggregated.copy()
    df["recommended_price"] = df.apply(
        lambda r: recommend_price(r, strategy=strategy, min_margin=min_margin, round_to=round_to, cost_floor_enabled=cost_floor_enabled),
        axis=1
    )

    # Flag below-cost (for visibility)
    if "cost_aed" in df.columns:
        cost = pd.to_numeric(df["cost_aed"], errors="coerce")
        rec  = pd.to_numeric(df["recommended_price"], errors="coerce")
        df["below_cost_flag"] = (cost > 0) & (~pd.isna(rec)) & (cost > rec)
    else:
        df["below_cost_flag"] = pd.NA

    # --- cost gap sanity (>30% by default; overridable in config/policy.yml) ---
    warn = 0.30
    try:
        import yaml
        pol = yaml.safe_load(open("config/policy.yml"))
        warn = pol.get("sanity", {}).get("cost_gap_warn_pct", warn)
    except Exception:
        pass

    cost = pd.to_numeric(df.get("cost_aed"), errors="coerce")
    rec  = pd.to_numeric(df.get("recommended_price"), errors="coerce")

    df["cost_gap_pct"] = (rec - cost) / cost
    df.loc[(cost <= 0) | cost.isna() | rec.isna(), "cost_gap_pct"] = pd.NA
    df["cost_gap_flag"] = (cost > 0) & (rec.notna()) & (df["cost_gap_pct"].abs() > warn)

    # Optional: add provenance so you can see why a price was chosen
    try:
        minm = pd.to_numeric(df.get("min_market"), errors="coerce")
        cost = pd.to_numeric(df.get("cost_aed"), errors="coerce")
        df["recommended_source"] = "undercut_min_1pct"
        if strategy == "undercut_min_1pct" and cost_floor_enabled:
            df.loc[(minm.notna()) & (cost.notna()) & (minm < cost), "recommended_source"] = "cost_floor"
    except Exception:
        pass

    # Ensure no legacy margin column
    if "margin_pct" in df.columns:
        df.drop(columns=["margin_pct"], inplace=True)

    return df
