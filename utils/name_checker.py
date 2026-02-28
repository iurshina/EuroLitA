from pathlib import Path
from typing import Dict, List, Any

import polars as pl

DATA_DIR = Path(__file__).parent.parent / "data"

country_codes_df = pl.read_csv(DATA_DIR / "country_codes.csv")
country_to_code = dict(zip(country_codes_df["country_name"], country_codes_df["country_code"]))
code_to_country = {v: k for k, v in country_to_code.items()}


def _safe_country_code(claimed_country: str) -> str:
    claimed_country = (claimed_country or "").strip()
    if claimed_country in country_to_code:
        return country_to_code[claimed_country]
    cc = claimed_country.upper()
    return cc if len(cc) == 2 else cc[:2]


def _count_expr(lf: pl.LazyFrame) -> pl.Expr:
    """
    Some files use 'count' instead of 'occurrences'.
    Pick whichever exists and normalize to 'count'.
    """
    cols = set(lf.collect_schema().names())
    if "occurrences" in cols:
        col = "occurrences"
    elif "count" in cols:
        col = "count"
    else:
        raise ValueError(f"Neither 'occurrences' nor 'count' found. Columns: {sorted(cols)}")

    return pl.col(col).fill_null(0).cast(pl.Int64).alias("count")


print("Loading name datasets (Polars streaming aggregate)...")

forenames_lf = pl.scan_csv(
    DATA_DIR / "forenames_eu.csv",
    dtypes={"gender": pl.String},
    low_memory=True,
)

forenames_agg = (
    forenames_lf.select(
        pl.col("country").cast(pl.Utf8),
        pl.col("forename").cast(pl.Utf8).str.to_lowercase().alias("name"),
        _count_expr(forenames_lf),
    )
    .group_by(["country", "name"])
    .agg(pl.col("count").sum().alias("count"))
    .collect(streaming=True)
)

surnames_lf_1 = pl.scan_csv(
    DATA_DIR / "surnames_eu_part1.csv",
    dtypes={"gender": pl.String},
    low_memory=True,
)
surnames_lf_2 = pl.scan_csv(
    DATA_DIR / "surnames_eu_part2.csv",
    dtypes={"gender": pl.String},
    low_memory=True,
)

surnames_lf = pl.concat([surnames_lf_1, surnames_lf_2], how="vertical_relaxed")

# assume same schema in both surname parts; take count column from part1
count_expr_surnames = _count_expr(surnames_lf_1)

surnames_agg = (
    surnames_lf.select(
        pl.col("country").cast(pl.Utf8),
        pl.col("surname").cast(pl.Utf8).str.to_lowercase().alias("name"),
        count_expr_surnames,
    )
    .group_by(["country", "name"])
    .agg(pl.col("count").sum().alias("count"))
    .collect(streaming=True)
)

# Totals by country (small)
forename_totals = forenames_agg.group_by("country").agg(pl.col("count").sum().alias("total_forenames"))
surname_totals = surnames_agg.group_by("country").agg(pl.col("count").sum().alias("total_surnames"))

countries_df = (
    forename_totals.join(surname_totals, on="country", how="inner")
    .select("country", "total_forenames", "total_surnames")
)

# Global totals (for baseline plausibility ratio)
GLOBAL_FORENAME_TOTAL = int(forenames_agg["count"].sum())
GLOBAL_SURNAME_TOTAL = int(surnames_agg["count"].sum())

print(f"Loaded aggregated tables for {countries_df.height} countries.")
print("Name datasets loaded.")


def check_plausibility(first: str, last: str, claimed_country: str) -> Dict[str, Any]:
    """
    Returns BOTH:
      1) plausibility_ratio: how typical the name pair is in claimed country vs EU baseline (ratio)
      2) posterior ranking list: normalized share across countries (sums to 1), plus rank

    Notes:
      - 'posterior_share' is a normalized share across countries, not a real demographic probability.
      - 'plausibility_ratio' answers “does it fit Germany?” better than posterior_share does.
    """
    code = _safe_country_code(claimed_country)
    first_lower = (first or "").lower().strip()
    last_lower = (last or "").lower().strip()

    if countries_df.height == 0:
        return {
            "country": claimed_country,
            "claimed_code": code,
            "message": "No country data available.",
            "plausibility_ratio": 0.0,
            "plausibility_label": "Unknown",
            "posterior_share_claimed": 0.0,
            "claimed_rank": "unknown",
            "top_country": None,
            "ranked_countries": [],
        }

    alpha = 0.5  # smoothing

    # Per-country counts for this query
    f_counts = (
        forenames_agg.filter(pl.col("name") == first_lower)
        .select(pl.col("country"), pl.col("count").alias("f_cnt"))
    )
    l_counts = (
        surnames_agg.filter(pl.col("name") == last_lower)
        .select(pl.col("country"), pl.col("count").alias("l_cnt"))
    )

    per_country = (
        countries_df.join(f_counts, on="country", how="left")
        .join(l_counts, on="country", how="left")
        .with_columns(
            pl.col("f_cnt").fill_null(0),
            pl.col("l_cnt").fill_null(0),
        )
        .with_columns(
            ((pl.col("f_cnt") + alpha) / (pl.col("total_forenames") + alpha)).alias("p_first"),
            ((pl.col("l_cnt") + alpha) / (pl.col("total_surnames") + alpha)).alias("p_last"),
        )
        .with_columns((pl.col("p_first") * pl.col("p_last")).alias("joint"))
        .select("country", "joint", "f_cnt", "l_cnt")
        .sort("joint", descending=True)
    )

    # If everything is empty (shouldn't happen with smoothing, but keep safe)
    if per_country.height == 0:
        return {
            "country": claimed_country,
            "claimed_code": code,
            "message": "No matching data for this name combination.",
            "plausibility_ratio": 0.0,
            "plausibility_label": "Unknown",
            "posterior_share_claimed": 0.0,
            "claimed_rank": "unknown",
            "top_country": None,
            "ranked_countries": [],
        }

    # "Posterior" shares across countries (normalize joint to sum 1)
    joint_sum = float(per_country["joint"].sum()) or 1.0
    per_country = per_country.with_columns((pl.col("joint") / joint_sum).alias("posterior_share"))

    # Global baseline probabilities for plausibility ratio
    global_first_count = int(
        forenames_agg.filter(pl.col("name") == first_lower)["count"].sum()
    )
    global_last_count = int(
        surnames_agg.filter(pl.col("name") == last_lower)["count"].sum()
    )

    p_first_global = (global_first_count + alpha) / (GLOBAL_FORENAME_TOTAL + alpha)
    p_last_global = (global_last_count + alpha) / (GLOBAL_SURNAME_TOTAL + alpha)
    p_global_joint = p_first_global * p_last_global

    # Claimed country row
    claimed_row = per_country.filter(pl.col("country") == code)
    claimed_joint = float(claimed_row["joint"][0]) if claimed_row.height else 0.0
    posterior_share_claimed = float(claimed_row["posterior_share"][0]) if claimed_row.height else 0.0
    f_count_claimed = int(claimed_row["f_cnt"][0]) if claimed_row.height else 0
    l_count_claimed = int(claimed_row["l_cnt"][0]) if claimed_row.height else 0

    plausibility_ratio = (claimed_joint / p_global_joint) if p_global_joint > 0 else 0.0

    # Friendly label for plausibility ratio
    if plausibility_ratio <= 0:
        plaus_label = "Unknown"
    elif plausibility_ratio < 0.3:
        plaus_label = "Very unusual"
    elif plausibility_ratio < 0.7:
        plaus_label = "Unusual"
    elif plausibility_ratio < 1.5:
        plaus_label = "Neutral"
    elif plausibility_ratio < 3.0:
        plaus_label = "Typical"
    else:
        plaus_label = "Very typical"

    # Rank of claimed (1-based) in the posterior ordering
    claimed_rank = "unknown"
    countries_sorted = per_country["country"].to_list()
    if code in countries_sorted:
        claimed_rank = countries_sorted.index(code) + 1

    top_country_code = per_country["country"][0]
    top_country = code_to_country.get(top_country_code, top_country_code)

    # Ranked list INCLUDING claimed country (top N)
    TOP_N = 8
    ranked_countries: List[Dict[str, Any]] = []
    for i, row in enumerate(per_country.head(TOP_N).iter_rows(named=True), start=1):
        ranked_countries.append(
            {
                "rank": i,
                "country": code_to_country.get(row["country"], row["country"]),
                "code": row["country"],
                "posterior_share_pct": round(100 * float(row["posterior_share"]), 2),
                "first_count": int(row["f_cnt"]),
                "last_count": int(row["l_cnt"]),
                "is_claimed": (row["country"] == code),
            }
        )

    message = (
        f"{claimed_country}: plausibility {plausibility_ratio:.2f}× EU baseline ({plaus_label}). "
        f"Posterior share {100*posterior_share_claimed:.2f}%."
        f" (first={f_count_claimed:,}, last={l_count_claimed:,})"
    )

    return {
        "country": claimed_country,
        "claimed_code": code,
        "message": message,
        # Fit metric (use this as your main “plausibility”)
        "plausibility_ratio": round(plausibility_ratio, 3),
        "plausibility_label": plaus_label,
        # Ranking metric (use this for the full list)
        "posterior_share_claimed": round(posterior_share_claimed, 6),
        "posterior_share_claimed_pct": round(100 * posterior_share_claimed, 2),
        "claimed_rank": claimed_rank,
        "top_country": top_country,
        "ranked_countries": ranked_countries,
        # keep these for debugging/inspection
        "counts_claimed": {"first": f_count_claimed, "last": l_count_claimed},
        "global_counts": {"first": global_first_count, "last": global_last_count},
    }