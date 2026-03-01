from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple

import polars as pl

DATA_DIR = Path(__file__).parent.parent / "data"
CACHE_DIR = Path(__file__).parent.parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

FORENAMES_PARQUET = CACHE_DIR / "forenames.parquet"
SURNAMES_PARQUET = CACHE_DIR / "surnames.parquet"
TOTALS_PARQUET = CACHE_DIR / "totals.parquet"
GLOBAL_JSON = CACHE_DIR / "global_totals.json"

# Cache tuples, not DataFrames
MAX_NAME_CACHE = 12_000          # per-name per-country counts (first OR last)
MAX_GLOBAL_COUNT_CACHE = 25_000  # global sum for a name

country_codes_df = pl.read_csv(DATA_DIR / "country_codes.csv")
country_to_code = dict(zip(country_codes_df["country_name"], country_codes_df["country_code"]))
code_to_country = {v: k for k, v in country_to_code.items()}


def _safe_country_code(claimed_country: str) -> str:
    claimed_country = (claimed_country or "").strip()
    if claimed_country in country_to_code:
        return country_to_code[claimed_country]
    cc = claimed_country.upper()
    return cc if len(cc) == 2 else cc[:2]


def _parquet_has_hash_column(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        cols = pl.scan_parquet(path).collect_schema().names()
        return "h" in cols and "name" in cols and "country" in cols and "count" in cols
    except Exception:
        return False


# The goal was to fit into 512 MB RAM :)
def _build_cache_if_missing() -> None:
    """
    Build Parquet caches once. This is the only heavy step.
    Also computes vocabulary sizes needed for additive smoothing.
    Includes a u64 hash column 'h' to speed up name lookups.
    """
    caches_exist = (
        FORENAMES_PARQUET.exists()
        and SURNAMES_PARQUET.exists()
        and TOTALS_PARQUET.exists()
        and GLOBAL_JSON.exists()
    )
    caches_ok = caches_exist and _parquet_has_hash_column(FORENAMES_PARQUET) and _parquet_has_hash_column(SURNAMES_PARQUET)
    if caches_ok:
        return

    print("Building Parquet cache (one-time)...")

    # Forenames
    forenames = (
        pl.scan_csv(DATA_DIR / "forenames_eu.csv", low_memory=True)
        .select(
            pl.col("country").cast(pl.Utf8),
            pl.col("forename").cast(pl.Utf8).str.to_lowercase().alias("name"),
            pl.col("count").fill_null(0).cast(pl.Int64),
        )
        .with_columns(pl.col("name").hash().alias("h"))  # u64 hash
        .group_by(["country", "name", "h"])
        .agg(pl.col("count").sum().alias("count"))
        .collect(streaming=True)
    )
    forenames.write_parquet(FORENAMES_PARQUET)

    # Surnames
    surnames = (
        pl.concat(
            [
                pl.scan_csv(DATA_DIR / "surnames_eu_part1.csv", low_memory=True),
                pl.scan_csv(DATA_DIR / "surnames_eu_part2.csv", low_memory=True),
            ],
            how="vertical_relaxed",
        )
        .select(
            pl.col("country").cast(pl.Utf8),
            pl.col("surname").cast(pl.Utf8).str.to_lowercase().alias("name"),
            pl.col("count").fill_null(0).cast(pl.Int64),
        )
        .with_columns(pl.col("name").hash().alias("h"))  # u64 hash
        .group_by(["country", "name", "h"])
        .agg(pl.col("count").sum().alias("count"))
        .collect(streaming=True)
    )
    surnames.write_parquet(SURNAMES_PARQUET)

    # Totals per country
    forename_totals = forenames.group_by("country").agg(pl.col("count").sum().alias("total_forenames"))
    surname_totals = surnames.group_by("country").agg(pl.col("count").sum().alias("total_surnames"))
    totals = (
        forename_totals.join(surname_totals, on="country", how="inner")
        .select("country", "total_forenames", "total_surnames")
    )
    totals.write_parquet(TOTALS_PARQUET)

    # Vocabulary sizes for proper smoothing
    V_FORENAMES = int(forenames.select(pl.col("name").n_unique()).item())
    V_SURNAMES = int(surnames.select(pl.col("name").n_unique()).item())

    global_totals = {
        "GLOBAL_FORENAME_TOTAL": int(forenames["count"].sum()),
        "GLOBAL_SURNAME_TOTAL": int(surnames["count"].sum()),
        "V_FORENAMES": V_FORENAMES,
        "V_SURNAMES": V_SURNAMES,
    }
    GLOBAL_JSON.write_text(json.dumps(global_totals), encoding="utf-8")

    print("Cache built.")


# Build caches once if needed
_build_cache_if_missing()

# Lazy scans (cheap, low RAM)
forenames_lf = pl.scan_parquet(FORENAMES_PARQUET)
surnames_lf = pl.scan_parquet(SURNAMES_PARQUET)
countries_df = pl.read_parquet(TOTALS_PARQUET)  # small; OK to load

_global = json.loads(GLOBAL_JSON.read_text(encoding="utf-8"))
GLOBAL_FORENAME_TOTAL = int(_global["GLOBAL_FORENAME_TOTAL"])
GLOBAL_SURNAME_TOTAL = int(_global["GLOBAL_SURNAME_TOTAL"])
V_FORENAMES = int(_global["V_FORENAMES"])
V_SURNAMES = int(_global["V_SURNAMES"])


# Hash helper (must match Parquet 'h' computation)
# Compute it using Polars' hash to match exactly
def _name_hash_u64(name: str) -> int:
    return int(pl.Series([name]).hash()[0])


# Cache results as Tuple[(country_code, count), ...] rather than DataFrames
@lru_cache(maxsize=MAX_NAME_CACHE)
def _forename_counts_tuples(name: str) -> Tuple[Tuple[str, int], ...]:
    if not name:
        return tuple()
    h = _name_hash_u64(name)
    df = (
        forenames_lf
        .filter(pl.col("h") == h)
        .filter(pl.col("name") == name)  # collision guard
        .select(pl.col("country"), pl.col("count"))
        .collect()
    )
    if df.is_empty():
        return tuple()
    # Keep only tiny Python primitives in cache
    return tuple((str(c), int(cnt)) for c, cnt in df.iter_rows())


@lru_cache(maxsize=MAX_NAME_CACHE)
def _surname_counts_tuples(name: str) -> Tuple[Tuple[str, int], ...]:
    if not name:
        return tuple()
    h = _name_hash_u64(name)
    df = (
        surnames_lf
        .filter(pl.col("h") == h)
        .filter(pl.col("name") == name)  # collision guard
        .select(pl.col("country"), pl.col("count"))
        .collect()
    )
    if df.is_empty():
        return tuple()
    return tuple((str(c), int(cnt)) for c, cnt in df.iter_rows())


@lru_cache(maxsize=MAX_GLOBAL_COUNT_CACHE)
def _global_forename_count(name: str) -> int:
    if not name:
        return 0
    h = _name_hash_u64(name)
    val = (
        forenames_lf
        .filter(pl.col("h") == h)
        .filter(pl.col("name") == name)
        .select(pl.col("count").sum())
        .collect()[0, 0]
    )
    return int(val or 0)


@lru_cache(maxsize=MAX_GLOBAL_COUNT_CACHE)
def _global_surname_count(name: str) -> int:
    if not name:
        return 0
    h = _name_hash_u64(name)
    val = (
        surnames_lf
        .filter(pl.col("h") == h)
        .filter(pl.col("name") == name)
        .select(pl.col("count").sum())
        .collect()[0, 0]
    )
    return int(val or 0)


def _tuples_to_df(tuples_: Tuple[Tuple[str, int], ...], col_name: str) -> pl.DataFrame:
    """
    Convert cached tuples into a small DataFrame for joining.
    This allocates per-request but remains small (countries where the name exists).
    """
    if not tuples_:
        return pl.DataFrame({"country": [], col_name: []}, schema={"country": pl.Utf8, col_name: pl.Int64})
    countries, counts = zip(*tuples_)
    return pl.DataFrame({"country": list(countries), col_name: list(counts)})


def check_plausibility(first: str, last: str, claimed_country: str) -> Dict[str, Any]:
    """
    Evaluate how plausible a first+last name pairing is for a claimed country.

    Returns:
    1) plausibility_ratio:
       Likelihood of the name pairing in the claimed country
       relative to the Europe-wide baseline.
       >1 means more typical than EU average.

    2) posterior_share:
       Normalized likelihood across all countries (sums to 1),
       used for ranking countries.

    Notes:
    - First and last names are assumed independent given country.
    - Uses proper additive smoothing with vocabulary size.
    - Guardrail: if (0 first AND 0 last) for a country, joint is set to 0.
    """
    code = _safe_country_code(claimed_country)
    first_lower = (first or "").lower().strip()
    last_lower = (last or "").lower().strip()

    alpha = 0.5

    # Cached per-name lookups (A), accelerated by hash (B)
    f_tuples = _forename_counts_tuples(first_lower)
    l_tuples = _surname_counts_tuples(last_lower)
    f_counts = _tuples_to_df(f_tuples, "f_cnt")
    l_counts = _tuples_to_df(l_tuples, "l_cnt")

    per_country = (
        countries_df
        .join(f_counts, on="country", how="left")
        .join(l_counts, on="country", how="left")
        .with_columns(
            pl.col("f_cnt").fill_null(0),
            pl.col("l_cnt").fill_null(0),
        )
        # Proper additive smoothing: denom includes alpha * V
        .with_columns(
            ((pl.col("f_cnt") + alpha) / (pl.col("total_forenames") + alpha * V_FORENAMES)).alias("p_first"),
            ((pl.col("l_cnt") + alpha) / (pl.col("total_surnames") + alpha * V_SURNAMES)).alias("p_last"),
        )
        .with_columns((pl.col("p_first") * pl.col("p_last")).alias("joint_raw"))
        # Guardrail: if both counts are 0, treat as "no evidence" => joint = 0
        .with_columns(
            pl.when((pl.col("f_cnt") == 0) & (pl.col("l_cnt") == 0))
            .then(0.0)
            .otherwise(pl.col("joint_raw"))
            .alias("joint")
        )
        .select("country", "joint", "f_cnt", "l_cnt")
        .sort("joint", descending=True)
    )

    joint_sum = float(per_country["joint"].sum()) or 1.0
    per_country = per_country.with_columns((pl.col("joint") / joint_sum).alias("posterior_share"))

    # Cached global baseline counts
    global_first_count = _global_forename_count(first_lower)
    global_last_count = _global_surname_count(last_lower)

    # Proper smoothing for global too
    p_first_global = (global_first_count + alpha) / (GLOBAL_FORENAME_TOTAL + alpha * V_FORENAMES)
    p_last_global = (global_last_count + alpha) / (GLOBAL_SURNAME_TOTAL + alpha * V_SURNAMES)
    p_global_joint = p_first_global * p_last_global

    claimed_row = per_country.filter(pl.col("country") == code)
    claimed_joint = float(claimed_row["joint"][0]) if claimed_row.height else 0.0
    posterior_share_claimed = float(claimed_row["posterior_share"][0]) if claimed_row.height else 0.0

    plausibility_ratio = (claimed_joint / p_global_joint) if p_global_joint > 0 else 0.0

    if plausibility_ratio < 0.3:
        plaus_label = "Very unusual"
    elif plausibility_ratio < 0.7:
        plaus_label = "Unusual"
    elif plausibility_ratio < 1.5:
        plaus_label = "Neutral"
    elif plausibility_ratio < 3.0:
        plaus_label = "Typical"
    else:
        plaus_label = "Very typical"

    countries_sorted = per_country["country"].to_list()
    claimed_rank = countries_sorted.index(code) + 1 if code in countries_sorted else "unknown"

    top_country_code = per_country["country"][0]
    top_country = code_to_country.get(top_country_code, top_country_code)

    ranked_countries: List[Dict[str, Any]] = []
    for i, row in enumerate(per_country.head(8).iter_rows(named=True), start=1):
        ranked_countries.append(
            {
                "rank": i,
                "country": code_to_country.get(row["country"], row["country"]),
                "posterior_share_pct": round(100 * float(row["posterior_share"]), 2),
                "first_count": int(row["f_cnt"]),
                "last_count": int(row["l_cnt"]),
                "is_claimed": (row["country"] == code),
            }
        )

    return {
        "country": claimed_country,
        "plausibility_ratio": round(plausibility_ratio, 3),
        "plausibility_label": plaus_label,
        "posterior_share_claimed_pct": round(100 * posterior_share_claimed, 2),
        "claimed_rank": claimed_rank,
        "top_country": top_country,
        "ranked_countries": ranked_countries,
    }