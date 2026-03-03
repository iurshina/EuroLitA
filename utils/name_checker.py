from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple

import polars as pl

from .name_normalizer import EuropeanNameNormalizer

logger = logging.getLogger("eurolita.name_checker")

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


try:
    country_codes_df = pl.read_csv(DATA_DIR / "country_codes.csv")
    country_to_code = dict(zip(country_codes_df["country_name"], country_codes_df["country_code"]))
    code_to_country = {v: k for k, v in country_to_code.items()}
except Exception:
    logger.exception("Failed to load country codes CSV from %s", DATA_DIR / "country_codes.csv")
    raise


def _safe_country_code(claimed_country: str) -> str:
    claimed_country = (claimed_country or "").strip()
    if claimed_country in country_to_code:
        return country_to_code[claimed_country]
    cc = claimed_country.upper()
    return cc if len(cc) == 2 else cc[:2]


def _parquet_has_hash_columns(path: Path) -> bool:
    """
    We require both primary + ascii keys (and their hashes) in the cache schema.
    """
    if not path.exists():
        return False
    try:
        cols = pl.scan_parquet(path).collect_schema().names()
        required = {"country", "count", "name", "h", "name_ascii", "h_ascii"}
        return required.issubset(set(cols))
    except Exception:
        logger.warning("Failed to inspect parquet schema for %s", path, exc_info=True)
        return False


def _name_hash_u64(name: str) -> int:
    # must match Parquet 'h' computation
    return int(pl.Series([name]).hash()[0])


def _normalize_unique_names(df: pl.DataFrame, raw_col: str, *, de_transliteration: bool) -> pl.DataFrame:
    """
    Build a mapping DataFrame: raw -> name (primary) + name_ascii (fallback)
    Normalizes only unique raw strings to keep work bounded.
    """
    norm = EuropeanNameNormalizer(keep_apostrophe=False, de_transliteration=de_transliteration)
    uniq = df.select(pl.col(raw_col).unique()).get_column(raw_col).to_list()

    raw_vals: List[str] = []
    prim_vals: List[str] = []
    ascii_vals: List[str] = []

    for x in uniq:
        x_str = "" if x is None else str(x)
        vars_ = norm.variants(x_str)
        primary = vars_[0] if vars_ else ""
        ascii_ = vars_[1] if len(vars_) > 1 else primary

        raw_vals.append(x_str)
        prim_vals.append(primary)
        ascii_vals.append(ascii_)

    return pl.DataFrame({raw_col: raw_vals, "name": prim_vals, "name_ascii": ascii_vals})


# The goal was to fit into 512 MB RAM :)
def _build_cache_if_missing() -> None:
    """
    Build Parquet caches once. This is the only heavy step.

    Cache schema (both forenames/surnames):
      - country: Utf8
      - name: primary normalized key
      - h: hash(name)
      - name_ascii: ascii fallback normalized key
      - h_ascii: hash(name_ascii)
      - count: Int64

    Also computes vocabulary sizes needed for additive smoothing (based on primary key).
    """
    caches_exist = (
        FORENAMES_PARQUET.exists()
        and SURNAMES_PARQUET.exists()
        and TOTALS_PARQUET.exists()
        and GLOBAL_JSON.exists()
    )
    caches_ok = caches_exist and _parquet_has_hash_columns(FORENAMES_PARQUET) and _parquet_has_hash_columns(SURNAMES_PARQUET)
    if caches_ok:
        logger.info("Cache OK: using existing parquet files in %s", CACHE_DIR)
        return

    logger.info("Building Parquet cache (one-time). DATA_DIR=%s CACHE_DIR=%s", DATA_DIR, CACHE_DIR)

    try:
        raw_forenames = (
            pl.scan_csv(DATA_DIR / "forenames_eu.csv", low_memory=True)
            .select(
                pl.col("country").cast(pl.Utf8),
                pl.col("forename").cast(pl.Utf8).alias("raw"),
                pl.col("count").fill_null(0).cast(pl.Int64),
            )
            .collect(streaming=True)
        )

        # Important: we do NOT apply de_transliteration for the *whole* cache,
        # because the cache covers many countries; DE transliteration is applied at query time.
        forename_map = _normalize_unique_names(raw_forenames, "raw", de_transliteration=False)

        forenames = (
            raw_forenames
            .join(forename_map, on="raw", how="left")
            .drop("raw")
            .with_columns(
                pl.col("name").hash().alias("h"),
                pl.col("name_ascii").hash().alias("h_ascii"),
            )
            .group_by(["country", "name", "h", "name_ascii", "h_ascii"])
            .agg(pl.col("count").sum().alias("count"))
        )
        forenames.write_parquet(FORENAMES_PARQUET)

        raw_surnames = (
            pl.concat(
                [
                    pl.scan_csv(DATA_DIR / "surnames_eu_part1.csv", low_memory=True),
                    pl.scan_csv(DATA_DIR / "surnames_eu_part2.csv", low_memory=True),
                ],
                how="vertical_relaxed",
            )
            .select(
                pl.col("country").cast(pl.Utf8),
                pl.col("surname").cast(pl.Utf8).alias("raw"),
                pl.col("count").fill_null(0).cast(pl.Int64),
            )
            .collect(streaming=True)
        )

        surname_map = _normalize_unique_names(raw_surnames, "raw", de_transliteration=False)

        surnames = (
            raw_surnames
            .join(surname_map, on="raw", how="left")
            .drop("raw")
            .with_columns(
                pl.col("name").hash().alias("h"),
                pl.col("name_ascii").hash().alias("h_ascii"),
            )
            .group_by(["country", "name", "h", "name_ascii", "h_ascii"])
            .agg(pl.col("count").sum().alias("count"))
        )
        surnames.write_parquet(SURNAMES_PARQUET)

        forename_totals = forenames.group_by("country").agg(pl.col("count").sum().alias("total_forenames"))
        surname_totals = surnames.group_by("country").agg(pl.col("count").sum().alias("total_surnames"))
        totals = (
            forename_totals.join(surname_totals, on="country", how="inner")
            .select("country", "total_forenames", "total_surnames")
        )
        totals.write_parquet(TOTALS_PARQUET)

        # Use PRIMARY key vocab sizes (keeps your model sharper)
        v_forenames = int(forenames.select(pl.col("name").n_unique()).item())
        v_surnames = int(surnames.select(pl.col("name").n_unique()).item())

        global_totals = {
            "GLOBAL_FORENAME_TOTAL": int(forenames["count"].sum()),
            "GLOBAL_SURNAME_TOTAL": int(surnames["count"].sum()),
            "V_FORENAMES": v_forenames,
            "V_SURNAMES": v_surnames,
        }
        GLOBAL_JSON.write_text(json.dumps(global_totals), encoding="utf-8")

        logger.info(
            "Cache built: forenames=%s surnames=%s totals=%s global=%s",
            FORENAMES_PARQUET.name,
            SURNAMES_PARQUET.name,
            TOTALS_PARQUET.name,
            GLOBAL_JSON.name,
        )

    except Exception:
        logger.exception("Cache build failed. Check that data files exist and are readable in %s", DATA_DIR)
        raise


# Build caches once if needed
_build_cache_if_missing()

try:
    # Lazy scans (cheap, low RAM)
    forenames_lf = pl.scan_parquet(FORENAMES_PARQUET)
    surnames_lf = pl.scan_parquet(SURNAMES_PARQUET)
    countries_df = pl.read_parquet(TOTALS_PARQUET)  # small; OK to load
except Exception:
    logger.exception("Failed to load parquet cache files from %s", CACHE_DIR)
    raise

try:
    _global = json.loads(GLOBAL_JSON.read_text(encoding="utf-8"))
    GLOBAL_FORENAME_TOTAL = int(_global["GLOBAL_FORENAME_TOTAL"])
    GLOBAL_SURNAME_TOTAL = int(_global["GLOBAL_SURNAME_TOTAL"])
    V_FORENAMES = int(_global["V_FORENAMES"])
    V_SURNAMES = int(_global["V_SURNAMES"])
except Exception:
    logger.exception("Failed to load global totals JSON from %s", GLOBAL_JSON)
    raise


@lru_cache(maxsize=MAX_NAME_CACHE)
def _forename_counts_tuples2(name: str, name_ascii: str) -> Tuple[Tuple[str, int], ...]:
    if not name and not name_ascii:
        return tuple()
    try:
        frames: List[pl.LazyFrame] = []

        if name:
            h = _name_hash_u64(name)
            frames.append(
                forenames_lf
                .filter(pl.col("h") == h)
                .filter(pl.col("name") == name)  # collision guard
                .select("country", "count")
            )

        if name_ascii and name_ascii != name:
            h_ascii = _name_hash_u64(name_ascii)
            frames.append(
                forenames_lf
                .filter(pl.col("h_ascii") == h_ascii)
                .filter(pl.col("name_ascii") == name_ascii)  # collision guard
                .select("country", "count")
            )

        if not frames:
            return tuple()

        df = pl.concat([f.collect() for f in frames], how="vertical_relaxed")
        if df.is_empty():
            return tuple()

        df = df.group_by("country").agg(pl.col("count").sum().alias("count"))
        return tuple((str(c), int(cnt)) for c, cnt in df.iter_rows())
    except Exception:
        logger.exception("Forename lookup failed")
        return tuple()


@lru_cache(maxsize=MAX_NAME_CACHE)
def _surname_counts_tuples2(name: str, name_ascii: str) -> Tuple[Tuple[str, int], ...]:
    if not name and not name_ascii:
        return tuple()
    try:
        frames: List[pl.LazyFrame] = []

        if name:
            h = _name_hash_u64(name)
            frames.append(
                surnames_lf
                .filter(pl.col("h") == h)
                .filter(pl.col("name") == name)  # collision guard
                .select("country", "count")
            )

        if name_ascii and name_ascii != name:
            h_ascii = _name_hash_u64(name_ascii)
            frames.append(
                surnames_lf
                .filter(pl.col("h_ascii") == h_ascii)
                .filter(pl.col("name_ascii") == name_ascii)  # collision guard
                .select("country", "count")
            )

        if not frames:
            return tuple()

        df = pl.concat([f.collect() for f in frames], how="vertical_relaxed")
        if df.is_empty():
            return tuple()

        df = df.group_by("country").agg(pl.col("count").sum().alias("count"))
        return tuple((str(c), int(cnt)) for c, cnt in df.iter_rows())
    except Exception:
        logger.exception("Surname lookup failed")
        return tuple()


@lru_cache(maxsize=MAX_GLOBAL_COUNT_CACHE)
def _global_forename_count2(name: str, name_ascii: str) -> int:
    if not name and not name_ascii:
        return 0
    try:
        total = 0

        if name:
            h = _name_hash_u64(name)
            val = (
                forenames_lf
                .filter(pl.col("h") == h)
                .filter(pl.col("name") == name)
                .select(pl.col("count").sum())
                .collect()[0, 0]
            )
            total += int(val or 0)

        if name_ascii and name_ascii != name:
            h_ascii = _name_hash_u64(name_ascii)
            val2 = (
                forenames_lf
                .filter(pl.col("h_ascii") == h_ascii)
                .filter(pl.col("name_ascii") == name_ascii)
                .select(pl.col("count").sum())
                .collect()[0, 0]
            )
            total += int(val2 or 0)

        return total
    except Exception:
        logger.exception("Global forename count failed")
        return 0


@lru_cache(maxsize=MAX_GLOBAL_COUNT_CACHE)
def _global_surname_count2(name: str, name_ascii: str) -> int:
    if not name and not name_ascii:
        return 0
    try:
        total = 0

        if name:
            h = _name_hash_u64(name)
            val = (
                surnames_lf
                .filter(pl.col("h") == h)
                .filter(pl.col("name") == name)
                .select(pl.col("count").sum())
                .collect()[0, 0]
            )
            total += int(val or 0)

        if name_ascii and name_ascii != name:
            h_ascii = _name_hash_u64(name_ascii)
            val2 = (
                surnames_lf
                .filter(pl.col("h_ascii") == h_ascii)
                .filter(pl.col("name_ascii") == name_ascii)
                .select(pl.col("count").sum())
                .collect()[0, 0]
            )
            total += int(val2 or 0)

        return total
    except Exception:
        logger.exception("Global surname count failed")
        return 0


def _tuples_to_df(tuples_: Tuple[Tuple[str, int], ...], col_name: str) -> pl.DataFrame:
    if not tuples_:
        return pl.DataFrame({"country": [], col_name: []}, schema={"country": pl.Utf8, col_name: pl.Int64})
    countries, counts = zip(*tuples_)
    return pl.DataFrame({"country": list(countries), col_name: list(counts)})


def check_plausibility(first: str, last: str, claimed_country: str) -> Dict[str, Any]:
    """
    Evaluate how plausible a first+last name pairing is for a claimed country.
    """
    code = _safe_country_code(claimed_country)

    # German transliteration only for DE/AT/CH (query-time only)
    normalizer = EuropeanNameNormalizer(
        keep_apostrophe=False,
        de_transliteration=(code in {"DE", "AT", "CH"}),
    )

    first_vars = normalizer.variants(first or "")
    last_vars = normalizer.variants(last or "")

    first_primary = first_vars[0] if first_vars else ""
    first_ascii = first_vars[1] if len(first_vars) > 1 else first_primary

    last_primary = last_vars[0] if last_vars else ""
    last_ascii = last_vars[1] if len(last_vars) > 1 else last_primary

    alpha = 0.5

    try:
        f_tuples = _forename_counts_tuples2(first_primary, first_ascii)
        l_tuples = _surname_counts_tuples2(last_primary, last_ascii)

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
            .with_columns(
                ((pl.col("f_cnt") + alpha) / (pl.col("total_forenames") + alpha * V_FORENAMES)).alias("p_first"),
                ((pl.col("l_cnt") + alpha) / (pl.col("total_surnames") + alpha * V_SURNAMES)).alias("p_last"),
            )
            .with_columns((pl.col("p_first") * pl.col("p_last")).alias("joint_raw"))
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

        global_first_count = _global_forename_count2(first_primary, first_ascii)
        global_last_count = _global_surname_count2(last_primary, last_ascii)

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

    except Exception:
        logger.exception("check_plausibility failed (claimed_country=%r)", claimed_country)
        raise