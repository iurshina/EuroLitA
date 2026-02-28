from collections import defaultdict
from pathlib import Path
import pandas as pd
import math
from collections import defaultdict
from typing import Dict, List, Tuple


DATA_DIR = Path(__file__).parent.parent / "data"

print("Loading name datasets...") 

# Load CSVs
forenames_df = pd.read_csv(
    DATA_DIR / "forenames_eu.csv",
    low_memory=False,
    dtype={"gender": "string"}
)

surnames_df = pd.concat([
    pd.read_csv(DATA_DIR / "surnames_eu_part1.csv", low_memory=False, dtype={"gender": "string"}),
    pd.read_csv(DATA_DIR / "surnames_eu_part2.csv", low_memory=False, dtype={"gender": "string"})
], ignore_index=True)

# Load country mapping (full name -> code)
country_codes_df = pd.read_csv(DATA_DIR / "country_codes.csv")
# Your mapping uses 'country_name' -> 'country_code' (adjust columns if different)
country_to_code = dict(zip(country_codes_df['country_name'], country_codes_df['country_code']))

forename_lookup: dict = defaultdict(lambda: defaultdict(int))
for _, row in forenames_df.iterrows():
    code = row['country']                     
    name = str(row['forename']).lower()
    count = int(row.get('occurrences', row.get('count', 0)))
    forename_lookup[code][name] += count

surname_lookup: dict = defaultdict(lambda: defaultdict(int))
for _, row in surnames_df.iterrows():
    code = row['country']
    name = str(row['surname']).lower()
    count = int(row.get('occurrences', row.get('count', 0)))
    surname_lookup[code][name] += count

print("Name datasets loaded.")

code_to_country = {v: k for k, v in country_to_code.items()}


def _safe_country_code(claimed_country: str) -> str:
    """Prefer explicit mapping; only accept 2-letter fallback if it *is* a code."""
    claimed_country = (claimed_country or "").strip()
    if claimed_country in country_to_code:
        return country_to_code[claimed_country]
    cc = claimed_country.upper()
    return cc if len(cc) == 2 else cc[:2] 


# This is still an approximation: it assumes first/last are independent within a country.
def check_plausibility(first: str, last: str, claimed_country: str) -> Dict:
    """
    Approximate joint probability under independence:
      score_country ∝ P(first|country) * P(last|country)
    where P(name|country) is approximated from counts:
      P(first|c) ≈ f_cnt / total_forenames_in_c
      P(last|c)  ≈ l_cnt / total_surnames_in_c

    Returns:
      - score: 5..95
      - joint_prob_claimed: normalized joint probability for claimed country
      - other_countries: top 4 countries by joint probability
    """
    code = _safe_country_code(claimed_country)

    first_lower = (first or "").lower().strip()
    last_lower = (last or "").lower().strip()

    # Countries present in both datasets
    countries = sorted(set(forename_lookup) & set(surname_lookup))

    if not countries:
        return {
            "score": 5,
            "rarity": "Unknown",
            "country": claimed_country,
            "message": "No country data loaded — cannot assess plausibility.",
            "other_countries": [],
        }

    # Precompute denominators (total counts) per country
    # NOTE: this assumes forename/surname tables are within-country occurrence counts
    total_forenames_by_country = {
        c: sum(forename_lookup[c].values()) for c in countries
    }
    total_surnames_by_country = {
        c: sum(surname_lookup[c].values()) for c in countries
    }

    # Small smoothing to avoid zero-prob collapse when one side is missing
    # (0 means "no smoothing")
    alpha_f = 1.0
    alpha_l = 1.0

    # Compute joint probability per country
    # joint(c) = P(first|c) * P(last|c)
    joint_by_country: List[Tuple[str, float, int, int]] = []
    for c in countries:
        f_cnt = forename_lookup[c].get(first_lower, 0)
        l_cnt = surname_lookup[c].get(last_lower, 0)

        denom_f = total_forenames_by_country.get(c, 0)
        denom_l = total_surnames_by_country.get(c, 0)

        if denom_f <= 0 or denom_l <= 0:
            continue

        # Laplace-style smoothing: (cnt + alpha) / (denom + alpha*V)
        # We don't know V (vocab size), so we do a light approximation:
        # use denom + alpha (keeps smoothing gentle and bounded).
        p_first = (f_cnt + alpha_f) / (denom_f + alpha_f)
        p_last = (l_cnt + alpha_l) / (denom_l + alpha_l)

        joint = p_first * p_last

        # Keep countries even if counts are zero on one side: smoothing makes them >0,
        # but they will be tiny and rank low.
        joint_by_country.append((c, joint, f_cnt, l_cnt))

    if not joint_by_country:
        return {
            "score": 5,
            "rarity": "Non-existent",
            "country": claimed_country,
            "message": "This name appears nowhere in the European dataset.",
            "other_countries": [],
        }

    # Rank by joint probability (descending)
    joint_by_country.sort(key=lambda x: x[1], reverse=True)

    # Normalize joint probabilities across countries so they sum to 1
    joint_sum = sum(j for _, j, _, _ in joint_by_country) or 1.0
    joint_norm = {c: (j / joint_sum) for c, j, _, _ in joint_by_country}

    # Claimed country stats
    f_count_claimed = forename_lookup.get(code, {}).get(first_lower, 0)
    l_count_claimed = surname_lookup.get(code, {}).get(last_lower, 0)
    joint_prob_claimed = joint_norm.get(code, 0.0)

    # Claimed rank (1-indexed), if present in ranking list
    claimed_rank = None
    for i, (c, _, _, _) in enumerate(joint_by_country):
        if c == code:
            claimed_rank = i + 1
            break

    # Top country joint mass (for suspicion comparisons)
    top_country, _, _, _ = joint_by_country[0]
    top_joint_norm = joint_norm.get(top_country, 0.0)

    # Scoring based on normalized joint probability
    # Use log scale because joint probs are tiny.
    # score increases with joint_prob_claimed, and also with closeness to the top country.
    eps = 1e-18
    logp = math.log10(joint_prob_claimed + eps)  # ~ negative
    # Map logp from [-18, -2] roughly into [5, 80]
    base = 5 + 75 * ((logp + 18) / 16)
    base = max(5, min(80, base))

    # Bonus for being close to top country's joint probability
    rel_to_top = (joint_prob_claimed / top_joint_norm) if top_joint_norm > 0 else 0.0
    rel_bonus = 15 * (rel_to_top ** 0.8)

    score = int(base + rel_bonus)
    score = max(5, min(95, score))

    # Rarity/message based on joint share
    # joint_prob_claimed is a share across countries, so thresholds are small.
    if joint_prob_claimed <= 0:
        rarity = "Non-existent here"
        message = f"No evidence for this name pairing in {claimed_country} (by joint estimate)."
        score = min(score, 10)
    elif joint_prob_claimed < 0.005:
        rarity = "Very Rare"
        message = f"Very low joint plausibility in {claimed_country}."
    elif joint_prob_claimed < 0.03:
        rarity = "Rare"
        message = f"Low joint plausibility in {claimed_country}."
    elif joint_prob_claimed < 0.10:
        rarity = "Uncommon"
        message = f"Moderate joint plausibility in {claimed_country}."
    else:
        rarity = "Common"
        message = f"High joint plausibility in {claimed_country}."

    # Add concrete counts too (helps debugging/UX)
    message += f" (first={f_count_claimed}, last={l_count_claimed})"

    # Suspicion: if another country dominates the joint probability
    if top_country != code and top_joint_norm > joint_prob_claimed * 4 and top_joint_norm > 0.05:
        message += " This pairing is much more plausible in other European countries."
        score = max(5, score - 25)

    if claimed_rank and claimed_rank > 3:
        message += f" It ranks only #{claimed_rank} by joint plausibility."

    # Other countries (top 4 excluding claimed)
    other_countries: List[Dict] = []
    for c, j, f_cnt, l_cnt in joint_by_country:
        if c == code:
            continue
        if len(other_countries) >= 4:
            break
        other_countries.append({
            "country": code_to_country.get(c, c),
            "joint_share": round(100 * joint_norm.get(c, 0.0), 2),  # percent of joint mass
            "first_count": f_cnt,
            "last_count": l_cnt,
        })

    return {
        "score": score,
        "rarity": rarity,
        "country": claimed_country,
        "message": message.strip(),
        "other_countries": other_countries,
        "claimed_rank": claimed_rank or "unknown",
        "top_country": code_to_country.get(top_country, top_country),
        "joint_prob_claimed": joint_prob_claimed,  # 0..1
    }