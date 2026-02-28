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

def check_plausibility(first: str, last: str, claimed_country: str) -> Dict:
    code = country_to_code.get(claimed_country, claimed_country.upper()[:2])
    first_lower = (first or "").lower().strip()
    last_lower = (last or "").lower().strip()

    f_count_claimed = forename_lookup.get(code, {}).get(first_lower, 0)
    l_count_claimed = surname_lookup.get(code, {}).get(last_lower, 0)
    total_claimed = f_count_claimed + l_count_claimed

    country_scores: List[Tuple[str, int]] = []

    for c_code in (set(forename_lookup) & set(surname_lookup)):
        f_cnt = forename_lookup[c_code].get(first_lower, 0)
        l_cnt = surname_lookup[c_code].get(last_lower, 0)
        total = f_cnt + l_cnt
        if total > 0:
            country_scores.append((c_code, total))

    if not country_scores:
        return {
            "score": 5,
            "rarity": "Non-existent",
            "country": claimed_country,
            "message": "This name (first+last) appears nowhere in the European dataset.",
            "other_countries": []
        }

    country_scores.sort(key=lambda x: x[1], reverse=True)
    max_total_any = country_scores[0][1]
    total_all = sum(cnt for _, cnt in country_scores)

    base = 15 + 55 * (math.log1p(total_claimed) / math.log1p(1_000_000))
    share = total_claimed / total_all if total_all else 0
    share_bonus = 25 * (share ** 0.7)
    rel_to_max = total_claimed / max_total_any if max_total_any else 0
    rel_bonus = 20 * (rel_to_max ** 2)

    score = int(base + share_bonus + rel_bonus)
    score = max(5, min(95, score))

    if total_claimed == 0:
        rarity = "Non-existent here"
        message = f"Zero occurrences in {claimed_country}."
        score = min(score, 10)  # optional: make zero-in-claimed-country clearly low
    elif total_claimed < 50:
        rarity = "Very Rare"
        message = f"Only {total_claimed} occurrences in {claimed_country}."
    elif total_claimed < 500:
        rarity = "Rare"
        message = f"Rare in {claimed_country} ({total_claimed} occurrences)."
    elif total_claimed < 5000:
        rarity = "Uncommon"
        message = f"Uncommon in {claimed_country} ({total_claimed} occurrences)."
    elif total_claimed < 20000:
        rarity = "Moderately common"
        message = f"Moderately common in {claimed_country} ({total_claimed} occurrences)."
    else:
        rarity = "Common"
        message = f"Common in {claimed_country} ({total_claimed} occurrences)."

    other_countries: List[Dict] = []
    claimed_rank = None
    for i, (c_code, cnt) in enumerate(country_scores):
        if c_code == code:
            claimed_rank = i + 1
            continue
        if len(other_countries) >= 4:
            break
        other_countries.append({
            "country": code_to_country.get(c_code, c_code),
            "occurrences": cnt,
            "percent_of_total": round(100 * cnt / total_all, 1) if total_all else 0
        })

    if max_total_any > total_claimed * 3.5 and max_total_any > 300:
        message += " This name is significantly more frequent in other European countries."
        score = max(5, score - 30)

    if claimed_rank and claimed_rank > 3:
        message += f" It ranks only #{claimed_rank} most common among European countries."

    return {
        "score": score,
        "rarity": rarity,
        "country": claimed_country,
        "message": message.strip(),
        "other_countries": other_countries,
        "claimed_rank": claimed_rank or "unknown",
        "max_in_any_country": max_total_any
    }