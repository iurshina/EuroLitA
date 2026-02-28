from collections import defaultdict
from pathlib import Path
import pandas as pd


DATA_DIR = Path(__file__).parent.parent / "data"

print("Loading name datasets...") 

# Load CSVs
forenames_df = pd.read_csv(
    DATA_DIR / "forenames_eu.csv",
    low_memory=False,
    dtype={"gender": "string"}
)

surnames_df = pd.read_csv(
    DATA_DIR / "surnames_eu.csv",
    low_memory=False,
    dtype={"gender": "string"}
)

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

def check_plausibility(first: str, last: str, claimed_country: str) -> dict:
    code = country_to_code.get(claimed_country, claimed_country.upper()[:2])
    first_lower = first.lower().strip()
    last_lower = last.lower().strip()

    f_count_claimed = forename_lookup[code].get(first_lower, 0)
    l_count_claimed = surname_lookup[code].get(last_lower, 0)
    total_claimed = f_count_claimed + l_count_claimed

    # Cross-country stats
    f_counts_all = [forename_lookup[c].get(first_lower, 0) for c in forename_lookup]
    l_counts_all = [surname_lookup[c].get(last_lower, 0) for c in surname_lookup]
    total_all_countries = sum(f_counts_all) + sum(l_counts_all)

    if total_all_countries == 0:
        return {
            "score": 5,
            "rarity": "Non-existent",
            "country": claimed_country,
            "message": "This name appears nowhere in the European dataset — very implausible."
        }

    max_total_any_country = max(
        forename_lookup[c].get(first_lower, 0) + surname_lookup[c].get(last_lower, 0)
        for c in forename_lookup
    )

    # ────────────────────────────────────────────────
    # Improved scoring
    # ────────────────────────────────────────────────

    # Base score: reward high absolute count (log scale to avoid extremes)
    import math
    max_possible = 1_000_000  # rough upper bound for most common names in Europe
    base = 30 + 65 * (math.log1p(total_claimed) / math.log1p(max_possible))

    # Share bonus: how much of total occurrences are in this country
    share_in_country = total_claimed / total_all_countries if total_all_countries > 0 else 0
    share_bonus = 20 * share_in_country

    # Relative to the most common country
    relative_to_max = total_claimed / max_total_any_country if max_total_any_country > 0 else 0
    relative_bonus = 15 * (relative_to_max ** 1.5)  # ^1.5 = gentler curve

    score = int(base + share_bonus + relative_bonus)
    score = max(10, min(98, score))

    # ────────────────────────────────────────────────
    # Rarity & message
    # ────────────────────────────────────────────────
    if total_claimed == 0:
        rarity = "Extremely Rare"
        message = f"Zero occurrences in {claimed_country}."
    elif total_claimed < 20:
        rarity = "Very Rare"
        message = f"Only {total_claimed} occurrences in {claimed_country}."
    elif total_claimed < 200:
        rarity = "Rare"
        message = f"Uncommon in {claimed_country} ({total_claimed} occurrences)."
    elif total_claimed < 2000:
        rarity = "Uncommon"
        message = f"Somewhat typical in {claimed_country} ({total_claimed} occurrences)."
    elif total_claimed < 15000:
        rarity = "Common"
        message = f"Common in {claimed_country} ({total_claimed} occurrences)."
    else:
        rarity = "Very Common"
        message = f"Very frequent in {claimed_country} ({total_claimed} occurrences)."

    # Mild suspicion only when clearly dominated elsewhere
    if max_total_any_country > total_claimed * 8 and max_total_any_country > 1000:
        message += " However, significantly more common in other European countries — a bit suspicious?"
        score = max(20, score - 20)  # lighter penalty

    return {
        "score": score,
        "rarity": rarity,
        "country": claimed_country,
        "message": message
    }