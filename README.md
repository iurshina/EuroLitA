# EuroLitA

**Are you really from where you say you are? 😏**

EuroLitA is a small satirical web app that estimates how plausible a claimed European origin is based on first and last name frequency distributions.

It compares submitted names against publicly available national name dataset and calculates whether the combination is statistically typical for the selected country.

This project is inspired by public reporting about **TraLitA**, a German Federal Office for Migration and Refugees (BAMF) tool designed to standardise transliteration and analyse name distributions in asylum procedures.

⚠️ This is a transparency experiment — not an identity verification system.

---

# What This Project Does

* Accepts first name, last name, and claimed country
* Looks up name frequencies in the country-level dataset
* Computes plausibility score
* Returns a lightweight statistical estimate

It does **not**:

* Identify individuals
* Store user data
* Verify identity
* Infer ethnicity
* Access government systems

---

# Inspiration: What is TraLitA?

TraLitA (Transliteration Tool) is referenced in German parliamentary documents and migration-tech discussions as a system used by the **Bundesamt für Migration und Flüchtlinge (BAMF)**.

It is described as:

* Standardising Arabic-to-Latin transliteration
* Comparing name distributions across origin regions
* Supporting asylum decision-making

Public references include:

* BAMF / EMN Working Paper 90 (Data management in asylum context):
  [https://www.bamf.de/SharedDocs/Anlagen/EN/EMN/Studien/wp90-datenmanagement.pdf](https://www.bamf.de/SharedDocs/Anlagen/EN/EMN/Studien/wp90-datenmanagement.pdf)

* German Bundestag inquiry discussing migration IT systems:
  [https://dserver.bundestag.de](https://dserver.bundestag.de)

* Council of Europe reference to migration data tools:
  [https://assembly.coe.int](https://assembly.coe.int)

EuroLitA is **not affiliated with BAMF** and does not replicate TraLitA.
It is an open demonstration using publicly available data.

---

# Data Source

This project currently uses one dataset only:

📦 Forenames and Surnames with Gender and Country
Kaggle dataset by erpel1
https://www.kaggle.com/datasets/erpel1/forenames-and-surnames-with-gender-and-country

The dataset includes:

* First names
* Surnames
* Gender (for forenames)
* Associated country labels

⚠️ Important:
This dataset is community-compiled and not an official national statistics source. It may contain:

* Incomplete coverage
* Bias toward certain countries
* Uneven sampling
* No population weighting

EuroLitA does not supplement this dataset with any additional official registry data (yet).

# How The Model Works (Simplified)

1. Load country-specific first name dataset
2. Load country-specific surname dataset
3. Normalize case
4. Match input name against dataset
5. Compute frequency presence score
6. Return plausibility classification

This is purely frequency-based — not predictive AI.

---

# How To Run The Project

## 1️⃣ Clone the repository
```bash
git clone https://github.com/iurshina/eurolita.git
cd eurolita
```

## 2️⃣ Install dependencies (choose one)

Option A – Fast & fun (uv)

```bash
# Install uv once if needed: curl -LsSf https://astral.sh/uv/install.sh | sh 

uv sync
```

Option B – Classic (venv + pip)

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install fastapi uvicorn[standard] jinja2 pandas python-multipart
```

## 3️⃣ Add data

Place datasets inside:

```
/data/
    contry_codes.csv
    surnames.csv
    forenames.csv
```

## 4️⃣ Run server

With uv

```bash
uv run fastapi dev
# or
uv run uvicorn main:app --reload
```

With venv + pip 

```bash
uvicorn main:app --reload
```

Open in browser:

```
http://127.0.0.1:8000
```

---

# Ethical Note

Name-based plausibility analysis can intersect with:

* Migration politics
* Border technologies
* Bias in statistical systems
* Automated decision-making

This project exists for:

* Transparency
* Technical exploration
* Investigative understanding

It is not designed for:

* Identity screening
* Immigration control
* Profiling

# Limitations (obvious for us but not for everyone it seems...)

* Incomplete datasets
* Name overlap across countries
* Migration effects distort distribution
* Transliteration inconsistencies
* Double surnames and diacritics
* Data freshness issues

Statistical presence ≠ origin truth.

---

# Metrics

## 1️⃣ Plausibility Ratio — “Does this name fit this country?”

We approximate:

```
P(first, last | country) ≈ P(first | country) * P(last | country)
```

where:

```
P(first | country) = (count(first, country) + α) / (total_forenames(country) + α)

P(last | country)  = (count(last, country) + α) / (total_surnames(country) + α)
```

* `count(first, country)` = frequency of the first name in that country
* `total_forenames(country)` = total forename counts in that country
* same logic for surnames
* `α = 0.5` (Laplace smoothing to avoid zeros)

We then compare this to the Europe-wide baseline:

```
plausibility_ratio =
    P(first, last | country)
    --------------------------------
    P(first, last)  (EU pooled)
```

### Interpretation

| Ratio | Meaning                      |
| ----- | ---------------------------- |
| > 1.0 | More typical than EU average |
| ≈ 1.0 | Neutral                      |
| < 1.0 | Less typical than EU average |

Examples:

* `2.0` → name is 2× more typical in that country than EU average
* `0.5` → half as typical as EU average

This is the **main plausibility metric**, because it does not penalize large countries for having large total name counts.

---

## 2️⃣ Posterior Share — “Which country fits best?”

We compute the same likelihood for every country and normalize:

```
posterior_share(country) =
    P(first, last | country)
    -----------------------------------------
    sum over all countries P(first, last | c)
```

This produces a distribution across countries that sums to 1.

### Important

* This is a **relative ranking signal**
* It is **not** a real-world probability
* It does **not** mean "% of population"
* It simply shows which country fits best among the dataset

A country can:

* Have a high plausibility ratio (good fit)
* But still rank #3 among all countries

This is not contradictory — the metrics answer different questions.

---

## Why Two Metrics?

| Metric             | Question it answers                     |
| ------------------ | --------------------------------------- |
| Plausibility Ratio | Does this name look typical in Germany? |
| Posterior Share    | Which country fits best overall?        |

You need both to avoid misleading interpretations.

---

## Modeling Assumptions

* First and last names are treated as independent given country.
* Uses Laplace smoothing (α = 0.5).
* All countries are treated equally (no population priors).
* Counts reflect the dataset, not necessarily full population distributions.

---

## Performance & Memory Design

* Uses Polars lazy scanning with streaming aggregation.
* Aggregates `(country, name, count)` tables once at startup.
* Avoids large Python dictionaries.
* Query-time computation is lightweight.

Designed to operate within moderate memory constraints (~500MB).

---

# License

MIT