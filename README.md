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

First names
Surnames
Gender (for forenames)
Associated country labels

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

## 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

## 4️⃣ Add data

Place datasets inside:

```
/data/
    contry_codes.csv
    surnames.csv
    forenames.csv
```

## 5️⃣ Run server

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

---

# Limitations (obvious for us but not for everyone it seems...)

* Incomplete datasets
* Name overlap across countries
* Migration effects distort distribution
* Transliteration inconsistencies
* Double surnames and diacritics
* Data freshness issues

Statistical presence ≠ origin truth.

---

# License

MIT