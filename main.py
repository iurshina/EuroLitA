from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
from pathlib import Path

app = FastAPI()
templates = Jinja2Templates(directory="templates")

DATA_DIR = Path("data")
forenames = pd.read_csv(DATA_DIR / "forenames.csv")
surnames = pd.read_csv(DATA_DIR / "surnames.csv")

# For demo: simple in-memory lookup function (replace with your real logic)
def check_plausibility(first: str, last: str, country: str) -> dict:
    # Dummy logic – adapt to your frequency/rarity calc
    name = f"{first} {last}".strip().lower()
    # Example: pretend we search dataframes
    count = len(forenames[(forenames['forename'].str.lower() == first.lower()) & 
                          (forenames['country'] == country)])
    rarity = "Very Rare" if count < 5 else "Common" if count > 50 else "Average"
    score = min(95, max(10, 100 - count * 2))  # satirical inversion
    return {
        "score": score,
        "rarity": rarity,
        "country": country,
        "message": "Suspiciously Polish-sounding for a German claim…" if country == "Germany" else "Seems plausible… or is it?"
    }


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "title": "EuroLitA – Satirical Name Checker"}
    )


@app.post("/check", response_class=HTMLResponse)
async def check(
    request: Request,
    first_name: str = Form(...),
    last_name: str = Form(...),
    country: str = Form(...)
):
    result = check_plausibility(first_name, last_name, country)
    return templates.TemplateResponse(
        "result.html",
        {"request": request, **result}
    )