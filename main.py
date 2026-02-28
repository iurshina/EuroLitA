from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from utils.name_checker import check_plausibility

app = FastAPI()
templates = Jinja2Templates(directory="templates")


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