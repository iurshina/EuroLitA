import logging
import time
from collections import Counter

from fastapi import FastAPI, Request, Form, status
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from utils.name_checker import check_plausibility


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("eurolita")


app = FastAPI()
templates = Jinja2Templates(directory="templates")


STATS = Counter()


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start = time.perf_counter()

    try:
        response = await call_next(request)
        return response
    finally:
        duration_ms = (time.perf_counter() - start) * 1000

        # Only count actual check submissions
        if request.url.path == "/check" and request.method == "POST":
            STATS["check_total"] += 1

        logger.info(
            "path=%s method=%s status=%s duration_ms=%.2f total_checks=%s",
            request.url.path,
            request.method,
            getattr(locals().get("response", None), "status_code", "unknown"),
            duration_ms,
            STATS.get("check_total", 0),
        )


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "title": "EuroLitA – Satirical Name Checker"},
    )


@app.post("/check", response_class=HTMLResponse)
async def check(
    request: Request,
    first_name: str = Form(...),
    last_name: str = Form(...),
    country: str = Form(...),
):
    try:
        result = check_plausibility(first_name, last_name, country)
    except Exception:
        STATS["check_error"] += 1
        logger.exception("check_plausibility failed (country=%s)", country)

        return templates.TemplateResponse(
            "error.html",
            {"request": request, "message": "Temporary error. Please try again."},
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    return templates.TemplateResponse(
        "result.html",
        {"request": request, **result},
    )