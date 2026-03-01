import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def configure_logging(log_file: Path, level: int = logging.INFO) -> None:
    root = logging.getLogger()
    root.setLevel(level)

    if getattr(root, "_eurolita_configured", False):
        return

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(formatter)

    log_file.parent.mkdir(exist_ok=True, parents=True)
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=5_000_000,  # 5MB
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    root.addHandler(console)
    root.addHandler(file_handler)

    root._eurolita_configured = True