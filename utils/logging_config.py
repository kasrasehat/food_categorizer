import logging
from pathlib import Path


LOG_DIR = Path(__file__).resolve().parent / "data"
LOG_FILE = LOG_DIR / "log.txt"


def get_logger(name: str = "ai_services") -> logging.Logger:
    """
    Return a configured logger that writes to utils/data/log.txt.

    Configuration is applied only once per process; subsequent calls
    with any name will reuse the same handlers.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")

    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.propagate = False

    return logger


