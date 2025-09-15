import logging
import sys

def configure_logging(level: str = "INFO"):
    logger = logging.getLogger()
    logger.handlers.clear()
    h = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    h.setFormatter(fmt)
    logger.addHandler(h)
    logger.setLevel(level.upper())
