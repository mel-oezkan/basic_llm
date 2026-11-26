import logging
from pathlib import Path


def setup_logging(level=logging.INFO, logfile: str | None = None) -> None:
    """Configure the root logger: console output, plus a log file if given.

    Call once at the start of a script. Modules should log through
    `logging.getLogger(__name__)` and never configure logging themselves.
    """
    handlers: list[logging.Handler] = [logging.StreamHandler()]

    if logfile:
        Path(logfile).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(logfile))

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,  
    )

    # silence noisy third-party loggers (httpx logs every HF Hub request at INFO)
    for name in ("transformers", "datasets", "torch", "accelerate", "huggingface_hub"):
        logging.getLogger(name).setLevel(logging.WARNING)
