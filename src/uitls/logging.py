import logging 


def setup_logging(level=logging.INFO, logfile: str | None = None) -> None:
    """Setup logging configuration."""
    handlers: list[logging.Handler] = [logging.StreamHandler()]

    if logfile:
        handlers.append(logging.FileHandler(logfile))

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )

    for name in ("transformers", "datasets", "torch", "accelerate"):
        logging.getLogger(name).setLevel(logging.WARNING)


logger = logging.getLogger(__name__)