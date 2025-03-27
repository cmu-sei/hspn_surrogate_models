import logging
import os


def setup_logging(log_dir: str = "logs", log_level: str = "INFO") -> None:
    """Configure logging.

    Args:
        log_dir: Directory to store log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    os.makedirs(log_dir, exist_ok=True)

    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(f"{log_dir}/deeponet.log"), logging.StreamHandler()],
    )
    logging.info(f"Logging configured with level {log_level}")

