"""
Logger setup
"""

from logging import getLogger, INFO, StreamHandler, Formatter, Logger

logger = getLogger(__name__)


def setup_logger():
  logger.setLevel(INFO)
  handler = StreamHandler()
  handler.setFormatter(Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
  logger.addHandler(handler)
  return logger