import logging.config
import os

import yaml

logger = logging.getLogger(__name__)
SCRIPT_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


def configure_logger():
    with open(os.path.join(SCRIPT_DIR, 'conf', 'logger.yaml')) as f:
        config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
