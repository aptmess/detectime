import yaml
import logging
import pathlib
from logging import config

with open(pathlib.Path(__file__).parents[1] / 'log_config.yml') as f:
    log_config = yaml.load(
        f,
        Loader=yaml.FullLoader
    )
    logging.config.dictConfig(log_config)
