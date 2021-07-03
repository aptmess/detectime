import logging
import yaml
from detectime.detectime import detectron
from definitions import ROOT_DIR
from detectime.utils import convert_dict_to_tuple

log = logging.getLogger(__name__)

CONFIG_PATH = 'config.yml'


def main():
    with open(ROOT_DIR / CONFIG_PATH) as f:
        data = yaml.safe_load(f)
    config = convert_dict_to_tuple(dictionary=data)
    detectron(config)


if __name__ == '__main__':
    main()
