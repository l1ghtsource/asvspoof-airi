import yaml
from typing import Dict
import os


def load_config(config_path: str, model_config_path: str) -> Dict:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    with open(model_config_path, 'r') as f:
        model_config = yaml.safe_load(f)

    config.update(model_config)

    return config
