import ml_collections

from src.experiments.configs import common


def get_config() -> ml_collections.ConfigDict:

    config = common.get_config()
    
    return config
