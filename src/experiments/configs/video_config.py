import ml_collections

from src.experiments.configs import video


def get_config() -> ml_collections.ConfigDict:

    config = video.get_config()
    
    return config
