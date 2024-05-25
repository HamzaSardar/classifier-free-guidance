import ml_collections
from ml_collections.config_dict import placeholder


def get_wandb_config() -> ml_collections.ConfigDict:

    config = ml_collections.ConfigDict()

    config.entity = 'hamzasardar'
    config.project = 'cfg-diffusion'
    config.group = placeholder(str)
    config.mode = placeholder(str)

    return config


def log_to_wandb(wandb_config: ml_collections.ConfigDict) -> bool:

    if wandb_config.entity and wandb_config.project:
        return True

    return False


WANDB_CONFIG = get_wandb_config()

