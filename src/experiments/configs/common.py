from pathlib import Path
import ml_collections


def get_config() -> ml_collections.ConfigDict:

    config = ml_collections.ConfigDict()

    # network parameters
    config.network = ml_collections.ConfigDict()
    config.network.in_channels = 1
    config.network.inner_channels = 32
    config.network.norm_groups = 16
    config.network.dropout_rate = 0.0

    # diffusion parameters 
    config.diffusion = ml_collections.ConfigDict()
    config.diffusion.lambda_min = -20
    config.diffusion.lambda_max = 20
    config.diffusion.p_uncond = 0.1

    # training parameters 
    config.training = ml_collections.ConfigDict()
    config.training.batch_size = 8
    config.training.n_iters = 250000
    config.training.learning_rate = 1e-4
    
    return config

