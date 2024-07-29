import itertools
from pathlib import Path
import torch
from torch.utils.data import DataLoader, TensorDataset
from accelerate import Accelerator
from absl import app, flags
import wandb # type: ignore
from wandb.sdk.lib.disabled import RunDisabled # type: ignore
from wandb.sdk.wandb_run import Run # type: ignore
from src.cfg_diffusion.diffusion import ContinousDiffusion
from src.cfg_diffusion.models.unet_periodic import UNet
import src.cfg_diffusion.utils.flags as cfg_flags
from ml_collections import config_flags

from .configs.wandb import WANDB_CONFIG # type: ignore


FLAGS = flags.FLAGS

_CONFIG = config_flags.DEFINE_config_file('config')
_WANDB_CONFIG = config_flags.DEFINE_config_dict('wandb', WANDB_CONFIG)

_RESULTS_PATH = cfg_flags.DEFINE_path(
    'results_path',
    '/home/hamzakage/Data/cfg_diffusion/models/',
    'Directory to store saved models.'
)

_TRAIN_PATH = cfg_flags.DEFINE_path(
    'train_path',
    '/home/hamzakage/Data/Datasets/rb_dedalus/rb_train_xsmall_1e9.pt',
    'Path to training data .pt file.'
)

_VAL_PATH = cfg_flags.DEFINE_path(
    'val_path',
    '/home/hamzakage/Data/Datasets/rb_dedalus/rb_val_xsmall_1e9.pt',
    'Path to validation data .pt file.'
)

_LOG_FREQUENCY = flags.DEFINE_integer(
    'logging_frequency',
    200,
    'Frequency at which to log results.'
)

_SAVE_FREQUENCY = flags.DEFINE_integer(
    'saving_frequency',
    5000,
    'Frequency at which to save model.'
)

_STANDARDISE = flags.DEFINE_bool(
    'standardise_data',
    False,
    'Whether to standardise individual samples.'
)

def load_data(pt_path: Path) -> TensorDataset:

    ds = torch.load(pt_path)
    if len(ds.shape) == 4:
        ds = ds.unsqueeze(2)

    ds = TensorDataset(ds[0].clone(), ds[1].clone())

    return ds


def get_dataloader(ds: TensorDataset, batch_size: int) -> DataLoader:
    dl = DataLoader(ds,batch_size=batch_size, pin_memory=True)
    return dl


def initialise_wandb() -> Run | RunDisabled | None:

    wandb_run = None
    wandb_config = FLAGS.wandb.to_dict()

    if wandb_config['group']:
        wandb_config.update({'group': str(wandb_config['group'])})

    if wandb_config['mode']:
        wandb_run = wandb.init(config=FLAGS.config.to_dict(), **wandb_config)

    return wandb_run


def main(_):

    torch.backends.cudnn.benchmark = True    

    accelerator = Accelerator(gradient_accumulation_steps=8)
    #accelerator = Accelerator()

    """
    @accelerator.on_local_main_process
    def _save_distributed(accelerator: Accelerator, 
                          model: ContinousDiffusion, 
                          wandb_instance: Run, 
                          iters: int,
                          results_path=FLAGS.results_path) -> None:
        
        accelerator.save(accelerator.unwrap_model(model).denoise_model.state_dict(), Path(results_path) / f'model_{wandb_instance.name}_{iters}.h5')
    """

    FLAGS.results_path.mkdir(parents=True, exist_ok=True)

    config = FLAGS.config

    run = initialise_wandb()
    
    print(config)
    train_ds = load_data(FLAGS.train_path)
    val_ds = load_data(FLAGS.val_path)
    
    train_loader = get_dataloader(train_ds, config.training.batch_size)
    val_loader = get_dataloader(val_ds, config.training.batch_size)

    for x, y in train_loader:
        sample = x
        break

    unet = UNet(
        in_channel=config.network.in_channels,
        out_channel=config.network.in_channels // 2,
        inner_channel=config.network.inner_channels,
        norm_groups=config.network.norm_groups,
        dropout=config.network.dropout_rate,
        image_size=tuple(sample.shape[-2:]) # type: ignore
    )
    
    unet.load_state_dict(torch.load('/mnt/mace01-cfd-home01/mmapzhs5/dedalus_rb/cfg_results/model_silvery-wave-47_115000.h5'))
    
    diffusion = ContinousDiffusion(
        denoise_model=unet,
        lambda_min=config.diffusion.lambda_min,
        lambda_max=config.diffusion.lambda_max,
        p_uncond=config.diffusion.p_uncond
    )

    optim = torch.optim.Adam(diffusion.denoise_model.parameters(), # type: ignore
                             lr=config.training.learning_rate,
                             weight_decay=0.0001)

    n_epochs = config.training.n_iters // (len(train_loader))

    train_loader, val_loader, diffusion, optim = accelerator.prepare(
        train_loader, val_loader, diffusion, optim
    )

    iters = 0
    standardise = lambda x: ((x - torch.min(torch.abs(x))) / (torch.max(torch.abs(x)) - torch.min(torch.abs(x))))

    standardise_fns = {True: standardise, False: lambda x:x}
    standardise_fn = standardise_fns[FLAGS.standardise_data]

    for epoch in range(n_epochs):
        for idx, zipped_data in enumerate(zip(train_loader, itertools.cycle(val_loader))):
            data, val_data = zipped_data
            diffusion.train()    

            with accelerator.accumulate(diffusion):
                optim.zero_grad(set_to_none=True)
                x, y = data
                loss = diffusion(standardise_fn(x), standardise_fn(y))

                torch.nn.utils.clip_grad_norm_(diffusion.parameters(), max_norm=1.0)
                accelerator.backward(loss)
                optim.step()

            diffusion.eval()
            with torch.no_grad():
                x, y = val_data
                val_loss = diffusion(standardise_fn(x), standardise_fn(y))

            if iters % FLAGS.logging_frequency == 0:
                metrics_dict = {'epoch': epoch, 'iter': iters, 'train loss': loss, 'val loss': val_loss}
                print(metrics_dict)
                if isinstance(run, Run):
                    run.log(metrics_dict) # type: ignore

            iters += 1
            
            if iters % FLAGS.saving_frequency == 0:
                if isinstance(run, Run):
                    if accelerator.num_processes > 1:
                        _save_distributed(accelerator, diffusion, run, iters)
                    else:
                        accelerator.save(accelerator.unwrap_model(diffusion).denoise_model.state_dict(), Path(FLAGS.results_path) / f'model_{run.name}_{iters}.h5')
                else:
                    accelerator.save(accelerator.unwrap_model(diffusion).denoise_model.state_dict(), Path(FLAGS.results_path) / f'model_{iters}.h5')


    # finish
    if isinstance(run, Run):
        run.finish() # type: ignore

    
if __name__=='__main__':
    app.run(main)

