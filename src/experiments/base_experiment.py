import itertools
from pathlib import Path
import torch
from torch.utils.data import DataLoader, TensorDataset
from accelerate import Accelerator
from absl import app, flags
from src.cfg_diffusion.diffusion import ContinousDiffusion
from src.cfg_diffusion.models.unet_periodic import UNet
import src.cfg_diffusion.utils.flags as cfg_flags
from ml_collections import config_flags


FLAGS = flags.FLAGS

_CONFIG = config_flags.DEFINE_config_file('config')

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


def load_data(pt_path: Path) -> TensorDataset:

    ds = torch.load(pt_path)
    ds = TensorDataset(ds[0].clone(), ds[1].clone())

    return ds

def get_dataloader(ds: TensorDataset, batch_size: int) -> DataLoader:
    dl = DataLoader(ds,batch_size=batch_size, pin_memory=True)
    return dl

def main(_):

    accelerator = Accelerator()
    FLAGS.results_path.mkdir(parents=True, exist_ok=True)

    config = FLAGS.config
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
        image_size=tuple(sample.shape[-2:])
    )

    diffusion = ContinousDiffusion(
        denoise_model=unet,
        lambda_min=config.diffusion.lambda_min,
        lambda_max=config.diffusion.lambda_max
    )

    optim = torch.optim.Adam(diffusion.denoise_model.parameters(),
                             lr=config.training.learning_rate,
                             weight_decay=0.0001)

    n_epochs = config.training.n_iters // (len(train_loader))

    train_loader, val_loader, diffusion, optim = accelerator.prepare(
        train_loader, val_loader, diffusion, optim
    )

    iters = 0

    for epoch in range(n_epochs):
        train_mse = 0.0
        val_mse = 0.0

        for idx, zipped_data in enumerate(zip(train_loader, itertools.cycle(val_loader))):
            data, val_data = zipped_data

            with accelerator.accumulate(diffusion):
                diffusion.train()    
                optim.zero_grad()

                x, y = data
                loss = diffusion(x, y)

                torch.nn.utils.clip_grad_norm_(diffusion.parameters(), max_norm=0.01)
                accelerator.backward(loss)
                optim.step()

            diffusion.eval()
            with torch.no_grad():
                x, y = val_data
                val_loss = diffusion(x, y)

            if iters % FLAGS.logging_frequency == 0:
                metrics_dict = {'epoch': epoch, 'iter': iters, 'train loss': loss, 'val loss': val_loss}
                print(metrics_dict)

            iters += 1
            
            if iters % FLAGS.saving_frequency == 0:
                accelerator.save(accelerator.unwrap_model(diffusion).denoise_model.state_dict(), Path(FLAGS.results_path) / f'model_64_128_{iters}.h5')

    
if __name__=='__main__':
    app.run(main)
