import torch
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader
from accelerate import Accelerator

from .diffusion import ContinuousForwardDiffusion

def main():

    def dummy_fn(x, y):
        return x, y

    cfd = ContinuousForwardDiffusion(denoise_model=dummy_fn)
    ds = torch.load(
        '/home/hamzakage/Data/Datasets/kol/nu_0p0045_100seeds_8x_val.pt'
    )
    x, y = ds[0, :10].clone(), ds[1, :10].clone()
    ds = TensorDataset(x, y)
    dl = DataLoader(ds, batch_size=10)

    accel = Accelerator()
    cfd, dl = accel.prepare(cfd, dl)

    for x, y in dl:
        y_noisy, _, _ = cfd(x, y)
        
    return y_noisy


if __name__=='__main__':

    y_noisy = main() 
    torch.save(y_noisy, 'test_noising.pt')

