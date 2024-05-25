import torch
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm
from typing import Callable
from functools import partial
from torch.special import expm1


class ContinuousForwardDiffusion(nn.Module):
    def __init__(self, 
                 denoise_model: Callable,
                 lambda_min: int=-10,  
                 lambda_max: int=15
                ) -> None:
        """
        Forward diffusion using the continuous formulation, i.e.
        predicting noise by conditioning on log(SNR).

        Parameters:
        -----------
        denoise_model: Callable
            Denoising UNet. 
        lambda_min: int
            Minimum log(SNR), given a default of -10.
        lambda_max: int
            Maximum log(SNR), given a default of 15.
        """
        super().__init__()
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.denoise_model = denoise_model

    @staticmethod
    def _alpha(log_snr: Tensor) -> Tensor:
        return torch.sqrt(1 / (1 + torch.exp(-log_snr)))

    @staticmethod
    def _sigma_2(alpha: Tensor) -> Tensor:
        return 1. - alpha**2

    def noising(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Given an input tensor, x, output the noised \tilde{x}.

        Parameters:
        -----------
        x: Tensor
            Input tensor.

        Returns:
        --------
        x_noisy, log_snr_sample, noise: tuple[Tensor, Tensor, Tensor]
            Tuple containing noised image, the log_snr used to noise the image, and the sample of unit Gaussian noise used.
        
        """

        # log(SNR) uses the even sampling of p(\lambda) over [\lambda_{min}, \lambda_{max}, given in appendix I.1 of Variational Diffusion Models.

        scaling = partial(lambda u, l_max, l_min: u*(l_max - l_min) + l_min, l_min=self.lambda_min, l_max=self.lambda_max)
        
        u = torch.rand((1))
        b = x.shape[0]
        log_snr_sample = scaling(torch.tensor([((u + i/b)%1) for i in range(1, b+1)], device=x.device))

        alpha = self._alpha(log_snr_sample).view(-1, *tuple(1 for i in range(len(x.shape) -1)))
        std = self._sigma_2(alpha).view(-1, *tuple(1 for i in range(len(x.shape) -1))).sqrt() 
        noise = torch.randn_like(x, device=x.device)

        x_noisy = alpha*x + std*noise

        return x_noisy, log_snr_sample, noise

    def forward(self, x: Tensor, y0: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Compute the forward pass of a diffusion model.

        Parameters:
        -----------
        x: Tensor
            Input LR image.
        y: Tensor
            Input HR image.

        Returns:
        --------
        y_noisy, eps_hat, eps: tuple[Tensor, Tensor, Tensor]
            Noised HR image, predicted noise, and actual noise.
        """
        y_noisy, log_snr_sample, eps = self.noising(y0)

        eps_hat = self.denoise_model(
            torch.cat([x, y_noisy], dim=1), 
            log_snr_sample.view(-1, 1, 1, 1)
        )
        
        return y_noisy, eps_hat, eps


class ContinuousReverseDiffusion(nn.Module):
    def __init__(self, 
                 denoise_model: Callable, 
                 num_steps: int=1000, 
                 lambda_min: int=-10, 
                 lambda_max: int=15
             ) -> None:

        super().__init__()
        self.denoise_model = denoise_model
        self.num_steps = num_steps
        
        # _next variables needed for mean and variance computation
        self.lambdas = torch.linspace(lambda_min, lambda_max, num_steps)
        self.lambdas_next = torch.cat((self.lambdas[1:], torch.tensor([lambda_max], device=self.lambdas.device)), dim=0)
        self.alphas = torch.sqrt(1 / (1 + torch.exp(-self.lambdas)))
        self.alphas_next = torch.cat((self.alphas[1:], torch.tensor([0], device=self.lambdas.device)), dim=0)
        self.sigmas = 1 - self.alphas ** 2
        self.sigmas_next = torch.cat((self.sigmas[1:], torch.tensor([1], device=self.lambdas.device)), dim=0)

    @staticmethod
    def _predict_y0(y_noisy: Tensor, sigma_lambda: Tensor, alpha_lambda: Tensor, eps_hat: Tensor) -> Tensor:
        pred_y0 = (y_noisy - sigma_lambda*eps_hat)/alpha_lambda
        return pred_y0
    
    def reverse_mean_variance(self, t: int, x: Tensor, y_lambda: Tensor) -> tuple[Tensor, Tensor]:

        eps_hat = self.denoise_model(torch.cat([x, y_lambda], dim=1), self.lambdas[t].view(-1, 1, 1, 1).to(x.device))
        
        y_hat0 = self._predict_y0(y_lambda, self.sigmas[t].sqrt(), self.alphas[t], eps_hat).clamp_(-1, 1)
        
        # Eq. 3 from classifier-free guidance paper
        d_lambda = self.lambdas[t] - self.lambdas_next[t]
        const = -expm1(d_lambda) # numerically stable operation of 1 - torch.exp(d_lambda)
        
        mean = self.alphas_next[t] * (y_lambda * (1 - const) / self.alphas[t] + const * y_hat0)
        variance = self.sigmas_next[t] * const

        return mean, variance

    def reverse_sample(self, t: int, x: Tensor, y_lambda: Tensor) -> Tensor:
        mean, variance = self.reverse_mean_variance(t, x, y_lambda)
        noise = torch.randn_like(y_lambda)
        if t != self.num_steps - 1:
            return mean + noise*variance.sqrt()
        else:
            return mean
    
    @torch.no_grad()
    def forward(self, x: Tensor) -> list[Tensor]:
        shape = x.shape
        img = torch.randn(shape).to(x.device)
        imgs = []
        for i in tqdm((range(0, self.num_steps))):
            img = self.reverse_sample(i, x, img)
            imgs.append(img.detach().cpu())
        return imgs


class ContinousDiffusion(nn.Module):
    def __init__(self,
                 denoise_model: Callable,
                 lambda_min: int = -20, 
                 lambda_max: int = 20,
                 num_timesteps: int = 2000,
                 loss_fn=nn.L1Loss(reduction='mean')
                ) -> None:
        
        super().__init__()
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.denoise_model = denoise_model
        self.loss_fn = loss_fn
        self.forward_process = ContinuousForwardDiffusion(denoise_model=denoise_model, lambda_min=lambda_min, lambda_max=lambda_max)
        self.reverse_process = ContinuousReverseDiffusion(denoise_model=denoise_model, num_steps=num_timesteps, lambda_min=lambda_min, lambda_max=lambda_max)  
    
    @staticmethod
    def _neg_one_to_one(img: Tensor) -> Tensor:
        return (img * 2) - 1.
    
    @staticmethod
    def _one_to_zero(img: Tensor) -> Tensor:
        return (img + 1.) * 0.5
    
    def super_resolution(self, x) -> Tensor:
        if torch.min(x) > 0:
            x = self._neg_one_to_one(x)
        x_sr = self.reverse_process(x)
        
        return self._one_to_zero(x_sr[-2])
    
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        if torch.min(x) > 0 and torch.min(y) > 0:
            x = self._neg_one_to_one(x)
            y = self._neg_one_to_one(y)
        
        _, eps_hat, eps = self.forward_process(x, y)
        
        loss = self.loss_fn(eps, eps_hat)
        
        return loss

