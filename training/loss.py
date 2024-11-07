# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Loss functions used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import torch
from torch_utils import persistence
import ambient_utils
from training.sampler import edm_sampler

#----------------------------------------------------------------------------
# Improved loss function proposed in the paper "Elucidating the Design Space
# of Diffusion-Based Generative Models" (EDM).

@persistence.persistent_class
class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5, 
                 num_primes=4, num_consistency_steps=4, consistency_coeff=1.0, 
                 consistency_batch_size_per_gpu=4, with_weight=True, with_grad=False):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        
        self.num_consistency_steps = num_consistency_steps
        self.num_primes = num_primes
        self.consistency_coeff = consistency_coeff
        self.consistency_batch_size = consistency_batch_size_per_gpu
        self.with_weight = with_weight
        self.with_grad = with_grad

    def __call__(self, net, images, labels=None, current_sigma=0.0, augment_pipe=None):

        net._set_static_graph()
        current_sigma = current_sigma.unsqueeze(1).unsqueeze(1).unsqueeze(1)

        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        # sample a sigma in [current_sigma, sigma_T]
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()        
        sigma = torch.clamp(sigma, min=current_sigma)
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        
        # add additional noise to reach the level sigma
        n = torch.randn_like(y) * torch.sqrt(sigma ** 2 - current_sigma ** 2)
        noisy_input = y + n
        x0_pred = net(noisy_input, sigma, labels, augment_labels=augment_labels)
        # make it xtn prediction
        D_yn = ambient_utils.from_x0_pred_to_xnature_pred_ve_to_ve(x0_pred, noisy_input, sigma, current_sigma)
        
        # loss weight depends on sigma
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        loss = weight * ((D_yn - y) ** 2)

    
        # consistency loss
        if self.consistency_coeff > 0:
            # sample a new_sigma in [sigma, 0]
            new_sigma = (rnd_normal * self.P_std + self.P_mean).exp()
            new_sigma = torch.clamp(new_sigma, max=sigma)

            # we will only keep the first batch_size / self.num_primes part of the batch
            consistency_batch_size = self.consistency_batch_size
            noisy_input = noisy_input[:consistency_batch_size]
            sigma = sigma[:consistency_batch_size]
            new_sigma = new_sigma[:consistency_batch_size]
            labels = labels[:consistency_batch_size]
            # repeat everything num_primes times
            noisy_input = noisy_input.repeat_interleave(self.num_primes, dim=0)
            sigma = sigma.repeat_interleave(self.num_primes, dim=0)
            new_sigma = new_sigma.repeat_interleave(self.num_primes, dim=0)
            labels = labels.repeat_interleave(self.num_primes, dim=0)

            # run sampler from sigma -> new_sigma
            with torch.no_grad() if not self.with_grad else torch.enable_grad():
                x_t_prime = edm_sampler(net, noisy_input, class_labels=labels, num_steps=self.num_consistency_steps, 
                                        sigma_min=new_sigma, sigma_max=sigma) 
            # get predictions for x_t_prime
            x0_pred_prime = net(x_t_prime, new_sigma, labels)
            # group together predictions
            x0_pred_prime = x0_pred_prime.reshape(consistency_batch_size, self.num_primes, *x0_pred_prime.shape[1:])
            # average predictions
            average_x0_pred_prime = x0_pred_prime.mean(dim=1)
            # check difference to x0_pred
            consistency_loss = ((average_x0_pred_prime - x0_pred[:consistency_batch_size]) ** 2)
            consistency_weight = weight[:consistency_batch_size] if self.with_weight else 1.0
            loss[:consistency_batch_size] += self.consistency_coeff * consistency_weight * consistency_loss
        return loss, x0_pred

#----------------------------------------------------------------------------
