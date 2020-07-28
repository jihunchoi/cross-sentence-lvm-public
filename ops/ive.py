"""
Taken from https://github.com/nicola-decao/s-vae-pytorch.
"""

from numbers import Number

import torch

import numpy as np
from scipy import special


class IveFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, v, z):

        assert isinstance(v, Number), 'v must be a scalar'

        ctx.save_for_backward(z)
        ctx.v = v
        z_cpu = z.data.cpu().numpy()

        if np.isclose(v, 0):
            output = special.i0e(z_cpu, dtype=z_cpu.dtype)
        elif np.isclose(v, 1):
            output = special.i1e(z_cpu, dtype=z_cpu.dtype)
        else:  # v > 0
            output = special.ive(v, z_cpu, dtype=z_cpu.dtype)

        return torch.Tensor(output).to(z.device)

    @staticmethod
    def backward(ctx, grad_output):
        z = ctx.saved_tensors[-1]
        return None, grad_output * (
                    ive(ctx.v - 1, z) - ive(ctx.v, z) * (ctx.v + z) / z)


ive = IveFunction.apply
