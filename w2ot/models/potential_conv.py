# Copyright (c) Meta Platforms, Inc. and affiliates.

from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax import dtypes
from jax.random import PRNGKey

from flax import linen as nn
from typing import Any, Optional, Callable, Sequence, Tuple, Union

from w2ot import utils


class ConvPotential(nn.Module):
    act: str = 'elu'

    mean = jnp.array([0.485, 0.456, 0.406])
    std = jnp.array([0.229, 0.224, 0.225])

    @nn.compact
    def __call__(self, x):
        single = x.ndim == 1
        if single:
            x = jnp.expand_dims(x, 0)
        assert x.ndim == 2 # Should be flattened
        num_batch = x.shape[0]

        x_flat = x # Save for taking the quadratic at the end.

        # Reshape and renormalize
        x = x.reshape(-1, 3, 64, 64).transpose(0, 2, 3, 1)
        x = (x + 1.)/2.
        x = (x-self.mean) / self.std
        y = x

        act_fn = utils.get_act(self.act)

        conv = nn.Conv(128, kernel_size=[4,4], strides=2)
        y = act_fn(conv(y))

        conv = nn.Conv(128, kernel_size=[4,4], strides=2)
        y = act_fn(conv(y))

        conv = nn.Conv(256, kernel_size=[4,4], strides=2)
        y = act_fn(conv(y))

        conv = nn.Conv(512, kernel_size=[4,4], strides=2)
        y = act_fn(conv(y))

        conv = nn.Conv(1024, kernel_size=[4,4], strides=2)
        y = act_fn(conv(y))

        conv = nn.Conv(
            1, kernel_size=[2,2], padding='VALID', strides=1)
        y = act_fn(conv(y))
        y = y.squeeze([1,2,3])

        assert y.shape == (num_batch,)

        log_alpha = self.param(
            'log_alpha', nn.initializers.constant(0), [])
        y += 0.5*jnp.exp(log_alpha)*utils.batch_dot(x_flat, x_flat)

        if single:
            y = jnp.squeeze(y, 0)
        return y
