# Copyright (c) Meta Platforms, Inc. and affiliates.

from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax import dtypes
from jax.random import PRNGKey

from flax import linen as nn
from flax.linen.linear import default_kernel_init, \
     _conv_dimension_numbers

from dataclasses import dataclass, field

from typing import Any, Optional, Callable, Sequence, Tuple, Union

from w2ot import utils

class InitNN(nn.Module):
    dim_hidden: Sequence[int]
    act: str = 'elu'

    @nn.compact
    def __call__(self, x):
        single = x.ndim == 1
        if single:
            x = jnp.expand_dims(x, 0)
        assert x.ndim == 2
        n_input = x.shape[-1]

        act_fn = utils.get_act(self.act)

        z = x
        for n_hidden in self.dim_hidden:
            Wx = nn.Dense(n_hidden, use_bias=True)
            z = act_fn(Wx(z))

        Wx = nn.Dense(n_input, use_bias=True)
        z = x + Wx(z) # Encourage identity initialization.

        if single:
            z = jnp.squeeze(z, 0)
        return z
