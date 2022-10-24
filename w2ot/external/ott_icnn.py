# This file is modified from the following files and remains under the
# original licensing.
#
# https://github.com/ott-jax/ott/blob/main/ott/core/icnn.py
# and
# https://github.com/ott-jax/ott/blob/main/ott/core/layers.py
#
#
# Copyright 2022 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax import dtypes
from jax.random import PRNGKey

from flax import linen as nn

from typing import Any, Optional, Callable, Sequence, Tuple, Union
ModuleDef = Any

from w2ot import utils


class PositiveDense(nn.Module):
    dim_hidden: int
    dtype: Any = jnp.float32
    precision: Any = None
    kernel_init: Any = None

    @nn.compact
    def __call__(self, inputs):
        inputs = jnp.asarray(inputs, self.dtype)
        kernel = self.param(
            'kernel', self.kernel_init, (inputs.shape[-1], self.dim_hidden))
        kernel = jnp.asarray(kernel, self.dtype)
        kernel = nn.softplus(kernel)
        y = jax.lax.dot_general(
            inputs, kernel, (((inputs.ndim - 1,), (0,)), ((), ())),
            precision=self.precision)

        gain = 1./inputs.shape[-1]
        y *= gain
        return y


class ICNN(nn.Module):
    dim_hidden: Sequence[int]
    act: str = 'elu'
    actnorm: bool = True

    def setup(self):
        kernel_init = nn.initializers.variance_scaling(
            1., "fan_in", "truncated_normal")

        num_hidden = len(self.dim_hidden)

        w_zs = list()
        for i in range(1, num_hidden):
            w_zs.append(PositiveDense(self.dim_hidden[i], kernel_init=kernel_init))
        w_zs.append(PositiveDense(1, kernel_init=kernel_init))
        self.w_zs = w_zs

        w_xs = list()
        for i in range(num_hidden):
            w_xs.append(nn.Dense(
                self.dim_hidden[i], use_bias=True,
                kernel_init=kernel_init))

        w_xs.append(nn.Dense(1, use_bias=True, kernel_init=kernel_init))
        self.w_xs = w_xs


    @nn.compact
    def __call__(self, x):
        single = x.ndim == 1
        if single:
            x = jnp.expand_dims(x, 0)
        assert x.ndim == 2
        n_input = x.shape[-1]

        act_fn = utils.get_act(self.act)

        z = act_fn(self.w_xs[0](x))
        for Wz, Wx in zip(self.w_zs[:-1], self.w_xs[1:-1]):
            z = Wz(z) + Wx(x)
            if self.actnorm:
                z = utils.ActNorm()(z)
            z = act_fn(z)

        # An activation on this last layer is really helpful sometimes.
        y = act_fn(self.w_zs[-1](z) + self.w_xs[-1](x))
        y = jnp.squeeze(y, -1)

        log_alpha = self.param(
            'log_alpha', nn.initializers.constant(0), [])
        y += jnp.exp(log_alpha)*0.5*utils.batch_dot(x, x)

        if single:
            y = jnp.squeeze(y, 0)
        return y
