# Copyright (c) Meta Platforms, Inc. and affiliates.

import jax
import jax.numpy as jnp
from jax import dtypes
from flax import linen as nn

from functools import partial

batch_dot = jax.vmap(jnp.dot)

class RunningAverageMeter(object):
    def __init__(self, momentum=0.999):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


def push_grad(net, params, data, **forward_kwargs):
    vars_in = {'params': params}
    return jax.vmap(lambda X: jax.grad(net.apply, argnums=1)(
        vars_in, X, **forward_kwargs))(data)


def vmap_apply(net, params, data, **forward_kwargs):
    vars_in = {'params': params}
    return jax.vmap(lambda x: net.apply(
        vars_in, x, **forward_kwargs))(data)

def get_act(name):
    if name.startswith('leaky_relu'):
        params = name[11:]
        if len(params) > 0:
            negative_slope = float(params)
        else:
            negative_slope = 0.01
        return partial(jax.nn.leaky_relu, negative_slope=negative_slope)
    elif name.startswith('relu'):
        return jax.nn.relu
    elif name.startswith('elu'):
        return jax.nn.elu
    elif name.startswith('selu'):
        return jax.nn.selu
    elif name.startswith('celu'):
        params = name[5:]
        if len(params) > 0:
            alpha = float(params)
        else:
            alpha = 1.0
        return partial(jax.nn.celu, alpha=1.0)
    elif name == 'softplus':
        return nn.softplus
    else:
        assert False

class ActNorm(nn.Module):
    @nn.compact
    def __call__(self, x):
        n = x.shape[-1]

        def init_log_scale(dtype=jnp.float_):
            def init(key, shape, dtype=dtype):
                assert x.ndim == 2
                dtype = dtypes.canonicalize_dtype(dtype)
                return jnp.log(x.std(axis=0))
            return init

        def init_bias(dtype=jnp.float_):
            def init(key, shape, dtype=dtype):
                assert x.ndim == 2
                dtype = dtypes.canonicalize_dtype(dtype)
                return -x.mean(axis=0)
            return init

        log_scale = self.param('log_scale', init_log_scale(), [n])
        bias = self.param('bias', init_bias(), [n])
        x = (x + bias) / jnp.exp(log_scale)
        return x
