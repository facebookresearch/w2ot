# Copyright (c) Meta Platforms, Inc. and affiliates.

import copy
import functools

import jax
import jax.numpy as jnp
from jax.lax import stop_gradient
from flax import linen as nn
from flax.training import train_state

from dataclasses import dataclass
from typing import Tuple, Sequence, Optional

import optax

from w2ot import utils, conjugate_solver
from w2ot.models.icnn import ICNN
from w2ot.amortization import BaseAmortization, NoAmortization, \
     RegressionAmortization, ObjectiveAmortization, W2GNAmortization


@dataclass
class DualTrainer:
    input_dim: int
    num_train_iter: int
    batch_size: int
    pretrain_lr: float
    key: jax.random.PRNGKey
    data_bounds: Tuple[float, float]

    # For evaluating the true dual objective.
    conjugate_solver: conjugate_solver.SolverLBFGS

    D: nn.Module # Dual potential

    # Initial \hat x for the conjugate (and the inverse transport map)
    # If not set, don't predict \hat x
    H: Optional[nn.Module]

    # Keyword arguments to pass into the optimizers
    lr_schedule_kwargs: dict
    adamw_kwargs: dict

    amortization: BaseAmortization # Amortization configuration
    num_pretrain_iter: int = 10001

    def __post_init__(self):
        # Initialize the dual potential D and conjugate amortization H
        k1, k2, k3, self.key = jax.random.split(self.key, 4)
        minval, maxval = self.data_bounds
        init_batch = jax.random.uniform(
            k1, [self.batch_size, self.input_dim],
            minval=minval, maxval=maxval)
        D_init = self.D.init(k2, init_batch)
        self.D_params = D_init['params']

        if self.H is not None:
            # H can be a potential that predicts the conjugate value or
            # it can directly try to predict the argmin/solution.
            H_out, H_init = self.H.init_with_output(k3, init_batch)
            self.H_params = H_init['params']

            # Push using the gradient of H (otherwise use H's output directly)
            self.H_push_grad = jnp.squeeze(H_out).ndim == 1
        else:
            self.H_params = None
        self.jit_fns = self.init_jit_fns()


    def init_train_state(self, pretrain=True):
        if pretrain:
            self.D_params = self.pretrain_identity(self.D, self.D_params)
            if self.H is not None:
                if type(self.D) == type(self.H):
                    self.H_params = self.D_params
                else:
                    self.H_params = self.pretrain_identity(
                            self.H, self.H_params, push_grad=False)

        lr_schedule = optax.cosine_decay_schedule(
            **self.lr_schedule_kwargs)
        opt = optax.adamw(learning_rate=lr_schedule, **self.adamw_kwargs)

        state_D = train_state.TrainState.create(
            apply_fn=self.D.apply, params=self.D_params, tx=opt)

        if self.H is not None:
            state_H = train_state.TrainState.create(
                apply_fn=self.H.apply, params=self.H_params, tx=opt)
            return [state_D, state_H]
        else:
            return [state_D]


    def create_update_fn(self):
        def train_loss_fn(D_params, H_params, X, Y):
            batch_size = X.shape[0]

            if self.H is not None:
                if self.H_push_grad:
                    init_X_hat = utils.push_grad(self.H, H_params, Y)
                else:
                    init_X_hat = self.H.apply({'params': H_params}, Y)
            else:
                init_X_hat = Y

            if self.amortization.finetune_during_training:
                X_hat_detach, num_opt_iter = stop_gradient(
                    self.jit_fns['push_D_conj'](D_params, Y, init_X_hat))
            else:
                assert self.H is not None, 'No conjugate prediction or finetuning enabled'
                X_hat_detach = stop_gradient(init_X_hat)
                num_opt_iter = jnp.zeros([])

            D_apply = lambda x: self.D.apply({'params': D_params} , x)
            D_x = D_apply(X)
            D_y = utils.batch_dot(X_hat_detach, Y) - D_apply(X_hat_detach)
            dual_x = D_x.mean()
            dual_y = D_y.mean()
            dual_loss = dual_x + dual_y

            metrics = {
                'dual_loss': dual_loss,
                'mean_x_potential': dual_x,
                'mean_y_potential': dual_y,
                'num_opt_iter': num_opt_iter.mean(),
            }

            loss = dual_loss

            if self.H is not None:
                # Add the amortization loss
                if isinstance(self.amortization, RegressionAmortization):
                    assert self.amortization.finetune_during_training, \
                        'Trying to regress without finetuning'
                    amor_loss = ((init_X_hat - X_hat_detach)**2).mean()
                elif isinstance(self.amortization, ObjectiveAmortization):
                    D_apply_parameters_detached = lambda x: self.D.apply(
                        {'params': stop_gradient(D_params)} , x)
                    amor_loss = (D_apply_parameters_detached(init_X_hat) - \
                                 utils.batch_dot(init_X_hat, Y)).mean()
                elif isinstance(self.amortization, W2GNAmortization):
                    if self.amortization.only_amortize_H:
                        # The amortization loss will not update D
                        Y_hat_hat = utils.push_grad(
                            self.D, stop_gradient(D_params), init_X_hat)
                    else:
                        # The default W2GN implementation:
                        # D /can/ be updated to better-amortize the conjugate.
                        Y_hat_hat = utils.push_grad(self.D, D_params, init_X_hat)

                    amor_loss = ((Y_hat_hat - Y) ** 2).mean()

                    if self.amortization.regularize_D:
                        if self.amortization.only_amortize_H:
                            Y_hat = utils.push_grad(self.D, stop_gradient(D_params), X)
                        else:
                            Y_hat = utils.push_grad(self.D, D_params, X)
                        X_hat_hat = self.jit_fns['push_D_conj_init'](H_params, Y_hat)
                        amor_loss += ((X_hat_hat - X) ** 2).mean()

                    weighted_amor_loss = self.amortization.cycle_loss_weight * amor_loss
                    amor_loss = weighted_amor_loss
                elif isinstance(self.amortization, NoAmortization):
                    amor_loss = 0.
                else:
                    raise NotImplementedError(
                        f'Amortization loss not recognized: {self.amortization}')

                if not isinstance(self.amortization, NoAmortization):
                    metrics['amor_loss'] = amor_loss
                    loss += amor_loss

            return loss, [metrics]


        @jax.jit
        def update(X, Y, *states):
            if self.H is not None:
                state_D, state_H = states
                H_params = state_H.params
            else:
                state_D, = states
                H_params = None
            grad_fn = jax.value_and_grad(
                train_loss_fn, argnums=(0,1), has_aux=True)
            (loss, aux), (grads_D, grads_H) = grad_fn(
                state_D.params, H_params, X, Y)

            metrics, = aux
            flat_grads_D = jax.flatten_util.ravel_pytree(grads_D)[0]
            metrics['grad_D_mean'] = flat_grads_D.mean()
            metrics['grad_D_maxabs'] = jnp.abs(flat_grads_D).max()

            if self.H is not None:
                flat_grads_H = jax.flatten_util.ravel_pytree(grads_H)[0]
                metrics['grad_H_mean'] = flat_grads_H.mean()
                metrics['grad_H_maxabs'] = jnp.abs(flat_grads_H).max()

            new_states = [state_D.apply_gradients(grads=grads_D)]
            if self.H is not None:
                new_states.append(state_H.apply_gradients(grads=grads_H))
            return [loss, metrics] + new_states

        return update


    def store_params(self, *states):
        if self.H is not None:
            state_D, state_H = states
            self.H_params = state_H.params
        else:
            state_D, = states
        self.D_params = state_D.params


    def dual_obj(self, X, Y, warmstart=True):
        return self.jit_fns['dual_obj'](
            self.D_params, self.H_params, X, Y, warmstart=warmstart)


    def push(self, X):
        single = X.ndim == 1
        if single:
            X = jnp.expand_dims(X, 0)

        Y_hat = self.jit_fns['push_D'](self.D_params, X)
        if single:
            Y_hat = Y_hat.squeeze(0)
        return Y_hat


    def push_inv(self, Y, warmstart=True):
        single = Y.ndim == 1
        if single:
            Y = jnp.expand_dims(Y, 0)

        if warmstart and self.H is not None:
            init_X_hat = self.jit_fns['push_D_conj_init'](self.H_params, Y)
            X_hat, num_opt_iter = stop_gradient(
                self.jit_fns['push_D_conj'](self.D_params, Y, init_X_hat))
        else:
            X_hat = self.jit_fns['push_D_conj'](self.D_params, Y, None)[0]

        if single:
            X_hat = X_hat.squeeze(0)
        return X_hat


    def pretrain_identity(self, net, params, push_grad=True):
        print('pre-training to satisfy push(net, x) \\approx x')
        k1, self.key = jax.random.split(self.key, 2)
        opt = optax.adam(learning_rate=self.pretrain_lr)
        state = train_state.TrainState.create(
            apply_fn=net.apply, params=params, tx=opt)

        def pretrain_loss_fn(params, x, key):
            k1, k2, key = jax.random.split(key, 3)

            if push_grad:
                push_x = utils.push_grad(net, params, x)
            else:
                push_x = net.apply({'params': params}, x)
            loss = ((push_x-x)**2).sum(axis=1).mean()
            return loss

        @jax.jit
        def pretrain_update(state, key):
            minval, maxval = self.data_bounds
            k1, key = jax.random.split(key, 2)
            X = jax.random.uniform(
                k1, [self.batch_size, self.input_dim],
                minval=minval, maxval=maxval)
            grad_fn = jax.value_and_grad(pretrain_loss_fn)
            loss, grads = grad_fn(state.params, X, key)
            return loss, state.apply_gradients(grads=grads)

        for i in range(self.num_pretrain_iter):
            k1, self.key = jax.random.split(self.key, 2)
            loss, state = pretrain_update(state, k1)
            if i % 1000 == 0:
                print(f'iter={i} pretrain_loss={loss:.2e}')
            if i > 1000 and loss < 1e-6:
                print('pretrain loss < 1e-6, exiting early')
                break

        return state.params


    # This function jits commonly used operations such as the pushforward,
    # conjugate, and dual objective and returns them in a dict so can
    # be easily called without needing to be jitted in every new scenario.
    def init_jit_fns(self):
        jit_fns = {}
        jit_fns['push_D'] = jax.jit(functools.partial(utils.push_grad, self.D))

        if self.H is not None:
            if self.H_push_grad:
                jit_fns['push_D_conj_init'] = jax.jit(
                    functools.partial(utils.push_grad, self.H))
            else:
                jit_fns['push_D_conj_init'] = jax.jit(
                    functools.partial(utils.vmap_apply, self.H))

        def D_conj(D_params, y, x_init):
            D_apply = lambda x: self.D.apply({'params': D_params} , x)
            out = self.conjugate_solver.solve(D_apply, y, x_init=x_init)
            return out.grad, out.num_iter

        jit_fns['push_D_conj'] = jax.jit(
            jax.vmap(D_conj, in_axes=[None, 0, 0]))

        def dual_obj(D_params, H_params, X, Y, warmstart=True):
            if warmstart and self.H is not None:
                init_X_hat = self.jit_fns['push_D_conj_init'](H_params, Y)
            else:
                init_X_hat = None
            X_hat, _ = stop_gradient(self.jit_fns['push_D_conj'](D_params, Y, init_X_hat))

            D_apply = lambda x: self.D.apply({'params': D_params} , x)
            dual = D_apply(X).mean() + \
                 (utils.batch_dot(X_hat, Y) - D_apply(X_hat)).mean()
            return -dual
        jit_fns['dual_obj'] = jax.jit(dual_obj, static_argnames=['warmstart'])

        return jit_fns


    def __getstate__(self):
        # Delete parts of the object that can't be pickled.
        d = copy.copy(self.__dict__)
        del d['jit_fns']
        return d


    def __setstate__(self, d):
        # Re-initialize the deleted parts of the object.
        self.__dict__ = d
        self.jit_fns = self.init_jit_fns()

