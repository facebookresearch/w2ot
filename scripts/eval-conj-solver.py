#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.

import argparse

import jax
import jax.numpy as jnp

import numpy as np

import pickle as pkl
import os
import functools

from w2ot import conjugate_solver, utils

import sys
import w2ot.run_train as train
sys.modules['train'] = train # Legacy

import matplotlib.pyplot as plt
plt.style.use('bmh')
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": ["Computer Modern Roman"]})

import sys
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(
    mode='Plain', color_scheme='Neutral', call_pdb=1)

DIR = os.path.dirname(os.path.realpath(__file__))

from jax.config import config
config.update("jax_enable_x64", True)

def main():
    import sys
    from IPython.core import ultratb
    sys.excepthook = ultratb.FormattedTB(mode='Verbose',
                                         color_scheme='Linux',
                                         call_pdb=1)

    parser = argparse.ArgumentParser()
    parser.add_argument('exp_root', type=str)
    parser.add_argument('--pkl_tag', type=str, default='best')
    args = parser.parse_args()

    exp_path = f'{args.exp_root}/{args.pkl_tag}.pkl'
    assert os.path.exists(exp_path)
    print('-- loading exp')
    with open(exp_path, 'rb') as f:
        exp = pkl.load(f)
    print('-- done')

    dual_trainer = exp.dual_trainer
    key = jax.random.PRNGKey(0)
    n_batch = 10
    X = exp.data.X_sampler.sample(key, n_batch)
    Y = exp.data.Y_sampler.sample(key, n_batch)

    in_vars = {'params': dual_trainer.D_params}
    D_apply = lambda x: dual_trainer.D.apply(in_vars, x)

    s_lbfgs_armijo = conjugate_solver.SolverLBFGS(
        gtol=1e-1, max_iter=50, ls_base=1.5, max_linesearch_iter=10, ls_method='armijo')
    solve_lbfgs_armijo_jit = jax.jit(jax.vmap(
        functools.partial(
            s_lbfgs_armijo.solve,
            D_apply, track_hist=True),
    ))

    print('--- LBFGS')
    out = solve_lbfgs_armijo_jit(Y)

    s_adam = conjugate_solver.SolverAdam(gtol=1e-1, max_iter=50)
    solve_adam_jit = jax.jit(jax.vmap(
        functools.partial(
            s_adam.solve,
            D_apply, track_hist=True),
    ))

    print('--- Adam')
    out_adam = solve_adam_jit(Y)

    if dual_trainer.H is not None:
        print('--- LBFGS + Warmstart')
        init_X_hats = dual_trainer.jit_fns['push_D_conj_init'](
            dual_trainer.H_params, Y)
        out_warm = solve_lbfgs_armijo_jit(Y, x_init=init_X_hats)


    fig, ax = plt.subplots(1, 1, figsize=(4,2.5))
    colors = plt.style.library['bmh']['axes.prop_cycle'].by_key()['color']

    for i in range(n_batch):
        n_iter_default = out.num_iter[i]
        vals_default = out.val_hist[i][:n_iter_default]
        vals_adam = out_adam.val_hist[i]
        all_vals = [vals_default, vals_adam]


        if dual_trainer.H is not None:
            n_iter_warm = out_warm.num_iter[i]
            vals_warm = out_warm.val_hist[i][:n_iter_warm]
            all_vals.append(vals_warm)

        all_vals = jnp.concatenate(all_vals)
        obj_min, obj_max = all_vals.min(), all_vals.max()
        obj_range = obj_max - obj_min

        def norm(x):
            return (x - obj_min) / obj_range

        ax.plot(norm(vals_default), color=colors[0], alpha=0.3)
        ax.plot(norm(vals_adam), color=colors[2], alpha=0.3)
        if dual_trainer.H is not None:
            ax.plot(norm(vals_warm), color=colors[1], alpha=0.3)
        ax.set_xlabel('Solver Iteration')
        ax.set_ylabel('Normalized $J(x; y)$')

    fname = args.exp_root + '/conj-convergence.png'
    print(f'Saving to {fname}')
    fig.tight_layout()
    fig.savefig(fname, transparent=True)
    plt.close(fig)



if __name__ == '__main__':
    main()
