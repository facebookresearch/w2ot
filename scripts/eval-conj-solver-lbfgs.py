#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.

import argparse
import functools
import glob
import os
import yaml
import time

from collections import namedtuple

import numpy as np
import pickle as pkl

import pandas as pd
pd.set_option("display.precision", 2)

import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.style.use('bmh')
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": ["Computer Modern Roman"]})

from w2ot import conjugate_solver, utils

import sys
import w2ot.run_train as train
sys.modules['train'] = train # Legacy

import sys
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(
    mode='Plain', color_scheme='Neutral', call_pdb=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_root', type=str)
    args = parser.parse_args()

    pkl_tag = 'latest'
    exp_path = f'{args.exp_root}/{pkl_tag}.pkl'
    assert os.path.exists(exp_path)
    with open(exp_path, 'rb') as f:
        exp = pkl.load(f)

    dual_trainer = exp.dual_trainer
    key = jax.random.PRNGKey(0)
    n_batch = 1024
    X = exp.data.X_sampler.sample(key, n_batch)
    Y = exp.data.Y_sampler.sample(key, n_batch)

    in_vars = {'params': dual_trainer.D_params}
    D_apply = lambda x: dual_trainer.D.apply(in_vars, x)

    gtol = 0.1
    max_iter = 100
    num_evaluations = 15
    ls_base = 1.5

    solvers = {
        'Jax Backtracking Armijo': conjugate_solver.SolverLBFGS(
            gtol=gtol, max_iter=max_iter,
            ls_method='armijo',
            ls_kwargs=dict(ls_base=ls_base, num_evaluations=num_evaluations, parallel=False)),
        'Jax Parallel Armijo': conjugate_solver.SolverLBFGS(
            gtol=gtol, max_iter=max_iter,
            ls_method='armijo',
            ls_kwargs=dict(ls_base=ls_base, num_evaluations=num_evaluations, parallel=True)),
        'Jax Strong Wolfe Zoom': conjugate_solver.SolverLBFGS(
            gtol=gtol, max_iter=max_iter,
            ls_method='wolfe',
            ls_kwargs=dict(maxls=num_evaluations)),
        'JaxOpt Strong Wolfe Zoom': conjugate_solver.SolverLBFGSJaxOpt(
            gtol=gtol, max_iter=max_iter,
            max_linesearch_iter=num_evaluations,
            linesearch_type='zoom', ls_method='strong-wolfe'
        ),
        'JaxOpt Backtracking Strong Wolfe': conjugate_solver.SolverLBFGSJaxOpt(
            gtol=gtol, max_iter=max_iter,
            max_linesearch_iter=num_evaluations,
            decrease_factor=1./ls_base,
            linesearch_type = 'backtracking',
            ls_method = 'strong-wolfe'),
        'JaxOpt Backtracking Wolfe': conjugate_solver.SolverLBFGSJaxOpt(
            gtol=gtol, max_iter=max_iter,
            max_linesearch_iter=num_evaluations,
            decrease_factor=1./ls_base,
            linesearch_type = 'backtracking',
            ls_method = 'wolfe'),
        'JaxOpt Backtracking Armijo': conjugate_solver.SolverLBFGSJaxOpt(
            gtol=gtol, max_iter=max_iter,
            max_linesearch_iter=num_evaluations,
            decrease_factor=1./ls_base,
            linesearch_type = 'backtracking',
            ls_method = 'armijo'),
    }

    X_inits = {
        # 'Scratch': None,
        'Warmstart': dual_trainer.jit_fns['push_D_conj_init'](
            dual_trainer.H_params, Y),
    }

    rows = []
    for X_init_name, X_init in X_inits.items():
        for solver_name, solver in solvers.items():
            # print(f'--- {solver_name}')
            row = profile(D_apply, solver=solver, Y=Y, X_init=X_init)
            row['solver'] = solver_name
            # row['X_init'] = X_init_name
            rows.append(row)
            # df = pd.DataFrame(rows).set_index(['solver', 'X_init'])
            df = pd.DataFrame(rows).set_index('solver')
            print(df.to_latex(
                columns=['mean_runtime', 'mean_num_iters'],
                header=['Runtime (ms)', '\#Iter'],
                float_format='${:.2f}$'.format,
                escape=False))



ProfResults = namedtuple('ProfResults', 'vals times')
def profile(D_apply, solver, Y, X_init=None, num_trials=10):
    times = []

    solve_jit = jax.jit(jax.vmap(
        functools.partial(solver.solve, D_apply, return_grad_norm=True)))
    for i in range(num_trials+1):
        start = time.time()
        solve_out = solve_jit(Y, x_init=X_init)
        grad_norms = solve_out.grad_norm.block_until_ready()
        t = time.time() - start
        num_iters = solve_out.num_iter

        if i > 1: # Ignore first iteration that runs the jit
            times.append(t*1000) # Milliseconds

    times = jnp.array(times)
    return {
        'mean_runtime': jnp.mean(times).item(),
        'mean_grad_norms': jnp.mean(grad_norms).item(),
        'mean_num_iters': jnp.mean(num_iters).item(),
        # 'max_grad_norms': jnp.max(grad_norms),
    }


if __name__ == '__main__':
    main()
