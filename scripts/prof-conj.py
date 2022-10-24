#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.

import argparse

import jax
import jax.numpy as jnp

import numpy as np

import pickle as pkl
import os

import functools
from collections import defaultdict

import time

from w2ot import conjugate_solver, utils

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

    maximin_trainer = exp.maximin_trainer
    key = jax.random.PRNGKey(0)
    n_batch = 1024
    X = exp.data.X_sampler.sample(key, n_batch)
    Y = exp.data.Y_sampler.sample(key, n_batch)

    in_vars = {'params': maximin_trainer.D_params}
    D_apply = lambda x: maximin_trainer.D.apply(in_vars, x)

    s_lbfgs = conjugate_solver.SolverLBFGS(
        gtol=-1, max_iter=10, max_linesearch_iter=10, ls_method='wolfe')
    solve_lbfgs_jit = jax.jit(jax.vmap(
        functools.partial(s_lbfgs.solve, D_apply)))

    s_lbfgs_armijo = conjugate_solver.SolverLBFGS(
        gtol=-1, max_iter=10, ls_base=1.5, max_linesearch_iter=10, ls_method='armijo')
    solve_lbfgs_armijo_jit = jax.jit(jax.vmap(
        functools.partial(s_lbfgs_armijo.solve, D_apply)))

    s_lbfgs_armijo = conjugate_solver.SolverLBFGS(
        gtol=-1, max_iter=10, ls_base=1.5, max_linesearch_iter=10, ls_method='armijo')
    solve_lbfgs_armijo_jit = jax.jit(jax.vmap(
        functools.partial(s_lbfgs_armijo.solve, D_apply)))

    times = defaultdict(list)
    n_trial = 10
    for i in range(n_trial+1):
        print('--- lbfgs (Wolfe)')
        start = time.time()
        out_lbfgs = solve_lbfgs_jit(Y)
        t = time.time() - start
        print(f'time: {t:.2e}s')

        if i > 1:
            times['lbfgs_wolfe'].append(t)

        print('--- lbfgs (Armijo)')
        start = time.time()
        out_lbfgs_armijo = solve_lbfgs_armijo_jit(Y)
        t = time.time() - start
        print(f'time: {t:.2e}s')

        if i > 1:
            times['lbfgs_armijo'].append(t)


if __name__ == '__main__':
    main()
