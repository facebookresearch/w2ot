#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.

import argparse
import functools
import glob
import os
import yaml
import time

from collections import defaultdict

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
    parser.add_argument('exp_roots', type=str, nargs='+')
    parser.add_argument('--tables', action='store_true')

    args = parser.parse_args()

    def clean_key(k):
        k_parts = k.split('.')
        if k_parts[0] in ['data', 'agent'] and len(k_parts) > 1:
            return '.'.join(k_parts[1:])
        elif k == 'maximin_trainer':
            return 'dual_trainer'
        else:
            return k

    rows = []

    for exp_root in args.exp_roots:
        exp_paths = glob.glob(exp_root + '/*')

        for exp_path in exp_paths:
            row = {}

            fname = exp_path + '/.hydra/overrides.yaml'
            if not os.path.exists(fname):
                continue

            with open(fname, 'r') as f:
                overrides = yaml.load(f, Loader=yaml.Loader)
                overrides = dict([x.split('=') for x in overrides])
                filter_keys = ['data']
                overrides = {
                    clean_key(k): v for (k,v) in overrides.items()
                    if k not in filter_keys}
                row.update(overrides)

            fname = exp_path + '/log.csv'
            if not os.path.exists(fname):
                continue

            try:
                log = pd.read_csv(fname)
            except:
                continue

            n_iter = log['iter'].max()
            row['UVP_fwd_best'] = log['UVP_fwd'].min()
            row['UVP_fwd_final'] = log['UVP_fwd'].values[-1]
            row['num_iter'] = n_iter
            # row['cos_fwd_best'] = log['cos_fwd'].max()
            # row['cos_fwd_final'] = log['cos_fwd'].values[-1]
            if 'num_opt_iter' in log:
                row['final_opt_iter'] = log['num_opt_iter'].values[-1]
            if 'conj_time' in log:
                row['conj_time'] = log['conj_time'].values[-1]

            row['runtime_hours'] = log['elapsed_time_s'].values[-1]/60/60
            row['id'] = int(exp_path.split('/')[-1])
            row['exp_root'] = exp_root

            rows.append(row)

    print('Number of experiments loaded: ', len(rows))
    if len(rows) == 0:
        import sys; sys.exit(-1)

    df = pd.DataFrame(rows)
    df = df.apply(pd.to_numeric, errors='ignore')

    def get_best_paths(dual_trainer):
        best_paths = []
        ax_labels = []
        if 'hd_benchmark' in dual_trainer:
            for input_dim in [64, 128, 256]:
                I = (df.dual_trainer == dual_trainer) & \
                    (df.input_dim == input_dim) & \
                    (df.amortization == 'regression')
                selected_df = df[I]
                best_id = selected_df.id.iloc[selected_df.UVP_fwd_final.argmin()]
                best_path = selected_df.exp_root.values[0] + '/' + str(best_id)
                best_paths.append(best_path)
                ax_labels.append(f'$D={input_dim}$')
        elif 'image_benchmark' in dual_trainer:
            ax_labels = ['Early', 'Mid', 'Late']
            for which in ax_labels:
                I = (df.dual_trainer == dual_trainer) & \
                    (df.which == which) & \
                    (df.amortization == 'regression')
                selected_df = df[I]
                best_id = selected_df.id.iloc[selected_df.UVP_fwd_final.argmin()]
                best_path = selected_df.exp_root.values[0] + '/' + str(best_id)
                best_paths.append(best_path)
        else:
            assert False

        return best_paths, ax_labels


    for dual_trainer in ['icnn_hd_benchmark', 'nn_hd_benchmark', 'image_benchmark']:
        nrow, ncol = 1, 3
        fig, axs = plt.subplots(nrow, ncol, figsize=(4*ncol, 2*nrow),
                                gridspec_kw = {'wspace': 0.05, 'hspace':0.1})
        axs = axs.ravel()

        best_paths, ax_labels = get_best_paths(dual_trainer)
        assert len(axs) == len(best_paths)

        times = []
        for ax, best_path, ax_label in zip(axs, best_paths, ax_labels):
            print(f'=== {dual_trainer} {ax_label}: {best_path}')
            add_plot(ax, best_path)
            ax.set_title(ax_label)

            batch_size = 1024 if 'hd_benchmark' in dual_trainer else 64
            prof_results = profile(ax, best_path, batch_size=batch_size)
            prof_results['tag'] = ax_label
            times.append(prof_results)

        times = pd.DataFrame.from_records(times, index='tag')
        times['Improvement Factor'] = times.Wolfe/times.Armijo
        print(times.T.to_latex(escape=False))

        for ax in axs[1:]:
            ax.set_yticklabels([])

        axs[0].set_ylabel('Normalized $J(x; y)$')

        for ax in axs:
            iter_lim = 50.01 if 'hd_benchmark' in dual_trainer else 10.01
            ax.set_xlim(0, iter_lim)
            ax.set_ylim(-0.01, 1.)

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)

            ax.set_xlabel('Solver Iteration')

        fname = f'paper/fig/conjugations/{dual_trainer}.pdf'
        print(f'saving to {fname}')
        fig.savefig(fname, transparent=True, bbox_inches='tight')
        os.system(f'pdfcrop {fname} {fname}')



def add_plot(ax, exp_root, pkl_tag='latest'):
    exp_path = f'{exp_root}/{pkl_tag}.pkl'
    assert os.path.exists(exp_path)
    with open(exp_path, 'rb') as f:
        exp = pkl.load(f)

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

    out = solve_lbfgs_armijo_jit(Y)

    s_adam = conjugate_solver.SolverAdam(max_iter=100, gtol=1e-1)
    solve_adam_jit = jax.jit(jax.vmap(
        functools.partial(
            s_adam.solve,
            D_apply, track_hist=True),
    ))

    out_adam = solve_adam_jit(Y)

    if dual_trainer.H is not None:
        init_X_hats = dual_trainer.jit_fns['push_D_conj_init'](
            dual_trainer.H_params, Y)
        out_warm = solve_lbfgs_armijo_jit(Y, x_init=init_X_hats)


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

def profile(ax, exp_root, batch_size, pkl_tag='latest'):
    exp_path = f'{exp_root}/{pkl_tag}.pkl'
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

    wolfe_solver = conjugate_solver.SolverLBFGS(
        gtol=-1, max_iter=10, max_linesearch_iter=10, ls_method='wolfe')
    armijo_solver = conjugate_solver.SolverLBFGS(
        gtol=-1, max_iter=10, ls_base=1.5, max_linesearch_iter=10, ls_method='armijo')

    times = defaultdict(list)
    n_trial = 2

    for solver, name in zip([wolfe_solver, armijo_solver], ['Wolfe', 'Armijo']):
        solve_jit = jax.jit(jax.vmap(
            functools.partial(solver.solve, D_apply)))
        for i in range(n_trial+1):
            start = time.time()
            solve_jit(Y).val.block_until_ready()
            t = time.time() - start

            if i > 1:
                # Ignore first iteration that runs the jit
                times[name].append(t*1000)

    times = {k: sum(v)/len(v) for k,v in times.items()}
    return times


if __name__ == '__main__':
        main()
