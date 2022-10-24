#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.

import glob
import os
import yaml
import numpy as np
import pandas as pd
pd.set_option("display.precision", 2)
import argparse

import sys
import w2ot.run_train as train
sys.modules['train'] = train # Legacy

import sys
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(
    mode='Plain', color_scheme='Neutral', call_pdb=1)


parser = argparse.ArgumentParser()
parser.add_argument('exp_roots', type=str, nargs='+')
parser.add_argument('--no_tables', action='store_true')

args = parser.parse_args()


def clean_key(k):
    k_parts = k.split('.')
    if k_parts[0] in ['data', 'agent'] and len(k_parts) > 1:
        return '.'.join(k_parts[1:])
    elif k == 'maximin_trainer':
        return 'dual_trainer'
    # elif k_parts[0] in ['conjugate_solver'] and len(k_parts) > 1:
    #     return '.'.join(k_parts[2:])
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
        # row['UVP_fwd_best'] = log['UVP_fwd'].min()
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

if 'conjugate_solver' not in df:
    df['conjugate_solver'] = 'lbfgs'

I = df['conjugate_solver'].isna()
df['conjugate_solver'][I] = 'lbfgs'

# Can optionally filter
# df = df[df.agent == 'maximin_benchmark_hd_icnn_no_init']
# df = df[df.input_dim == 256]
# print(df)
# import sys; sys.exit(-1)

if 'input_dim' in overrides:
    main_key = 'input_dim'
elif 'which' in overrides:
    main_key = 'which'
    idx_order = {
        'early': 0,
        'mid': 1,
        'late': 2,
    }
else:
    assert False


# Set the correct multiindex
del overrides['seed'], overrides[main_key]
# del overrides['conjugate_solver']
idx = [main_key] + list(overrides.keys())
df = df.set_index(idx)

def sort_key(val):
    return idx_order[val.lower()]

def sort_columns(df):
    if main_key == 'which':
        df = df.reindex(sorted(df.columns, key=sort_key), axis=1)
    return df

def sort_index_fn(index):
    if index.name == 'which':
        return index.map(sort_key)
    else:
        return index
df = df.sort_index(key=sort_index_fn)

print(df.to_string())

mean_df = df.groupby(df.index.names).mean()
mean_df.UVP_fwd_final[mean_df.UVP_fwd_final > 100] = '>100'
mean_df = mean_df.sort_index(key=sort_index_fn)
print(mean_df.to_string())
if args.no_tables:
    sys.exit(0)

df = df.reset_index()
def get_tag(amortization, conjugate_solver):
    pretty_names = {
        'lbfgs': 'L-BFGS',
        'adam': 'Adam',
    }
    conjugate_solver = pretty_names[conjugate_solver]
    if amortization == 'objective' or amortization == 'w2gn':
        conjugate_solver = 'None'
    if '_finetune' in amortization:
        amortization = amortization.replace('_finetune', '')
    pretty_names = {
        'none': 'None',
        'objective': 'Objective',
        'regression': 'Regression',
        'w2gn': 'Cycle',
    }
    amortization = pretty_names[amortization]
    s = ' & ' + amortization + ' & ' + conjugate_solver
    if main_key == 'which':
        s += ' & Conv & F '
    return s

df['tag'] = df.apply(
    lambda x: get_tag(x.amortization, x.conjugate_solver), axis=1)


def print_subdf_tables(df):
    del df['dual_trainer']

    def aggfunc(vals):
        mean_val = np.mean(vals)
        if mean_val > 100:
            return '>100'
        else:
            return f'\\pair{{{mean_val:.2f}}}{{{np.std(vals):.2f}}}'

    uvps = pd.pivot_table(
        df.reset_index(),
        values='UVP_fwd_final',
        index=['tag'],
        columns=[main_key],
        aggfunc={'UVP_fwd_final': aggfunc},
        sort=False
    )
    idx = np.concatenate((uvps.index[-2:], uvps.index[:-2]))
    uvps = sort_columns(uvps.reindex(idx))
    print(uvps)
    print(uvps.to_latex(escape=False))

    if main_key == 'input_dim':
        # Only look at the largest settings
        df = df[df.input_dim >= 64]

    def aggfunc(vals):
        agg = np.median(vals)
        return f'{agg:.2f}'

    runtimes = pd.pivot_table(
        df.reset_index(),
        values='runtime_hours',
        index=['tag'],
        columns=[main_key],
        aggfunc={'runtime_hours': aggfunc},
    )
    idx = np.concatenate((runtimes.index[-2:], runtimes.index[:-2]))
    runtimes = sort_columns(runtimes.reindex(idx))
    # print(runtimes)
    print('--runtimes')
    print(runtimes.to_latex(escape=False))

    def aggfunc(vals):
        agg = np.mean(vals)
        return f'{agg:.2f}'

    conjs = pd.pivot_table(
        df.reset_index(),
        values='final_opt_iter',
        index=['tag'],
        columns=[main_key],
        aggfunc={'final_opt_iter': aggfunc},
    )
    idx = np.concatenate((conjs.index[-2:], conjs.index[:-2]))
    conjs = sort_columns(conjs.reindex(idx))
    # print(conjs)
    print('--conj num iter')
    print(conjs.to_latex(escape=False))

    conjs = pd.pivot_table(
        df.reset_index(),
        values='conj_time',
        index=['tag'],
        columns=[main_key],
        aggfunc={'conj_time': aggfunc},
    )
    idx = np.concatenate((conjs.index[-2:], conjs.index[:-2]))
    conjs = sort_columns(conjs.reindex(idx))
    # print(conjs)
    print('--conj runtime')
    print(conjs.to_latex(escape=False))

partitions = df.dual_trainer.unique()
for partition in partitions:
    print(f'=== {partition}')
    print_subdf_tables(df[df.dual_trainer == partition])

