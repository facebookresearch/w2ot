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
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(
    mode='Plain', color_scheme='Neutral', call_pdb=1)


parser = argparse.ArgumentParser()
parser.add_argument('exp_root', type=str)
args = parser.parse_args()

exp_paths = glob.glob(args.exp_root + '/*')

def clean_key(k):
    k_parts = k.split('.')
    if k_parts[0] in ['data', 'agent', 'dual_trainer'] and len(k_parts) > 1:
        return '.'.join(k_parts[1:])
    # if k_parts[0] in ['data', 'agent'] and len(k_parts) > 1:
    #     return '.'.join(k_parts[1:])
    else:
        return k

rows = []
for exp_path in exp_paths:
    row = {}

    fname = exp_path + '/.hydra/overrides.yaml'
    if not os.path.exists(fname):
        continue

    with open(fname, 'r') as f:
        overrides = yaml.load(f, Loader=yaml.Loader)
        overrides = dict([x.split('=') for x in overrides])
        filter_keys = []
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
    row['best_obj'] = log['dual_obj'].max()
    row['num_iter'] = n_iter
    row['eval_fwd_error_last'] = log['eval_fwd_error'].values[-1]
    row['eval_inv_error_last'] = log['eval_inv_error'].values[-1]
    # row['eval_fwd_error_best'] = log['eval_fwd_error'].values.min()
    # row['eval_inv_error_best'] = log['eval_inv_error'].values.min()
    # if 'num_opt_iter' in log:
    #     row['final_opt_iter'] = log['num_opt_iter'].values[-1]
    # if 'conj_time' in log:
    #     row['conj_time'] = log['conj_time'].values[-1]

    # row['runtime_hours'] = log['elapsed_time_s'].values[-1]/60/60
    row['id'] = int(exp_path.split('/')[-1])

    rows.append(row)

print('Number of experiments loaded: ', len(rows))
if len(rows) == 0:
    import sys; sys.exit(-1)

df = pd.DataFrame(rows)
df = df.apply(pd.to_numeric, errors='ignore')
# df = df.set_index('id')

# Can optionally filter
# df = df[df.agent == 'dual_benchmark_hd_icnn_no_init']
# df = df[df.input_dim == 256]
# print(df)
# for data in df.data.unique():
#     print(f'\n=== {data}')
#     sub_df = df[df.data == data]
#     print(sub_df)
    # import ipdb; ipdb.set_trace()
# import sys; sys.exit(-1)


# Set the correct multiindex
del overrides['data']
idx = ['data'] + list(overrides.keys())
df = df.set_index(idx).sort_index()
print(df.to_string())

# import ipdb; ipdb.set_trace()
