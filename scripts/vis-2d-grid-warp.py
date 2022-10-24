#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.

import argparse

import jax
import jax.numpy as jnp

import numpy as np

import pickle as pkl
import os
import shutil
import functools

from w2ot import conjugate_solver, utils

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
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

def main():
    import sys
    from IPython.core import ultratb
    sys.excepthook = ultratb.FormattedTB(mode='Verbose',
                                         color_scheme='Linux',
                                         call_pdb=1)

    parser = argparse.ArgumentParser()
    parser.add_argument('exp_root', type=str)
    parser.add_argument('--pkl_tag', type=str, default='best')
    args = parser.parse_arg()

    exp_path = f'{args.exp_root}/{args.pkl_tag}.pkl'
    assert os.path.exists(exp_path)
    with open(exp_path, 'rb') as f:
        exp = pkl.load(f)

    dual_trainer = exp.dual_trainer

    key = jax.random.PRNGKey(0)
    n_batch = 500
    X = exp.data.X_sampler.sample(key, n_batch)
    Y = exp.data.Y_sampler.sample(key, n_batch)

    xylim = (-5, 5, -5, 5)
    plot_grid_warping(args, dual_trainer, X, Y, xylim)
    plot_grid_warping_video(args, dual_trainer, X, Y, xylim)


def _add_grid(x,y, ax, **kwargs):
    segs1 = np.stack((x,y), axis=2)
    segs2 = segs1.transpose(1,0,2)
    ax.add_collection(LineCollection(segs1, **kwargs))
    ax.add_collection(LineCollection(segs2, **kwargs))
    ax.autoscale()


def plot_grid_warping(args, dual_trainer, X, Y, xylim):
    xmin, xmax, ymin, ymax = xylim

    num_interp = 7
    nrow, ncol = 1, num_interp
    fig, axs = plt.subplots(nrow, ncol, figsize=(4*ncol,4*nrow),
                            gridspec_kw = {'wspace':0, 'hspace':0})
    key = jax.random.PRNGKey(0)

    # Plot the grid warped by the conjugate potential's gradient
    ts = np.linspace(0, 1, num=num_interp)
    n_sample = 30
    x,y = np.meshgrid(np.linspace(xmin, xmax, n_sample), np.linspace(ymin, ymax, n_sample))
    x_flat = x.ravel()
    y_flat = y.ravel()
    xy_flat = jnp.stack((x_flat, y_flat), axis=1)
    warped_xy = dual_trainer.push_inv(xy_flat)

    for ax, t in zip(axs, ts):
        grads = (1-t)*xy_flat + t*warped_xy
        grads_x = grads[:,0].reshape(x.shape)
        grads_y = grads[:,1].reshape(y.shape)
        _add_grid(grads_x, grads_y, ax=ax, color="C0")

    fig.tight_layout()
    for ax in axs:
        # ax.set_xticklabels([])
        # ax.set_yticklabels([])
        ax.grid(False)
        ax.set_axis_off()

    fname = f'{args.exp_root}/grid-warp.png'
    fig.savefig(fname, transparent=True)
    plt.close(fig)
    os.system(f'convert -trim {fname} {fname}')
    print(f'Saving to {fname}')


def plot_grid_warping_video(args, dual_trainer, X, Y, xylim):
    frames_dir = f'{args.exp_root}/grid-warp-frames'
    if os.path.exists(frames_dir):
        shutil.rmtree(frames_dir)
    os.mkdir(frames_dir)

    xmin, xmax, ymin, ymax = xylim

    num_interp = 32
    key = jax.random.PRNGKey(0)

    # Plot the grid warped by the conjugate potential's gradient
    ts = np.linspace(0, 1, num=num_interp)
    n_sample = 20
    x,y = np.meshgrid(np.linspace(xmin, xmax, n_sample), np.linspace(ymin, ymax, n_sample))
    x_flat = x.ravel()
    y_flat = y.ravel()
    xy_flat = jnp.stack((x_flat, y_flat), axis=1)

    warped_xy = dual_trainer.push_inv(xy_flat)

    for frame_num, t in enumerate(ts):
        grads = (1-t)*xy_flat + t*warped_xy
        grads_x = grads[:,0].reshape(x.shape)
        grads_y = grads[:,1].reshape(y.shape)
        fig, ax = plt.subplots(1, 1, figsize=(5, 5),
                               gridspec_kw = {'wspace':0, 'hspace':0})
        _add_grid(grads_x, grads_y, ax=ax, color="C0")

        fig.tight_layout()
        ax.grid(False)
        ax.set_axis_off()

        fname = f'{frames_dir}/{frame_num:05d}.png'
        fig.savefig(fname, transparent=True)
        plt.close(fig)
        os.system(f'convert -trim {fname} {fname}')


    os.system(f"ffmpeg -i {frames_dir}/%05d.png {frames_dir}/forward.mp4 -y")
    os.system(f"ffmpeg -i {frames_dir}/forward.mp4 -vf reverse {frames_dir}/reverse.mp4 -y")

    gif_fname = f'{args.exp_root}/grid.gif'
    print(f'Saving to {gif_fname}')
    os.system(f'convert -loop 0 {frames_dir}/forward.mp4 {frames_dir}/reverse.mp4 {gif_fname}')

    shutil.rmtree(frames_dir)


if __name__ == '__main__':
    main()
