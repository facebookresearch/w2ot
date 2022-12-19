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
    parser.add_argument('--colorbars', action='store_true')
    parser.add_argument('--quivers', action='store_true')
    args = parser.parse_args()

    exp_path = f'{args.exp_root}/{args.pkl_tag}.pkl'
    assert os.path.exists(exp_path)
    with open(exp_path, 'rb') as f:
        exp = pkl.load(f)

    # Use high-precision solver.
    # dual_trainer.conjugate_solver = conjugate_solver.SolverLBFGSJaxOpt(
    #     gtol=1e-4, max_iter=100,
    #     max_linesearch_iter=30,
    #     linesearch_type='backtracking', ls_method='strong-wolfe'
    # )

    key = jax.random.PRNGKey(0)

    # Sample a lot of transportations between the images
    n_sample = 5000000
    n_batch = 100000
    X, Y, X_push, Y_push = [], [], [], []
    for i in range(n_sample // n_batch):
        k1, k2, key = jax.random.split(key, 3)
        Xi = exp.data.X_sampler.sample(k2, n_batch)
        Yi = exp.data.Y_sampler.sample(k2, n_batch)
        X_push_i = exp.dual_trainer.push(Xi)
        Y_push_i = exp.dual_trainer.push_inv(Yi)
        X.append(Xi)
        Y.append(Yi)
        X_push.append(X_push_i)
        Y_push.append(Y_push_i)

    X = jnp.concatenate(X)
    Y = jnp.concatenate(Y)
    X_push = jnp.concatenate(X_push)
    Y_push = jnp.concatenate(Y_push)

    plot_transport_video(args, exp, X, Y, X_push, Y_push, conjugate=True)
    plot_transport_video(args, exp, X, Y, X_push, Y_push, conjugate=False)
    plot_transport_video_bi(args, exp, X, Y, X_push, Y_push)


def get_image(Y, scale):
    Y = (Y + (scale/2)) / scale
    img, _, _ = jnp.histogram2d(Y[:,1], Y[:,0], bins=250, range=[[0,1],[0,1]])
    img = (img > 1).astype(np.float32)
    img = np.flipud(img)
    return img

def plot_transport_video(args, exp, X, Y, X_push, Y_push, conjugate=True):
    scale = exp.data.scale

    tag = 'inverse' if conjugate else 'forward'
    frames_dir = f'{args.exp_root}/transport-samples-frames-{tag}'
    if os.path.exists(frames_dir):
        shutil.rmtree(frames_dir)
    os.mkdir(frames_dir)


    num_interp = 25
    ts = np.linspace(0, 1, num=num_interp)

    # Colors for the interpolation
    if exp.data.reverse:
        c1 = np.array([28,150,232])/256
        c0 = np.array([93,81,232])/256
    else:
        c0 = np.array([28,150,232])/256
        c1 = np.array([93,81,232])/256

    for frame_num, t in enumerate(ts):
        nrow, ncol = 1, 1
        fig, ax = plt.subplots(nrow, ncol, figsize=(3*ncol,3*nrow),
                                gridspec_kw = {'wspace':0, 'hspace':0})

        if conjugate:
            push_t = (1.-t)*Y + t*Y_push
            color_vals = t*c0 + (1-t)*c1
        else:
            push_t = (1.-t)*X + t*X_push
            color_vals = (1-t)*c0 + t*c1

        img_t = get_image(push_t, scale)
        img_t = np.expand_dims(img_t, 2) * color_vals
        img_t[img_t == 0] = 1.
        ax.imshow(img_t, origin='upper',
                  interpolation='none')

        ax.grid(False)
        ax.set_axis_off()
        fig.tight_layout()

        fname = f'{frames_dir}/{frame_num:05d}.png'
        fig.savefig(fname, transparent=True, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        # os.system(f'convert -trim {fname} {fname}')
        os.system(f'convert {fname} -background White -alpha remove -alpha off {fname}')


    os.system(f"ffmpeg -i {frames_dir}/%05d.png {frames_dir}/forward.mp4 -y")
    os.system(f"ffmpeg -i {frames_dir}/forward.mp4 -vf reverse {frames_dir}/reverse.mp4 -y")

    gif_fname = f'{args.exp_root}/transport-samples-{tag}.gif'
    print(f'Saving to {gif_fname}')
    os.system(f'convert -loop 0 {frames_dir}/forward.mp4 {frames_dir}/reverse.mp4 {gif_fname}')

    shutil.rmtree(frames_dir)


def plot_transport_video_bi(args, exp, X, Y, X_push, Y_push):
    # Bidirectional transport: combines the forward and
    # inverse paths to best-show the marginals.
    scale = exp.data.scale

    frames_dir = f'{args.exp_root}/transport-samples-frames-bi'
    if os.path.exists(frames_dir):
        shutil.rmtree(frames_dir)
    os.mkdir(frames_dir)

    num_interp = 25
    ts = np.linspace(0, 1, num=num_interp)

    # Colors for the interpolation
    if exp.data.reverse:
        c0 = np.array([28,150,232])/256
        c1 = np.array([93,81,232])/256
    else:
        c1 = np.array([28,150,232])/256
        c0 = np.array([93,81,232])/256

    for frame_num, t in enumerate(ts):
        nrow, ncol = 1, 1
        fig, ax = plt.subplots(nrow, ncol, figsize=(3*ncol,3*nrow),
                                gridspec_kw = {'wspace':0, 'hspace':0})

        # Compute both paths at the same time
        inverse_push_t = get_image((1.-t)*Y + t*Y_push, scale)
        forward_push_t = get_image(t*X + (1.-t)*X_push, scale)

        sigmoid_bound = 12.
        sigmoid_t = 1./(1.+jnp.exp(-(t*sigmoid_bound - sigmoid_bound//2.)))
        # Merge the 2 paths
        img_t = (1-sigmoid_t)*inverse_push_t + sigmoid_t*forward_push_t
        combined_threshold = 0.7
        img_t = (img_t > combined_threshold).astype(jnp.float32)

        color_vals = (1-t)*c0 + t*c1
        img_t = np.expand_dims(img_t, 2) * color_vals
        img_t[img_t == 0] = 1.

        ax.imshow(img_t, origin='upper', interpolation='none')

        fig.tight_layout()
        ax.grid(False)
        ax.set_axis_off()

        fname = f'{frames_dir}/{frame_num:05d}.png'
        fig.savefig(fname, transparent=True, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        # os.system(f'convert -trim {fname} {fname}')
        os.system(f'convert {fname} -background White -alpha remove -alpha off {fname}')


    os.system(f"ffmpeg -i {frames_dir}/%05d.png {frames_dir}/forward.mp4 -y")
    os.system(f"ffmpeg -i {frames_dir}/forward.mp4 -vf reverse {frames_dir}/reverse.mp4 -y")

    gif_fname = f'{args.exp_root}/transport-samples-bi.gif'
    print(f'Saving to {gif_fname}')
    os.system(f'convert -loop 0 {frames_dir}/forward.mp4 {frames_dir}/reverse.mp4 {gif_fname}')

    shutil.rmtree(frames_dir)



if __name__ == '__main__':
    main()
