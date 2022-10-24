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

    dual_trainer = exp.dual_trainer

    # Use high-precision solver.
    # dual_trainer.conjugate_solver = conjugate_solver.SolverLBFGSJaxOpt(
    #     gtol=1e-4, max_iter=100,
    #     max_linesearch_iter=30,
    #     linesearch_type='backtracking', ls_method='strong-wolfe'
    # )

    key = jax.random.PRNGKey(0)
    n_batch = 500
    X = exp.data.X_sampler.sample(key, n_batch)
    Y = exp.data.Y_sampler.sample(key, n_batch)

    plot_potential_single(args, dual_trainer, X, Y)
    xylim = plot_transport(args, dual_trainer, X, Y)
    plot_conj(args, dual_trainer, X, Y, xylim)

    plot_transport_video(args, dual_trainer, X, Y)
    plot_conj_transport_path_video(args, dual_trainer, X, Y)

    key = jax.random.PRNGKey(0)
    n_batch = 5000
    X = exp.data.X_sampler.sample(key, n_batch)
    Y = exp.data.Y_sampler.sample(key, n_batch)
    plot_conj_transport_samples(args, dual_trainer, X, Y)
    plot_conj_transport_samples_video(args, dual_trainer, X, Y)


def plot_potential_single(args, dual_trainer, X, Y):
    nrow, ncol = 1, 1
    fig, ax = plt.subplots(nrow, ncol, figsize=(4*ncol,4*nrow),
                            gridspec_kw = {'wspace':0, 'hspace':0})
    key = jax.random.PRNGKey(0)
    k1, k2, key = jax.random.split(key, 3)
    X_push = dual_trainer.push(X)
    Y_push = dual_trainer.push_inv(Y)

    s = 5
    ax.scatter(X[:,0], X[:,1], s=s, color='#1A254B', zorder=10)
    # ax.scatter(Y[:,0], Y[:,1], s=s, color='#1A254B', zorder=10)

    xmin, xmax = -5, 5
    ymin, ymax = -5, 5

    n = 300
    all_data = jnp.concatenate((X.ravel(), Y.ravel()))
    b = jnp.abs(all_data).max().item() * 1.2
    x1 = np.linspace(xmin, xmax, n)
    x2 = np.linspace(ymin, ymax, n)
    X1, X2 = np.meshgrid(x1, x2)
    X1flat = np.ravel(X1)
    X2flat = np.ravel(X2)
    X12flat = np.stack((X1flat, X2flat)).T
    Zflat = utils.vmap_apply(dual_trainer.D, dual_trainer.D_params, X12flat)
    Z = np.array(Zflat.reshape(X1.shape))

    if args.exp_root.endswith('/12'):
        # Hack to better-visualize the colors
        Zmin, Zmax = -5, 20.
        Z = Z.clip(min=Zmin, max=Zmax)
        CS = ax.contourf(X1, X2, Z, cmap='Blues', levels=np.linspace(Zmin, Zmax, num=8))
    else:
        CS = ax.contourf(X1, X2, Z, cmap='Blues')

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes('right', size='5%', pad=0.05)
    # fig.colorbar(CS, cax=cax)

    ax.grid(False)
    ax.set_axis_off()
    # ax.set_xticks([])
    # ax.set_yticks([])
    # for d in ['top', 'right', 'bottom', 'left']:
    #     ax.spines[d].set_edgecolor('k')


    fname = f'{args.exp_root}/vis-potential.png'
    fig.savefig(fname, transparent=True, dpi=200)
    plt.close(fig)
    os.system(f'convert -trim {fname} -resize 240x {fname}')
    print(f'Saving to {fname}')

    return (xmin, xmax, ymin, xmax)


def plot_lines(ax, A, B):
    xs = np.vstack((A[:,0], B[:,0]))
    ys = np.vstack((A[:,1], B[:,1]))
    ax.plot(xs, ys, color=[0.5, 0.5, 1], alpha=0.1)


def plot_transport(args, dual_trainer, X, Y):
    nrow, ncol = 1, 4
    fig, axs = plt.subplots(nrow, ncol, figsize=(4*ncol,4*nrow),
                            gridspec_kw = {'wspace':0, 'hspace':0})
    key = jax.random.PRNGKey(0)
    k1, k2, key = jax.random.split(key, 3)
    X_push = dual_trainer.push(X)
    Y_push = dual_trainer.push_inv(Y)

    # Plot the data and transported samples
    ax = axs[0]
    s = 5
    ax.scatter(X[:,0], X[:,1], s=s, color='#A7BED3')
    ax.scatter(Y[:,0], Y[:,1], s=s, color='#1A254B')
    ax.scatter(X_push[:,0], X_push[:,1], s=s, color='#F2545B')
    plot_lines(ax, X, X_push)
    xmin, xmax = axs[0].get_xlim()
    ymin, ymax = axs[0].get_ylim()

    ax = axs[1]
    ax.scatter(X[:,0], X[:,1], s=s, color='#1A254B')
    ax.scatter(Y[:,0], Y[:,1], s=s, color='#A7BED3')
    ax.scatter(Y_push[:,0], Y_push[:,1], s=s, color='#F2545B')
    plot_lines(ax, Y, Y_push)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    ax = axs[2]
    s = 5
    ax.scatter(X[:,0], X[:,1], s=s, color='#A7BED3', zorder=10)
    ax.scatter(Y[:,0], Y[:,1], s=s, color='#1A254B', zorder=10)

    # Plot the values of the potential
    n = 300
    all_data = jnp.concatenate((X.ravel(), Y.ravel()))
    b = jnp.abs(all_data).max().item() * 1.2
    x1 = np.linspace(xmin, xmax, n)
    x2 = np.linspace(ymin, ymax, n)
    X1, X2 = np.meshgrid(x1, x2)
    X1flat = np.ravel(X1)
    X2flat = np.ravel(X2)
    X12flat = np.stack((X1flat, X2flat)).T
    Zflat = utils.vmap_apply(dual_trainer.D, dual_trainer.D_params, X12flat)
    Z = np.array(Zflat.reshape(X1.shape))
    # Z = Z.clamp(15, 3gcc

    CS = ax.contourf(X1, X2, Z, cmap='Blues')
    if args.colorbars:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(CS, cax=cax)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    if args.quivers:
        # Add quiver of gradient field
        in_vars = {'params': dual_trainer.D_params}
        D_apply = lambda x: dual_trainer.D.apply(in_vars, x)
        D_apply_grad = jax.jit(jax.vmap(jax.grad(D_apply)))

        n_sample = 15
        x,y = np.meshgrid(np.linspace(xmin, xmax, n_sample), np.linspace(ymin, ymax, n_sample))
        x_flat = x.ravel()
        y_flat = y.ravel()
        xy_flat = jnp.stack((x_flat, y_flat), axis=1)

        grads = D_apply_grad(xy_flat) - xy_flat
        grads_x = grads[:,0].reshape(x.shape)
        grads_y = grads[:,1].reshape(y.shape)
        ax.quiver(x, y, grads_x, grads_y)

    ax = axs[3]
    s = 5
    ax.scatter(X[:,0], X[:,1], s=s, color='#A7BED3', zorder=10)
    ax.scatter(Y[:,0], Y[:,1], s=s, color='#1A254B', zorder=10)

    n = 300
    all_data = jnp.concatenate((X.ravel(), Y.ravel()))
    b = jnp.abs(all_data).max().item() * 1.2
    x1 = np.linspace(xmin, xmax, n)
    x2 = np.linspace(ymin, ymax, n)
    X1, X2 = np.meshgrid(x1, x2)
    X1flat = np.ravel(X1)
    X2flat = np.ravel(X2)
    X12flat = np.stack((X1flat, X2flat)).T
    conj_flat = dual_trainer.push_inv(X12flat)
    Zflat = -(utils.vmap_apply(
        dual_trainer.D, dual_trainer.D_params, conj_flat) - \
        utils.batch_dot(X12flat, conj_flat))

    Z = np.array(Zflat.reshape(X1.shape))

    CS = ax.contourf(X1, X2, Z, cmap='Blues')
    if args.colorbars:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(CS, cax=cax)

    if args.quivers:
        # Add quiver of gradient field
        n_sample = 15
        x,y = np.meshgrid(np.linspace(xmin, xmax, n_sample), np.linspace(ymin, ymax, n_sample))
        x_flat = x.ravel()
        y_flat = y.ravel()
        xy_flat = jnp.stack((x_flat, y_flat), axis=1)

        grads = dual_trainer.push_inv(xy_flat) - xy_flat
        # grads = grads.clip(-5, 5)
        grads_x = grads[:,0].reshape(x.shape)
        grads_y = grads[:,1].reshape(y.shape)
        ax.quiver(x, y, grads_x, grads_y)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    fig.tight_layout()
    for ax in axs:
        # ax.set_xticklabels([])
        # ax.set_yticklabels([])
        ax.grid(False)
        ax.set_axis_off()
        # ax.set_xticks([])
        # ax.set_yticks([])
        # for d in ['top', 'right', 'bottom', 'left']:
        #     ax.spines[d].set_edgecolor('k')


    fname = f'{args.exp_root}/vis-transport.png'
    fig.savefig(fname, transparent=True)
    plt.close(fig)
    os.system(f'convert -trim {fname} {fname}')
    print(f'Saving to {fname}')

    return (xmin, xmax, ymin, xmax)


def plot_conj(args, dual_trainer, X, Y, xylim):
    xmin, xmax, ymin, ymax = xylim

    in_vars = {'params': dual_trainer.D_params}
    D_apply = lambda x: dual_trainer.D.apply(in_vars, x)

    solver = dual_trainer.conjugate_solver
    solve_jit = jax.jit(
        functools.partial(
            solver.solve,
            D_apply, track_hist=True),
    )

    nrow, ncol = 1, 5
    fig, axs = plt.subplots(nrow, ncol, figsize=(4*ncol,4*nrow),
                            gridspec_kw = {'wspace':0, 'hspace':0})
    axs = axs.ravel()

    for i, ax in enumerate(axs):
        print('====')
        s = 5
        ax.scatter(X[:,0], X[:,1], s=s, color='#A7BED3', zorder=10)
        ax.scatter(Y[:,0], Y[:,1], s=s, color='#1A254B', zorder=10)

        y = Y[i]
        ax.scatter(y[0], y[1], color='#F2545B', zorder=10, s=100)

        def conj_min_obj(x):
            return dual_trainer.D.apply({'params': dual_trainer.D_params}, x) - x.dot(y)

        conj_min_obj_batch = jax.vmap(conj_min_obj)

        n = 300
        x1 = np.linspace(xmin, xmax, n)
        x2 = np.linspace(ymin, ymax, n)
        X1, X2 = np.meshgrid(x1, x2)
        X1flat = np.ravel(X1)
        X2flat = np.ravel(X2)
        X12flat = np.stack((X1flat, X2flat)).T
        Zflat = conj_min_obj_batch(X12flat)
        Z = np.array(Zflat.reshape(X1.shape))

        init_conj_val = conj_min_obj(y) + 1
        Z[Z > init_conj_val] = np.infty

        CS = ax.contourf(X1, X2, Z, cmap='Blues')
        out = solve_jit(y, x_init=X[i])
        # out = solve_jit(y, x_init=y)
        conj_grad = out.grad
        ax.scatter(conj_grad[0], conj_grad[1], color='#F2545B', zorder=10, marker='*', s=200)

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        print(f'-- n_iter: {out.num_iter}')
        print(f'-- min found: {conj_min_obj(conj_grad):.2e} / sampled: {np.min(Z):.2e}')

    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
        for d in ['top', 'right', 'bottom', 'left']:
            ax.spines[d].set_edgecolor('k')

    fname = f'{args.exp_root}/conj.png'
    fig.savefig(fname, transparent=True)
    plt.close(fig)
    os.system(f'convert -trim {fname} {fname}')
    print(f'-- Saving to {fname}')


def plot_transport_video(args, dual_trainer, X, Y):
    frames_dir = f'{args.exp_root}/transport-frames'
    if os.path.exists(frames_dir):
        shutil.rmtree(frames_dir)
    os.mkdir(frames_dir)

    key = jax.random.PRNGKey(0)
    k1, k2, key = jax.random.split(key, 3)
    X_push = dual_trainer.push(X)
    Y_push = dual_trainer.push_inv(Y)

    num_interp = 20
    ts = np.linspace(0, 1, num=num_interp)


    for frame_num, t in enumerate(ts):
        nrow, ncol = 1, 4
        fig, axs = plt.subplots(nrow, ncol, figsize=(4*ncol,4*nrow),
                                gridspec_kw = {'wspace':0, 'hspace':0})

        X_push_t = (1.-t)*X + t*X_push
        Y_push_t = (1.-t)*Y + t*Y_push

        # Plot the data and transported samples
        ax = axs[0]
        s = 5
        ax.scatter(X[:,0], X[:,1], s=s, color='#A7BED3')
        ax.scatter(Y[:,0], Y[:,1], s=s, color='#1A254B')
        ax.scatter(X_push_t[:,0], X_push_t[:,1], s=s, color='#F2545B')
        plot_lines(ax, X, X_push_t)

        if frame_num == 0:
            # Save axis limits and compute potential values only the first time.
            xmin, xmax = axs[0].get_xlim()
            ymin, ymax = axs[0].get_ylim()

            n = 300
            all_data = jnp.concatenate((X.ravel(), Y.ravel()))
            b = jnp.abs(all_data).max().item() * 1.2
            x1 = np.linspace(xmin, xmax, n)
            x2 = np.linspace(ymin, ymax, n)
            X1, X2 = np.meshgrid(x1, x2)
            X1flat = np.ravel(X1)
            X2flat = np.ravel(X2)
            X12flat = np.stack((X1flat, X2flat)).T
            Zflat = utils.vmap_apply(dual_trainer.D, dual_trainer.D_params, X12flat)
            Z = np.array(Zflat.reshape(X1.shape))

            conj_flat = dual_trainer.push_inv(X12flat)
            Zflat = -(utils.vmap_apply(
                dual_trainer.D, dual_trainer.D_params, conj_flat) - \
                utils.batch_dot(X12flat, conj_flat))
            Z_conj = np.array(Zflat.reshape(X1.shape))
        else:
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)

        ax = axs[1]
        ax.scatter(X[:,0], X[:,1], s=s, color='#1A254B')
        ax.scatter(Y[:,0], Y[:,1], s=s, color='#A7BED3')
        ax.scatter(Y_push_t[:,0], Y_push_t[:,1], s=s, color='#F2545B')
        plot_lines(ax, Y, Y_push_t)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        # Plot the values of the potential and conjugate
        ax = axs[2]
        s = 5
        ax.scatter(X[:,0], X[:,1], s=s, color='#A7BED3', zorder=10)
        ax.scatter(Y[:,0], Y[:,1], s=s, color='#1A254B', zorder=10)
        CS = ax.contourf(X1, X2, Z, cmap='Blues')
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        ax = axs[3]
        s = 5
        ax.scatter(X[:,0], X[:,1], s=s, color='#A7BED3', zorder=10)
        ax.scatter(Y[:,0], Y[:,1], s=s, color='#1A254B', zorder=10)
        CS = ax.contourf(X1, X2, Z_conj, cmap='Blues')
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        fig.tight_layout()
        for ax in axs:
            ax.grid(False)
            ax.set_axis_off()

        fname = f'{frames_dir}/{frame_num:05d}.png'
        fig.savefig(fname, transparent=True)
        plt.close(fig)
        os.system(f'convert -trim {fname} {fname}')
        os.system(f'convert {fname} -background White -alpha remove -alpha off {fname}')
        os.system(f'convert {DIR}/transport-header.png {fname} {DIR}/transport-footer.png -resize 1200x -append {fname}')

    pad = 10
    for frame_num in range(num_interp, num_interp+pad):
        shutil.copy(
            f'{frames_dir}/{num_interp-1:05d}.png',
            f'{frames_dir}/{frame_num:05d}.png'
        )


    os.system(f"ffmpeg -i {frames_dir}/%05d.png {frames_dir}/forward.mp4 -y")

    gif_fname = f'{args.exp_root}/transport.gif'
    print(f'Saving to {gif_fname}')
    os.system(f'convert -loop 0 {frames_dir}/forward.mp4 {gif_fname}')

    shutil.rmtree(frames_dir)


def plot_conj_transport_path_video(args, dual_trainer, X, Y):
    # Can horizontally concatenate with:
    # ffmpeg -i 1/conj-transport.gif -i 2/conj-transport.gif -filter_complex hstack=inputs=2 -f image2pipe -vcodec ppm - | convert -delay 10 -loop 0 - t.gif

    frames_dir = f'{args.exp_root}/conj-transport-frames'
    if os.path.exists(frames_dir):
        shutil.rmtree(frames_dir)
    os.mkdir(frames_dir)

    key = jax.random.PRNGKey(0)
    k1, k2, key = jax.random.split(key, 3)
    # X_push = dual_trainer.push(X)
    Y_push = dual_trainer.push_inv(Y)

    num_interp = 20
    ts = np.linspace(0, 1, num=num_interp)

    XY = jnp.vstack((X, Y))*1.05
    xmin, xmax = XY[:,0].min(), XY[:,0].max()
    ymin, ymax = XY[:,1].min(), XY[:,1].max()

    for frame_num, t in enumerate(ts):
        nrow, ncol = 1, 1
        fig, ax = plt.subplots(nrow, ncol, figsize=(3*ncol,3*nrow),
                                gridspec_kw = {'wspace':0, 'hspace':0})

        # X_push_t = (1.-t)*X + t*X_push
        Y_push_t = (1.-t)*Y + t*Y_push

        # Plot the data and transported samples
        s = 5
        ax.scatter(X[:,0], X[:,1], s=s, color='#1A254B')
        ax.scatter(Y[:,0], Y[:,1], s=s, color='#A7BED3')
        ax.scatter(Y_push_t[:,0], Y_push_t[:,1], s=s, color='#F2545B')
        plot_lines(ax, Y, Y_push_t)

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        fig.tight_layout()
        ax.grid(False)
        ax.set_axis_off()

        fname = f'{frames_dir}/{frame_num:05d}.png'
        fig.savefig(fname, transparent=True)
        plt.close(fig)
        # os.system(f'convert -trim {fname} {fname}')
        os.system(f'convert {fname} -background White -alpha remove -alpha off {fname}')

    pad = 10
    for frame_num in range(num_interp, num_interp+pad):
        shutil.copy(
            f'{frames_dir}/{num_interp-1:05d}.png',
            f'{frames_dir}/{frame_num:05d}.png'
        )


    os.system(f"ffmpeg -i {frames_dir}/%05d.png {frames_dir}/forward.mp4 -y")

    gif_fname = f'{args.exp_root}/conj-transport-path.gif'
    print(f'Saving to {gif_fname}')
    os.system(f'convert -loop 0 {frames_dir}/forward.mp4 {gif_fname}')

    shutil.rmtree(frames_dir)


def plot_conj_transport_samples(args, dual_trainer, X, Y):
    nrow, ncol = 1, 7
    fig, axs = plt.subplots(nrow, ncol, figsize=(4*ncol,4*nrow),
                            gridspec_kw = {'wspace':0, 'hspace':0})
    key = jax.random.PRNGKey(0)
    k1, k2, key = jax.random.split(key, 3)

    Y_push = dual_trainer.push_inv(Y)

    XY = jnp.concatenate((X, Y), axis=0)

    xmin, xmax = XY[:,0].min(), XY[:,0].max()
    ymin, ymax = XY[:,1].min(), XY[:,1].max()

    gammas = np.linspace(0, 1, ncol)
    for i, gamma in enumerate(gammas):
        ax = axs[i]
        s = 5
        mid = gamma*Y_push + (1.-gamma)*Y
        ax.scatter(mid[:,0], mid[:,1], s=s, color='k') #, alpha=0.5)

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        for ax in axs:
            ax.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.grid(False)

    fname = f'{args.exp_root}/conj-transport-samples.png'
    fig.savefig(fname, transparent=True)
    plt.close(fig)
    os.system(f'convert -trim {fname} {fname}')
    print(f'Saving to {fname}')


def plot_conj_transport_samples_video(args, dual_trainer, X, Y):
    # Can horizontally concatenate with:
    # ffmpeg -i 1/conj-transport.gif -i 2/conj-transport.gif -filter_complex hstack=inputs=2 -f image2pipe -vcodec ppm - | convert -delay 10 -loop 0 - t.gif

    frames_dir = f'{args.exp_root}/conj-transport-samples-frames'
    if os.path.exists(frames_dir):
        shutil.rmtree(frames_dir)
    os.mkdir(frames_dir)

    key = jax.random.PRNGKey(0)
    k1, k2, key = jax.random.split(key, 3)
    # X_push = dual_trainer.push(X)
    Y_push = dual_trainer.push_inv(Y)

    num_interp = 20
    ts = np.linspace(0, 1, num=num_interp)

    XY = jnp.vstack((X, Y_push)) #*1.05
    xmin, xmax = XY[:,0].min(), XY[:,0].max()
    ymin, ymax = XY[:,1].min(), XY[:,1].max()

    for frame_num, t in enumerate(ts):
        nrow, ncol = 1, 1
        fig, ax = plt.subplots(nrow, ncol, figsize=(3*ncol,3*nrow),
                                gridspec_kw = {'wspace':0, 'hspace':0})

        # X_push_t = (1.-t)*X + t*X_push
        Y_push_t = (1.-t)*Y + t*Y_push

        # Plot the data and transported samples
        s = 5
        ax.scatter(Y_push_t[:,0], Y_push_t[:,1], s=s, color='k', alpha=0.5)

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        fig.tight_layout()
        ax.grid(False)
        ax.set_axis_off()

        fname = f'{frames_dir}/{frame_num:05d}.png'
        fig.savefig(fname, transparent=True)
        plt.close(fig)
        # os.system(f'convert -trim {fname} {fname}')
        os.system(f'convert {fname} -background White -alpha remove -alpha off {fname}')

    # pad = 5
    # for frame_num in range(num_interp, num_interp+pad):
    #     shutil.copy(
    #         f'{frames_dir}/{num_interp-1:05d}.png',
    #         f'{frames_dir}/{frame_num:05d}.png'
    #     )


    os.system(f"ffmpeg -i {frames_dir}/%05d.png {frames_dir}/forward.mp4 -y")
    os.system(f"ffmpeg -i {frames_dir}/forward.mp4 -vf reverse {frames_dir}/reverse.mp4 -y")

    gif_fname = f'{args.exp_root}/conj-transport-samples.gif'
    print(f'Saving to {gif_fname}')
    os.system(f'convert -loop 0 {frames_dir}/forward.mp4 {frames_dir}/reverse.mp4 {gif_fname}')

    shutil.rmtree(frames_dir)



if __name__ == '__main__':
    main()
