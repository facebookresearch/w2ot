# This file is from
# https://github.com/iamalexkorotin/Wasserstein1Benchmark/commit/647a1acc85f88e207733d087cbe87987cc0dea06
# and remains under the original licensing.

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from .tools import ewma, freeze

import torch
import gc

def plot_benchmark_emb(benchmark, emb_X, emb_Y, D, D_conj, size=1024):
    freeze(D); freeze(D_conj)
    fig, axes = plt.subplots(2, 2, figsize=(12, 12), dpi=100, sharex=True, sharey=True)
    
    Y = benchmark.output_sampler.sample(size); Y.requires_grad_(True)
    X = benchmark.input_sampler.sample(size); X.requires_grad_(True)

    Y_inv = emb_X.transform(D_conj.push_nograd(Y).cpu().numpy())
    Y = emb_Y.transform(Y.cpu().detach().numpy())
    
    X_push = emb_Y.transform(D.push_nograd(X).cpu().numpy())
    X = emb_X.transform(X.cpu().detach().numpy())

    axes[0, 0].scatter(X[:, 0], X[:, 1], edgecolors='black')
    axes[0, 1].scatter(Y[:, 0], Y[:, 1], edgecolors='black')
    axes[1, 0].scatter(Y_inv[:, 0], Y_inv[:, 1], c='peru', edgecolors='black')
    axes[1, 1].scatter(X_push[:, 0], X_push[:, 1], c='peru', edgecolors='black')

    axes[0, 0].set_title(r'Ground Truth Input $\mathbb{P}$', fontsize=12)
    axes[0, 1].set_title(r'Ground Truth Output $\mathbb{Q}$', fontsize=12)
    axes[1, 0].set_title(r'Inverse Map $\nabla\overline{\psi_{\omega}}\circ\mathbb{Q}$', fontsize=12)
    axes[1, 1].set_title(r'Forward Map $\nabla\psi_{\theta}\circ\mathbb{P}$', fontsize=12)
    
    fig.tight_layout()
    torch.cuda.empty_cache(); gc.collect()
    return fig, axes

def plot_W2(benchmark, W2):
    fig, ax = plt.subplots(1, 1, figsize=(12, 3), dpi=100)
    ax.set_title('Wasserstein-2', fontsize=12)
    ax.plot(ewma(W2), c='blue', label='Estimated Cost')
    if hasattr(benchmark, 'linear_cost'):
        ax.axhline(benchmark.linear_cost, c='orange', label='Bures-Wasserstein Cost')
    if hasattr(benchmark, 'cost'):
        ax.axhline(benchmark.cost, c='green', label='True Cost')    
    ax.legend(loc='lower right')
    fig.tight_layout()
    return fig, ax

def plot_benchmark_metrics(benchmark, metrics, baselines=None):
    fig, axes = plt.subplots(2, 2, figsize=(12,6), dpi=100)
    cmap = {'identity' : 'red', 'linear' : 'orange', 'constant' : 'magenta'}
    
    if baselines is None:
        baselines = {}
    
    for i, metric in enumerate(["L2_UVP_fwd", "L2_UVP_inv", "cos_fwd", "cos_inv"]):
        axes.flatten()[i].set_title(metric, fontsize=12)
        axes.flatten()[i].plot(ewma(metrics[metric]), label='Fitted Transport')
        for baseline in cmap.keys():
            if not baseline in baselines.keys():
                continue
            axes.flatten()[i].axhline(baselines[baseline][metric], label=f'{baseline} baseline', c=cmap[baseline])
            
        axes.flatten()[i].legend(loc='upper left')
    
    fig.tight_layout()
    return fig, axes

def vecs_to_plot(input, shape=(3, 64, 64)):
    return input.reshape(-1, *shape).permute(0, 2, 3, 1).mul(0.5).add(0.5).cpu().numpy().clip(0, 1)


def _plot_images(X, Y, D, D_conj, shape=(3, 64, 64)):
    freeze(D); freeze(D_conj)
    
    X_push = D.push_nograd(X); X_push.requires_grad_(True)
    X_push_inv = D_conj.push_nograd(X_push).cpu()
    X_push = X_push.cpu().detach()
    
    Y_push = D_conj.push_nograd(Y); Y_push.requires_grad_(True)
    Y_push_inv = D.push_nograd(Y_push).cpu()
    Y_push = Y_push.cpu().detach()
    
    with torch.no_grad():
        vecs = torch.cat([X.cpu(), X_push, X_push_inv, Y.cpu(), Y_push, Y_push_inv])
    imgs = vecs_to_plot(vecs)

    fig, axes = plt.subplots(6, 10, figsize=(12, 7), dpi=100)
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(imgs[i])
        ax.get_xaxis().set_visible(False)
        ax.set_yticks([])
        
    axes[0, 0].set_ylabel('X', fontsize=12)
    axes[1, 0].set_ylabel('Pushed X', fontsize=12)
    axes[2, 0].set_ylabel('Inverse\nPushed X', fontsize=12)

    axes[3, 0].set_ylabel('Y', fontsize=12)
    axes[4, 0].set_ylabel('Pushed Y', fontsize=12)
    axes[5, 0].set_ylabel('Inverse\nPushed Y', fontsize=12)
    fig.tight_layout()
    
    return fig, axes

def plot_benchmark_images(benchmark, D, D_conj, shape=(3, 64, 64)):
    X = benchmark.input_sampler.sample(10); X.requires_grad_(True)
    Y = benchmark.output_sampler.sample(10); Y.requires_grad_(True)
    return _plot_images(X, Y, D, D_conj, shape=shape)


def plot_generated_images(G, Z_sampler, Y_sampler, D, D_conj, shape=(3, 64, 64)):
    freeze(G);
    Z = Z_sampler.sample(10);
    X = G(Z).reshape(-1, np.prod(shape)).detach(); X.requires_grad_(True)
    Y = Y_sampler.sample(10); Y.requires_grad_(True)
    return _plot_images(X, Y, D, D_conj, shape=shape)

def plot_generated_emb(Z_sampler, G, Y_sampler, D, D_conj, emb, size=512, n_pairs=3, show=True):
    freeze(G); freeze(D); freeze(D_conj)
    Z = Z_sampler.sample(size)
    with torch.no_grad():
        X = G(Z).reshape(len(Z), -1)
    X.requires_grad_(True)
    pca_X = emb.transform(X.cpu().detach().numpy().reshape(len(X), -1))
    pca_D_push_X = emb.transform(D.push_nograd(X).cpu().numpy().reshape(len(X), -1))
    
    Y = Y_sampler.sample(size)
    Y.requires_grad_(True)
    pca_Y = emb.transform(Y.cpu().detach().numpy().reshape(len(Y), -1))
    pca_D_conj_push_Y = emb.transform(D_conj.push_nograd(Y).cpu().numpy())
    
    fig, axes = plt.subplots(n_pairs, 4, figsize=(12, 3 * n_pairs), sharex=True, sharey=True)
    
    for n in range(n_pairs):
        axes[n, 0].set_ylabel(f'Component {2*n+1}', fontsize=12)
        axes[n, 0].set_xlabel(f'Component {2*n}', fontsize=12)
        axes[n, 0].set_title(f'G(Z)', fontsize=15)
        axes[n, 1].set_title('D.push(G(Z))', fontsize=15)
        axes[n, 2].set_title('Y', fontsize=15)
        axes[n, 3].set_title('D_conj.push(Y)', fontsize=15)

        axes[n, 0].scatter(pca_X[:, 2*n], pca_X[:, 2*n+1], color='b', alpha=0.5)
        axes[n, 1].scatter(pca_D_push_X[:, 2*n], pca_D_push_X[:, 2*n+1], color='r', alpha=0.5)
        axes[n, 2].scatter(pca_Y[:, 2*n], pca_Y[:, 2*n+1], color='g', alpha=0.5)
        axes[n, 3].scatter(pca_D_conj_push_Y[:, 2*n], pca_D_conj_push_Y[:, 2*n+1], color='orange', alpha=0.5)
        
    fig.tight_layout()
    
    torch.cuda.empty_cache(); gc.collect()
    return fig, axes
