#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.

import hydra

import csv
import time

import warnings
warnings.filterwarnings('ignore')

from collections import defaultdict
import copy

import numpy as np

import pickle as pkl

import jax
import jax.numpy as jnp
import optax

import os
import sys
import shutil

from setproctitle import setproctitle
setproctitle('w2ot')

try:
    if os.isatty(sys.stdout.fileno()):
        from IPython.core import ultratb
        sys.excepthook = ultratb.FormattedTB(
            mode='Plain', color_scheme='Linux', call_pdb=1)
except:
    pass

from flax.training.checkpoints import save_checkpoint, restore_checkpoint

from w2ot.utils import RunningAverageMeter
from w2ot import data
from w2ot.dual_trainer import DualTrainer
from w2ot.amortization import BaseAmortization, RegressionAmortization, W2GNAmortization

DIR = os.path.dirname(os.path.realpath(__file__))


class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        self.train_iter = 0
        self.output = 0

        print('instantiating data loader')
        self.data = hydra.utils.instantiate(self.cfg.data)
        self.cfg.dual_trainer.input_dim = self.data.input_dim
        self.set_defaults()

        self.key = jax.random.PRNGKey(self.cfg.seed)
        k1, self.key = jax.random.split(self.key, 2)
        self.dual_trainer = hydra.utils.instantiate(
            self.cfg.dual_trainer,
            key=k1,
            data_bounds=self.data.bounds,
            conjugate_solver=self.cfg.conjugate_solver,
            batch_size=self.data.batch_size,
            amortization=self.cfg.amortization
        )

        self.loss_meter = RunningAverageMeter()
        self.train_iter = 0

        self.train_states = self.dual_trainer.init_train_state()
        self.aux_meters = defaultdict(RunningAverageMeter)
        self.best_dual_obj = None

        self.elapsed_time_s = 0.
        print('initialization finished')


    def run(self):
        print('starting train loop')
        self.last_time = time.time()
        update_fn = self.dual_trainer.create_update_fn()

        batch_size = self.data.batch_size

        dual_key = jax.random.PRNGKey(0)
        X_dual = self.data.X_sampler.sample(dual_key, batch_size)
        Y_dual = self.data.Y_sampler.sample(dual_key, batch_size)

        writer = None
        while self.train_iter < self.dual_trainer.num_train_iter:
            k1, k2, k3, self.key = jax.random.split(self.key, 4)
            X = self.data.X_sampler.sample(k1, batch_size)
            Y = self.data.Y_sampler.sample(k2, batch_size)
            loss, aux, *self.train_states = update_fn(X, Y, *self.train_states)

            self.loss_meter.update(loss.item())
            for k, v in aux.items():
                self.aux_meters[k].update(v.item())
            self.train_iter += 1

            if self.train_iter == 1 or self.train_iter % self.cfg.log_freq == 0:
                self.dual_trainer.store_params(*self.train_states)
                dual_obj = self.dual_trainer.dual_obj(X_dual, Y_dual)
                current_time = time.time()
                time_chunk = current_time - self.last_time
                self.last_time = current_time
                self.elapsed_time_s += time_chunk
                stats_dict = {
                    'iter': self.train_iter, 'train_loss': self.loss_meter.avg,
                    'dual_obj': dual_obj,
                    'elapsed_time_s': self.elapsed_time_s,
                }
                stats_str = f'iter={self.train_iter} train_loss={self.loss_meter.avg:.2e} dual_obj={dual_obj:.2e}'
                for k, meter in self.aux_meters.items():
                    stats_str += f' {k}={meter.avg:.2e}'
                    stats_dict[k] = meter.avg

                conj_time = self.profile_conj()
                stats_dict['conj_time'] = conj_time
                stats_str += f' conj_time: {conj_time:.2e}s'

                stats_str += f' time: {self.elapsed_time_s:.2e}s'
                print(stats_str)

                loc = f'{self.train_iter:06d}.png' if self.cfg.save_all_plots else 'latest.png'
                self.data.plot(self.dual_trainer, loc=loc)
                stats_dict.update(self.data.eval_extra(self.dual_trainer))
                if writer is None:
                    logf, writer = self._init_logging(fieldnames=stats_dict.keys())
                writer.writerow(stats_dict)
                logf.flush()

                if self.cfg.save_all_plots and os.path.exists(loc):
                    shutil.copy(loc, 'latest.png')

                self.save()
                if self.best_dual_obj is None or dual_obj > self.best_dual_obj:
                    self.best_dual_obj = dual_obj
                    with open('best_dual_obj', 'w') as f:
                        f.write(str(self.best_dual_obj))
                    self.save('best')

                if jnp.abs(loss.item()) > 1e5 or jnp.isnan(loss.item()):
                    print('exiting early, loss is too high')
                    import sys; sys.exit(-1)


    def set_defaults(self):
        if not isinstance(self.data, data.BenchmarkImagePair) and self.cfg.dual_trainer.D.dim_hidden is None:
            if isinstance(self.data, data.BenchmarkHDPair):
                base_sz = max(self.data.input_dim, 32)
                self.cfg.dual_trainer.D.dim_hidden = [2*base_sz, 2*base_sz, base_sz]
                if self.cfg.dual_trainer.H is not None and \
                        self.cfg.dual_trainer.H.dim_hidden is None:
                    self.cfg.dual_trainer.H.dim_hidden = [2*base_sz, 2*base_sz, base_sz]
            else:
                self.cfg.dual_trainer.D.dim_hidden = [512, 512]
                if self.cfg.dual_trainer.H is not None and \
                        self.cfg.dual_trainer.H.dim_hidden is None:
                    self.cfg.dual_trainer.H.dim_hidden = [512, 512]
            print(f'dual_trainer.D.dim_hidden: {self.cfg.dual_trainer.D.dim_hidden}')
            if self.cfg.dual_trainer.H is not None:
                print(f'dual_trainer.H.dim_hidden: {self.cfg.dual_trainer.H.dim_hidden}')

        if 'W2GN' in self.cfg['amortization']['_target_'] and \
                self.cfg['amortization']['cycle_loss_weight'] is None:
            if isinstance(self.data, data.BenchmarkImagePair):
                w = 1e4
            else:
                w = self.data.input_dim

            self.cfg['amortization']['cycle_loss_weight'] = w

            print(f'Defaulting cycle loss weight for W2GN amortization to {w}')


    def profile_conj(self, num_trials=1):
        times = []

        for i in range(num_trials+1):
            k1, self.key = jax.random.split(self.key, 2)
            Y_batch = self.data.Y_sampler.sample(
                k1, self.data.batch_size).block_until_ready()
            start = time.time()
            X_hat = self.dual_trainer.push_inv(
                Y_batch, warmstart=True).block_until_ready()
            if i > 0: # Warmup with first iterate
                times.append(time.time() - start)

        return np.mean(times)


    def save(self, tag='latest'):
        self.dual_trainer.store_params(*self.train_states)

        self.checkpoint_tag = tag
        if not os.path.exists('ckpt'):
            os.mkdir('ckpt')
        save_checkpoint(
            'ckpt', self.train_states, 0, prefix=tag, keep=1, overwrite=True)
        path = os.path.join(self.work_dir, f'{tag}.pkl')
        with open(path, 'wb') as f:
            pkl.dump(self, f)


    def _init_logging(self, fieldnames):
        logf = open('log.csv', 'a')
        writer = csv.DictWriter(logf, fieldnames=fieldnames)
        if os.stat('log.csv').st_size == 0:
            writer.writeheader()
            logf.flush()
        return logf, writer


    def __getstate__(self):
        d = copy.copy(self.__dict__)
        del d['train_states']
        return d


    def __setstate__(self, d):
        self.__dict__ = d

        if not hasattr(self, 'dual_trainer'):
            self.dual_trainer = self.maximin_trainer

        target_states = self.dual_trainer.init_train_state(pretrain=False)
        self.train_states = restore_checkpoint(
            'ckpt', prefix=self.checkpoint_tag, target=target_states)


@hydra.main(config_path='../config', config_name='train.yaml')
def main(cfg):
    from w2ot.run_train import Workspace as W # For pickling
    fname = os.getcwd() + '/latest.pkl'
    if os.path.exists(fname):
        print(f'Resuming fom {fname}')
        with open(fname, 'rb') as f:
            workspace = pkl.load(f)
    else:
        workspace = W(cfg)

    workspace.run()


if __name__ == '__main__':
    main()
