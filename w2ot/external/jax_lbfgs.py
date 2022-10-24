# This file is modified from
# https://github.com/google/jax/blob/ba557d5/jax/_src/scipy/optimize/_lbfgs.py
# and remains under the original licensing.
#
#
# Copyright 2020 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from typing import Any, Callable, NamedTuple, Optional, Union
from functools import partial

import jax
import jax.numpy as jnp
from jax import lax
from jax._src.scipy.optimize.line_search import line_search
from jax.experimental.host_callback import call

_dot = partial(jnp.dot, precision=lax.Precision.HIGHEST)

Array = Any

class LBFGSResults(NamedTuple):
  """Results from L-BFGS optimization

  Parameters:
    converged: True if minimization converged
    failed: True if non-zero status and not converged
    k: integer number of iterations of the main loop (optimisation steps)
    nfev: integer total number of objective evaluations performed.
    ngev: integer total number of jacobian evaluations
    x_k: array containing the last argument value found during the search. If
      the search converged, then this value is the argmin of the objective
      function.
    f_k: array containing the value of the objective function at `x_k`. If the
      search converged, then this is the (local) minimum of the objective
      function.
    g_k: array containing the gradient of the objective function at `x_k`. If
      the search converged the l2-norm of this tensor should be below the
      tolerance.
    status: integer describing the status:
      0 = nominal  ,  1 = max iters reached  ,  2 = max fun evals reached
      3 = max grad evals reached  ,  4 = insufficient progress (ftol)
      5 = line search failed
    ls_status: integer describing the end status of the last line search
  """
  converged: Union[bool, Array]
  failed: Union[bool, Array]
  k: Union[int, Array]
  nfev: Union[int, Array]
  ngev: Union[int, Array]
  x_k: Array
  f_k: Array
  g_k: Array
  s_history: Array
  y_history: Array
  rho_history: Array
  gamma: Union[float, Array]
  status: Union[int, Array]
  ls_status: Union[int, Array]
  f_hist: Optional[Array]


def _minimize_lbfgs(
    fun: Callable,
    x0: Array,
    maxiter: Optional[float] = None,
    norm=jnp.inf,
    maxcor: int = 10,
    ftol: float = 2.220446049250313e-09,
    gtol: float = 1e-05,
    maxfun: Optional[float] = None,
    maxgrad: Optional[float] = None,
    track_hist: bool = False,
    ls_method: str = 'wolfe',
    ls_kwargs: dict = field(default_factory=dict),
):
  """
  Minimize a function using L-BFGS

  Implements the L-BFGS algorithm from
    Algorithm 7.5 from Wright and Nocedal, 'Numerical Optimization', 1999, pg. 176-185
  And generalizes to complex variables from
     Sorber, L., Barel, M.V. and Lathauwer, L.D., 2012.
     "Unconstrained optimization of real functions in complex variables"
     SIAM Journal on Optimization, 22(3), pp.879-898.

  Args:
    fun: function of the form f(x) where x is a flat ndarray and returns a real scalar.
      The function should be composed of operations with vjp defined.
    x0: initial guess
    maxiter: maximum number of iterations
    norm: order of norm for convergence check. Default inf.
    maxcor: maximum number of metric corrections ("history size")
    ftol: terminates the minimization when `(f_k - f_{k+1}) < ftol`
    gtol: terminates the minimization when `|g_k|_norm < gtol`
    maxfun: maximum number of function evaluations
    maxgrad: maximum number of gradient evaluations

  Returns:
    Optimization results.
  """
  d = len(x0)
  dtype = jnp.dtype(x0)

  # ensure there is at least one termination condition
  if (maxiter is None) and (maxfun is None) and (maxgrad is None):
    maxiter = d * 200

  # set others to inf, such that >= is supported
  if maxiter is None:
    maxiter = jnp.inf
  if maxfun is None:
    maxfun = jnp.inf
  if maxgrad is None:
    maxgrad = jnp.inf

  # initial evaluation
  f_0, g_0 = jax.value_and_grad(fun)(x0)
  if track_hist:
    f_hist = jnp.zeros((maxiter))
    f_hist = f_hist.at[0].set(f_0)
  else:
      f_hist = None

  state_initial = LBFGSResults(
    converged=False,
    failed=False,
    k=0,
    nfev=1,
    ngev=1,
    x_k=x0,
    f_k=f_0,
    g_k=g_0,
    s_history=jnp.zeros((maxcor, d), dtype=dtype),
    y_history=jnp.zeros((maxcor, d), dtype=dtype),
    rho_history=jnp.zeros((maxcor,), dtype=dtype),
    gamma=1.,
    status=0,
    ls_status=0,
    f_hist=f_hist,
  )

  def cond_fun(state: LBFGSResults):
    return (~state.converged) & (~state.failed)

  def body_fun(state: LBFGSResults):
    p_k = _two_loop_recursion(state) # search direction

    if ls_method == 'wolfe':
      maxls = ls_kwargs.get('maxls', 20)
      ls_results = line_search(
        f=fun,
        xk=state.x_k,
        pk=p_k,
        old_fval=state.f_k,
        gfk=state.g_k,
        maxiter=maxls,
      )
      s_k = ls_results.a_k.astype(p_k.dtype) * p_k
    elif ls_method == 'armijo':
      ls_base = ls_kwargs.get('ls_base', 1.5)
      armijo_gamma = ls_kwargs.get('armijo_gamma', 1e-4)
      parallel = ls_kwargs.get('parallel', True)
      num_evaluations = ls_kwargs.get('num_evaluations', 10)
      armijo_line_search = armijo_line_search_parallel if parallel else armijo_line_search_backtracking
      ls_results = armijo_line_search(
        f=fun,
        xk=state.x_k,
        pk=p_k,
        old_fval=state.f_k,
        gfk=state.g_k,
        ls_base=ls_base,
        armijo_gamma=armijo_gamma,
        num_evaluations=num_evaluations,
      )
      s_k = ls_results.s_k
    else:
        raise NotImplementedError

    # evaluate at next iterate
    x_kp1 = state.x_k + s_k
    f_kp1 = ls_results.f_k
    g_kp1 = ls_results.g_k
    y_k = g_kp1 - state.g_k
    rho_k_inv = jnp.real(_dot(y_k, s_k))
    rho_k = jnp.reciprocal(rho_k_inv).astype(y_k.dtype)
    gamma = rho_k_inv / jnp.real(_dot(jnp.conj(y_k), y_k))

    # replacements for next iteration
    status = 0
    status = jnp.where(state.f_k - f_kp1 < ftol, 4, status)
    status = jnp.where(state.ngev >= maxgrad, 3, status)  # type: ignore
    status = jnp.where(state.nfev >= maxfun, 2, status)  # type: ignore
    status = jnp.where(state.k >= maxiter, 1, status)  # type: ignore
    status = jnp.where(ls_results.failed, 5, status)

    # Add these for when the line search returns a bad update
    # TODO(bda): Better-handle when approximation becomes indefinite
    status = jnp.where(jnp.isnan(gamma.sum()), 6, status)
    status = jnp.where(jnp.isnan(x_kp1.sum()), 7, status)
    x_kp1 = jnp.where(jnp.isnan(x_kp1.sum()), state.x_k, x_kp1)

    converged = jnp.linalg.norm(g_kp1, ord=norm) < gtol

    if track_hist:
      next_f_hist = state.f_hist.at[state.k+1].set(ls_results.f_k)
    else:
      next_f_hist = None


    # TODO(jakevdp): use a fixed-point procedure rather than type-casting?
    state = state._replace(
      converged=converged,
      failed=(status > 0) & (~converged),
      k=state.k + 1,
      nfev=state.nfev + ls_results.nfev,
      ngev=state.ngev + ls_results.ngev,
      x_k=x_kp1.astype(state.x_k.dtype),
      f_k=f_kp1.astype(state.f_k.dtype),
      g_k=g_kp1.astype(state.g_k.dtype),
      s_history=_update_history_vectors(history=state.s_history, new=s_k),
      y_history=_update_history_vectors(history=state.y_history, new=y_k),
      rho_history=_update_history_scalars(history=state.rho_history, new=rho_k),
      gamma=gamma,
      status=jnp.where(converged, 0, status),
      ls_status=ls_results.status,
      f_hist=next_f_hist,
    )

    return state

  state = lax.while_loop(cond_fun, body_fun, state_initial)
  # if track_hist:
  #   state = state._replace(f_hist=state.f_hist[:state.k+1])

  return state


class LineSearchState(NamedTuple):
    iteration: int
    a_proposal: float
    s_k: float
    f: float

class LineSearchResults(NamedTuple):
    failed: Union[bool, jnp.ndarray]
    nfev: Union[int, jnp.ndarray]
    ngev: Union[int, jnp.ndarray]
    a_k: Union[int, jnp.ndarray]
    f_k: jnp.ndarray
    g_k: jnp.ndarray
    s_k: jnp.ndarray
    status: Union[bool, jnp.ndarray]


def armijo_line_search_backtracking(f, xk, pk, old_fval, gfk,
                                    armijo_gamma=1e-4, num_evaluations=10,
                                    ls_base=1.5):
    def check_satisfied(state):
        sufficient_decrease_val = old_fval + armijo_gamma * state.a_proposal * gfk.dot(pk)
        satisfied = state.f < sufficient_decrease_val
        return satisfied

    def cond_fun(state):
        return (~check_satisfied(state)) & (state.iteration < num_evaluations)

    def body_fun(state):
        new_a_proposal = ls_base ** (-state.iteration)
        new_s_k = new_a_proposal*pk
        new_f = f(xk+new_s_k)
        return LineSearchState(
            iteration=state.iteration+1,
            a_proposal=new_a_proposal,
            s_k=new_s_k,
            f=new_f,
        )

    state_initial = LineSearchState(iteration=0, a_proposal=1., s_k=pk, f=f(xk+pk))
    out = lax.while_loop(cond_fun, body_fun, state_initial)
    failed = ~check_satisfied(out)

    a_k = (1-failed)*out.a_proposal
    s_k = (1-failed)*out.s_k
    f_k = (1-failed)*out.f + failed*old_fval
    g_k = (1-failed)*jax.grad(f)(xk+s_k) + failed*gfk

    return LineSearchResults(
        failed=failed,
        nfev=out.iteration+1,
        ngev=out.iteration+1,
        a_k=a_k,
        f_k=f_k,
        g_k=g_k,
        s_k=s_k,
        status=0,
    )

def armijo_line_search_parallel(f, xk, pk, old_fval, gfk,
                                armijo_gamma=1e-4, num_evaluations=10,
                                ls_base=1.5):
    a_proposals = float(ls_base)**(-jnp.flip(jnp.arange(num_evaluations)))
    proposed_steps = jnp.expand_dims(a_proposals, 1) * jnp.expand_dims(pk, 0)
    fs = jax.vmap(f)(xk + proposed_steps)

    sufficient_decrease_vals = old_fval + armijo_gamma * a_proposals * gfk.dot(pk)
    armijo_satisfied = (fs < sufficient_decrease_vals)
    failed = (~armijo_satisfied).all()

    # Take the largest step size satisfying the armijo condition,
    # or return the previous iterate if it failed.
    I = jnp.argmax(a_proposals * armijo_satisfied)
    a_k = (1-failed)*a_proposals[I]
    s_k = (1-failed)*proposed_steps[I]
    f_k = (1-failed)*fs[I] + failed*old_fval
    g_k = (1-failed)*jax.grad(f)(xk+s_k) + failed*gfk

    return LineSearchResults(
        failed=failed,
        nfev=num_evaluations,
        ngev=num_evaluations,
        a_k=a_k,
        f_k=f_k,
        g_k=g_k,
        s_k=s_k,
        status=0,
    )


def _two_loop_recursion(state: LBFGSResults):
  dtype = state.rho_history.dtype
  his_size = len(state.rho_history)
  curr_size = jnp.where(state.k < his_size, state.k, his_size)
  q = -jnp.conj(state.g_k)
  a_his = jnp.zeros_like(state.rho_history)

  def body_fun1(j, carry):
    i = his_size - 1 - j
    _q, _a_his = carry
    a_i = state.rho_history[i] * _dot(jnp.conj(state.s_history[i]), _q).real.astype(dtype)
    _a_his = _a_his.at[i].set(a_i)
    _q = _q - a_i * jnp.conj(state.y_history[i])
    return _q, _a_his

  q, a_his = lax.fori_loop(0, curr_size, body_fun1, (q, a_his))
  q = state.gamma * q

  def body_fun2(j, _q):
    i = his_size - curr_size + j
    b_i = state.rho_history[i] * _dot(state.y_history[i], _q).real.astype(dtype)
    _q = _q + (a_his[i] - b_i) * state.s_history[i]
    return _q

  q = lax.fori_loop(0, curr_size, body_fun2, q)
  return q


def _update_history_vectors(history, new):
  # TODO(Jakob-Unfried) use rolling buffer instead? See #6053
  return jnp.roll(history, -1, axis=0).at[-1, :].set(new)


def _update_history_scalars(history, new):
  # TODO(Jakob-Unfried) use rolling buffer instead? See #6053
  return jnp.roll(history, -1, axis=0).at[-1].set(new)
