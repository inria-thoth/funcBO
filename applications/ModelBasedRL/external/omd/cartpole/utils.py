import numpy as np
from functools import partial
from collections import namedtuple

import jax
import jax.numpy as jnp
import optax
from jax.scipy.sparse.linalg import cg
from jax.lax import stop_gradient

import haiku as hk



def evaluate(agent, eval_env, rng, num_eval_episodes=10):
  average_episode_reward = 0
  for episode in range(num_eval_episodes):
    obs = eval_env.reset()
    done = False
    episode_reward = 0
    while not done:
      rng, _ = jax.random.split(rng)
      action = agent.act(agent.params_Q, obs, rng).item()
      obs, reward, done, _ = eval_env.step(action)
      episode_reward += reward
    average_episode_reward += episode_reward
  average_episode_reward /= num_eval_episodes
  return average_episode_reward

@partial(jax.custom_vjp, nondiff_argnums=(0, 5))
def root_solve(param_func, init_xs, params, replay, rng, solvers):
  # to mimic two_phase_solve API
  fwd_solver = solvers[0]
  return fwd_solver( init_xs, params, replay, rng)

def root_solve_fwd(param_func, init_xs, params, replay, rng, solvers):
  sol = root_solve(param_func, init_xs, params, replay, rng, solvers)
  tpQ = jax.lax.stop_gradient(sol.target_params_Q)
  return sol, (sol.params_Q, params, replay, rng, tpQ)

def root_solve_bwd(param_func, solvers, res, g):
  pQ, params, replay, rng, tpQ = res
  _, vdp_fun = jax.vjp(lambda y: param_func(y, pQ, replay, rng, tpQ), params)
  g_main = g[0] if isinstance(g, tuple) else g
  # if args.with_inv_jac:
  #   # _, vds_fun = jax.vjp(lambda x: param_func(params, x), pQ)
  #   # (J)^-1 -> (J+cI)^-1
  #   _, vds_fun = jax.vjp(lambda x: jax.tree_map(
  #     lambda y,z: y + 1e-5*z, param_func(params, x, replay, rng, tpQ), x), pQ)
  #   vdsinv = cg(lambda z: vds_fun(z)[0], g_main, maxiter=100)[0]
  #   vdp = vdp_fun(vdsinv)[0]
  # else:
  vdp = vdp_fun(g_main)[0]
  z_sol, z_replay, z_rng = jax.tree_map(jnp.zeros_like, (pQ, replay, rng))
  return z_sol, jax.tree_map(lambda x: -x, vdp), z_replay, z_rng

root_solve.defvjp(root_solve_fwd, root_solve_bwd)


InnnerSol = namedtuple('InnerSol', 'val_Q val_target_Q loss_Q vals_Q grad_norm_Q entropy_Q params_Q target_params_Q opt_state_Q next_obs_nll')

@partial(jax.custom_vjp, nondiff_argnums=(0,1,7))
def inner_solution(inner_loss, inner_model, init_xs, params, replay, rng, params_dual_Q, solvers):
  # to mimic two_phase_solve API
  fwd_solver = solvers[0]
  sol = fwd_solver(init_xs, params, replay, rng)

  obs, action, reward, next_obs, not_done, not_done_no_max = replay

  val_Q = inner_model.apply(jax.lax.stop_gradient(sol.params_Q), obs)
  val_target_Q = inner_model.apply(jax.lax.stop_gradient(sol.target_params_Q), next_obs)  
  


  return InnnerSol(val_Q, val_target_Q, sol.loss_Q, sol.vals_Q,
              sol.grad_norm_Q, sol.entropy_Q, sol.params_Q, sol.target_params_Q, 
              sol.opt_state_Q, sol.next_obs_nll)





def argmin_fwd(inner_loss, inner_model, init_xs, params, replay, rng, params_dual_Q, solvers):
  sol = inner_solution(inner_loss, inner_model, init_xs, params, replay, rng, params_dual_Q , solvers)
  tpQ = jax.lax.stop_gradient(sol.target_params_Q)
  val_target_Q = jax.lax.stop_gradient(sol.val_target_Q)
  val_Q = jax.lax.stop_gradient(sol.val_Q)
  return sol, (val_Q, init_xs, params, replay, rng, val_target_Q, params_dual_Q)

def argmin_bwd(inner_loss, inner_model, solvers, res, g):
  pQ, init_xs, params, replay, rng, tpQ, params_dual_Q = res
  g_main = g[0] if isinstance(g, tuple) else g
  bwd_solver = solvers[1]
  sol = bwd_solver(params_dual_Q, replay, rng , g_main)

  _, vdp_fun = jax.vjp(lambda y: inner_loss(y, pQ, replay, rng, tpQ), params)
  

  
  vdp = vdp_fun(sol.val_dual_Q)[0]
  

  ####### Define gradients for the things that require a gradient: TODO

  z_sol, z_replay, z_rng, z_dual = jax.tree_map(jnp.zeros_like, (init_xs, replay, rng, params_dual_Q))
  return z_sol, jax.tree_map(lambda x: -x, vdp), z_replay, z_rng, z_dual


def argmin_bwd_exact(inner_loss, inner_model, solvers, res, g):
  pQ, init_xs, params, replay, rng, tpQ, params_dual_Q = res
  g_main = g[0] if isinstance(g, tuple) else g
  #sol = bwd_solver(params_dual_Q, pQ, tpQ, params, replay, rng , g_main)

  _, vdp_fun = jax.vjp(lambda y: inner_loss(y, pQ, replay, rng, tpQ), params)
  
  vdp = vdp_fun(g_main)[0]
  

  ####### Define gradients for the things that require a gradient: TODO

  z_sol, z_replay, z_rng, z_dual = jax.tree_map(jnp.zeros_like, (init_xs, replay, rng, params_dual_Q))
  return z_sol, jax.tree_map(lambda x: -x, vdp), z_replay, z_rng, z_dual



#inner_solution.defvjp(root_solve_fwd, root_solve_bwd)
inner_solution.defvjp(argmin_fwd, argmin_bwd)
















def add_dict(d, k, v):
  if not isinstance(v, list):
    v = [v]
  if k in d:
    d[k].extend(v)
  else:
    d[k] = v

@jax.jit
def soft_update_params(tau, params, target_params):
  return jax.tree_map(
    lambda p, tp: tau * p + (1 - tau) * tp, 
    params, target_params)

@jax.jit
def tree_norm(tree):
  return jnp.sqrt(sum((x**2).sum() for x in jax.tree_util.tree_leaves(tree)))

def net_fn(args, net_type, dims, x):
  obs_dim, action_dim, hidden_dim = dims
  activation = jax.nn.relu
  init = hk.initializers.Orthogonal(scale=jnp.sqrt(2.0))
  layers = [
    hk.Linear(hidden_dim, w_init=init), activation,
    hk.Linear(hidden_dim, w_init=init), activation,
  ]
  final_init = hk.initializers.Orthogonal(scale=1e-2)
  # T -- model, Q and V -- value functions
  if args.agent_type == 'vep':
    if net_type == 'V':
      ensemble = []
      for i in range(args.num_ensemble_vep):
        layers = [
          hk.Linear(hidden_dim, w_init=init), activation,
          hk.Linear(hidden_dim, w_init=init), activation,
          hk.Linear(1, w_init=final_init)
        ]
        mlp = hk.Sequential(layers)
        ensemble.append(mlp(x))
      return ensemble
  if net_type == 'V':
    layers += [hk.Linear(1, w_init=final_init)] 
  elif net_type == 'Q' or net_type == 'dual_Q':
    layers += [hk.Linear(action_dim, w_init=final_init)]
  elif net_type == 'T':
    out_dim = 2 * obs_dim if args.prob_model else obs_dim
    layers += [hk.Linear(out_dim, w_init=final_init)]

  if (net_type == 'Q' or net_type == 'dual_Q') and not args.no_double:
    layers2 = [
      hk.Linear(hidden_dim, w_init=init), activation,
      hk.Linear(hidden_dim, w_init=init), activation,
      hk.Linear(action_dim, w_init=final_init)
    ]
    mlp1, mlp2 = hk.Sequential(layers), hk.Sequential(layers2)
    return mlp1(x), mlp2(x)
  elif net_type == 'T' and not args.no_learn_reward:
    layers2 = [
      hk.Linear(hidden_dim, w_init=init), activation,
      hk.Linear(hidden_dim, w_init=init), activation,
      hk.Linear(1, w_init=final_init)
    ]
    mlp1, mlp2 = hk.Sequential(layers), hk.Sequential(layers2)
    return mlp1(x), mlp2(x)
  else:
    mlp = hk.Sequential(layers)
    return mlp(x)

def init_net_opt(args, net_type, dims):
  net = hk.without_apply_rng(hk.transform(partial(net_fn, args, net_type, dims)))
  if net_type == 'Q':
    opt = optax.adam(args.inner_lr)
  elif net_type=='dual_Q':
    opt = optax.adam(args.dual_lr)
  else:
    opt = optax.adam(args.lr)
  Model = namedtuple(net_type, 'net opt')
  return Model(net, opt)
