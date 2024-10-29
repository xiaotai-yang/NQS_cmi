
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
import jax.random as random
from jax.random import PRNGKey, split, categorical
import jax.lax as lax
from jax.lax import scan
import jax.nn as nn
import time
from tqdm import tqdm
from functools import partial
from jax.nn.initializers import he_normal, he_uniform, glorot_normal, glorot_uniform
from model.model_utlis import *

@jax.jit
def layer_norm_T(x, w, b):
    mean = jnp.mean(x)
    std =  jnp.sqrt((jnp.sum((x - mean)**2) + 1e-8)/(x.size-1))
    return (x - mean)/ std * w + b
@jax.jit
def linear(x, w, b):
    return w@x+b
@jax.jit
def attention(q, k, v, loc, mask):
    a = q.shape[0]
    QKV = nn.softmax(jnp.matmul(q, k.T)/jnp.sqrt(a) + mask[loc]) @ v
    return QKV

@partial(jax.jit, static_argnames=("num_layer"))
def TF_step(x, K, V, loc, num_layer, params, mask):
    (Wemb, Wi, bi, Wq, bq, Wk, bk, Wv, bv, Wo, bo, a1, a2, b1, b2, Wfh, bfh, Whf, bhf, Whh1, bhh1, Whh2, bhh2, Who1, bho1, Who2, bho2) = params
    _, y_ = scan(partial(TF_cell, cell_params = (Wq, bq, Wk, bk, Wv, bv, Wo, bo, a1, a2, b1, b2, Wfh, bfh, Whf, bhf), loc = loc, mask = mask)
    , (x, K, V) , jnp.arange(num_layer))
    state1 = nn.relu(Whh1 @ _[0][-1] + bhh1)
    prob = nn.softmax(Who1 @ state1 + bho1)
    state2 = nn.relu(Whh2 @ _[0][-1] + bhh2)
    phase = jnp.arctan(Who2 @ state2 + bho2)
    return _, prob, phase

@jax.jit
def TF_cell(x_, l, cell_params, loc, mask):
    Wq, bq, Wk, bk, Wv, bv, Wo, bo, a1, a2, b1, b2, Wfh, bfh, Whf, bhf = cell_params
    x, K, V = x_
    Q = linear (x[l], Wq[l], bq[l])
    K.at[l, loc].set(linear(x[l], Wk[l], bk[l]))
    V.at[l, loc].set(linear(x[l], Wv[l], bv[l]))

    # Now Q is of shape (head, L, units/head)
    out = linear(vmap(attention,(0, 1, 1, None, None))(Q, K[l], V[l], loc, mask).ravel(), Wo[l], bo[l])
    z = layer_norm_T(x[l] + out, a1[l], b1[l])
    x = x.at[l+1].set(layer_norm_T(z + linear(nn.relu(linear(z, Wfh[l], bfh[l])), Whf[l], bhf[l]), a2[l], b2[l]))
    return (x, K, V), None

def pos_2d(Ny, Nx, units):
    x_odd_f = jnp.repeat(jnp.array([1, 0, 0, 0]), units // 4)
    x_even_f = jnp.repeat(jnp.array([0, 1, 0, 0]), units // 4)
    y_odd_f = jnp.repeat(jnp.array([0, 0, 1, 0]), units // 4)
    y_even_f = jnp.repeat(jnp.array([0, 0, 0, 1]), units // 4)
    p = 2*jnp.arange(units)/units
    x = jnp.arange(Ny*Nx+1) %  Nx
    y = jnp.arange(Ny*Nx+1) // Nx

    return jnp.sin(jnp.outer(x, 1/10000**(p)))*x_odd_f + jnp.cos(jnp.outer(x, 1/10000**(p)))*x_even_f + jnp.sin(jnp.outer(y, 1/10000**(p)))*y_odd_f + jnp.cos(jnp.outer(y, 1/10000**(p)))*y_even_f

@partial(jax.jit, static_argnames=['fixed_params'])
def sample_prob_2DTQS(params, fixed_params, key):

    Ny, Nx, py, px, num_layer, units, head = fixed_params
    int_to_binary = partial(int_to_binary_array, num_bits=px * py)
    wemb, Wi, bi = params[0], params[1], params[2]
    wemb = jnp.concatenate([wemb, jnp.zeros((1, units))], axis=0)
    pos_encoding = pos_2d(Ny, Nx, units)
    mask = jnp.tril(-(jnp.ones((Ny*Nx, Ny*Nx)) - jnp.eye(Ny*Nx)) * 1e9).T
    def scan_fun(carry_1d, loc):
        input_, x, K, V, key = carry_1d
        x = x.at[0].set(nn.tanh(linear(wemb[input_[loc]] + pos_encoding[loc], Wi, bi)))
        (x, K, V), new_prob, new_phase = TF_step(x, K, V, loc, num_layer, params, mask)
        key, subkey = split(key)
        block_sample = categorical(subkey, jnp.log(new_prob))
        probs, phases = new_prob[block_sample], new_phase[block_sample]
        output = input_.at[loc+1].set(block_sample)

        return (output, x, K, V, key), (block_sample, probs, phases)

    # initialization
    init = -jnp.ones(Ny*Nx+1, dtype=int), jnp.zeros((num_layer+1, units)), jnp.zeros((num_layer, Ny*Nx, head, units//head)), jnp.zeros((num_layer, Ny*Nx, head, units//head)), key
    ny_nx_indices = jnp.array([i for i in range(Ny*Nx)])
    __, (samples, probs, phases) = scan(scan_fun, init, ny_nx_indices)
    samples = int_to_binary(samples)
    samples_log_amp = jnp.sum(jnp.log(probs)) / 2 + jnp.sum(phases) * 1j

    return samples, samples_log_amp

@partial(jax.jit, static_argnames=['fixed_params'])
def log_amp_2DTQS(samples, params, fixed_params):

    Ny, Nx, py, px, num_layer, units, head = fixed_params
    binary_to_int = partial(binary_array_to_int, num_bits=px * py)
    wemb, Wi, bi = params[0], params[1], params[2]
    wemb = jnp.concatenate([wemb, jnp.zeros((1, units))], axis=0)
    pos_encoding = pos_2d(Ny, Nx, units)
    mask = jnp.tril(-(jnp.ones((Ny*Nx, Ny*Nx)) - jnp.eye(Ny*Nx)) * 1e9).T
    def scan_fun(carry_1d, loc):

        input_ , x, K, V  = carry_1d
        x = x.at[0].set(nn.tanh(linear(wemb[input_[loc]] + pos_encoding[loc], Wi, bi)))
        (x, K, V), new_prob, new_phase = TF_step(x, K, V, loc, num_layer, params, mask)
        block_sample = binary_to_int(samples[loc])
        probs, phases = new_prob[block_sample], new_phase[block_sample]
        output = input_.at[loc+1].set(block_sample)

        return (output, x, K, V), (probs, phases)

    # initialization
    init = -jnp.ones(Ny*Nx+1, dtype=int), jnp.zeros((num_layer+1, units)), jnp.zeros((num_layer, Ny*Nx, head, units//head)), jnp.zeros((num_layer, Ny*Nx, head, units//head))
    ny_nx_indices = jnp.array([i for i in range(Ny*Nx)])
    __, (probs, phases) = scan(scan_fun, init, ny_nx_indices)
    log_amp = jnp.sum(jnp.log(probs)) / 2 + jnp.sum(phases) * 1j

    return log_amp
