
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

def layer_norm(x, w, b):
    mean = jnp.mean(x)
    std =  jnp.sqrt((jnp.sum((x - mean)**2) + 1e-8)/(x.size-1))
    return (x - mean)/ std * w + b

@jax.jit
def linear(x, w, b):
    return w@x+b
@jax.jit
def attention(q, k, v, loc):
    a = q.shape[0]

    N = k.shape[0]
    m =  jnp.tril(-(jnp.ones((N, N)) - jnp.eye(N))*1e9).T
    QKV = nn.softmax(jnp.matmul(q, k.T)/jnp.sqrt(a) + m[loc]) @ v
    return QKV

@partial(jax.jit, static_argnames=("num_layer"))
def TF_step(x, loc, num_layer, params):
    #(Wemb, Wq, bq, Wk, bk, Wv, bv, Wo, bo, a, b, Wfh, bfh, Whf, bhf, Whh, bhh, Who1, bho1) = params
    (Wemb, Wi, bi, Wq, bq, Wk, bk, Wv, bv, Wqm, bqm, Wkm, bkm, Wvm, bvm, Wo, bo, a1, a2, b1, b2, Wfh, bfh, Whf, bhf, Whh1, bhh1, Who1, bho1, Who2, bho2) = params
    _, y_ = scan(partial(TF_cell, cell_params = (Wq, bq, Wk, bk, Wv, bv, Wqm, bqm, Wkm, bkm, Wvm, bvm, Wo, bo, a1, a2, b1, b2, Wfh, bfh, Whf, bhf), loc = loc)
    , (x, x[:, loc]) , jnp.arange(num_layer))
    state = nn.relu(Whh1 @ _[1][-1] + bhh1)
    prob = nn.softmax(Who1 @ state + bho1)
    phase = jnp.pi*nn.soft_sign(Who2 @ state + bho2)
    return _[0], prob, phase

@jax.jit
def TF_cell(x_, l, cell_params, loc):
    Wq, bq, Wk, bk, Wv, bv, Wqm, bqm, Wkm, bkm, Wvm, bvm, Wo, bo, a1, a2, b1, b2, Wfh, bfh, Whf, bhf = cell_params
    x, y = x_
    Q = linear (y[l], Wq[l], bq[l])
    K = vmap(linear, (0, None, None), 0)(x[l], Wk[l], bk[l])
    V = vmap(linear, (0, None, None), 0)(x[l], Wv[l], bv[l])
    Qm = linear(Q, Wqm[l], bqm[l])
    Km = vmap(linear, (0, None, None), 1)(K, Wkm[l], bkm[l])
    Vm = vmap(linear, (0, None, None), 1)(V, Wvm[l], bvm[l])
    # Now Km, Vm  is of shape (head, L, units/head)
    out = linear(vmap(attention,(0, 0, 0, None))(Qm, Km, Vm, loc).ravel(), Wo[l], bo[l])
    z = layer_norm(y[l] + out, a1[l], b1[l])
    y = y.at[l+1].set(layer_norm(z + linear(nn.relu(linear(z, Wfh[l], bfh[l])), Whf[l], bhf[l]), a2[l], b2[l]))
    x = x.at[l+1, loc].set(y[l+1])
    return (x, y), None

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

    Ny, Nx, py, px, num_layer, units = fixed_params
    int_to_binary = partial(int_to_binary_array, num_bits=px * py)
    wemb, Wi, bi = params[0], params[1], params[2]
    wemb = jnp.concatenate([wemb, jnp.zeros((1, units))], axis=0)

    def scan_fun(carry_1d, loc):
        input_, x, key = carry_1d
        x = x.at[0].set(nn.tanh(vmap(linear, (0, None, None))(wemb[input_] + pos_2d(Ny, Nx, units), Wi, bi)))
        x, new_prob, new_phase = TF_step(x, loc, num_layer, params)
        key, subkey = split(key)
        block_sample = categorical(subkey, jnp.log(new_prob))
        probs, phases = new_prob[block_sample], new_phase[block_sample]
        output = input_.at[loc+1].set(block_sample)

        return (output, x, key), (block_sample, probs, phases)

    # initialization
    init = -jnp.ones(Ny*Nx+1, dtype=int), jnp.zeros((num_layer+1, Ny*Nx+1, units)), key
    ny_nx_indices = jnp.array([i for i in range(Ny*Nx)])
    __, (samples, probs, phases) = scan(scan_fun, init, ny_nx_indices)
    samples = int_to_binary(samples)
    samples_log_amp = jnp.sum(jnp.log(probs)) / 2 + jnp.sum(phases) * 1j

    return samples, samples_log_amp

@partial(jax.jit, static_argnames=['fixed_params'])
def log_amp_2DTQS(samples, params, fixed_params):

    Ny, Nx, py, px, num_layer, units = fixed_params
    binary_to_int = partial(binary_array_to_int, num_bits=px * py)
    wemb, Wi, bi = params[0], params[1], params[2]
    wemb = jnp.concatenate([wemb, jnp.zeros((1, units))], axis=0)
    def scan_fun(carry_1d, loc):

        input_ , x  = carry_1d
        x = x.at[0].set(nn.tanh(vmap(linear, (0, None, None))(wemb[input_] + pos_2d(Ny, Nx, units), Wi, bi)))
        x, new_prob, new_phase = TF_step(x, loc, num_layer, params)
        block_sample = binary_to_int(samples[loc])
        probs, phases = new_prob[block_sample], new_phase[block_sample]
        output = input_.at[loc+1].set(block_sample)

        return (output, x), (probs, phases)

    # initialization
    init = -jnp.ones(Ny*Nx+1, dtype=int), jnp.zeros((num_layer+1, Ny*Nx+1, units))
    ny_nx_indices = jnp.array([i for i in range(Ny*Nx)])
    __, (probs, phases) = scan(scan_fun, init, ny_nx_indices)

    log_amp = jnp.sum(jnp.log(probs)) / 2 + jnp.sum(phases) * 1j

    return log_amp
