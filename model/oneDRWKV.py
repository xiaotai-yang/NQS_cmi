
import jax.numpy as jnp
import jax.nn as nn
from jax import lax
from functools import partial
from model.model_utlis import *
from jax.random import PRNGKey, split, categorical
from jax.lax import scan
def time_mixing(x, t_state, decay, bonus, t_mix_k, t_mix_v, t_mix_r, Wk, Wv, Wr, Wout):
    last_x, last_alpha, last_beta = t_state

    k = Wk @ (x * t_mix_k + last_x * (1 - t_mix_k))
    v = Wv @ (x * t_mix_v + last_x * (1 - t_mix_v))
    r = Wr @ (x * t_mix_r + last_x * (1 - t_mix_r))
    wkv = (last_alpha + jnp.exp(bonus + k) * v) / \
          (last_beta + jnp.exp(bonus + k) + 1e-10)

    rwkv = nn.sigmoid(r) * wkv
    alpha = jnp.exp(-jnp.exp(decay)) * last_alpha + jnp.exp(k) * v
    beta = jnp.exp(-jnp.exp(decay)) * last_beta + jnp.exp(k)

    return Wout @ rwkv, (x, alpha, beta)

def channel_mixing(x, c_states, c_mix_k, c_mix_r, Wk, Wv, Wr):
    last_x = c_states #change tuple into array

    k = Wk @ (x * c_mix_k + last_x * (1 - c_mix_k))
    r = Wr @ (x * c_mix_r + last_x * (1 - c_mix_r))
    vk = Wv @ nn.selu(k)

    return nn.sigmoid(r) * vk, x

def rms_norm_R(x, w, b):
    return x/(jnp.sqrt(jnp.sum(x**2) + 1e-10)) * w + b

def layer_norm_R(x, w, b):
    mean = jnp.mean(x)
    std =  jnp.sqrt((jnp.sum((x - mean)**2) + 1e-10)/(x.size-1))
    return (x - mean)/ std * w + b

def RWKV_step(x, t_states, c_states, num_layer, RWKV_net_params, indices):
    n = indices
    w_in, b_in, whead, bhead, wln_out, bln_out, wprob1, bprob1, wphase1, bphase1, wprob2, bprob2, wphase2, bphase2, RWKV_cell_params = RWKV_net_params
    x = rms_norm_R(x, w_in[n], b_in[n])
    x , y = lax.scan(partial(RWKV_cell, params = tuple(p[n] for p in RWKV_cell_params), cell_t_states = t_states, cell_c_states = c_states), x, jnp.arange(num_layer))
    t_states, c_states = y
    x = whead[n] @ rms_norm_R(x, wln_out[n], bln_out[n]) + bhead[n]
    prob = nn.softmax(wprob2 @ nn.relu(wprob1 @ x + bprob1) + bprob2)
    phase = 2*jnp.pi*nn.soft_sign(wphase2 @ nn.relu(wphase1 @ x + bphase1) + bphase2)
    return x, t_states, c_states, prob, phase

def RWKV_cell(carry, i, params, cell_t_states, cell_c_states): # carry = (x, t_states, c_states)
    #modify it for different layer of t_state and c_state and x .
    x = carry
    wln_i, bln_i, wln_m, bln_m, decay, bonus, t_mix_k, t_mix_v, t_mix_r, t_wk, t_wv, t_wr, t_wout, c_mix_k, c_mix_r, c_wk, c_wv, c_wr = tuple(p[i] for p in params)
    layer_t_states = tuple(t[i] for t in cell_t_states)
    layer_c_states = cell_c_states[i]

    x_ = rms_norm_R(x, wln_i, bln_i)
    dx, output_t_states = time_mixing(x_, layer_t_states, decay, bonus, t_mix_k, t_mix_v, t_mix_r, t_wk, t_wv, t_wr, t_wout)
    x = x + dx
    x_ = rms_norm_R(x, wln_m, bln_m)
    dx, output_c_states = channel_mixing(x_, layer_c_states, c_mix_k, c_mix_r, c_wk, c_wv, c_wr)
    x = x + dx
    # carry need to be modified
    return x, (output_t_states, output_c_states)

def sample_prob_RWKV(params, fixed_params, key, dmrg):

    def scan_fun(carry, n):
        input, last_t, t_alpha, t_beta, last_c, key = carry
        x, t_states, c_states, out_prob, out_phase = RWKV_step(input, (last_t, t_alpha, t_beta),
        last_c, num_layer, RWKV_net_params, n)

        key, subkey = split(key)
        block_sample = categorical(subkey, jnp.log(out_prob))
        prob, phase = out_prob[block_sample], out_phase[block_sample]
        last_t, t_alpha, t_beta = t_states
        last_c= c_states
        input = wemb[block_sample]

        return (input, last_t, t_alpha, t_beta, last_c, key), (block_sample, prob, phase)

    N, p, h_size, num_layer, num_emb = fixed_params
    wemb = jnp.eye(num_emb)
    int_to_binary = partial(int_to_binary_array, num_bits=p)
    n_indices = jnp.array([i for i in range(N)])
    init = (params[0], params[1], jnp.zeros((num_layer, h_size)), jnp.zeros((num_layer, h_size)), params[2], key)
    RWKV_net_params = params[3:]
    __, (samples, probs, phase) = scan(scan_fun, init, n_indices)
    samples = int_to_binary(samples).reshape(N*p)
    log_probs, phase = jnp.sum(jnp.log(probs)), jnp.sum(phase)
    samples_log_amp = log_probs / 2 + 0. * 1j
    samples_log_amp = lax.cond(dmrg, lambda x: samples_log_amp, lambda x: samples_log_amp + phase * 1j, None)

    return samples, samples_log_amp

def log_amp_RWKV(samples, params, fixed_params, dmrg):
    def scan_fun(carry, n):
        input, last_t, t_alpha, t_beta, last_c = carry
        x, t_states, c_states, out_prob, out_phase = RWKV_step(input, (last_t, t_alpha, t_beta),
last_c, num_layer, RWKV_net_params, n)
        block_sample = binary_to_int(samples[n])
        prob, phase = out_prob[block_sample], out_phase[block_sample]
        last_t, t_alpha, t_beta = t_states
        last_c = c_states
        input = wemb[block_sample]

        return (input, last_t, t_alpha, t_beta, last_c), (block_sample, prob, phase)

    N, p, h_size, num_layer, num_emb = fixed_params
    wemb = jnp.eye(num_emb)
    n_indices = jnp.array([i for i in range(N)])
    binary_to_int = partial(binary_array_to_int, num_bits=p)
    init = (params[0], params[1], jnp.zeros((num_layer, h_size)), jnp.zeros((num_layer, h_size)), params[2])
    RWKV_net_params = params[3:]
    __, (samples, probs, phase) = scan(scan_fun, init, n_indices)

    log_probs, phase = jnp.sum(jnp.log(probs)), jnp.sum(phase)
    log_amp = log_probs / 2 + 0. * 1j
    log_amp = lax.cond(dmrg, lambda x: log_amp, lambda x: log_amp + phase * 1j, None)

    return log_amp