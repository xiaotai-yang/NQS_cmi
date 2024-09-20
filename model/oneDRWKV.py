
import jax.numpy as jnp
import jax.nn as nn
from jax import lax
from jax import vmap
from functools import partial
from model.model_utlis import *
from jax.random import PRNGKey, split, categorical
from jax.lax import scan

def rms_norm_R(x, w, b):
    return x/(jnp.sqrt(jnp.sum(x**2) + 1e-10)) * w + b

def layer_norm_R(x, w, b):
    mean = jnp.mean(x)
    std =  jnp.sqrt((jnp.sum((x - mean)**2) + 1e-10)/(x.size-1))
    return (x - mean)/ std * w + b

def group_norm_R(x, w, b):
    mean = jnp.mean(x, axis=1, keepdims=True)
    std =  jnp.sqrt((jnp.sum((x - mean)**2, axis=1, keepdims=True) + 1e-10)/(x.shape[1]-1))
    return (x - mean)/ std * w + b
def lora(x, lam, A, B):
    return lam + nn.tanh(x @ A) @ B

def lerp(a, b, mu):
    return a + (b - a) * mu
def ddlerp(a, b, m1, m2, A, B):
    return a + (b - a) * (m1 + nn.tanh((a + (b - a) * m2) @ A) @ B)
def time_mixing(x, t_state, t_params, head):
    mx_t, mw_t, mk_t, mv_t, mr_t, mg_t, Ww_t, Wk_t, Wv_t, Wr_t, Wg_t, Wo_t, Wd, Ax, Bx, decay, u, wln_t, bln_t = t_params
    last_x, state = t_state
    # state is of shape (head, D, D), u is  of shape (head, D)
    batch_dot = vmap(jnp.matmul, (0, 0), 0)
    batch_multiply = vmap(jnp.multiply, (0, 0), 0)
    print(lerp(last_x, x, mx_t).shape)
    print(Ax.shape)
    mr, mk, mv, mg, mw = batch_dot(jnp.sum(nn.tanh(lerp(last_x, x, mx_t)[:, None, None] * Ax), axis = 0), Bx)
    w = lerp(last_x, x, mw_t + mw) @ Ww_t
    k = (lerp(last_x, x, mk_t + mk) @ Wk_t).reshape(head, -1)
    v = (lerp(last_x, x, mv_t + mv) @ Wv_t).reshape(head, -1)
    r = (lerp(last_x, x, mr_t + mr) @ Wr_t).reshape(head, -1)
    g = (lerp(last_x, x, mg_t + mg) @ Wg_t).reshape(head, -1)
    w = (jnp.exp(-jnp.exp(decay + nn.tanh(w) @ Wd))).reshape(head, -1)
    kv = batch_dot(k[..., None], v[:, None])
    wkv = group_norm_R(batch_dot(r, (state + batch_multiply(u, kv))), wln_t, bln_t)
    state = batch_multiply(w, state) + kv

    return batch_multiply(nn.silu(g), wkv).ravel() @ Wo_t, (x, state)

def channel_mixing(x, c_states, c_params):
    mr_c, mk_c, Wk_c, Wv_c, Wr_c = c_params
    last_x = c_states

    r = lerp(x, last_x, mr_c) @ Wr_c
    k = lerp(x, last_x, mk_c) @ Wk_c
    v = nn.relu(k)**2 @ Wv_c

    return nn.sigmoid(r) * v, x

def RWKV_step(x, t_states, c_states, num_layer, RWKV_net_params, head):
    step_params, t_params, c_params, cell_params= RWKV_net_params
    wln_in, bln_in, whead, bhead, wln_out, bln_out, wprob1, bprob1, wphase1, bphase1, wprob2, bprob2, wphase2, bphase2 = step_params
    x = layer_norm_R(x, wln_in, bln_in)
    x , y = lax.scan(partial(RWKV_cell,
                             t_params = t_params,
                             c_params = c_params,
                             cell_params = cell_params,
                             cell_t_states = t_states,
                             cell_c_states = c_states,
                             head = head)
                     , x, jnp.arange(num_layer))
    last_x_state, t_state, c_states = y
    x = whead @ layer_norm_R(x, wln_out, bln_out) + bhead
    prob = nn.softmax(wprob2 @ nn.relu(wprob1 @ x + bprob1) + bprob2)
    phase = jnp.pi*nn.soft_sign(wphase2 @ nn.relu(wphase1 @ x + bphase1) + bphase2)
    return x, last_x_state, t_state, c_states, prob, phase

def RWKV_cell(carry, i, t_params, c_params, cell_params, cell_t_states, cell_c_states, head):
    # carry = (x, t_states, c_states)
    x = carry
    wln_i, bln_i, wln_m, bln_m = cell_params
    t_params_i = tuple(t[i] for t in t_params)
    c_params_i = tuple(t[i] for t in c_params)
    layer_t_states = tuple(t[i] for t in cell_t_states)
    layer_c_states = cell_c_states[i]

    x_ = layer_norm_R(x, wln_i[i], bln_i[i])
    dx, output_t_states = time_mixing(x_, layer_t_states, t_params_i, head)
    x = x + dx
    x_ = layer_norm_R(x, wln_m[i], bln_m[i])
    dx, output_c_states = channel_mixing(x_, layer_c_states, c_params_i)
    x = x + dx
    # carry need to be modified
    return x, (output_t_states[0], output_t_states[1], output_c_states)

def sample_prob_RWKV(params, fixed_params, key, dmrg):

    def scan_fun(carry, n):
        input, last_x, last_t, last_c, key = carry
        x, last_x, last_t, last_c, out_prob, out_phase = RWKV_step(input, (last_x, last_t), last_c, num_layer, RWKV_net_params, head)
        key, subkey = split(key)
        block_sample = categorical(subkey, jnp.log(out_prob))
        prob, phase = out_prob[block_sample], out_phase[block_sample]
        input = wemb[block_sample]

        return (input, last_x, last_t, last_c, key), (block_sample, prob, phase)

    N, p, num_layer, head = fixed_params
    int_to_binary = partial(int_to_binary_array, num_bits=p)
    n_indices = jnp.array([i for i in range(N)])
    wemb = params[0]
    xi, last_xi, ti, ci = params[1]
    RWKV_net_params = params[2:]
    __, (samples, probs, phase) = scan(scan_fun, (xi, last_xi, ti, ci, key), n_indices)

    samples = int_to_binary(samples).reshape(N*p)
    log_probs, phase = jnp.sum(jnp.log(probs)), jnp.sum(phase)
    samples_log_amp = log_probs / 2 + 0. * 1j
    samples_log_amp = lax.cond(dmrg, lambda x: samples_log_amp, lambda x: samples_log_amp + phase * 1j, None)

    return samples, samples_log_amp

def log_amp_RWKV(samples, params, fixed_params, dmrg):
    def scan_fun(carry, n):
        input, last_x, last_t, last_c = carry
        x, last_x, last_t, last_c, out_prob, out_phase = RWKV_step(input, (last_x, last_t), last_c, num_layer, RWKV_net_params, head)
        block_sample = binary_to_int(samples[n])
        prob, phase = out_prob[block_sample], out_phase[block_sample]
        input = wemb[block_sample]

        return (input, last_x, last_t, last_c), (block_sample, prob, phase)

    N, p, num_layer, head = fixed_params
    wemb = params[0]
    n_indices = jnp.array([i for i in range(N)])
    binary_to_int = partial(binary_array_to_int, num_bits=p)
    RWKV_net_params = params[2:]
    __, (samples, probs, phase) = scan(scan_fun, params[1], n_indices)

    log_probs, phase = jnp.sum(jnp.log(probs)), jnp.sum(phase)
    log_amp = log_probs / 2 + 0. * 1j
    log_amp = lax.cond(dmrg, lambda x: log_amp, lambda x: log_amp + phase * 1j, None)

    return log_amp