import jax.numpy as jnp
import jax.nn as nn
import jax
from jax import vmap
from jax.lax import scan
from jax.random import categorical, split, PRNGKey
from functools import partial
from model.model_utlis import *
def time_mixing(x, t_state, decay, bonus, t_mix_k, t_mix_v, t_mix_r, t_wlast_x, Wk, Wv, Wr, Wout, t_xout, t_alphaout, t_betaout):
    last_x, last_alpha, last_beta = t_state
    last_x = layer_norm_R(t_wlast_x @ last_x, 1, 0)
    k = Wk @ (x * t_mix_k + last_x * (1 - t_mix_k))
    v = Wv @ (x * t_mix_v + last_x * (1 - t_mix_v))
    r = Wr @ (x * t_mix_r + last_x * (1 - t_mix_r))
    wkv = (last_alpha + jnp.exp(bonus + k) * v) / \
          (last_beta + jnp.exp(bonus + k) + 1e-10)
    rwkv = nn.sigmoid(r) * wkv
    alpha = jnp.exp(-jnp.exp(decay)) * last_alpha + jnp.exp(k) * v
    beta = jnp.exp(-jnp.exp(decay)) * last_beta + jnp.exp(k)
    alpha = t_alphaout @ alpha
    beta = t_betaout @ beta
    x = t_xout @ x
    return Wout @ rwkv, (x, alpha, beta)

def channel_mixing(x, c_states, c_mix_k, c_mix_r, c_wlast_x, Wk, Wv, Wr, c_xout):
    last_x = c_states #change tuple into array
    last_x = layer_norm_R(c_wlast_x @ last_x, 1, 0)

    k = Wk @ (x * c_mix_k + last_x * (1 - c_mix_k))
    r = Wr @ (x * c_mix_r + last_x * (1 - c_mix_r))
    vk = Wv @ nn.selu(k)
    x = c_xout @ x

    return nn.sigmoid(r) * vk, x

def layer_norm_R(x, w, b):
    mean = jnp.mean(x)
    std =  jnp.sqrt((jnp.sum((x - mean)**2) + 1e-10)/(x.size-1))
    return (x - mean)/ std * w + b
def rms_norm_R(x, w, b):
    return x/(jnp.sqrt(jnp.sum(x**2) + 1e-10)) * w + b

def RWKV_step(x, t_states, c_states, num_layer, RWKV_net_params, indices):
    ny, nx = indices
    w_in, b_in, w_out, b_out, whead, bhead, wprob, bprob, wphase, bphase, RWKV_cell_params = RWKV_net_params
    x = layer_norm_R(x, w_in[ny, nx], b_in[ny, nx])
    x , y = lax.scan(partial(RWKV_cell, params = tuple(px[nx] for px in tuple(py[ny] for py in RWKV_cell_params)), cell_t_states = t_states, cell_c_states = c_states), x, jnp.arange(num_layer))
    t_states, c_states = y
    x = nn.relu(whead[ny, nx] @ layer_norm_R(x, w_out[ny, nx], b_out[ny, nx]) + bhead[ny, nx])
    prob = nn.softmax(wprob @ x + bprob)
    phase = jnp.pi*nn.soft_sign(wphase @ x + bphase)
    return x, t_states, c_states, prob, phase

def RWKV_cell(carry, i, params, cell_t_states, cell_c_states): # carry = (x, t_states, c_states)
    #modify it for different layer of t_state and c_state and x .
    x = carry
    wln_i, bln_i, wln_m, bln_m, decay, bonus, t_mix_k, t_mix_v, t_mix_r, t_wk, t_wv, t_wr, t_wout, t_wlast_x, c_mix_k, c_mix_r, c_wk, c_wv, c_wr, c_wlast_x, t_xout, t_alphaout, t_betaout, c_xout = tuple(p[i] for p in params)
    layer_t_states = tuple(t[i] for t in cell_t_states)
    layer_c_states = cell_c_states[i]

    x_ = layer_norm_R(x, wln_i, bln_i)
    dx, output_t_states = time_mixing(x_, layer_t_states, decay, bonus, t_mix_k, t_mix_v, t_mix_r, t_wlast_x, t_wk, t_wv, t_wr, t_wout,  t_xout, t_alphaout, t_betaout)
    x = x + dx
    x_ = layer_norm_R(x, wln_m, bln_m)
    dx, output_c_states = channel_mixing(x_, layer_c_states, c_mix_k, c_mix_r, c_wlast_x, c_wk, c_wv, c_wr, c_xout)
    x = x + dx
    # carry need to be modified
    return x, (output_t_states, output_c_states)

@partial(jax.jit, static_argnames=['fixed_params'])
def sample_prob_2DRWKV(params, fixed_params, key):
    Ny, Nx, py, px, num_layer = fixed_params
    int_to_binary = partial(int_to_binary_array, num_bits=px * py)

    def scan_fun_1d(carry_1d, indices):
        ny, nx = indices

        input_x, input_yi, t_last_x1, t_last_x1s, t_last_x1e, t_last_x2, t_last_y1i, t_last_y2i, t_alpha_state_x1, t_beta_state_x1, t_alpha_state_yi, t_beta_state_yi, c_last_x1, c_last_x1s, c_last_x1e, c_last_x2, c_last_y1i, c_last_y2i, key = carry_1d

        rnn_input = jnp.concatenate((input_yi, input_x[nx]), axis=0)
        last_t_state = jnp.concatenate((t_last_y2i, t_last_x1s[nx], t_last_x2[nx], t_last_x1e[nx]), axis=1)
        last_c_state = jnp.concatenate((c_last_y2i, c_last_x1s[nx], c_last_x2[nx], c_last_x1e[nx]), axis=1)
        t_alpha_state = jnp.concatenate((t_alpha_state_yi, t_alpha_state_x1[nx]), axis=1)
        t_beta_state = jnp.concatenate((t_beta_state_yi, t_beta_state_x1[nx]), axis=1)
        x, t_states, c_states, prob, phase = RWKV_step(rnn_input, (last_t_state, t_alpha_state, t_beta_state),
                                                       (last_c_state), num_layer, RWKV_net_params, jnp.array([ny, nx]))

        key, subkey = split(key)
        block_sample = categorical(subkey, jnp.log(prob))
        probs, phase = prob[block_sample], phase[block_sample]
        t_last_y2i = t_last_y1i
        c_last_y2i = c_last_y1i
        t_last_y1i, t_alpha_state_yi, t_beta_state_yi = t_states
        c_last_y1i = c_states
        input_yi = wemb[block_sample]
        return (
        input_x, input_yi, t_last_x1, t_last_x1s, t_last_x1e, t_last_x2, t_last_y1i, t_last_y2i, t_alpha_state_x1, t_beta_state_x1,
        t_alpha_state_yi, t_beta_state_yi, c_last_x1, c_last_x1s, c_last_x1e, c_last_x2, c_last_y1i, c_last_y2i, key), (
        t_last_y1i, t_alpha_state_yi, t_beta_state_yi, c_last_y1i, block_sample, probs, phase)

    def scan_fun_2d(carry_2d, indices):  # indices:[[0,0], [0,1], [0,2]...[0,Nx-1]]

        input_x, input_y, t_last_x1, t_last_x2, t_last_y1s, t_last_y1e, t_last_y2, t_alpha_state_x, t_beta_state_x, t_alpha_state_y, t_beta_state_y, c_last_x1, c_last_x2, c_last_y1s, c_last_y1e, c_last_y2, key = carry_2d
        ## The shape of last_y1s, last_y1e are (Ny+1)
        index = indices[0, 0]
        t_last_x1s = jnp.concatenate((t_last_y1s[index][None, ...], t_last_x1[:-1]), axis=0)
        c_last_x1s = jnp.concatenate((c_last_y1s[index][None, ...], c_last_x1[:-1]), axis=0)
        t_last_x1e = jnp.concatenate((t_last_x1[1:], t_last_y1e[index][None, ...]), axis=0)
        c_last_x1e = jnp.concatenate((c_last_x1[1:], c_last_y1e[index][None, ...]), axis=0)
        carry_1d = input_x, input_y[index], t_last_x1, t_last_x1s, t_last_x1e, t_last_x2, t_last_y1e[index + 1], t_last_y2[
            index], t_alpha_state_x, t_beta_state_x, t_alpha_state_y[index], t_beta_state_y[index], c_last_x1, c_last_x1s, c_last_x1e, c_last_x2, c_last_y1e[index + 1], c_last_y2[
            index], key
        _, y = scan(scan_fun_1d, carry_1d, indices)
        '''
        The stacked y1i becomes the x1 in the next row
        The stacked y2i becomes the x2e in the next row
        '''
        t_last_x2 = t_last_x1  # x2 for the next row
        c_last_x2 = c_last_x1  # x2 for the next row
        t_last_x1, t_alpha_state_x1, t_beta_state_x1, c_last_x1, row_block_sample, row_prob, row_phase = y
        key = _[-1]
        t_last_x2 = jnp.flip(t_last_x2, 0)
        t_last_x1 = jnp.flip(t_last_x1, 0)
        c_last_x2 = jnp.flip(c_last_x2, 0)
        c_last_x1 = jnp.flip(c_last_x1, 0)
        t_alpha_state_x1 = jnp.flip(t_alpha_state_x1, 0)
        t_beta_state_x1 = jnp.flip(t_beta_state_x1, 0)
        input_x = wemb[jnp.flip(row_block_sample)]
        row_block_sample = lax.cond(index % 2, lambda x: jnp.flip(x, 0), lambda x: x, row_block_sample)
        row_prob = lax.cond(index % 2, lambda x: jnp.flip(x, 0), lambda x: x, row_prob)
        row_phase = lax.cond(index % 2, lambda x: jnp.flip(x, 0), lambda x: x, row_phase)

        return (input_x, input_y, t_last_x1, t_last_x2, t_last_y1s, t_last_y1e, t_last_y2, t_alpha_state_x1, t_beta_state_x1,
                t_alpha_state_y, t_beta_state_y, c_last_x1, c_last_x2, c_last_y1s, c_last_y1e, c_last_y2, key), (row_block_sample, row_prob, row_phase)

    # initialization
    wemb = params[0]
    init = (*params[1:17], key)
    RWKV_net_params = params[17:]
    ny_nx_indices = jnp.array([[(i, j) for j in range(Nx)] for i in range(Ny)])
    __, (samples, probs, phase) = scan(scan_fun_2d, init, ny_nx_indices)
    samples = vmap(int_to_binary, 0)(samples)
    log_probs, phase = jnp.sum(jnp.log(probs)), jnp.sum(phase)
    samples_log_amp = log_probs / 2 + phase * 1j

    return samples, samples_log_amp


@partial(jax.jit, static_argnames=['fixed_params'])
def log_amp_2DRWKV(samples, params, fixed_params):
    Ny, Nx, py, px, num_layer = fixed_params
    int_to_binary = partial(int_to_binary_array, num_bits=px * py)
    binary_to_int = partial(binary_array_to_int, num_bits=px * py)
    def scan_fun_1d(carry_1d, indices):
        ny, nx = indices

        (input_x, input_yi, t_last_x1, t_last_x1s, t_last_x1e, t_last_x2, t_last_y1i, t_last_y2i, t_alpha_state_x1, t_beta_state_x1,
         t_alpha_state_yi, t_beta_state_yi, c_last_x1, c_last_x1s, c_last_x1e, c_last_x2, c_last_y1i, c_last_y2i) = carry_1d

        rnn_input = jnp.concatenate((input_yi, input_x[nx]), axis=0)
        last_t_state = jnp.concatenate((t_last_y2i, t_last_x1s[nx], t_last_x2[nx], t_last_x1e[nx]), axis=1)
        last_c_state = jnp.concatenate((c_last_y2i, c_last_x1s[nx], c_last_x2[nx], c_last_x1e[nx]), axis=1)
        t_alpha_state = jnp.concatenate((t_alpha_state_yi, t_alpha_state_x1[nx]), axis=1)
        t_beta_state = jnp.concatenate((t_beta_state_yi, t_beta_state_x1[nx]), axis=1)
        x, t_states, c_states, prob, phase = RWKV_step(rnn_input, (last_t_state, t_alpha_state, t_beta_state),
                                                       (last_c_state), num_layer, RWKV_net_params, jnp.array([ny, nx]))

        block_sample = binary_to_int(lax.cond(ny%2, lambda x: x[ny, -nx-1], lambda x: x[ny, nx], samples).ravel())
        probs, phase = prob[block_sample], phase[block_sample]

        t_last_y2i = t_last_y1i
        c_last_y2i = c_last_y1i
        t_last_y1i, t_alpha_state_yi, t_beta_state_yi = t_states
        c_last_y1i = c_states
        input_yi = wemb[block_sample]

        return ((input_x, input_yi, t_last_x1, t_last_x1s, t_last_x1e, t_last_x2, t_last_y1i, t_last_y2i, t_alpha_state_x1,
            t_beta_state_x1, t_alpha_state_yi, t_beta_state_yi, c_last_x1, c_last_x1s, c_last_x1e, c_last_x2, c_last_y1i, c_last_y2i),
            (t_last_y1i, t_alpha_state_yi, t_beta_state_yi, c_last_y1i, block_sample, probs, phase))

    def scan_fun_2d(carry_2d, indices):  # indices:[[0,0], [0,1], [0,2]...[0,Nx-1]]

        input_x, input_y, t_last_x1, t_last_x2, t_last_y1s, t_last_y1e, t_last_y2, t_alpha_state_x, t_beta_state_x, t_alpha_state_y, t_beta_state_y, c_last_x1, c_last_x2, c_last_y1s, c_last_y1e, c_last_y2 = carry_2d
        ## The shape of last_y1s, last_y1e are (Ny+1)
        index = indices[0, 0]

        t_last_x1s = jnp.concatenate((t_last_y1s[index][None, ...], t_last_x1[:-1]), axis=0)
        c_last_x1s = jnp.concatenate((c_last_y1s[index][None, ...], c_last_x1[:-1]), axis=0)
        t_last_x1e = jnp.concatenate((t_last_x1[1:], t_last_y1e[index][None, ...]), axis=0)
        c_last_x1e = jnp.concatenate((c_last_x1[1:], c_last_y1e[index][None, ...]), axis=0)

        carry_1d = (input_x, input_y[index], t_last_x1, t_last_x1s, t_last_x1e, t_last_x2, t_last_y1e[index + 1], \
        t_last_y2[index], t_alpha_state_x, t_beta_state_x, t_alpha_state_y[index], t_beta_state_y[index], c_last_x1,
        c_last_x1s, c_last_x1e, c_last_x2, c_last_y1e[index + 1], c_last_y2[index])
        _, y = scan(scan_fun_1d, carry_1d, indices)

        '''
        The stacked y1i becomes the x1 in the next row
        The stacked y2i becomes the x2e in the next row
        '''

        t_last_x2 = t_last_x1  # x2 for the next row
        c_last_x2 = c_last_x1  # x2 for the next row
        t_last_x1, t_alpha_state_x1, t_beta_state_x1, c_last_x1, row_block_sample, row_prob, row_phase = y
        key = _[-1]
        t_last_x2 = jnp.flip(t_last_x2, 0)
        t_last_x1 = jnp.flip(t_last_x1, 0)
        c_last_x2 = jnp.flip(c_last_x2, 0)
        c_last_x1 = jnp.flip(c_last_x1, 0)

        t_alpha_state_x1 = jnp.flip(t_alpha_state_x1, 0)
        t_beta_state_x1 = jnp.flip(t_beta_state_x1, 0)
        input_x = wemb[jnp.flip(row_block_sample)]
        row_prob = lax.cond(index % 2, lambda x: jnp.flip(x, 0), lambda x: x, row_prob)
        row_phase = lax.cond(index % 2, lambda x: jnp.flip(x, 0), lambda x: x, row_phase)


        return (input_x, input_y, t_last_x1, t_last_x2, t_last_y1s, t_last_y1e, t_last_y2, t_alpha_state_x1, t_beta_state_x1,
                t_alpha_state_y, t_beta_state_y, c_last_x1, c_last_x2, c_last_y1s, c_last_y1e, c_last_y2), (row_prob, row_phase)

    # initialization
    wemb = params[0]
    init = params[1:17]
    RWKV_net_params = params[17:]
    ny_nx_indices = jnp.array([[(i, j) for j in range(Nx)] for i in range(Ny)])
    __, (probs, phase) = scan(scan_fun_2d, init, ny_nx_indices)
    log_probs, phase = jnp.sum(jnp.log(probs)), jnp.sum(phase)
    log_amp = log_probs / 2 + phase * 1j

    return log_amp
