import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
import jax.random as random
from jax.random import PRNGKey, split, categorical
import jax.lax as lax
from jax.lax import scan
import jax.nn as nn
from functools import partial
import sys
import os
from model.model_utlis import *

def tensor_gru_rnn_step(local_inputs, local_states, params):  # local_input is already concantenated
    Wu, bu, Wr, br, Ws, bs, Wamp, bamp, Wphase, bphase = params
    rnn_inputs = jnp.outer(local_inputs, local_states).ravel()
    u = nn.sigmoid(jnp.dot(Wu, rnn_inputs) + bu)
    r = nn.tanh(jnp.dot(Wr, rnn_inputs) + br)
    s = jnp.dot(Ws, rnn_inputs) + bs
    new_state = u * r + (1 - u) * s
    prob = nn.softmax(jnp.dot(Wamp, new_state) + bamp)
    phase = jnp.pi * nn.soft_sign(jnp.dot(Wphase, new_state) + bphase)

    return new_state, prob, phase
@partial(jax.jit, static_argnames=['fixed_params'])
def sample_prob(params, fixed_params, key):

    Ny, Nx, py, px, units = fixed_params
    int_to_binary = partial(int_to_binary_array, num_bits=px * py)
    wemb = jnp.eye(2**(px*py))
    def scan_fun_1d(carry_1d, indices):
        ny, nx = indices
        rnn_states_x_1d, rnn_states_yi_1d, inputs_x_1d, inputs_yi_1d, key = carry_1d
        rnn_states = jnp.concatenate((rnn_states_yi_1d, rnn_states_x_1d[nx]), axis=0)
        rnn_inputs = jnp.concatenate((inputs_yi_1d, inputs_x_1d[nx]), axis=0)

        new_state, new_prob, new_phase = tensor_gru_rnn_step(rnn_inputs, rnn_states,  tuple(px[nx] for px in tuple(py[ny] for py in params)))
        key, subkey = split(key)
        block_sample = categorical(subkey, jnp.log(new_prob))
        probs, phase = new_prob[block_sample], new_phase[block_sample]
        inputs_yi_1d = wemb[block_sample]

        return (rnn_states_x_1d, new_state, inputs_x_1d, inputs_yi_1d, key), (block_sample, probs, phase, new_state)

    def scan_fun_2d(carry_2d, indices):  # indices:[[0,0], [0,1], [0,2]...[0,Nx-1]]
        rnn_states_x, rnn_states_y, inputs_x, inputs_y, key = carry_2d
        index = indices[0, 0]
        carry_1d = rnn_states_x, rnn_states_y[index], inputs_x, inputs_y[index], key
        _, y = scan(scan_fun_1d, carry_1d, indices)
        key = _[-1]
        row_block_sample, row_prob, row_phase, rnn_states_x = y
        rnn_states_x = jnp.flip(rnn_states_x, 0)  # reverse the direction of input of for the next line
        inputs_x = wemb[jnp.flip(row_block_sample)]
        row_block_sample = lax.cond(index % 2, lambda x: jnp.flip(x, 0), lambda x: x, row_block_sample)
        row_prob = lax.cond(index % 2, lambda x: jnp.flip(x, 0), lambda x: x, row_prob)
        row_phase = lax.cond(index % 2, lambda x: jnp.flip(x, 0), lambda x: x, row_phase)

        return (rnn_states_x, rnn_states_y, inputs_x, inputs_y, key), (row_block_sample, row_prob, row_phase)

    # initialization
    init = jnp.zeros((Nx, units)), jnp.zeros((Ny, units)), jnp.zeros((Nx, 2**(px*py))), jnp.zeros((Ny, 2**(px*py))), key
    ny_nx_indices = jnp.array([[(i, j) for j in range(Nx)] for i in range(Ny)])
    __, (samples, probs, phase) = scan(scan_fun_2d, init, ny_nx_indices)
    samples = vmap(int_to_binary, 0)(samples).reshape(Ny*py, Nx*px)
    log_probs, phase = jnp.sum(jnp.log(probs)), jnp.sum(phase)
    samples_log_amp = log_probs / 2 + phase * 1j

    return samples, samples_log_amp

@partial(jax.jit, static_argnames=['fixed_params'])
def log_amp(samples, params, fixed_params):

    Ny, Nx, py, px, units = fixed_params
    binary_to_int = partial(binary_array_to_int, num_bits=px * py)
    wemb = jnp.eye(2**(px*py))

    def scan_fun_1d(carry_1d, indices):
        ny, nx = indices
        rnn_states_x_1d, rnn_states_yi_1d, inputs_x_1d, inputs_yi_1d = carry_1d
        rnn_states = jnp.concatenate((rnn_states_yi_1d, rnn_states_x_1d[nx]), axis=0)
        rnn_inputs = jnp.concatenate((inputs_yi_1d, inputs_x_1d[nx]), axis=0)
        new_state, new_prob, new_phase = tensor_gru_rnn_step(rnn_inputs, rnn_states,  tuple(px[nx] for px in tuple(py[ny] for py in params)))
        block_sample = binary_to_int(lax.cond(ny%2, lambda x: x[ny, -nx-1], lambda x: x[ny, nx], samples).ravel())
        probs, phase = new_prob[block_sample], new_phase[block_sample]
        inputs_yi_1d = wemb[block_sample]
        return (rnn_states_x_1d, new_state, inputs_x_1d, inputs_yi_1d), (probs, phase, new_state)

    def scan_fun_2d(carry_2d, indices):  # indices:[[0,0], [0,1], [0,2]...[0,Nx-1]]
        rnn_states_x, rnn_states_y, inputs_x, inputs_y= carry_2d
        index = indices[0, 0]
        carry_1d = rnn_states_x, rnn_states_y[index], inputs_x, inputs_y[index]
        _, y = scan(scan_fun_1d, carry_1d, indices)
        row_prob, row_phase, rnn_states_x = y
        rnn_states_x = jnp.flip(rnn_states_x, 0)  # reverse the direction of input of for the next line
        row_input = lax.cond(index%2, lambda x: x[index], lambda x: jnp.flip(x[index], 0), samples)
        inputs_x = wemb[binary_to_int(row_input.reshape(Nx, py*px))]

        return (rnn_states_x, rnn_states_y, inputs_x, inputs_y), (row_prob, row_phase)

    # initialization
    init = jnp.zeros((Nx, units)), jnp.zeros((Ny, units)), jnp.zeros((Nx, 2**(px*py))), jnp.zeros((Ny, 2**(px*py)))
    ny_nx_indices = jnp.array([[(i, j) for j in range(Nx)] for i in range(Ny)])
    __, (probs, phase) = scan(scan_fun_2d, init, ny_nx_indices)
    log_probs, phase = jnp.sum(jnp.log(probs)), jnp.sum(phase)
    log_amp = log_probs / 2 + phase * 1j

    return log_amp
