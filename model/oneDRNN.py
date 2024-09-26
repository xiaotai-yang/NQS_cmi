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


def sample_prob(params, fixed_params, key, setting):

    def scan_fun(carry, indices):
        n = indices
        state, input, key, params = carry
        params_point = tuple(p[n] for p in params)
        state, new_prob, new_phase = tensor_gru_rnn_step(input, state, params_point)
        key, subkey = split(key)
        block_sample = categorical(subkey, jnp.log(new_prob)) #sampling
        prob, phase = new_prob[block_sample], new_phase[block_sample]
        input = wemb[block_sample]
        return (state, input, key, params), (block_sample, prob, phase)

    N, p, units = fixed_params
    dmrg, n_indices = setting
    wemb = jnp.eye(2**p)
    int_to_binary = partial(int_to_binary_array, num_bits=p)
    input_init, state_init = jnp.zeros(2**p), jnp.zeros((units))
    init = state_init, input_init, key, params

    __, (block_samples, probs, phase) = scan(scan_fun, init, n_indices)
    samples = int_to_binary(block_samples).reshape(N*p)
    log_probs, phase = jnp.sum(jnp.log(probs)), jnp.sum(phase)
    sample_amp = log_probs / 2 + 0. * 1j
    sample_amp = lax.cond(dmrg, lambda x: x, lambda x: x + phase * 1j, sample_amp)

    return samples, sample_amp

def log_amp(sample, params, fixed_params, setting):
    # samples : (num_samples, Ny, Nx)
    def scan_fun(carry, indices):
        n = indices
        state, input, samples, params = carry
        params_point = tuple(p[n] for p in params)
        state, new_prob, new_phase = tensor_gru_rnn_step(input, state, params_point)
        samples_output = binary_to_int(sample[n])
        prob = new_prob[samples_output]
        phase = new_phase[samples_output]
        input = wemb[samples_output] # one_hot_encoding of the sample

        return (state, input, samples, params), (prob, phase)

    #initialization
    N, p, units = fixed_params
    dmrg, n_indices = setting
    wemb = jnp.eye(2 ** p)
    binary_to_int = partial(binary_array_to_int, num_bits=p)
    state_init, input_init = jnp.zeros((units)), jnp.zeros(2**p)
    init = state_init, input_init, sample, params

    __, (probs, phase) = scan(scan_fun, init, n_indices)
    log_probs, phase = jnp.sum(jnp.log(probs)), jnp.sum(phase)
    log_amp = log_probs / 2 + 0. * 1j
    log_amp = lax.cond(dmrg, lambda x: x , lambda x: x + phase * 1j, log_amp)
   
    return log_amp




