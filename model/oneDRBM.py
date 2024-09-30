
import jax
import jax.numpy as jnp
from jax.random import PRNGKey, split, uniform
import flax.linen as nn
import netket as nk
from model.model_utlis import *


def uniform_init(scale):
    def init(key, shape, dtype=jnp.float32):
        return jax.random.uniform(key, shape, dtype=dtype, minval=-scale, maxval=scale)
    return init
def complex_uniform_init(scale=1e-2, dtype=jnp.complex64):
    def init(key, shape, dtype=dtype):
        key_real, key_imag = jax.random.split(key)
        real_part = uniform(key_real, shape, dtype=jnp.float32, minval=-scale.real, maxval=scale.real)
        imag_part = uniform(key_imag, shape, dtype=jnp.float32, minval=-scale.real, maxval=scale.real)

        return real_part + 1j * imag_part

    return init
class RBM_dmrg_model(nn.Module):
    func: callable
    alpha: int
    L: int
    def setup(self):
        self.dense = nn.Dense(
            features=self.alpha * self.L,
            param_dtype=jnp.float32,
            kernel_init=uniform_init(1 / jnp.sqrt(self.alpha * self.L*100)),
            bias_init=uniform_init(1 / jnp.sqrt(self.alpha * self.L*100))
        )
        self.ai = jnp.zeros(self.L, dtype=jnp.float32)
        self.scale = self.param("scale", nn.initializers.constant(0), (1,), jnp.float32)

    def __call__(self, x):
        y = self.dense(x)  # x shape: (batch_size, input_dim)
        y = nk.nn.activation.log_cosh(y)
        y_sum = jnp.sum(y, axis=-1).astype(jnp.complex64)
        # Apply batch_log_phase_dmrg to x
        phase_corrections = self.func(x)
        y_sum += phase_corrections
        y_sum += jnp.dot(x, self.ai)
        y_sum += self.scale
        return y_sum
class ComplexRBM(nn.Module):
    n_hidden_units: int  # Number of hidden units
    dtype: jnp.dtype

    @nn.compact
    def __call__(self, x):
        # Get the number of visible units from input
        n_visible_units = x.shape[-1]

        # Define complex weights matrix W (hidden units, visible units)
        W = self.param("W", complex_uniform_init(), (n_visible_units, self.n_hidden_units), self.dtype)
        # Define complex hidden biases b (hidden units)
        b = self.param("b", complex_uniform_init(), (self.n_hidden_units,), self.dtype)
        # Define complex visible biases a (visible units)
        a = self.param("a", complex_uniform_init(), (n_visible_units,), self.dtype)
        scale_ = self.param("scale", nn.initializers.constant(0), (1,), self.dtype)
        # Compute hidden layer pre-activation: W @ x + b
        hidden_pre_activation = jnp.dot(x, W) + b
        # Apply non-linearity (log_cosh for complex inputs)
        hidden_activation = nk.nn.activation.log_cosh(hidden_pre_activation)
        # Sum the activations over the hidden units (axis=-1)
        y_sum = jnp.sum(hidden_activation, axis=-1).astype(jnp.complex64)
        # Add the contribution from the visible bias term
        y_sum += jnp.dot(x, a)
        y_sum += scale_

        return y_sum