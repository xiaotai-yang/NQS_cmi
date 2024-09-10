import jax.numpy as jnp
import jax.lax as lax
from functools import partial

def int_to_binary_array(x, num_bits):
    """
    Converts an array of integers to their binary representation arrays with a fixed number of bits.
    This function is designed to be compatible with Jax's vmap for vectorization over an array of integers.

    Parameters:
    - x: An array of integers, the numbers to convert.
    - num_bits: Integer, the fixed number of bits for the binary representation.

    Returns:
    - A 2D Jax array where each row is the binary representation of an integer in 'x'.
    """
    # Create an array of bit positions: [2^(num_bits-1), 2^(num_bits-2), ..., 1]
    powers_of_two = 2 ** jnp.arange(num_bits - 1, -1, -1)

    # Expand dims of x and powers_of_two for broadcasting
    x_expanded = x[:, None]
    powers_of_two_expanded = powers_of_two[None, :]

    # Perform bitwise AND between each number and each power of two, then right shift to get the bit value
    binary_matrix = (x_expanded & powers_of_two_expanded) >> jnp.arange(num_bits - 1, -1, -1)

    return binary_matrix.astype(jnp.int32)  # Ensure the result is integer


def binary_array_to_int(binary_array, num_bits):
    """
    Converts a 2D array of binary representations to their decimal equivalents.

    Parameters:
    - binary_array: A 2D Jax array where each row represents a binary number.

    Returns:
    - A 1D Jax array of integers, the decimal equivalents of the binary representations.
    """
    powers_of_two = 2 ** jnp.arange(num_bits - 1, -1, -1)
    # Multiply each bit by its corresponding power of two and sum the results
    decimals = jnp.dot(binary_array, powers_of_two)
    return decimals

def log_phase_dmrg(samples, M0, M, Mlast):
    def scan_fun(vec, indices):
        n = indices
        vec = M[samples[n+1],:,:,n] @ vec
        return vec, None

    vec_init = M0[samples[0]]
    vec_last = Mlast[samples[-1]]
    N = samples.shape[0]
    n_indices = jnp.arange(N-2)
    amp_last, _ = lax.scan(scan_fun, vec_init, n_indices)
    amp = jnp.dot(amp_last, vec_last)
    sign = amp / jnp.abs(amp)
    log_phase = lax.cond(jnp.abs(amp)>1e-12, lambda x:(-sign+1)/2*jnp.pi*1j, lambda x: 0.+0.*1j, None)
    return log_phase