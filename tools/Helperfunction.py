import jax
import jax.numpy as jnp
import itertools

def all_coe(array, reference, x, y):
    # Function to calculate x^n * y^m for a single element
    def calculate_product(element):
        # Count the number of flips compared to the reference
        flips = jnp.sum(element != reference)
        same = jnp.sum(element == reference)
        return x ** same * y ** flips

    coe = jnp.apply_along_axis(calculate_product, 1, array)
    coe_len = coe.shape[0]
    comb_num = 2 ** (array.shape[1])
    return coe[:-int(coe_len/comb_num)]

def clip_grad(g, clip_norm=5.0):
    norm = jnp.linalg.norm(g)
    scale = jnp.minimum(1.0, clip_norm / (norm + 1e-6))
    return g * scale



