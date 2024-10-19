
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
import time
from math import ceil
import itertools
import numpy as np

def local_element_indices_1d(num_body, pauli_array, loc_array, rotation = False, angle = 0.0):
    '''

    Args:
        num_body: The length of each Pauli string.
        pauli_array: The corresponding Pauli string of the Hamiltonian term.
        Ex: [[3,1,1,1,1],....[3,1,1,1,1]]
        loc_array: The location that the Hamiltonian term acts on.
        Ex: [[1,1],[1, 2], [2, 1], [1, 0], [0 ,1]]....[[Ny-2,Nx-2], [Ny-2, Nx-1], [Ny-1, Nx-2], [Ny-2, Nx-3], [Ny-3, Nx-2]]

    Returns:
        xy_loc_arrays, yloc_arrays, zloc_arrays
        They are all dictionarys that: [pauli_z terms, pauli_x terms]: The location that the Hamiltonian term acts on.
        For example:
        zloc_bulk: {(4, 1): Array([[[1, 2], [2, 1], [1, 0], [0, 1]],...
        [[2, 3], [3, 2], [2, 1], [1, 2]]]}
     '''
    if pauli_array.shape[-1] != num_body:
        raise ValueError(f"Array has incorrect body of interactions {pauli_array.shape[-1]}. Expected body of interactions is {num_body}.")

    #Count the number of pauli_Z and pauli_X for each term in the Hamiltonian

    count_3s = jnp.sum(pauli_array == 3, axis = 1)
    count_1s = jnp.sum(pauli_array == 1, axis = 1)
    pauli_array_xz = {}
    xloc_arrays = {}
    zloc_arrays = {}
    yloc_arrays = {}
    xy_loc_arrays = {}
    coe_arrays = {}
    coe_arrays_odd = {}
    coe_arrays_even = {}
    for i in range(num_body+1):    #z_number
        for j in range (num_body+1-i):  #x_number]
            mask = ((count_3s == i)&(count_1s == j))  #mask for the interaction with i pauli_Z and j pauli_X
            pauli_array_xz[i, j] = pauli_array[mask]

            mask_x = (pauli_array_xz[i, j] == 1)
            mask_y = (pauli_array_xz[i, j] == 2)
            mask_z = (pauli_array_xz[i, j] == 3)
            ref_bulk_z_mask = jnp.array([False, True, False])
            ref_first_z_mask = jnp.array([True, False])
            ref_last_z_mask = jnp.array([False, False])
            ref_xzz_z_mask = jnp.array([False, True, True])
            x, y = jnp.cos(angle), jnp.sin(angle)
            if rotation:
                if (num_body == 3 and pauli_array.shape[0]!= 8):   #extract the bulk part
                    # adding a minus sign when Z go to X
                    coe_arrays[i, j] = -(-1)**(jnp.sum((mask_z == False) & (mask_z != ref_bulk_z_mask), axis = 1))*x**(jnp.sum(mask_z == ref_bulk_z_mask, axis = 1))*y**(jnp.sum(mask_z != ref_bulk_z_mask, axis = 1))
                elif num_body == 2 : #extract the first and the last part
                    coe_arrays_odd[i, j] = -(-1)**(jnp.sum((mask_z[::2] != ref_first_z_mask)&(mask_z[::2] == False), axis = 1))*x**(jnp.sum(mask_z[::2] == ref_first_z_mask, axis = 1))*y**(jnp.sum(mask_z[::2] != ref_first_z_mask, axis = 1)) # odd part represents the first ZX stablizer
                    coe_arrays_even[i, j] = -(-1)**(jnp.sum((mask_z[1::2] != ref_last_z_mask)&(mask_z[1::2] == False), axis = 1))*x**(jnp.sum(mask_z[1::2] == ref_last_z_mask, axis = 1))*y**(jnp.sum(mask_z[1::2] != ref_last_z_mask, axis = 1)) # even part represents the last XX stablizer
                    coe_arrays[i, j] = jnp.vstack((coe_arrays_odd[i, j], coe_arrays_even[i, j])).T.reshape(-1) #mix the odd part and the even part together
                elif num_body == 3 and pauli_array.shape[0]==8: #extract the XZZ part
                    coe_arrays[i, j] = -(-1)**(jnp.sum((mask_z == False) & (mask_z != ref_xzz_z_mask), axis = 1))*x**(jnp.sum(mask_z == ref_xzz_z_mask, axis = 1))*y**(jnp.sum(mask_z != ref_xzz_z_mask, axis = 1))
            if mask_x.sum() != 0:
                xloc_arrays[i, j] = loc_array[mask][mask_x].reshape(-1, j, 1)
            elif mask_y.sum() != 0 or mask_z.sum() !=0:
                if mask_y.sum()!=0:
                    xloc_arrays[i, j] = jnp.array([[]]).reshape(pauli_array_xz[i ,j].shape[0], 0, 1).astype(int)
                else:
                    xloc_arrays[i, j] = jnp.array([[]]).reshape(pauli_array_xz[i, j].shape[0], 0, 1).astype(int)

            if mask_y.sum() !=0:
                yloc_arrays[i, j] = loc_array[mask][mask_y].reshape(-1, num_body-i-j, 1).astype(int)
            elif mask_x.sum() != 0 or mask_z.sum() !=0:
                if mask_x.sum()!=0:
                    yloc_arrays[i, j] = jnp.array([[]]).reshape(pauli_array_xz[i, j].shape[0], 0, 1).astype(int)
                else:
                    yloc_arrays[i, j] = jnp.array([[]]).reshape(pauli_array_xz[i, j].shape[0], 0, 1).astype(int)

            if mask_z.sum()!=0:
                zloc_arrays[i, j] = loc_array[mask][mask_z].reshape(-1, i, 1).astype(int)
            elif mask_x.sum() != 0 or mask_y.sum() !=0:
                if mask_y.sum()!=0:
                    zloc_arrays[i, j] = jnp.array([[]]).reshape(pauli_array_xz[i, j].shape[0], 0, 1).astype(int)
                else:
                    zloc_arrays[i, j] = jnp.array([[]]).reshape(pauli_array_xz[i, j].shape[0], 0, 1).astype(int)

    for ind in (xloc_arrays):
        xy_loc_arrays[ind] = jnp.concatenate((xloc_arrays[ind], yloc_arrays[ind]), axis=1).astype(int)

    if rotation:
        return xy_loc_arrays, yloc_arrays, zloc_arrays, coe_arrays
    return  xy_loc_arrays, yloc_arrays, zloc_arrays

@jax.jit
def total_samples_1d(sample, xyloc_arrays):
    '''
        Get the sample we need for the VMC off-diagonal elements
    '''
    def single_samples_1d(sample, xy_loc_array_element):
        return sample.at[xy_loc_array_element[:, 0]].set((sample[xy_loc_array_element[:, 0]] + 1) % 2) # flip the corresponding site
    results = []
    vmap_single_samples_1d = vmap(single_samples_1d, (None, 0))
    for i in xyloc_arrays:
        xy_loc_arrays_elements = xyloc_arrays[i]
        tmp = vmap_single_samples_1d(sample, xy_loc_arrays_elements)
        results.append(tmp)
    concatenated_result = jnp.concatenate(results, axis=0)
    return concatenated_result

@partial(jax.jit, static_argnames=['rotation'])
def new_coe_1d(sample, coe_array_off_diag, yloc, zloc, rotation):
    # Get the corresponding coefficient for the off-diagonal elements in the Hamiltonian, here the ycoe is incorrect but there is no pauli-y terms
    # in the Hamiltonian so it doesn't matter.

    def ycoe(single_sample, yloc_arrays_element):
        return ((0-1j)**single_sample[yloc_arrays_element[:,0]]).prod()
    def zcoe(single_sample, zloc_arrays_element):
        return ((-1+0j)**single_sample[zloc_arrays_element[:,0]]).prod()

    tmp_y = []
    tmp_z = []
    tmp_array = []
    vmap_ycoe = vmap(ycoe, (None, 0))
    vmap_zcoe = vmap(zcoe, (None, 0))

    for i in yloc:
        tmp_y.append(lax.cond(yloc[i].shape[1] != 0, lambda: vmap_ycoe(sample, yloc[i]), lambda: jnp.repeat(jnp.array([1. + 0j]), yloc[i].shape[0])))
    for i in zloc:
        tmp_z.append(lax.cond(zloc[i].shape[1] != 0, lambda: vmap_zcoe(sample, zloc[i]), lambda: jnp.repeat(jnp.array([1. + 0j]), zloc[i].shape[0])))
        if rotation:
            tmp_array.append(coe_array_off_diag[i])  # the original coefficient of the interaction strength
    coe_tmp_y = jnp.concatenate(tmp_y, axis = 0)
    coe_tmp_z = jnp.concatenate(tmp_z, axis = 0)
    if rotation:
        coe_tmp_array = jnp.concatenate(tmp_array, axis = 0)


    if rotation:
        return coe_tmp_y * coe_tmp_z * coe_tmp_array
    else:
        return coe_tmp_y * coe_tmp_z * coe_array_off_diag

@jax.jit
def diag_coe(samples, zloc_bulk_diag, zloc_fl_diag, zloc_xzz_diag, coe_bulk_diag, coe_fl_diag, coe_xzz_diag):
    def scan_z(samples, z_array):
        return ((-1) ** samples[z_array]).prod()
    vmap_scan_z = vmap(scan_z, (None, 0))
    coe_bulk = (vmap_scan_z(samples, zloc_bulk_diag)*coe_bulk_diag).sum() if zloc_bulk_diag.size!=0 else 0
    coe_fl = (vmap_scan_z(samples, zloc_fl_diag)*coe_fl_diag).sum() if zloc_fl_diag.size!=0 else 0
    coe_xzz = (vmap_scan_z(samples, zloc_xzz_diag)*coe_xzz_diag).sum() if zloc_xzz_diag.size!=0 else 0
    return coe_bulk+coe_fl+coe_xzz

def loc_array_es(N):
    # Create a range of values from 0 to N-3
    i_values = jnp.arange(N - 3)[:, None]  # Reshape to use broadcasting
    # Create the loc_array_bulk using broadcasting
    loc_array_bulk = i_values + jnp.array([0, 1, 2])
    loc_array_first_last = jnp.array([[0 ,1],[N-2, N-1]])
    loc_array_left = jnp.array([[N-3, N-2, N-1]])
    return loc_array_bulk, loc_array_first_last, loc_array_left

def off_diag_count(xy_loc_bulk, xy_loc_fl, xy_loc_xzz):
    off_diag_bulk_count, off_diag_fl_count, off_diag_xzz_count = 0, 0, 0
    for i in xy_loc_bulk:
        if i[1] != 0:
            off_diag_bulk_count += xy_loc_bulk[i].shape[0]
        else:
            diag_bulk_count = 1
    for i in xy_loc_fl:
        if i[1] != 0:
            off_diag_fl_count += xy_loc_fl[i].shape[0]
        else:
            diag_fl_count = 1
    for i in xy_loc_xzz:
        if i[1] != 0:
            off_diag_xzz_count += xy_loc_xzz[i].shape[0]
        else:
            diag_xzz_count = 1
    return off_diag_bulk_count, off_diag_fl_count, off_diag_xzz_count

def vmc_off_diag_es(N, p, angle, basis_rotation):
    x, y = jnp.cos(angle), jnp.sin(angle)
    if (basis_rotation == False):
        '''
        # create pauli matrices, 1 stands for pauli x and 3 stands for pauli z, fl means first and last, l means the left one
        Bulk means the XZX terms for the bulk. There are (N-3) terms of them.
        fl means the first and last term which is ZX and XX acting on the first two sites and last two sites respectively
        xzz is the term XZZ acting on the last three sites
        '''
        pauli_array_bulk, pauli_array_fl, pauli_array_xzz = jnp.repeat(jnp.array([1, 3, 1])[None], (N*p - 3), axis=0), jnp.array([[3, 1], [1, 1]]), jnp.array( [[1, 3, 3]])
        loc_array_bulk, loc_array_fl, loc_array_xzz = loc_array_es(N*p)
    else:
        '''
        First repeat for each location then iterate over the combinations
        [[1,1,1]...,[1,1,1],[1,1,3], [1,1,3]..., [3,3,3],[3,3,3]]
        '''
        pauli_array_bulk, pauli_array_fl, pauli_array_xzz = (jnp.repeat(generate_combinations(3), (N*p - 3),axis=0),
                                                             jnp.repeat(generate_combinations(2), 2,axis=0),
                                                             jnp.repeat(generate_combinations(3), 1, axis=0))
        # The location that each Hamiltonian term acts on
        loc_array_bulk, loc_array_fl, loc_array_xzz = loc_array_es(N*p)
        loc_array_bulk, loc_array_fl, loc_array_xzz = jnp.tile(loc_array_bulk, (8, 1)), jnp.tile(loc_array_fl,(4, 1)), jnp.tile(loc_array_xzz, (8, 1))

    '''
    label_xxx[y, x] is a dict datatype and it is the location of loc_array_xxx 
    such that pauli_array_bulk.at[label[i][:,0].astype(int), label[i][:,1].astype(int)] will
    show the pauli matrix that acts on lattice location (y, x). This function coupled with pauli_cmi_pattern
    are used previously to change the measurement basis  with different density. We actually don't need this function here 
    
    We group the location that each Hamiltonian term acts on according to how many x,y,z they have in each term
    XX_loc_YYY is a dict datatype and its key is the number of Z-term and X-term (Z, X) and its value is the location
    of corresponding XX type of interaction acting on the lattice

    And off_diag_count is to count how many off-diagonal terms are there when we do VMC. It's just the total number of terms involving X and Y.
    off-diag_coe is the corresponding coeffiecient for each off-diagonal term. 
    '''
    if (basis_rotation == False):
        xy_loc_bulk, yloc_bulk, zloc_bulk = local_element_indices_1d(3, pauli_array_bulk, loc_array_bulk)
        xy_loc_fl, yloc_fl, zloc_fl = local_element_indices_1d(2, pauli_array_fl, loc_array_fl)
        xy_loc_xzz, yloc_xzz, zloc_xzz = local_element_indices_1d(3, pauli_array_xzz, loc_array_xzz)
        off_diag_bulk_count, off_diag_fl_count, off_diag_xzz_count = off_diag_count(xy_loc_bulk, xy_loc_fl, xy_loc_xzz)
        off_diag_bulk_coe, off_diag_fl_coe, off_diag_xzz_coe = -jnp.ones(off_diag_bulk_count), -jnp.ones(
            off_diag_fl_count), -jnp.ones(off_diag_xzz_count)
    else:
        xy_loc_bulk, yloc_bulk, zloc_bulk, off_diag_bulk_coe = local_element_indices_1d(3, pauli_array_bulk,
                                                                                        loc_array_bulk, rotation=True,
                                                                                        angle=angle)
        xy_loc_fl, yloc_fl, zloc_fl, off_diag_fl_coe = local_element_indices_1d(2, pauli_array_fl, loc_array_fl,
                                                                                rotation=True, angle=angle)
        xy_loc_xzz, yloc_xzz, zloc_xzz, off_diag_xzz_coe = local_element_indices_1d(3, pauli_array_xzz, loc_array_xzz,
                                                                                    rotation=True, angle=angle)

    zloc_bulk_diag, zloc_fl_diag, zloc_xzz_diag = jnp.array([]), jnp.array([]), jnp.array([])
    coe_bulk_diag, coe_fl_diag, coe_xzz_diag = jnp.array([]), jnp.array([]), jnp.array([])

    # Here we get the diagonal term and its coefficient of the Hamiltonian
    if (3, 0) in xy_loc_bulk:
        if zloc_bulk[(3, 0)].size != 0:
            zloc_bulk_diag = zloc_bulk[(3, 0)]  # label the diagonal term by zloc_bulk_diag
            if (basis_rotation == False):
                # it's fine here since no ZZ term exist in the original Hamiltonian
                coe_bulk_diag = -jnp.ones(zloc_bulk_diag.shape[0])
            else:
                # For xy_loc_bulk, the original term is XZX, for it rotate to ZZZ, it will obtain a cos(\theta)*sin^2(\theta) coeffiecient
                coe_bulk_diag = -jnp.ones(zloc_bulk_diag.shape[0]) * x * y ** 2
        del xy_loc_bulk[(3, 0)]
        del yloc_bulk[(3, 0)]
        del zloc_bulk[(3, 0)]
    if (2, 0) in xy_loc_fl:
        if zloc_fl[(2, 0)].size != 0:
            zloc_fl_diag = zloc_fl[(2, 0)]
            if (basis_rotation == False):
                # it's fine here since no ZZ term exist in the original Hamiltonian
                coe_fl_diag = jnp.ones(zloc_fl_diag.shape[0])
            else:
                # ZX term rotate to ZZ term, it will obtain a cos(\theta)*sin(\theta) coeffiecient
                # XX term rotate to ZZ term, it will obtain a sin^2(\theta) coeffiecient
                coe_fl_diag = jnp.concatenate((-jnp.ones(int(zloc_fl_diag.shape[0] / 2)) * x * y,
                                               -jnp.ones(int(zloc_fl_diag.shape[0] / 2)) * y ** 2))
        del xy_loc_fl[(2, 0)]
        del yloc_fl[(2, 0)]
        del zloc_fl[(2, 0)]
    if (3, 0) in xy_loc_xzz:
        if zloc_xzz[(3, 0)].size != 0:
            zloc_xzz_diag = zloc_xzz[(3, 0)]
            if (basis_rotation == False):
                # it's fine here since no ZZ term exist in the original Hamiltonian
                coe_xzz_diag = jnp.ones(zloc_xzz_diag.shape[0])
            else:
                # XZZ term rotate to ZZZ term, it will obtain a cos^2(\theta)*sin(\theta) coeffiecient
                coe_xzz_diag = -jnp.ones(zloc_xzz_diag.shape[0]) * x ** 2 * y
        del xy_loc_xzz[(3, 0)]
        del yloc_xzz[(3, 0)]
        del zloc_xzz[(3, 0)]
    return (xy_loc_bulk, xy_loc_fl, xy_loc_xzz, yloc_bulk, yloc_fl, yloc_xzz, zloc_bulk, zloc_fl,
            zloc_xzz, off_diag_bulk_coe, off_diag_fl_coe, off_diag_xzz_coe, zloc_bulk_diag, zloc_fl_diag,
            zloc_xzz_diag, coe_bulk_diag, coe_fl_diag, coe_xzz_diag)



def generate_combinations(length):
    set1 = [1] * length
    set2 = [3] * length
    combinations = list(itertools.product(*zip(set1, set2)))
    return jnp.array(combinations)


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

def clip_grad(g, clip_norm=10.0):
    norm = jnp.linalg.norm(g)
    scale = jnp.minimum(1.0, clip_norm / (norm + 1e-6))
    return g * scale

def schedule(step: float, min_lr: float, max_lr: float, period: float) -> float:
    """Compute a learning rate that oscillates sinusoidally between min_lr and max_lr."""
    oscillation = (jnp.sin(jnp.pi * step / period) + 1) / 2  # Will be between 0 and 1
    return min_lr + (max_lr - min_lr) * oscillation

