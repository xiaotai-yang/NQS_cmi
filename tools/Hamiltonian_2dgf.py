
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

def generate_combinations(length):
    set1 = [1] * length
    set2 = [3] * length
    combinations = list(itertools.product(*zip(set1, set2)))
    return jnp.array(combinations)
def local_element_indices_2d(num_body, pauli_array, loc_array, rotation = False, angle = 0.0):
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
    #jax.debug.print("pauli_array:{}", pauli_array)
    #jax.debug.print("loc_array:{}", loc_array)
    #Count the number of pauli_Z and pauli_X for each term in the Hamiltonian
    count_3s = jnp.sum(pauli_array == 3, axis = 1)
    count_1s = jnp.sum(pauli_array == 1, axis = 1)

    pauli_array_xz = {}
    xloc_arrays = {}
    zloc_arrays = {}
    yloc_arrays = {}
    xy_loc_arrays = {}
    coe_arrays = {}
    for i in range(num_body+1):    #z_number
        for j in range (num_body+1-i):  #x_number
            mask = ((count_3s == i)&(count_1s == j))  #mask for the interaction with i pauli_Z and j pauli_X
            pauli_array_xz[i, j] = pauli_array[mask]

            mask_x = (pauli_array_xz[i, j] == 1)
            mask_y = (pauli_array_xz[i, j] == 2)
            mask_z = (pauli_array_xz[i, j] == 3)
            ref_bulk_z_mask = jnp.array([True, False, False, False, False])
            ref_edge_z_mask = jnp.array([True, False, False, False])
            ref_corner_z_mask = jnp.array([True, False, False])
            x, y = jnp.cos(angle), jnp.sin(angle)
            if rotation:
                if num_body == 5:
                    coe_arrays[i, j] = -(-1)**(jnp.sum((mask_z == False) & (mask_z != ref_bulk_z_mask), axis = 1))*x**(jnp.sum(mask_z == ref_bulk_z_mask, axis = 1))*y**(jnp.sum(mask_z != ref_bulk_z_mask, axis = 1))
                elif num_body == 4:
                    coe_arrays[i, j] = -(-1)**(jnp.sum((mask_z == False) & (mask_z != ref_edge_z_mask), axis = 1))*x**(jnp.sum(mask_z == ref_edge_z_mask, axis = 1))*y**(jnp.sum(mask_z != ref_edge_z_mask, axis = 1))
                elif num_body == 3:
                    coe_arrays[i, j] = -(-1)**(jnp.sum((mask_z == False) & (mask_z != ref_corner_z_mask), axis = 1))*x**(jnp.sum(mask_z == ref_corner_z_mask, axis = 1))*y**(jnp.sum(mask_z != ref_corner_z_mask, axis = 1))
            if mask_x.sum() != 0:
                xloc_arrays[i, j] = loc_array[mask][mask_x].reshape(-1, j, 2)
            elif mask_y.sum() != 0 or mask_z.sum() !=0:
                if mask_y.sum()!=0:
                    xloc_arrays[i, j] = jnp.array([[]]).reshape(pauli_array_xz[i ,j].shape[0], 0, 2).astype(int)
                else:
                    xloc_arrays[i, j] = jnp.array([[]]).reshape(pauli_array_xz[i, j].shape[0], 0, 2).astype(int)

            if mask_y.sum() !=0:
                yloc_arrays[i, j] = loc_array[mask][mask_y].reshape(-1, num_body-i-j, 2).astype(int)
            elif mask_x.sum() != 0 or mask_z.sum() !=0:
                if mask_x.sum()!=0:
                    yloc_arrays[i, j] = jnp.array([[]]).reshape(pauli_array_xz[i, j].shape[0], 0, 2).astype(int)
                else:
                    yloc_arrays[i, j] = jnp.array([[]]).reshape(pauli_array_xz[i, j].shape[0], 0, 2).astype(int)

            if mask_z.sum()!=0:
                zloc_arrays[i, j] = loc_array[mask][mask_z].reshape(-1, i, 2).astype(int)
            elif mask_x.sum() != 0 or mask_y.sum() !=0:
                if mask_y.sum()!=0:
                    zloc_arrays[i, j] = jnp.array([[]]).reshape(pauli_array_xz[i, j].shape[0], 0, 2).astype(int)
                else:
                    zloc_arrays[i, j] = jnp.array([[]]).reshape(pauli_array_xz[i, j].shape[0], 0, 2).astype(int)

    for ind in (xloc_arrays):
        xy_loc_arrays[ind] = jnp.concatenate((xloc_arrays[ind], yloc_arrays[ind]), axis=1).astype(int)

    if rotation:
        return xy_loc_arrays, yloc_arrays, zloc_arrays, coe_arrays
    return  xy_loc_arrays, yloc_arrays, zloc_arrays

@jax.jit
def total_samples_2d(sample, xyloc_arrays):
    def single_samples_2d(sample, xy_loc_array_element):  #flip the spin for the location that the XY terms acts on
        return sample.at[xy_loc_array_element[:, 0], xy_loc_array_element[:, 1]].set((sample[xy_loc_array_element[:, 0], xy_loc_array_element[:, 1]] + 1) % 2)
    results = []
    vmap_single_samples_2d = vmap(single_samples_2d, (None, 0))
    for i in xyloc_arrays:
        xy_loc_arrays_elements = xyloc_arrays[i]
        tmp = vmap_single_samples_2d(sample, xy_loc_arrays_elements)
        results.append(tmp)
    # Concatenate all results along axis=0
    concatenated_result = jnp.concatenate(results, axis=0)
    return concatenated_result

@partial(jax.jit, static_argnames=['rotation'])
def new_coe_2d(sample, coe_array_off_diag, yloc, zloc, rotation):
    def ycoe(single_sample, yloc_arrays_element):
        return ((0-1j)**single_sample[yloc_arrays_element[:,0],yloc_arrays_element[:,1]]).prod()
    def zcoe(single_sample, zloc_arrays_element):
        return ((-1+0j)**single_sample[zloc_arrays_element[:,0], zloc_arrays_element[:,1]]).prod()

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
            tmp_array.append(coe_array_off_diag[i])
    coe_tmp_y = jnp.concatenate(tmp_y, axis = 0)
    coe_tmp_z = jnp.concatenate(tmp_z, axis = 0)
    if rotation:
        coe_tmp_array = jnp.concatenate(tmp_array, axis = 0)
    #jax.debug.print("zloc:{}", zloc)
    #jax.debug.print("sample:{}", sample)

    if rotation:
        #jax.debug.print("result:{}", (coe_tmp_y * coe_tmp_z * coe_tmp_array))
        return coe_tmp_y * coe_tmp_z * coe_tmp_array
    else:
        #jax.debug.print("result:{}", (coe_tmp_y * coe_tmp_z * coe_array_off_diag))
        return coe_tmp_y * coe_tmp_z * coe_array_off_diag
@jax.jit
def diag_coe(samples, zloc_bulk_diag, zloc_edge_diag, zloc_corner_diag, coe_bulk_diag, coe_edge_diag, coe_corner_diag):
    def scan_z(samples, z_array):
        return ((-1) ** samples[z_array[:, 0], z_array[:, 1]]).prod()
    vmap_scan_z = vmap(scan_z, (None, 0))
    coe_bulk = (vmap_scan_z(samples, zloc_bulk_diag)*coe_bulk_diag).sum() if zloc_bulk_diag.size!=0 else 0
    coe_edge = (vmap_scan_z(samples, zloc_edge_diag)*coe_edge_diag).sum() if zloc_edge_diag.size!=0 else 0
    coe_corner = (vmap_scan_z(samples, zloc_corner_diag)*coe_corner_diag).sum() if zloc_corner_diag.size!=0 else 0
    return coe_bulk+coe_edge+coe_corner
def location_pauli_label(loc_array_bulk, loc_array_edge, loc_array_corner, Ny, Nx):
    '''
    Args:
        loc_array_bulk, loc_array_edge, loc_array_corner: Each element is the "location" that the Hamiltonian term acts on.

    Returns:
        Three dictionaries that label the location of the Hamiltonian term.
    For example,
    1. (0, 1) [[0. 4.]]
    means that the Hamiltonian terms act on the 0th row and 1st column of the grid
    is the 4th element of the 0th term in the Hamiltonian.

    2. (1, 2) [[0. 1.]
              [1. 0.]
              [3. 4.]]
    means that the Hamiltonian terms act on the 1st row and 2nd column of the grid
    is the 1st, 0th, 4th element of the 0th, 1st, 3rd terms in the Hamiltonian.
    '''

    label_bulk = {}
    label_edge = {}
    label_corner = {}
    # yi, xi : the location to label how many operation is acted on
    # y , x : The location of the loc_array that we can use to access the location to label
    for yi in range (Ny):
        for xi in range(Nx):
            y = 0
            tmp = jnp.array([])
            for i in loc_array_bulk:
                x = 0
                for j in i:
                    if (j == jnp.array([yi, xi])).all():
                        tmp = jnp.append(tmp, jnp.array([y, x]))
                    x += 1
                y += 1
            label_bulk[yi, xi] = tmp.reshape(-1, 2)
    for yi in range (Ny):
        for xi in range(Nx):
            y = 0
            tmp = jnp.array([])
            for i in loc_array_edge:
                x = 0
                for j in i:
                    if (j == jnp.array([yi, xi])).all():
                        tmp = jnp.append(tmp, jnp.array([y, x]))
                    x += 1
                y += 1
            label_edge[yi, xi] = tmp.reshape(-1, 2)
    for yi in range (Ny):
        for xi in range(Nx):
            y = 0
            tmp = jnp.array([])
            for i in loc_array_corner:
                x = 0
                for j in i:
                    if (j == jnp.array([yi, xi])).all():
                        tmp = jnp.append(tmp, jnp.array([y, x]))
                    x += 1
                y += 1
            label_corner[yi, xi] = tmp.reshape(-1, 2)
    return label_bulk, label_edge, label_corner

def loc_array_gf(Ny, Nx):
    '''
    Inputs
    Ny, Nx : The size of the grid

    Outputs:
    The location that each Hamiltonian term acts on.
    Specifically,

    For loc_array bulk: it is [[[1, 1], [1, 2], [2, 1], [1, 0], [0 ,1]]....[[Ny-2,Nx-2], [Ny-2, Nx-1], [Ny-1, Nx-2], [Ny-2, Nx-3], [Ny-3, Nx-2]]
    For single element, the order inside is :
        4
    3   0   1
        2
    For the entire array, it follows that it moves right first until the boundary and then go to the first element of the next row.

    For loc_array corner: it is [[[0, 0],[0 ,1], [1, 0]]...[[Ny - 1, Nx - 1], [Ny - 1, Nx - 2], [Ny - 2, Nx - 1]]]
    For single element, the order inside follows that dealing with x-axis first and then y-axis.
    '''
    I, J = jnp.meshgrid(jnp.arange(1, Ny - 1), jnp.arange(1, Nx - 1), indexing='ij')
    bulk_coordinates = jnp.stack([I, J, I, J + 1, I + 1, J, I, J - 1, I - 1, J], axis=-1)
    loc_array_bulk = bulk_coordinates.reshape(-1, 5, 2)

    # Add edge coordinates excluding corners for the left and right sides of the grid
    edge_coordinates = []
    for i in range(1, Ny - 1):
        edge_coordinates.extend([[i, 0], [i, 1], [i + 1, 0], [i - 1, 0]])
        edge_coordinates.extend([[i, Nx - 1], [i, Nx - 2], [i + 1, Nx - 1], [i - 1, Nx - 1]])

    # Add edge coordinates excluding corners for the top and bottom of the grid
    for j in range(1, Nx - 1):
        edge_coordinates.extend([[0, j], [1, j], [0, j - 1], [0, j + 1]])
        edge_coordinates.extend([[Ny - 1, j], [Ny - 2, j], [Ny - 1, j - 1], [Ny - 1, j + 1]])

    # Convert list to a JAX array
    loc_array_edge = jnp.array(edge_coordinates).reshape(-1, 4, 2)

    loc_array_corner = jnp.array([[[0, 0], [0, 1], [1, 0]],
                                  [[0, Nx - 1], [0, Nx - 2], [1, Nx - 1]],
                                  [[Ny - 1, 0], [Ny - 1, 1], [Ny - 2, 0]],
                                  [[Ny - 1, Nx - 1], [Ny - 1, Nx - 2], [Ny - 2, Nx - 1]]])
    return loc_array_bulk, loc_array_edge, loc_array_corner
def off_diag_count(xy_loc_bulk, xy_loc_edge, xy_loc_corner):
    off_diag_bulk_count, off_diag_edge_count, off_diag_corner_count = 0, 0, 0
    for i in xy_loc_bulk:
        if i[1] != 0:
            off_diag_bulk_count += xy_loc_bulk[i].shape[0]
        else:
            diag_bulk_count = 1
    for i in xy_loc_edge:
        if i[1] != 0:
            off_diag_edge_count += xy_loc_edge[i].shape[0]
        else:
            diag_edge_count = 1
    for i in xy_loc_corner:
        if i[1] != 0:
            off_diag_corner_count += xy_loc_corner[i].shape[0]
        else:
            diag_corner_count = 1
    return off_diag_bulk_count, off_diag_edge_count, off_diag_corner_count

def vmc_off_diag_es(Ny, Nx, px, py, angle, basis_rotation):
    x, y = jnp.cos(angle), jnp.sin(angle)
    if (basis_rotation == False):
    # create pauli matrices, 1 stands for pauli x and 3 stands for pauli z

        pauli_array_bulk, pauli_array_edge, pauli_array_corner  = jnp.repeat(jnp.array([1,3,3,3,3])[None], (Ny*py-2)*(Nx*px-2), axis=0), jnp.repeat(jnp.array([1,3,3,3])[None], (Ny*py+Nx*px-4)*2, axis=0), jnp.repeat(jnp.array([1,3,3])[None], 4, axis=0)
        loc_array_bulk, loc_array_edge, loc_array_corner = loc_array_gf(Ny*py, Nx*px)
    else :
        total_samples =         '''
        First repeat for each location then iterate over the combinations
        [[1,1,1,1,1]...,[1,1,1,1,1],[1,1,1,1,3]...[3,3,3,3,3]]
        '''
        pauli_array_bulk, pauli_array_edge, pauli_array_corner = jnp.repeat(generate_combinations(5), (Ny*py-2)*(Nx*px-2), axis=0), jnp.repeat(generate_combinations(4),(Ny*py+Nx*px-4)*2, axis=0), jnp.repeat(generate_combinations(3), 4, axis=0)

        # The location that each Hamiltonian term acts on
        loc_array_bulk, loc_array_edge, loc_array_corner = loc_array_gf(Ny*py, Nx*px)
        loc_array_bulk, loc_array_edge, loc_array_corner  = jnp.tile(loc_array_bulk, (32, 1, 1)), jnp.tile(loc_array_edge, (16, 1, 1)), jnp.tile(loc_array_corner, (8, 1, 1 ))


    '''
    label_xxx[y, x] is a dict datatype and it is the location of loc_array_xxx 
    such that pauli_array_bulk.at[label[i][:,0].astype(int), label[i][:,1].astype(int)] will
    show the pauli matrix that acts on lattice location
    '''
    label_bulk, label_edge, label_corner = location_pauli_label(loc_array_bulk, loc_array_edge, loc_array_corner, Ny*py, Nx*px)

    '''
    We group the location that each Hamiltonian term acts on according to how many x,y,z they have in each term
    XX_loc_YYY is a dict datatype and its key is the number of Z-term and X-term (Z, X) and its value is the location
    of corresponding XX type of interaction acting on the lattice 
    '''
    if (basis_rotation == False):
        xy_loc_bulk, yloc_bulk, zloc_bulk = local_element_indices_2d(5, pauli_array_bulk, loc_array_bulk)
        xy_loc_edge, yloc_edge, zloc_edge = local_element_indices_2d(4, pauli_array_edge, loc_array_edge)
        xy_loc_corner, yloc_corner, zloc_corner = local_element_indices_2d(3, pauli_array_corner, loc_array_corner)
        off_diag_bulk_count, off_diag_edge_count, off_diag_corner_count = off_diag_count(xy_loc_bulk, xy_loc_edge, xy_loc_corner)
        off_diag_bulk_coe, off_diag_edge_coe, off_diag_corner_coe = -jnp.ones(off_diag_bulk_count), -jnp.ones(off_diag_edge_count), -jnp.ones(off_diag_corner_count)
    else :
        xy_loc_bulk, yloc_bulk, zloc_bulk, off_diag_bulk_coe = local_element_indices_2d(5, pauli_array_bulk, loc_array_bulk, rotation = True, angle = angle)
        xy_loc_edge, yloc_edge, zloc_edge, off_diag_edge_coe = local_element_indices_2d(4, pauli_array_edge, loc_array_edge, rotation = True, angle = angle)
        xy_loc_corner, yloc_corner, zloc_corner, off_diag_corner_coe = local_element_indices_2d(3, pauli_array_corner, loc_array_corner, rotation = True, angle = angle)

    zloc_bulk_diag, zloc_edge_diag, zloc_corner_diag = jnp.array([]), jnp.array([]), jnp.array([])
    coe_bulk_diag, coe_edge_diag, coe_corner_diag = jnp.array([]), jnp.array([]), jnp.array([])

    if (5, 0) in xy_loc_bulk:
        if zloc_bulk[(5, 0)].size!=0:
            zloc_bulk_diag = zloc_bulk[(5, 0)]     #label the diagonal term by zloc_bulk_diag
            # ZXXXX to ZZZZZ
            if (basis_rotation == False):
                coe_bulk_diag = -jnp.ones(zloc_bulk_diag.shape[0])
            else:
                coe_bulk_diag = -jnp.ones(zloc_bulk_diag.shape[0])*x*y**4
                #Here is the coefficient for the diagonal term. We can change it later if we want
        del xy_loc_bulk[(5, 0)]
        del yloc_bulk[(5, 0)]
        del zloc_bulk[(5, 0)]
    if (4, 0) in xy_loc_edge:
        if zloc_edge[(4, 0)].size!=0:
            zloc_edge_diag = zloc_edge[(4, 0)]
            # ZXXX to ZZZZ
            if (basis_rotation == False):
                coe_edge_diag = -jnp.ones(zloc_edge_diag.shape[0])
            else:
                coe_edge_diag = -jnp.ones(zloc_edge_diag.shape[0])*x*y**3
        del xy_loc_edge[(4, 0)]
        del yloc_edge[(4, 0)]
        del zloc_edge[(4, 0)]
    if (3, 0) in xy_loc_corner:
        if zloc_corner[(3, 0)].size!=0:
            zloc_corner_diag = zloc_corner[(3, 0)]
            # ZXX to ZZZ
            if (basis_rotation == False):
                coe_corner_diag = -jnp.ones(zloc_corner_diag.shape[0])
            else:
                coe_corner_diag = -jnp.ones(zloc_corner_diag.shape[0])*x*y**2
        del xy_loc_corner[(3, 0)]
        del yloc_corner[(3, 0)]
        del zloc_corner[(3, 0)]

    return (xy_loc_bulk, xy_loc_edge, xy_loc_corner, yloc_bulk, yloc_edge, yloc_corner, zloc_bulk, zloc_edge,
            zloc_corner, off_diag_bulk_coe, off_diag_edge_coe, off_diag_corner_coe, zloc_bulk_diag, zloc_edge_diag,
            zloc_corner_diag, coe_bulk_diag, coe_edge_diag, coe_corner_diag)
