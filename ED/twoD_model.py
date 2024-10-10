import jax
import netket as nk
import numpy as np
from netket.operator.spin import sigmax,sigmaz, sigmap, sigmam, identity
import time
from scipy.sparse.linalg import eigsh
import jax.numpy as jnp
import matplotlib.pyplot as plt
from twoD_tool import *
from jax.random import PRNGKey, categorical, split

L = 4
N = L*L
periodic = False
hi = nk.hilbert.Spin(s=1 / 2, N =  N)
model = "2DRyberg"

if model == "2DXXZ":
    int_ = "delta"
    params = [1.2, 1.05, 1., 0.95, 0.8, 0.2, -0.2, -0.8, -0.95, -1.0, -1.05, -1.2]  #sigmaz interaction
elif model == "2DJ1J2":
    int_ = "J2"
    params = [0.2, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.8, 1.0, 1.05, 1.2] #J2
elif model == "2DTFIM":
    int_ = "B"
    params =  [0, -0.5, -1.0, -1.2, -1.4, -1.6, -1.8, -2.0, -2.5, -4.0] #magnetic field
elif model == "2DRyberg":
    int_ = "delta"
    params = [ 0.0, 0.4, 0.8, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.8, 3.2, 3.6] #delta
for param in params :
    if model == "2DXXZ":
        H = sum([2*(sigmap(hi, y*L+x)*sigmam(hi, (y+1)*L+x)+sigmam(hi, y*L+x)*sigmap(hi, (y+1)*L+x))+param*sigmaz(hi, y*L+x)*sigmaz(hi, (y+1)*L+x) for y in range(L-1) for x in range(L)]) #up-down J1
        H += sum([2*(sigmap(hi, y*L+x)*sigmam(hi, y*L+x+1)+sigmam(hi, y*L+x)*sigmap(hi, y*L+x+1))+param*sigmaz(hi, y*L+x)*sigmaz(hi, y*L+x+1) for y in range(L) for x in range(L-1)]) #left-right J1
        if (periodic == True):
            H+= sum([2*(sigmap(hi, x)*sigmam(hi, (L-1)*L+x)+sigmam(hi, x)*sigmap(hi, (L-1)*L+x))+param*sigmaz(hi, x)*sigmaz(hi, (L-1)*L+x) for x in range(L)]) # last row - first row J1
            H+= sum([2*(sigmap(hi, y*L)*sigmam(hi, y*L+L-1)+sigmam(hi, y*L)*sigmap(hi, y*L+L-1))+param*sigmaz(hi, y*L)*sigmaz(hi, y*L+L-1) for y in range(L)]) # last column - first column J1
        H/=4
    elif model == "2DJ1J2":
        H = sum([2*(sigmap(hi, y*L+x)*sigmam(hi, (y+1)*L+x)+sigmam(hi, y*L+x)*sigmap(hi, (y+1)*L+x))+sigmaz(hi, y*L+x)*sigmaz(hi, (y+1)*L+x) for y in range(L-1) for x in range(L)]) #up-down J1
        H += sum([2*(sigmap(hi, y*L+x)*sigmam(hi, y*L+x+1)+sigmam(hi, y*L+x)*sigmap(hi, y*L+x+1))+sigmaz(hi, y*L+x)*sigmaz(hi, y*L+x+1) for y in range(L) for x in range(L-1)]) #left-right J1
        H += param*sum([2*(sigmap(hi, y*L+x)*sigmam(hi, (y+1)*L+x+1)+sigmam(hi, y*L+x)*sigmap(hi, (y+1)*L+x+1))+sigmaz(hi, y*L+x)*sigmaz(hi, (y+1)*L+x+1) for y in range(L-1) for x in range(L-1)]) #right-down J2
        H += param*sum([2*(sigmap(hi, y*L+x+1)*sigmam(hi, (y+1)*L+x)+sigmam(hi, y*L+x+1)*sigmap(hi, (y+1)*L+x))+sigmaz(hi, y*L+x+1)*sigmaz(hi, (y+1)*L+x) for y in range(L-1) for x in range(L-1)]) #left-down J2
        if (periodic == True):
        #periodic boundary conditions
            H+= sum([2*(sigmap(hi, x)*sigmam(hi, (L-1)*L+x)+sigmam(hi, x)*sigmap(hi, (L-1)*L+x))+sigmaz(hi, x)*sigmaz(hi, (L-1)*L+x) for x in range(L)]) # last row - first row J1
            H+= sum([2*(sigmap(hi, y*L)*sigmam(hi, y*L+L-1)+sigmam(hi, y*L)*sigmap(hi, y*L+L-1))+sigmaz(hi, y*L)*sigmaz(hi, y*L+L-1) for y in range(L)]) # last column - first column J1
            H+= sum([2*(sigmap(hi, y*L+L-1)*sigmam(hi, (y+1)*L)+sigmam(hi, y*L+L-1)*sigmap(hi, (y+1)*L))+sigmaz(hi, y*L+L-1)*sigmaz(hi, (y+1)*L) for y in range(L-1)]) # last column - first column J2 (right down)
            H+= param*sum([2*(sigmap(hi, y*L)*sigmam(hi, (y+2)*L-1)+sigmam(hi, y*L)*sigmap(hi, (y+2)*L-1))+sigmaz(hi, y*L)*sigmaz(hi, (y+2)*L-1) for y in range(L-1)]) #  last column - first column J2 (left down)
            H+= param*sum([2*(sigmap(hi, x+1)*sigmam(hi, (L-1)*L+x)+sigmam(hi, x+1)*sigmap(hi, (L-1)*L+x))+sigmaz(hi, x+1)*sigmaz(hi, (L-1)*L+x) for x in range(L-1)]) # last row - first row J2 (right down)
            H+= param*sum([2*(sigmap(hi, x)*sigmam(hi, (L-1)*L+x+1)+sigmam(hi, x)*sigmap(hi, (L-1)*L+x+1))+sigmaz(hi, x)*sigmaz(hi, (L-1)*L+x+1) for x in range(L-1)]) # last row - first row J2 (left down)
            H+= param*(2*(sigmap(hi, L*L-1)*sigmam(hi, 0)+sigmam(hi, L*L-1)*sigmap(hi, 0))+sigmaz(hi, L*L-1)*sigmaz(hi, 0)) # right down corner J2
            H+= param*(2*(sigmap(hi, L*(L-1))*sigmam(hi, L-1)+sigmam(hi, L*(L-1))*sigmap(hi, L-1))+sigmaz(hi, L*(L-1))*sigmaz(hi, L-1)) # left down corner J2
        H/=4
    elif model == "2DTFIM":
        H = -sum([sigmaz(hi, y*L+x)*sigmaz(hi, (y+1)*L+x) for y in range(L-1) for x in range(L)])  #up-down
        H -= sum([sigmaz(hi, y*L+x)*sigmaz(hi, y*L+x+1) for y in range(L) for x in range(L-1)]) #left-right
        H += 2*param*sum([sigmax(hi, y*L+x) for y in range(L) for x in range(L)]) # B
        if (periodic == True):
        #periodic boundary conditions
            H-= sum([sigmaz(hi, x)*sigmaz(hi, (L-1)*L+x) for x in range(L)]) # last row - first row
            H-= sum([sigmaz(hi, y*L)*sigmaz(hi, y*L+L-1) for y in range(L)]) # last column - first column
        H/=4
    elif model == "2DRyberg":
        numsamples = 1024
        batch_categorical = jax.vmap(categorical, in_axes=(0, None))
        batch_dot = jax.vmap(jnp.dot, (0, None))
        key = PRNGKey(1)
        Omega = 1.0
        Rb = 3.
        H = Omega/2*sum([sigmax(hi, y*L+x) for y in range (L) for x in range (L)]) #X
        H -= param/2*sum([(identity(hi)-sigmaz(hi, y*L+x)) for y in range(L) for x in range(L)])
        H += Omega*Rb/4*sum([((identity(hi)-sigmaz(hi, y1*L+x1))*(identity(hi)-sigmaz(hi, y1*L+x2)))/((x1-x2)**2)**3 \
                             for y1 in range(L) for x1 in range(L) for x2 in range(x1+1, L)])
        H += Omega*Rb/4*sum([((identity(hi)-sigmaz(hi, y1*L+x1))*(identity(hi)-sigmaz(hi, y2*L+x2)))/(((x1-x2)**2+(y1-y2)**2)**3) \
                             for y1 in range(L) for x1 in range(L) for y2 in range(y1+1, L) for x2 in range(L)])
    sp_h = H.to_sparse()
    eig_vals, eig_vecs = eigsh(sp_h, k=2, which="SA")
    print("eigenvalues with scipy sparse " +int_+"="+str(param) +":", eig_vals)
    prob_exact = eig_vecs[:,0]**2
    mag = np.sum(prob_exact*count_diff_ones_zeros(L**2))
    magH = sum([sigmaz(hi, y*L+x) for y in range(L) for x in range (L)])
    mag1 = eig_vecs[:,0] @ magH.to_sparse() @ eig_vecs[:,0]
    print(mag, mag1)
    shape = (2,) * (L**2)
    prob_exact = prob_exact.reshape(*shape)
    mean_corr, var_corr = correlation_all(prob_exact, L)
    cmi = cmi_(prob_exact, L)
    cmi_all = cmi_traceout(prob_exact, L)

    if model == "2DRyberg":
        key, subkey = split(key, 2)
        key_ = split(subkey, numsamples)
        samples = batch_categorical(key_, jnp.log(jnp.real(eig_vecs[:, 0].conj() * eig_vecs[:, 0])))
        samples_b = jnp.flip(jnp.unpackbits(samples.view('uint8'), bitorder='little').reshape(numsamples, -1), axis=1)[:, 64-L**2:]
        alt_m = create_alternating_matrix(L)
        stagger_mag = jnp.mean(jnp.abs(batch_dot(samples_b.reshape(numsamples, -1), create_alternating_matrix(L).ravel().T)) / (L ** 2))

    np.save("result/"+model+"/gap_"+model+"_L"+str(L)+"_"+int_+"_"+str(param)+"periodic_"+str(periodic)+".npy", np.array(eig_vals[1]-eig_vals[0]))
    np.save("result/"+model+"/cmi_"+model+"_L"+str(L)+"_"+int_+"_"+str(param)+"periodic_"+str(periodic)+".npy", cmi)
    np.save("result/"+model+"/mean_corr_"+model+"_L"+str(L)+"_"+int_+"_"+str(param)+"periodic_"+str(periodic)+".npy", mean_corr)
    np.save("result/"+model+"/var_corr_"+model+"_L"+str(L)+"_"+int_+"_"+str(param)+"periodic_"+str(periodic)+".npy", var_corr)
    np.save("result/"+model+"/cmi_traceout_"+model+"_L"+str(L)+"_"+int_+"_"+str(param)+"periodic_"+str(periodic)+".npy", cmi_all)
    np.save("result/"+model+"/mag_"+model+"_L"+str(L)+"_"+int_+"_"+str(param)+"periodic_"+str(periodic)+".npy", mag)
    np.save("result/"+model+"/stagger_mag_"+model+"_L"+str(L)+"_"+int_+"_"+str(param)+"periodic_"+str(periodic)+".npy", stagger_mag)