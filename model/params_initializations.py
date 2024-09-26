import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
import jax.random as random
from jax.random import PRNGKey, split, categorical
from jax.nn.initializers import he_normal, he_uniform, glorot_normal, glorot_uniform, orthogonal
re = jnp.repeat
ru = jax.random.uniform
gu = glorot_uniform()
def random_layer_params(N, m, n, key):
    w_key, b_key = random.split(key)
    return  (2*random.uniform(w_key, (N, m, n))-1)/jnp.sqrt((m+n)/2),  (2*random.normal(b_key, (N, m))-1)/jnp.sqrt((m+n)/2)

def init_network_params(sizes, N,  key):
    keys = random.split(key, len(sizes))
    outkey = keys[0]
    return outkey, [random_layer_params(N, m, n, k) for m, n, k in zip(sizes[1:], sizes[:-1], keys[1:])]

def init_tensor_gru_params(input_size, units, N, key):
    # input is already concantenated
    key, u_params = init_network_params([(units * input_size), units], N, key)
    key, r_params = init_network_params([ (units * input_size), units], N, key)
    key, s_params = init_network_params([(units * input_size), units], N, key)
    key, amp_params = init_network_params([units, input_size], N, key)
    key, phase_params = init_network_params([units, input_size], N, key)

    Wu, bu, Wr, br, Ws, bs, = u_params[0][0], u_params[0][1], r_params[0][0], r_params[0][1], s_params[0][0], s_params[0][1],
    Wamp, bamp, Wphase, bphase = amp_params[0][0] * 0, amp_params[0][1] * 0, phase_params[0][0], phase_params[0][1]

    return (Wu, bu, Wr, br, Ws, bs, Wamp, bamp, Wphase, bphase)

def random_2dlayer_params(ny, nx, m, n, key):
    w_key, b_key = random.split(key)
    # outkey1, outkey2 = random.split(w_key)
    return (2 * random.uniform(w_key, (ny, nx, m, n)) - 1) / jnp.sqrt(3 * (m + n) / 2), (
                2 * random.normal(b_key, (ny, nx, m)) - 1) / jnp.sqrt(3 * (m + n) / 2)

def init_2dnetwork_params(sizes, ny, nx, key):
    keys = random.split(key, len(sizes))
    outkey = keys[0]
    return outkey, [random_2dlayer_params(ny, nx, m, n, k) for m, n, k in zip(sizes[1:], sizes[:-1], keys[1:])]

def init_2dtensor_gru_params(input_size, units, Ny, Nx, key):
    # input is already concantenated
    key, u_params = init_2dnetwork_params( [ 4*(units * input_size), units], Ny, Nx, key)
    key, r_params = init_2dnetwork_params( [ 4*(units * input_size), units], Ny, Nx, key)
    key, s_params = init_2dnetwork_params( [ 4*(units * input_size), units], Ny, Nx, key)
    key, amp_params = init_2dnetwork_params([units, input_size], Ny, Nx, key)
    key, phase_params = init_2dnetwork_params([units, input_size], Ny, Nx, key)

    Wu, bu, Wr, br, Ws, bs = u_params[0][0], u_params[0][1], r_params[0][0], r_params[0][1], s_params[0][0], s_params[0][1]
    Wamp, bamp, Wphase, bphase = amp_params[0][0] * 0, amp_params[0][1] * 0, phase_params[0][0], phase_params[0][1]

    return (Wu, bu, Wr, br, Ws, bs, Wamp, bamp, Wphase, bphase)

def init_RWKV_params(input_size, emb_size, h_size, head, ff_size, num_layer, key):

    (k_, k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14) = split(key, 16)
    decay_size = 64
    D = h_size // head
    r0 = jnp.arange(num_layer)/(num_layer-1)
    r1 = 1 - jnp.arange(num_layer)/num_layer

    wemb = ru(k_, (input_size, emb_size), minval = -5e-4, maxval = 5e-4)

    # Initialization parameters
    x_init = ru(k0, (emb_size,), minval = -5e-4, maxval = 5e-4)
    last_x_init = ru(k1, (num_layer, emb_size), minval = -5e-4, maxval = 5e-4)
    t_init = ru(k2, (num_layer, head, D, D), minval = -5e-4, maxval = 5e-4)
    c_init = ru(k3, (num_layer, emb_size), minval = -5e-4, maxval = 5e-4)
    init_params = (x_init, last_x_init, t_init, c_init)

    # time mixing params
    # m are of the shape (num_layer, head, D)
    base = (jnp.arange(emb_size) / emb_size) [None, :]
    base1 = (jnp.arange(h_size)/(h_size-1))
    mx_t = 1 - re(base, num_layer, axis = 0) ** (r1)[:, None]
    mw_t = 1 - re(base, num_layer, axis = 0) ** (r1)[:, None]
    mk_t = 1 - re(base, num_layer, axis = 0) ** (r1)[:, None]
    mv_t = 1 - re(base, num_layer, axis = 0) ** (r1)[:, None] + 0.3 * r0[:, None]
    mr_t = 1 - re(base, num_layer, axis = 0) ** (r1 * 0.5)[:, None]
    mg_t = 1 - re(base, num_layer, axis = 0) ** (r1 * 0.5)[:, None]

    u = (r0[:, None] @ (1 - base1 + 0.1 * ((jnp.arange(h_size) + 1) % 3))[None, :]).reshape(num_layer, head, D)
    decay = -6 + 5 * re(base1[None, :], num_layer, axis = 0) ** ((0.7 + 1.3 * r0)[:, None])

    Ax, Bx = jnp.zeros((num_layer, emb_size, 5, 64)), ru(k4, (num_layer, 5, 64, emb_size), minval=-1e-2, maxval=1e-2)
    Wk_t, Wv_t, Wr_t, Wg_t, Wo_t = gu(k4, (num_layer, emb_size, h_size)), gu(k5, (num_layer, emb_size, h_size)), gu(k6, (num_layer, emb_size, h_size)), gu(k7, (num_layer, emb_size, h_size)), gu(k8, (num_layer, h_size, emb_size))
    Ww_t, Wd = jnp.zeros((num_layer, emb_size, decay_size)), ru(k9, (num_layer, decay_size, h_size), minval=-1e-2, maxval=1e-2)
    # WKV GroupNorm weights initialized with constant value ((l + i) / L)^0.7
    wln_t, bln_t = re(re((((1 + jnp.arange(num_layer))/num_layer) ** (0.7))[:, None], head, -1)[..., None], D, -1) , jnp.zeros((num_layer, head, D))
    t_params = mx_t, mw_t, mk_t, mv_t, mr_t, mg_t, Ww_t, Wk_t, Wv_t, Wr_t, Wg_t, Wo_t, Wd, Ax, Bx, decay, u, wln_t, bln_t

    # channel mixing params
    mr_c = 1 - re(base, num_layer, axis = 0) ** (r1)[:, None]
    mk_c = 1 - re(base, num_layer, axis = 0) ** (r1)[:, None]
    Wk_c = gu(k11, (num_layer, emb_size, 4*emb_size))
    Wv_c = gu(k12, (num_layer, 4*emb_size, emb_size))
    Wr_c = gu(k13, (num_layer, emb_size, emb_size))
    c_params = mr_c, mk_c, Wk_c, Wv_c, Wr_c
    # output params

    whead, bhead = orthogonal(scale = 0.5)(k12, (2 * emb_size, emb_size)), jnp.zeros((2 * emb_size))
    wprob1, bprob1 = gu(k13, (ff_size, 2 * emb_size)), jnp.zeros((ff_size))
    wphase1, bphase1 = gu(k14, (ff_size, 2 * emb_size)), jnp.zeros((ff_size))
    wprob2, bprob2 = jnp.zeros((input_size, ff_size)), jnp.zeros((input_size))
    wphase2, bphase2 = jnp.zeros((input_size, ff_size)), jnp.zeros((input_size))
    wln_in, bln_in, wln_out, bln_out = jnp.ones(emb_size), jnp.zeros(emb_size), jnp.ones(emb_size), jnp.zeros(emb_size)  # in&out layer_norm params
    step_params = wln_in, bln_in, whead, bhead, wln_out, bln_out, wprob1, bprob1, wphase1, bphase1, wprob2, bprob2, wphase2, bphase2

    wln_i, bln_i, wln_m, bln_m = jnp.ones((num_layer, emb_size)), jnp.zeros((num_layer, emb_size)), jnp.ones((num_layer, emb_size)), jnp.zeros((num_layer, emb_size))  # in&out layer_norm params
    cell_params = wln_i, bln_i, wln_m, bln_m

    return (wemb, init_params, step_params, t_params, c_params, cell_params)
def init_2DRWKV_params(input_size, emb_size, h_size,  num_layer, ff_size, Ny, Nx, key):
    (key, emb_key, init_x_key, init_y_key, t_last_x1_key, t_last_x2_key, t_last_y1s_key, t_last_y1e_key, t_last_y2_key,  key_tout, key_txout, key_talpha_out,
     c_last_x1_key, c_last_x2_key, c_last_y1s_key, c_last_y1e_key, c_last_y2_key, key_tbeta_out, key_tlast_x, key_c_wv, key_clast_x, key_cxout, key_whead) = split(key, 23)
    wemb = random.uniform(emb_key, (input_size, emb_size), minval=-1e-4, maxval=1e-4) #input hasn't been connected
    x_init = random.uniform(init_x_key, (Nx, emb_size), minval=-1e-4, maxval=1e-4)
    y_init = random.uniform(init_y_key, (Ny, emb_size), minval=-1e-4, maxval=1e-4)
    t_last_x1_init = random.uniform(t_last_x1_key, (Nx, num_layer, emb_size), minval=-1e-4, maxval=1e-4)
    t_last_x2_init = random.uniform(t_last_x2_key, (Nx, num_layer, emb_size), minval=-1e-4, maxval=1e-4)
    t_last_y1s_init = random.uniform(t_last_y1s_key, (Ny+1, num_layer, emb_size), minval=-1e-4, maxval=1e-4)
    t_last_y1e_init = random.uniform(t_last_y1e_key, (Ny+1, num_layer, emb_size), minval=-1e-4, maxval=1e-4)
    t_last_y2_init = random.uniform(t_last_y2_key, (Ny, num_layer, emb_size), minval=-1e-4, maxval=1e-4)
    c_last_x1_init = random.uniform(c_last_x1_key, (Nx, num_layer, emb_size), minval=-1e-4, maxval=1e-4)
    c_last_x2_init = random.uniform(c_last_x2_key, (Nx, num_layer, emb_size), minval=-1e-4, maxval=1e-4)
    c_last_y1s_init = random.uniform(c_last_y1s_key, (Ny + 1, num_layer, emb_size), minval=-1e-4, maxval=1e-4)
    c_last_y1e_init = random.uniform(c_last_y1e_key, (Ny + 1, num_layer, emb_size), minval=-1e-4, maxval=1e-4)
    c_last_y2_init = random.uniform(c_last_y2_key, (Ny, num_layer, emb_size), minval=-1e-4, maxval=1e-4)
    t_alpha_init_x, t_alpha_init_y, t_beta_init_x, t_beta_init_y = jnp.zeros((Nx, num_layer, h_size)), jnp.zeros((Ny, num_layer, h_size)), jnp.zeros((Nx, num_layer, h_size)), jnp.zeros((Ny, num_layer, h_size))
    t_xout = random.normal(key_tout, (Ny, Nx, num_layer, emb_size, 2*emb_size))
    t_alphaout = 0.5*jnp.tile(jnp.eye(h_size), (Ny, Nx, num_layer, 1, 2))
    t_betaout = 0.5*jnp.tile(jnp.eye(h_size), (Ny, Nx, num_layer, 1, 2))
    #t_betaout = random.uniform(key_tbeta_out, (Ny, Nx, num_layer, h_size, 2*h_size), minval = 0.3/h_size, maxval = 0.6/h_size)
    c_xout = random.normal(key_cxout, (Ny, Nx, num_layer, emb_size, 2*emb_size))/emb_size
    emb_size, h_size = 2*emb_size, 2*h_size #tensor product the input from two directions and concantenate the hidden state
    wln_in, bln_in, wln_out, bln_out = jnp.ones((Ny, Nx, emb_size)), jnp.zeros((Ny, Nx, emb_size)), jnp.ones((Ny, Nx, emb_size)), jnp.zeros((Ny, Nx, emb_size))  #in&out layer_norm params
    wln, bln = jnp.ones((2, Ny, Nx, num_layer, emb_size)), jnp.zeros((2, Ny, Nx, num_layer, emb_size))  #time&channel layer_norm params

    # time mixing params
    decay = jnp.tile(-5 + jnp.array([8*(jnp.arange(h_size)/(h_size-1))**(0.7 + 1.3*i/(num_layer-1)) for i in range(num_layer)]), (Ny, Nx, 1, 1))
    bonus = jnp.tile(0.5*(jnp.arange(h_size)%3-1)+jnp.log(0.3), (Ny, Nx, num_layer, 1))
    t_mix_k = jnp.tile(jnp.array([(jnp.arange(emb_size) / emb_size) ** (1 - i / num_layer) for i in range(num_layer)]), (Ny, Nx, 1, 1))
    t_mix_v = t_mix_k + jnp.transpose(jnp.tile(jnp.arange(num_layer) * 0.3 / (num_layer - 1), (Ny, Nx, emb_size, 1)), (0, 1, 3, 2))
    t_mix_r = 0.5 * t_mix_k
    t_wk, t_wv, t_wr = jnp.zeros((Ny, Nx, num_layer, h_size, emb_size)), jnp.zeros((Ny, Nx, num_layer, h_size, emb_size)), jnp.zeros((Ny, Nx, num_layer, h_size, emb_size))
    t_wout = jnp.sqrt(h_size/emb_size)*random.normal(key_tout, (Ny, Nx, num_layer, emb_size, h_size))
    t_wlast_x = random.normal(key_tlast_x, (Ny, Nx, num_layer, emb_size, 2*emb_size)) #since last_x is twice larger than x

    # channel mixing params
    c_mix_k =  jnp.tile(jnp.array([(jnp.arange(emb_size) / emb_size) ** (1 - i / num_layer) for i in range(num_layer)]), (Ny, Nx, 1, 1))
    c_mix_r =  jnp.tile(jnp.array([(jnp.arange(emb_size) / emb_size) ** (1 - i / num_layer) for i in range(num_layer)]), (Ny, Nx, 1, 1))
    c_wr, c_wv, c_wk = jnp.zeros((Ny, Nx, num_layer, emb_size, emb_size)), jnp.sqrt(h_size/emb_size)*random.normal(key_c_wv, (Ny, Nx, num_layer, emb_size, emb_size)), jnp.zeros((Ny, Nx, num_layer, emb_size, emb_size))
    c_wlast_x = random.normal(key_clast_x, (Ny, Nx, num_layer, emb_size, 2*emb_size)) #since last_x is twice larger than x
    # output params
    whead, bhead = random.uniform(key_whead, (Ny, Nx, ff_size, emb_size)) * jnp.sqrt(6 / (emb_size + ff_size)), jnp.zeros((Ny, Nx, ff_size))
    wprob, bprob, wphase, bphase = jnp.zeros((input_size, ff_size)), jnp.zeros((input_size)), jnp.zeros((input_size, ff_size)), jnp.zeros((input_size))
    RWKV_cell_params = wln[0], bln[0], wln[1], bln[1], decay, bonus, t_mix_k, t_mix_v, t_mix_r, t_wk, t_wv, t_wr, t_wout, t_wlast_x, c_mix_k, c_mix_r, c_wk, c_wv, c_wr, c_wlast_x, t_xout, t_alphaout, t_betaout, c_xout

    return (wemb, x_init, y_init, t_last_x1_init, t_last_x2_init, t_last_y1s_init, t_last_y1e_init, t_last_y2_init, t_alpha_init_x, t_alpha_init_y,
            t_beta_init_x, t_beta_init_y, c_last_x1_init, c_last_x2_init, c_last_y1s_init, c_last_y1e_init, c_last_y2_init,  wln_in, bln_in, wln_out, bln_out, whead, bhead, wprob, bprob, wphase, bphase, RWKV_cell_params)
def init_1DTQS_params(input_size, T, ff_size, units, head, key):
    # input is already concantenated
    i1, i2 = (he_normal(), glorot_normal())
    key_encode, key_i, keyq, keyk, keyv, keyo, keyfh, keyhf, keyhh1, keyhh2, keyho = random.split(key, 11)
    Wemb = random.uniform(key_encode, (input_size, units), minval=-1e-4, maxval=1e-4)
    Wi, bi = i2(key_i, (units, units)), jnp.zeros((units))
    Wq, bq = i2(keyq,(T, head, int(units/head) , units)), jnp.zeros((T, head, int(units/head)))
    Wk, bk = i2(keyk,(T, head, int(units/head) , units)), jnp.zeros((T, head, int(units/head)))
    Wv, bv = i2(keyv,(T, head, int(units/head) , units)), jnp.zeros((T, head, int(units/head)))
    Wo, bo = i2(keyo,(T, units, units)), jnp.zeros((T, units))
    a1, a2, b1, b2  = jnp.ones((T, units)), jnp.ones((T, units)), jnp.zeros((T, units)), jnp.zeros((T, units))
    #a, b = jnp.ones((T, units)), jnp.ones((T, units))
    Wfh, bfh = i1(keyfh,(T, ff_size, units)), jnp.zeros((T, ff_size))
    Whf, bhf = i1(keyhf,(T, units, ff_size)), jnp.zeros((T, units))
    Whh1, bhh1 = i1(keyhh1,(units, units)), jnp.zeros((units))
    Whh2, bhh2 = i1(keyhh2,(units, units)), jnp.zeros((units))
    Who1, bho1 = jnp.zeros((input_size, units)), jnp.zeros((input_size))
    Who2, bho2 = jnp.zeros((input_size, units)), jnp.zeros((input_size))

    return (Wemb, Wi, bi, Wq, bq, Wk, bk, Wv, bv, Wo, bo, a1, a2, b1, b2, Wfh, bfh, Whf, bhf, Whh1, bhh1, Whh2, bhh2, Who1, bho1, Who2, bho2)


def init_2DTQS_params(input_size, T, ff_size, units, head, key):
    # input is already concantenated
    i1, i2 = (he_uniform(), glorot_uniform())

    key_encode, key_i, keyq, keyk, keyv, keyqm, keykm, keyvm, keyo, keyfh, keyhf, keyhh1, keyhh2, keyho = random.split(  key, 14)
    Wemb = random.uniform(key_encode, (input_size, units), minval=-1e-4, maxval=1e-4)
    Wi, bi = i2(key_i, (units, units)), jnp.zeros((units))
    Wq, bq = i2(keyq, ( T, units, units)), jnp.zeros(( T, units))
    Wk, bk = i2(keyk, ( T, units, units)), jnp.zeros(( T, units))
    Wv, bv = i2(keyv, ( T, units, units)), jnp.zeros(( T, units))
    Wqm, bqm = i2(keyqm, ( T, head, int(units / head), units)), jnp.zeros(( T, head, int(units / head)))
    Wkm, bkm = i2(keykm, ( T, head, int(units / head), units)), jnp.zeros(( T, head, int(units / head)))
    Wvm, bvm = i2(keyvm, ( T, head, int(units / head), units)), jnp.zeros(( T, head, int(units / head)))
    Wo, bo = i2(keyo,( T, units, units)), jnp.zeros(( T, units))
    a1, a2, b1, b2  = jnp.ones(( T, units)), jnp.ones(( T, units)), jnp.zeros(( T, units)), jnp.zeros(( T,units))
    Wfh, bfh = i1(keyfh,( T, ff_size, units)), jnp.zeros(( T, ff_size))
    Whf, bhf = i2(keyhf,( T, units, ff_size)), jnp.zeros(( T, units))
    Whh1, bhh1 = i1(keyhh1,(ff_size, units)), jnp.zeros((ff_size))
    Who1, bho1 = jnp.zeros((input_size, ff_size)), jnp.zeros((input_size))
    Who2, bho2 = jnp.zeros((input_size, ff_size)), jnp.zeros((input_size))
    #return (Wemb, Wq, bq, Wk, bk, Wv, bv, Wo, bo, a, b, Wfh, bfh, Whf, bhf, Whh, bhh, Who1, bho1)
    return (Wemb, Wi, bi, Wq, bq, Wk, bk, Wv, bv, Wqm, bqm, Wkm, bkm, Wvm, bvm, Wo, bo, a1, a2, b1, b2, Wfh, bfh, Whf, bhf, Whh1, bhh1, Who1, bho1, Who2, bho2)


