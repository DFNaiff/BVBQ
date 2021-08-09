# -*- coding: utf-8 -*-
import functools

import numpy as np
import jax
import jax.numpy as jnp
import cholupdates.rank_1


def sample_unit_ball(*shape):
    d = shape[-1]
    x = np.random.randn(*shape)
    x /= np.linalg.norm(x,axis=-1,keepdims=True)
    u = np.random.rand(*(shape[:-1] + (1,)))
    x *= u**(1./d)
    return x


def nspherical_to_cartesian(r,theta,*phi):
    x = r*np.ones(len(phi)+2)
    for i,phi_i in enumerate(phi):
        x[i+1:] *= np.sin(phi_i)
    x[:-2] *= np.cos(phi)
    x[-2] *= np.cos(theta)
    x[-1] *= np.sin(theta)
    return x


@jax.jit
def jittering(M,sigma):
    diag_elements = jnp.diag_indices_from(M)
    Mj = jax.ops.index_update(M,diag_elements,M.diagonal()+sigma)
    return Mj

@jax.jit
def woodbudy_identity(Ainv,Cinv,U,V):
    #Inverse of (A+UCV), with Ainv (and Cinv) known
    M = jax.scipy.linalg.inv(Cinv + V@Ainv@U)
    K = Ainv - Ainv@U@M@V@Ainv
    return K

@jax.jit
def block_matrix_inversion(Ainv,D,B,C):
    #Inverse of 
    #[[A B]
    # [C D]],
    #With Ainv known
    S = jax.scipy.linalg.inv(D - C@Ainv@B)
    P11 = Ainv + Ainv@B@S@C@Ainv
    P12 = -Ainv@B@S
    P21 = -S@C@Ainv
    P22 = S
    P = jnp.block([[P11,P12],
                   [P21,P22]])
    return P


def rankn_update_upper(U,V):
    L = U.T.astype(np.double)
    V = V.astype(np.double)
    for i in range(V.shape[1]):
        v = V[:,i]
        L = cholupdates.rank_1.update(L,v)
    return L.T.astype(np.float32)


def delete_submatrix(M,j):
    return jnp.delete(jnp.delete(M,j,axis=0),j,axis=1)


def inds_mean(x,inds,indrange):
    def ind_mean(i):
        return jnp.nan_to_num(jnp.mean(x[inds==i]))
    return jnp.stack([ind_mean(i) for i in indrange])



def logsumexp(logx,weights=None,axis=-1,keepdims=False):
    return jax.scipy.special.logsumexp(logx,axis=axis,b=weights,keepdims=keepdims)


@jax.jit
def logbound(logx,logdelta):
    clipx = jnp.clip(logx,logdelta,None)
    boundx = clipx + jnp.log(jnp.exp(logx-clipx) + \
                             jnp.exp(logdelta-clipx))
    return boundx


@jax.jit
def lb_mvn_mixmvn_cross_entropy(mean,var,mixmeans,mixvars,mixweights,logdelta=-20):
    #mean : (n,)
    #var : (n,)
    #mixmeans : (m,n)
    #mixvars : (m,n)
    #mixweights : (m,)
    #-\log(\sum_j (\prod_k \sqrt(2 \pi (\sigma_k^2 + \sigma_j,k^2))
    w = mixweights*jnp.prod(jnp.sqrt(2*jnp.pi*(var + mixvars)),axis=-1)
    logz = -0.5*jnp.sum(((mean-mixmeans)/(var + mixvars))**2,axis=-1)
    res = -jnp.log(jnp.sum(w*jnp.exp(logz)) + jnp.exp(logdelta))
    return res


@jax.jit
def dmvn_samples_from_zsamples(mean,var,zsamples):
    std = jnp.sqrt(var)
    return mean + std*zsamples


@jax.jit
def mixdmvn_samples_from_zsamples_catinds(means,variances,z,catinds):
    stds = jnp.sqrt(variances)
    res = means[catinds,:] + stds[catinds,:]*z
    return res