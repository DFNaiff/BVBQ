# -*- coding: utf-8 -*-
import functools

import jax
import jax.numpy as jnp
import numpy as np

from . import kernelfunctions

LOCS_25,WEIGHTS_25 = [jnp.array(x,dtype=jnp.float32) for x in \
                   np.polynomial.hermite.hermgauss(25)]
LOCS_50,WEIGHTS_50 = [jnp.array(x,dtype=jnp.float32) for x in \
                   np.polynomial.hermite.hermgauss(50)]
    
    
def monte_carlo_bayesian_quadrature(gp,distrib,nsamples,return_var=True):
    samples1 = distrib.sample(nsamples)
    samples2 = distrib.sample(nsamples)
    samples3 = distrib.sample(nsamples)
    z = gp.kernel_function(gp.X,samples1).mean(axis=-1,keepdims=True) #(m,1)
    if return_var:
        gamma = gp.kernel_function(samples2,samples3,diagonal=True).mean()
    else:
        gamma = None
    mean,var = calculate_bq_mean_var(gp,z,gamma)
    if var is None:
        return mean
    else:
        return mean,var

def separable_dmvn_bq(gp,mean,var,nhermite=25,return_var=True):
    if nhermite == 25:
        hlocs,hweights = LOCS_25,WEIGHTS_25
    elif nhermite == 50:
        hlocs,hweights = LOCS_50,WEIGHTS_50
    else:
        hlocs,weights = np.polynomial.hermite.hermgauss(nhermite)
        hlocs = jnp.array(hlocs,dtype=jnp.float32)
        hweights = jnp.array(hweights,dtype=jnp.float32)
    if return_var:
        mean,var = _separable_dmvn_bq_mean_var(mean,var,hlocs,hweights,
                                               gp.X,gp.y_,gp.mean,
                                               gp.theta,gp.lengthscale,
                                               gp.upper_chol_matrix,gp.kind)
        mean = mean + gp.ymax
        return mean,var
    else:
        mean = _separable_dmvn_bq_mean(mean,var,hlocs,hweights,
                                       gp.X,gp.y_,gp.mean,
                                       gp.theta,gp.lengthscale,
                                       gp.upper_chol_matrix,gp.kind)
        mean = mean + gp.ymax
        return mean
    

def separable_mixdmvn_bq(gp,means,variances,weights,nhermite=25,return_var=True):
    if nhermite == 25:
        hlocs,hweights = LOCS_25,WEIGHTS_25
    elif nhermite == 50:
        hlocs,hweights = LOCS_50,WEIGHTS_50
    else:
        hlocs,weights = np.polynomial.hermite.hermgauss(nhermite)
        hlocs = jnp.array(hlocs,dtype=jnp.float32)
        hweights = jnp.array(hweights,dtype=jnp.float32)
    if return_var:
        mean,var = _separable_mixdmvn_bq_mean_var(means,variances,weights,
                                                  hlocs,hweights,
                                                  gp.X,gp.y_,gp.mean,
                                                  gp.theta,gp.lengthscale,
                                                  gp.upper_chol_matrix,gp.kind)
        mean = mean + gp.ymax
        return mean,var
    else:
        mean = _separable_mixdmvn_bq_mean(means,variances,weights,
                                          hlocs,hweights,
                                          gp.X,gp.y_,gp.mean,
                                          gp.theta,gp.lengthscale,
                                          gp.upper_chol_matrix,gp.kind)
        mean = mean + gp.ymax
        return mean


@functools.partial(jax.jit,static_argnums=(10,))
def _separable_dmvn_bq_mean_var(mean,var,hlocs,hweights,
                                xdata,y_,gpmean,theta,lengthscale,
                                upper_chol_matrix,kind):
    hlocs_ = hlocs.reshape(-1,1)*jnp.sqrt(var)*jnp.sqrt(2) + mean
    hweights_ = jnp.expand_dims(hweights,(-2,-1)) #(k,1,1)
    hweights__ = jnp.expand_dims(hweights,(0,-1)) #(1,k,1)

    kernel_tensor = kernelfunctions.kernel_function_separated(hlocs_,
                                                              xdata,
                                                              lengthscale,
                                                              theta,
                                                              kind) #(k,n,d)
    kernel_tensor *= hweights_ #(k,n,d)
    kernel_matrix = jnp.sum(kernel_tensor,axis=0)*1/jnp.sqrt(jnp.pi) #(n,d)
    z = jnp.prod(kernel_matrix,axis=-1,keepdims=True)

    kernel_tensor_2 = kernelfunctions.kernel_function_separated(hlocs_,
                                                                hlocs_,
                                                                lengthscale,
                                                                theta,
                                                                kind) #(k,k,d)
    kernel_tensor_2 *= hweights_*hweights__
    gamma = jnp.prod(1/jnp.pi*jnp.sum(jnp.sum(kernel_tensor_2,0),0))
    bqmean,bqvar = calculate_bq_mean_var(y_,gpmean,upper_chol_matrix,z,gamma)
    return bqmean,bqvar


@functools.partial(jax.jit,static_argnums=(10,))
def _separable_dmvn_bq_mean(mean,var,hlocs,hweights,
                                xdata,y_,gpmean,theta,lengthscale,
                                upper_chol_matrix,kind):
    hlocs_ = hlocs.reshape(-1,1)*jnp.sqrt(var)*jnp.sqrt(2) + mean
    hweights_ = jnp.expand_dims(hweights,(-2,-1)) #(k,1,1)

    kernel_tensor = kernelfunctions.kernel_function_separated(hlocs_,
                                                              xdata,
                                                              lengthscale,
                                                              theta,
                                                              kind) #(k,n,d)
    kernel_tensor *= hweights_ #(k,n,d)
    kernel_matrix = jnp.sum(kernel_tensor,axis=0)*1/jnp.sqrt(jnp.pi) #(n,d)
    z = jnp.prod(kernel_matrix,axis=-1,keepdims=True)

    bqmean = calculate_bq_mean(y_,gpmean,upper_chol_matrix,z)
    return bqmean


@functools.partial(jax.jit,static_argnums=(11,))
def _separable_mixdmvn_bq_mean_var(means,variances,weights,hlocs,hweights,
                                   xdata,y_,gpmean,theta,lengthscale,
                                   upper_chol_matrix,kind):
    hlocs_ = jnp.expand_dims(hlocs,(-2,-1))*jnp.sqrt(variances)*jnp.sqrt(2) + means #(k,m,d)
    hweights_ = jnp.expand_dims(hweights,(-3,-2,-1)) #(k,1,1,1)
    hweights__ = jnp.expand_dims(hweights,(-2,-1)) #(k,1,1)
    hweights___ = jnp.expand_dims(hweights,(-4,-3,-2,-1)) #(k,1,1,1,1)
    kernel_tensor = kernelfunctions.kernel_function_separated(hlocs_,
                                                              xdata,
                                                              lengthscale,
                                                              theta,
                                                              kind) #(k,m,n,d)
    kernel_tensor *= hweights_ #(k,m,n,d)
    kernel_matrices = jnp.sum(kernel_tensor,axis=0)*1/jnp.sqrt(jnp.pi) #(m,n,d)
    z = jnp.expand_dims(
                    jnp.sum(
                        jnp.prod(kernel_matrices,axis=-1)*\
                        jnp.expand_dims(weights,-1),
                     axis=0),-1) #(m,n,d) -> (n,d) -> (d,) -> (d,1)

    kernel_tensor_2 = kernelfunctions.kernel_function_separated(hlocs_,
                                                                hlocs_,
                                                                lengthscale,
                                                                theta,
                                                                kind) #(k,m,k,m,d)
    kernel_tensor_2 *= hweights__*hweights___
    gamma_matrix = jnp.prod(1/jnp.pi*jnp.sum(jnp.sum(kernel_tensor_2,0),1),-1)
    gamma = jnp.sum(gamma_matrix*weights*jnp.expand_dims(weights,-1))

    bqmean,bqvar = calculate_bq_mean_var(y_,gpmean,upper_chol_matrix,z,gamma)
    return bqmean,bqvar


@functools.partial(jax.jit,static_argnums=(11,))
def _separable_mixdmvn_bq_mean(means,variances,weights,hlocs,hweights,
                                   xdata,y_,gpmean,theta,lengthscale,
                                   upper_chol_matrix,kind):
    hlocs_ = jnp.expand_dims(hlocs,(-2,-1))*jnp.sqrt(variances)*jnp.sqrt(2) + means #(k,m,d)
    hweights_ = jnp.expand_dims(hweights,(-3,-2,-1)) #(k,1,1,1)
    kernel_tensor = kernelfunctions.kernel_function_separated(hlocs_,
                                                              xdata,
                                                              lengthscale,
                                                              theta,
                                                              kind) #(k,m,n,d)
    kernel_tensor *= hweights_ #(k,m,n,d)
    kernel_matrices = jnp.sum(kernel_tensor,axis=0)*1/jnp.sqrt(jnp.pi) #(m,n,d)
    z = jnp.expand_dims(
                    jnp.sum(
                        jnp.prod(kernel_matrices,axis=-1)*\
                        jnp.expand_dims(weights,-1),
                     axis=0),-1) #(m,n,d) -> (n,d) -> (d,) -> (d,1)

    bqmean = calculate_bq_mean(y_,gpmean,upper_chol_matrix,z)
    return bqmean


@jax.jit
def calculate_bq_mean(gpy_,gpmean,upper_chol_matrix,z):
    y_ = jax.scipy.linalg.solve_triangular(upper_chol_matrix,
                                           gpy_-gpmean,
                                           trans='T') #(m,1)
    z_ = jax.scipy.linalg.solve_triangular(upper_chol_matrix,
                                           z,
                                           trans='T') #(m,1)
    mean = (gpmean + z_.transpose()@y_)[0][0] #(1,1) -> (,)
    return mean


@jax.jit
def calculate_bq_mean_var(gpy_,mean,upper_chol_matrix,z,gamma):
    y_ = jax.scipy.linalg.solve_triangular(upper_chol_matrix,
                                           gpy_-mean,
                                           trans='T') #(m,1)
    z_ = jax.scipy.linalg.solve_triangular(upper_chol_matrix,
                                           z,
                                           trans='T') #(m,1)
    mean = (mean + z_.transpose()@y_)[0][0] #(1,1) -> (,)
    var = (gamma - z_.transpose()@z_)[0][0] #(1,1) -> (,)
    return mean,var