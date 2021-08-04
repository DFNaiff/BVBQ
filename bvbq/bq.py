# -*- coding: utf-8 -*-
import jax
import jax.numpy as jnp
import numpy as np

from . import kernelfunctions


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


def separable_mvn_bq(gp,mean,var,nhermite=30,return_var=True):
    locs,weights = np.polynomial.hermite.hermgauss(nhermite)
    locs = jnp.array(locs,dtype=jnp.float32)
    weights = jnp.array(weights,dtype=jnp.float32)
    locs_ = locs.reshape(-1,1)*jnp.sqrt(var)*jnp.sqrt(2) + mean
    weights_ = jnp.expand_dims(weights,(-2,-1)) #(k,1,1)
    weights__ = jnp.expand_dims(weights,(0,-1)) #(1,k,1)
    
    xdata = gp.X
    kernel_tensor = kernelfunctions.kernel_function_separated(locs_,
                                                              xdata,
                                                              kind=gp.kind,
                                                              theta=gp.theta,
                                                              l=gp.lengthscale) #(k,n,d)
    kernel_tensor *= weights_ #(k,n,d)
    kernel_matrix = jnp.sum(kernel_tensor,axis=0)*1/jnp.sqrt(jnp.pi) #(n,d)
    z = jnp.prod(kernel_matrix,axis=-1,keepdims=True)
    if return_var:
        kernel_tensor_2 = kernelfunctions.kernel_function_separated(locs_,
                                                                    locs_,
                                                                    kind=gp.kind,
                                                                    theta=gp.theta,
                                                                    l=gp.lengthscale) #(k,k,d)
        kernel_tensor_2 *= weights_*weights__
        gamma = jnp.prod(1/jnp.pi*jnp.sum(jnp.sum(kernel_tensor_2,0),0))
    else:
        gamma = None
    mean,var = calculate_bq_mean_var(gp,z,gamma)
    if var is None:
        return mean
    else:
        return mean,var


def separable_mixmvn_bq(gp,means,variances,weights,nhermite=30,return_var=True):
    hlocs,hweights = np.polynomial.hermite.hermgauss(nhermite)
    hlocs = jnp.array(hlocs,dtype=jnp.float32)
    hweights = jnp.array(hweights,dtype=jnp.float32)
    hlocs_ = jnp.expand_dims(hlocs,(-2,-1))*jnp.sqrt(variances)*jnp.sqrt(2) + means #(k,m,d)
    hweights_ = jnp.expand_dims(hweights,(-3,-2,-1)) #(k,1,1,1)
    hweights__ = jnp.expand_dims(hweights,(-2,-1)) #(k,1,1)
    hweights___ = jnp.expand_dims(hweights,(-4,-3,-2,-1)) #(k,1,1,1,1)
    xdata = gp.X
    kernel_tensor = kernelfunctions.kernel_function_separated(hlocs_,
                                                              xdata,
                                                              kind=gp.kind,
                                                              theta=gp.theta,
                                                              l=gp.lengthscale) #(k,m,n,d)
    kernel_tensor *= hweights_ #(k,m,n,d)
    kernel_matrices = jnp.sum(kernel_tensor,axis=0)*1/jnp.sqrt(jnp.pi) #(m,n,d)
    z = jnp.expand_dims(
                    jnp.sum(
                        jnp.prod(kernel_matrices,axis=-1)*\
                        jnp.expand_dims(weights,-1),
                     axis=0),-1) #(m,n,d) -> (n,d) -> (d,) -> (d,1)
    if return_var:
        kernel_tensor_2 = kernelfunctions.kernel_function_separated(hlocs_,
                                                                    hlocs_,
                                                                    kind=gp.kind,
                                                                    theta=gp.theta,
                                                                    l=gp.lengthscale) #(k,m,k,m,d)
        kernel_tensor_2 *= hweights__*hweights___
        gamma_matrix = jnp.prod(1/jnp.pi*jnp.sum(jnp.sum(kernel_tensor_2,0),1),-1)
        gamma = jnp.sum(gamma_matrix*weights*jnp.expand_dims(weights,-1))
    else:
        gamma = None
    mean,var = calculate_bq_mean_var(gp,z,gamma)
    if var is None:
        return mean
    else:
        return mean,var


def calculate_bq_mean_var(gp,z,gamma=None):
    y_ = jax.scipy.linalg.solve_triangular(gp.upper_chol_matrix,
                                           gp.y-gp.mean,
                                           trans='T') #(m,1)
    z_ = jax.scipy.linalg.solve_triangular(gp.upper_chol_matrix,
                                           z,
                                           trans='T') #(m,1)
    mean = gp.mean + z_.transpose()@y_
    if gamma is None:
        var = None
    else:
        var = gamma - z_.transpose()@z_
    return mean,var