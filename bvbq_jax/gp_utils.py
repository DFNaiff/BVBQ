# -*- coding: utf-8 -*-
import functools

import jax
import jax.numpy as jnp

from . import kernelfunctions
from . import utils


@functools.partial(jax.jit,static_argnums=(9,))
def predict_mean_and_cov(xpred,X,y_,theta,lengthscale,mean,
                         upper_chol_matrix,noise,min_jitter,kind):
    kxpred = make_kernel_matrix(X,xpred,theta,lengthscale,kind) #(m,n)
    y_ = jax.scipy.linalg.solve_triangular(upper_chol_matrix,
                                           y_-mean,
                                           trans='T') #(m,1)
    kxpred_ = jax.scipy.linalg.solve_triangular(upper_chol_matrix,
                                                kxpred,
                                                trans='T') #(m,n)
    pred_mean = (kxpred_.transpose()@y_) + mean# + ymax
    Kxpxp = make_kernel_matrix(xpred,xpred,theta,lengthscale,kind)
    Kxpxp = utils.jittering(Kxpxp,noise**2+min_jitter)
    pred_cov = Kxpxp - kxpred_.transpose()@kxpred_
    return pred_mean,pred_cov

@functools.partial(jax.jit,static_argnums=(9,))
def predict_mean_and_var(xpred,X,y_,theta,lengthscale,mean,
                         upper_chol_matrix,noise,min_jitter,kind):
    kxpred = make_kernel_matrix(X,xpred,theta,lengthscale,kind) #(m,n)
    y_ = jax.scipy.linalg.solve_triangular(upper_chol_matrix,
                                           y_-mean,
                                           trans='T') #(m,1)
    kxpred_ = jax.scipy.linalg.solve_triangular(upper_chol_matrix,
                                                kxpred,
                                                trans='T') #(m,n)
    pred_mean = (kxpred_.transpose()@y_) + mean# + ymax
    Kxpxp = make_kernel_diagonal(xpred,xpred,theta,lengthscale,kind) #(...)
    Kxpxp = Kxpxp + noise**2 + min_jitter #(...)
    pred_var = Kxpxp + jnp.sum((kxpred_**2),axis=-1)
    return pred_mean,pred_var

@functools.partial(jax.jit,static_argnums=(7,))
def predict_mean(xpred,X,y_,theta,lengthscale,mean,
                 upper_chol_matrix,kind):
    kxpred = make_kernel_matrix(X,xpred,theta,lengthscale,kind) #(m,n)
    y_ = jax.scipy.linalg.solve_triangular(upper_chol_matrix,
                                           y_-mean,
                                           trans='T') #(m,1)
    kxpred_ = jax.scipy.linalg.solve_triangular(upper_chol_matrix,
                                                kxpred,
                                                trans='T') #(m,n)
    pred_mean = (kxpred_.transpose()@y_) + mean# + ymax
    return pred_mean

@functools.partial(jax.jit,static_argnums=(7,))
def loglikelihood(X,y_,theta,lengthscale,noise,mean,
                  min_jitter,kind):
    ndata = X.shape[0]
    kernel_matrix = make_kernel_matrix(X,X,theta,lengthscale,kind)
    kernel_matrix = utils.jittering(kernel_matrix,noise**2+min_jitter)
    upper_chol_matrix = jax.scipy.linalg.cholesky(kernel_matrix)
    y_ = jax.scipy.linalg.solve_triangular(upper_chol_matrix,
                                           y_-mean,
                                           trans='T') #(m,1)
    term1 = -0.5*jnp.sum(y_**2)
    term2 = -jnp.sum(jnp.log(jnp.diag(upper_chol_matrix)))
    term3 = -0.5*ndata*jnp.log(2*jnp.pi)
    return term1 + term2 + term3
    
@functools.partial(jax.jit,static_argnums=(4,))
def make_kernel_matrix(X1,X2,theta,lengthscale,kind):
    return kernelfunctions.kernel_function(X1,X2,theta,lengthscale,
                                    kind=kind,output='pairwise')
    

@functools.partial(jax.jit,static_argnums=(4,))
def make_kernel_diagonal(X1,X2,theta,lengthscale,kind):
    return kernelfunctions.kernel_function(X1,X2,theta,lengthscale,
                                    kind=kind,output='diagonal')
