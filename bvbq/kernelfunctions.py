# -*- coding: utf-8 -*-
import functools

import jax
import jax.numpy as jnp


SEPARABLE_KERNELS = ['smatern12','smatern32','smatern52','sqe']


def kernel_function(x1,x2,theta=1.0,l=1.0,kind='sqe',
                    output='pairwise'):
    #if output == 'pairwise'
    #x1 : (...,d)
    #x2 : (...*,d)
    #return : (...,...*)
    #elif output == 'diagonal'
    #x1 : (...,d)
    #x2 : (...,d)
    #return : (...,)
    if output == 'pairwise':
        difference = jnp.expand_dims(x1,tuple(range(-1-(x2.ndim-1),-1))) - x2
    elif output == 'diagonal':
        difference = x1 - x2
    else:
        raise ValueError
    if kind in ['sqe','matern12','matern32','matern52']:
        r2 = jnp.sum((difference/l)**2,axis=-1) #(...,...*)
        if kind == 'sqe':
            return theta*_sqe(r2)
        if kind == 'matern12':
            return theta*_matern12rhalf(r2)
        elif kind == 'matern32':
            return theta*_matern32rhalf(r2)
        elif kind == 'matern52':
            return theta*_matern52rhalf(r2)
    elif kind in ['smatern12','smatern32','smatern52']:
        r2 = (difference/l)**2 #(...,...*,d)
        if kind == 'smatern12':
            return theta*jnp.prod(_matern12rhalf(r2),axis=-1)
        elif kind == 'smatern32':
            return theta*jnp.prod(_matern32rhalf(r2),axis=-1)
        elif kind == 'smatern52':
            return theta*jnp.prod(_matern52rhalf(r2),axis=-1)
    else:
        raise NotImplementedError
            

def kernel_function_separated(x1,x2,theta=1.0,l=1.0,kind='sqe',
                              output='pairwise'):
    #x1 : (...,d)
    #x2 : (...*,d)
    #return : (...,...*,d) or (...,d)
    assert kind in SEPARABLE_KERNELS
    if output == 'pairwise': #(...,...*,d)
        if kind == 'sqe':
            return _pairwise_kernel_function_separated_sqe(x1,x2,theta,l)
        elif kind == 'smatern12':
            return _pairwise_kernel_function_separated_smatern12(x1,x2,theta,l)
        elif kind == 'smatern32':
            return _pairwise_kernel_function_separated_smatern32(x1,x2,theta,l)
        elif kind == 'smatern52':
            return _pairwise_kernel_function_separated_smatern52(x1,x2,theta,l)
    elif output == 'diagonal':
        if kind == 'sqe':
            return _diagonal_kernel_function_separated_sqe(x1,x2,theta,l)
        elif kind == 'smatern12':
            return _diagonal_kernel_function_separated_smatern12(x1,x2,theta,l)
        elif kind == 'smatern32':
            return _diagonal_kernel_function_separated_smatern32(x1,x2,theta,l)
        elif kind == 'smatern52':
            return _diagonal_kernel_function_separated_smatern52(x1,x2,theta,l)


@jax.jit
def _diagonal_kernel_function_sqe(x1,x2,theta,l):
    difference = x1 - x2
    r2 = jnp.sum((difference/l)**2,axis=-1) #(...,...*)
    return theta*_sqe(r2)

@jax.jit
def _pairwise_kernel_function_sqe(x1,x2,theta,l):
    difference = _make_pairwise_difference(x1,x2)
    r2 = jnp.sum((difference/l)**2,axis=-1) #(...,...*)
    return theta*_sqe(r2)

@jax.jit
def _diagonal_kernel_function_smatern12(x1,x2,theta,l):
    difference = x1 - x2
    r2 = (difference/l)**2
    return theta*jnp.prod(*_matern12rhalf(r2),axis=-1)


@jax.jit
def _diagonal_kernel_function_smatern32(x1,x2,theta,l):
    difference = x1 - x2
    r2 = (difference/l)**2
    return theta*jnp.prod(*_matern32rhalf(r2),axis=-1)


@jax.jit
def _diagonal_kernel_function_smatern52(x1,x2,theta,l):
    difference = x1 - x2
    r2 = (difference/l)**2
    return theta*jnp.prod(*_matern52rhalf(r2),axis=-1)

@jax.jit
def _pairwise_kernel_function_smatern12(x1,x2,theta,l):
    difference = _make_pairwise_difference(x1,x2)
    r2 = (difference/l)**2
    return theta*jnp.prod(*_matern12rhalf(r2),axis=-1)


@jax.jit
def _pairwise_kernel_function_smatern32(x1,x2,theta,l):
    difference = _make_pairwise_difference(x1,x2)
    r2 = (difference/l)**2
    return theta*jnp.prod(*_matern32rhalf(r2),axis=-1)


@jax.jit
def _pairwise_kernel_function_smatern52(x1,x2,theta,l):
    difference = _make_pairwise_difference(x1,x2)
    r2 = (difference/l)**2
    return theta*jnp.prod(*_matern52rhalf(r2),axis=-1)


@jax.jit
def _diagonal_kernel_function_separated_sqe(x1,x2,theta,l):
    difference = x1 - x2
    r2 = (difference/l)**2
    d = r2.shape[-1]
    return theta**(1.0/d)*_sqe(r2)


@jax.jit
def _diagonal_kernel_function_separated_smatern12(x1,x2,theta,l):
    difference = x1 - x2
    r2 = (difference/l)**2
    d = r2.shape[-1]
    return theta**(1.0/d)*_matern12rhalf(r2)


@jax.jit
def _diagonal_kernel_function_separated_smatern32(x1,x2,theta,l):
    difference = x1 - x2
    r2 = (difference/l)**2
    d = r2.shape[-1]
    return theta**(1.0/d)*_matern32rhalf(r2)


@jax.jit
def _diagonal_kernel_function_separated_smatern52(x1,x2,theta,l):
    difference = x1 - x2
    r2 = (difference/l)**2
    d = r2.shape[-1]
    return theta**(1.0/d)*_matern52rhalf(r2)


@jax.jit
def _pairwise_kernel_function_separated_sqe(x1,x2,theta,l):
    difference = _make_pairwise_difference(x1,x2)
    r2 = (difference/l)**2
    d = r2.shape[-1]
    return theta**(1.0/d)*_sqe(r2)


@jax.jit
def _pairwise_kernel_function_separated_smatern12(x1,x2,theta,l):
    difference = _make_pairwise_difference(x1,x2)
    r2 = (difference/l)**2
    d = r2.shape[-1]
    return theta**(1.0/d)*_matern12rhalf(r2)


@jax.jit
def _pairwise_kernel_function_separated_smatern32(x1,x2,theta,l):
    difference = _make_pairwise_difference(x1,x2)
    r2 = (difference/l)**2
    d = r2.shape[-1]
    return theta**(1.0/d)*_matern32rhalf(r2)


@jax.jit
def _pairwise_kernel_function_separated_smatern52(x1,x2,theta,l):
    difference = _make_pairwise_difference(x1,x2)
    r2 = (difference/l)**2
    d = r2.shape[-1]
    return theta**(1.0/d)*_matern52rhalf(r2)


@jax.jit
def _make_pairwise_difference(x1,x2,l):
    difference = jnp.expand_dims(x1,tuple(range(-1-(x2.ndim-1),-1))) - x2
    return difference


@jax.jit
def _sqe(r2):
    return jnp.exp(-0.5*r2)


@jax.custom_jvp
@jax.jit
def _matern12rhalf(r2):
    r = jnp.sqrt(r2)
    return jnp.exp(-r)


@jax.custom_jvp
@jax.jit
def _matern32rhalf(r2):
    r = jnp.sqrt(r2)
    return (1+jnp.sqrt(3)*r)*jnp.exp(-jnp.sqrt(3)*r)


@jax.custom_jvp
@jax.jit
def _matern52rhalf(r2):
    r = jnp.sqrt(r2)
    return (1+jnp.sqrt(5)*r+5./3*r2)*jnp.exp(-jnp.sqrt(5)*r)


@_matern32rhalf.defjvp
@jax.jit
def _matern32rhalf_jvp(primals, tangents):
    r2, = primals
    r2_dot, = tangents
    ans = _matern32rhalf(r2)
    r = jnp.sqrt(r2)
    ans_dot = -3./2*jnp.exp(-jnp.sqrt(3)*r) * r2_dot
    return ans, ans_dot


@_matern52rhalf.defjvp
@jax.jit
def _matern52rhalf_jvp(primals, tangents):
    r2, = primals
    r2_dot, = tangents
    ans = _matern52rhalf(r2)
    r = jnp.sqrt(r2)
    ans_dot = -5./6*(1+jnp.sqrt(5)*r)*jnp.exp(-jnp.sqrt(5)*r)*r2_dot
    return ans, ans_dot