# -*- coding: utf-8 -*-
import jax
import jax.numpy as jnp


def kernel_function(x1,x2,kind='sqe',**kwargs):
    #x1 : (...,d)
    #x2 : (...*,d)
    #return : (...,...*)
    theta = kwargs.get("theta",1.0) #(,)
    l = kwargs.get("l",1.0) #(,) or #(d,)
    difference = jnp.expand_dims(x1,tuple(range(-1-(x2.ndim-1),-1))) - x2
    if kind in ['sqe','matern12','matern32','matern52']:
        r2 = jnp.sum((difference/l)**2,axis=-1) #(...,...*)
        if kind == 'sqe':
            return theta*jnp.exp(-r2)
        if kind == 'matern12':
            return theta*matern12rhalf(r2)
        elif kind == 'matern32':
            return theta*matern32rhalf(r2)
        elif kind == 'matern52':
            return theta*matern52rhalf(r2)
    elif kind in ['smatern12','smatern32','smatern52']:
        r2 = (difference/l)**2 #(...,...*,d)
        if kind == 'smatern12':
            return theta*jnp.prod(matern12rhalf(r2),axis=-1)
        elif kind == 'smatern32':
            return theta*jnp.prod(matern32rhalf(r2),axis=-1)
        elif kind == 'smatern52':
            return theta*jnp.prod(matern52rhalf(r2),axis=-1)
    else:
        raise NotImplementedError
            

def kernel_function_2(x1,x2,kind='sqe',**kwargs):
    #x1 : (...,d)
    #x2 : (...,d)
    #return : (...)
    theta = kwargs.get("theta",1.0) #(,)
    l = kwargs.get("l",1.0) #(,) or #(d,)
    difference = (x1 - x2) #(...,d)
    r2 = jnp.sum((difference/l)**2,axis=-1) #(...)
    if kind == 'sqe':
        return theta*jnp.exp(-r2)
    if kind == 'matern12':
        return theta*matern12rhalf(r2)
    elif kind == 'matern32':
        return theta*matern32rhalf(r2)
    elif kind == 'matern52':
        return theta*matern52rhalf(r2)
    else:
        raise NotImplementedError

@jax.custom_jvp
def matern12rhalf(r2):
    r = jnp.sqrt(r2)
    return jnp.exp(-r)

@jax.custom_jvp
def matern32rhalf(r2):
    r = jnp.sqrt(r2)
    return (1+jnp.sqrt(3)*r)*jnp.exp(-jnp.sqrt(3)*r)

@jax.custom_jvp
def matern52rhalf(r2):
    r = jnp.sqrt(r2)
    return (1+jnp.sqrt(5)*r+5./3*r2)*jnp.exp(-jnp.sqrt(5)*r)

@matern12rhalf.defjvp
def matern12rhalf_jvp(primals, tangents):
    r2, = primals
    r2_dot, = tangents
    ans = matern12rhalf(r2)
    r = jnp.sqrt(r2)
    ans_dot = -jnp.exp(-r)/(2*r) * r2_dot
    return ans, ans_dot

@matern32rhalf.defjvp
def matern32rhalf_jvp(primals, tangents):
    r2, = primals
    r2_dot, = tangents
    ans = matern32rhalf(r2)
    r = jnp.sqrt(r2)
    ans_dot = -3./2*jnp.exp(-jnp.sqrt(3)*r) * r2_dot
    return ans, ans_dot

@matern52rhalf.defjvp
def matern52rhalf_jvp(primals, tangents):
    r2, = primals
    r2_dot, = tangents
    ans = matern52rhalf(r2)
    r = jnp.sqrt(r2)
    ans_dot = -5./6*(1+jnp.sqrt(5)*r)*jnp.exp(-jnp.sqrt(5)*r)*r2_dot
    return ans, ans_dot