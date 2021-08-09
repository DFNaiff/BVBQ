# -*- coding: utf-8 -*-
import math
import torch


SEPARABLE_KERNELS = ['smatern12','smatern32','smatern52','sqe']


def kernel_function(x1,x2,kind='sqe',
                    output='pairwise',
                    **kwargs):
    #if output == 'pairwise'
    #x1 : (...,d)
    #x2 : (...*,d)
    #return : (...,...*)
    #elif output == 'diagonal'
    #x1 : (...,d)
    #x2 : (...,d)
    #return : (...,)
    theta = kwargs.get("theta",1.0) #(,)
    l = kwargs.get("l",1.0) #(,) or #(d,)
    if output == 'pairwise':
        x1_ = x1[(slice(None),)*(x1.ndim-1) + (None,)*(x2.ndim-1)]
        difference = x1_ - x2
    elif output == 'diagonal':
        difference = x1 - x2
    else:
        raise ValueError
    if kind in ['sqe','matern12','matern32','matern52']:
        r2 = torch.sum((difference/l)**2,dim=-1) #(...,...*)
        if kind == 'sqe':
            return theta*sqe(r2)
        if kind == 'matern12':
            return theta*matern12rhalf(r2)
        elif kind == 'matern32':
            return theta*matern32rhalf(r2)
        elif kind == 'matern52':
            return theta*matern52rhalf(r2)
    elif kind in ['smatern12','smatern32','smatern52']:
        r2 = (difference/l)**2 #(...,...*,d)
        if kind == 'smatern12':
            return theta*torch.prod(matern12rhalf(r2),dim=-1)
        elif kind == 'smatern32':
            return theta*torch.prod(matern32rhalf(r2),dim=-1)
        elif kind == 'smatern52':
            return theta*torch.prod(matern52rhalf(r2),dim=-1)
    else:
        raise NotImplementedError
            

def kernel_function_separated(x1,x2,theta=1.0,l=1.0,kind='sqe',
                              output='pairwise',
                              **kwargs):
    #x1 : (...,d)
    #x2 : (...*,d)
    #return : (...,...*,d) or (...,d)
    assert kind in SEPARABLE_KERNELS
    if output == 'pairwise': #(...,...*,d)
        x1_ = x1[(slice(None),)*(x1.ndim-1) + (None,)*(x2.ndim-1)]
        difference = x1_ - x2
    elif output == 'diagonal':
        difference = x1 - x2 #(...,d)
    else:
        raise ValueError
    r2 = (difference/l)**2 #(...,...*,d) or (...,d)
    d = r2.shape[-1]
    if kind == 'sqe':
        return theta**(1.0/d)*sqe(r2)
    elif kind == 'smatern12':
        return theta**(1.0/d)*matern12rhalf(r2)
    elif kind == 'smatern32':
        return theta**(1.0/d)*matern32rhalf(r2)
    elif kind == 'smatern52':
        return theta**(1.0/d)*matern52rhalf(r2)


def sqe(r2):
    return torch.exp(-0.5*r2)


def matern12rhalf(r2):
    r = torch.sqrt(r2)
    return torch.exp(-r)


def matern32rhalf(r2):
    r = torch.sqrt(r2)
    return (1+math.sqrt(3)*r)*torch.exp(-math.sqrt(3)*r)


def matern52rhalf(r2):
    r = torch.sqrt(r2)
    return (1+math.sqrt(5)*r+5./3*r2)*torch.exp(-math.sqrt(5)*r)