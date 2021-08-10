# -*- coding: utf-8 -*-
import collections

import torch


def jittering(K,d):
    K[range(len(K)),range(len(K))] += d
    return K


def dict_minimize_torch_wrapper(f,*args):
    def dwrapper(params,*args):
        obj = f(params,*args)
        grads = torch.autograd.grad(obj,list(params.values()))
        d_obj = collections.OrderedDict(
                [(key,grads[i]) for i,key in enumerate(params.keys())])
        return obj,d_obj
    return dwrapper


def tensor_convert(x):
    return torch.tensor(x,dtype=torch.float32) if not torch.is_tensor(x) else x


def tensor_convert_(*args):
    return [tensor_convert(x) for x in args]


def logbound(logx,logdelta):
    clipx = torch.clip(logx,logdelta,None)
    boundx = clipx + torch.log(torch.exp(logx-clipx) + \
                               torch.exp(logdelta-clipx))
    return boundx