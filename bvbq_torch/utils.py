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