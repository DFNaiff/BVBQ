# -*- coding: utf-8 -*-
import math
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


def numpy_to_torch_wrapper(f):
    def g(x,*args,**kwargs):
        x = x.detach().numpy()
        res = f(x,*args,**kwargs)
        res = tensor_convert(res)
        return res
    return g

        
def lb_mvn_mixmvn_cross_entropy(mean,var,mixmeans,mixvars,mixweights,logdelta=-20):
    #mean : (n,)
    #var : (n,)
    #mixmeans : (m,n)
    #mixvars : (m,n)
    #mixweights : (m,)
    #-\log(\sum_j (\prod_k \sqrt(2 \pi (\sigma_k^2 + \sigma_j,k^2))
    w = mixweights*torch.prod(torch.sqrt(2*math.pi*(var + mixvars)),dim=-1)
    logz = -0.5*torch.sum(((mean-mixmeans)/(var + mixvars))**2,dim=-1)
    res = -torch.log(torch.sum(w*torch.exp(logz)) + math.exp(logdelta))
    return res


def cut_components_mixmvn(mixmeans,mixvars,mixweights,cutoff_limit=1e-6):
    remain_inds = mixweights > cutoff_limit
    if len(remain_inds) == 0:
        print("No component passed the cutoff. Returning original components")
        return mixmeans,mixvars,mixweights
    else:
        mixmeans_cut = mixmeans[remain_inds,:]
        mixvars_cut = mixvars[remain_inds,:]
        mixweights_cut = mixweights[remain_inds]
        mixweights_cut = mixweights_cut/torch.sum(mixweights_cut) #Normalization
        return mixmeans_cut,mixvars_cut,mixweights_cut
    

def crop_fixed_params_gp(params,fixed_params):
    params_list = set(params.keys())
    for param in params_list:
        if param[:4] == 'raw_':
            param = param[4:]
        if param in fixed_params:
            params.pop(param,None)
            params.pop('raw_'+param,None)
    return params


def softplus(x):
    return torch.nn.functional.softplus(x)


def invsoftplus(x):
    bound = 20.0
    xa = torch.clamp(x,None,bound)
    res = torch.log(torch.exp(xa)-1.0)*(xa < bound) + xa*(xa >= bound)
    return res


def get_subdict(d,param):
    return dict([(key,d[key][param]) for key in d.keys()])