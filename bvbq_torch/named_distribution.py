# -*- coding: utf-8 -*-
import collections

import torch

from . import utils


class NamedDistribution(object):
    def __init__(self,params_name,params_dim,params_bound,basedistrib,params_scale=None):
        self.basedistrib = basedistrib
        assert self.basedistrib.ndim == sum(params_dim)
        self.set_param_distrib(params_name,params_dim,params_bound,params_scale)
        
    def logprob(self,params):
        #logdensity_{X}(x) = logdensity_{warp(X)}(warp(X)) + logdwarp(X) 
        #Ordering
        params_ = collections.OrderedDict([(key,params[key]) for key in self.names])
        joint_and_warped_x = self.join_and_warp_parameters(params)
        uncorrected_res = self.basedistrib.logprob(joint_and_warped_x)
        corrections = [self.logdwarpf(key)(value) for key,value in params_.items()]
        correction = torch.sum(torch.cat(corrections,dim=-1),dim=-1)
        print([self.logdwarpf(key)(value) for key,value in params_.items()])
        res = uncorrected_res + correction
        return res
    
    def sample(self,n):
        warped_samples = self.basedistrib.sample(n)
        samples = self.split_and_unwarp_parameters(warped_samples)
        return samples
    
    def set_param_distrib(self,params_name,params_dim,params_bound,params_scale):
        if params_scale is None:
            params_scale = dict()
        param_dict = collections.OrderedDict()
        for i,(name,dim,bounds) in enumerate(zip(params_name,params_dim,params_bound)):
            lb,ub = bounds
            scale = utils.tensor_convert(params_scale.get(name,1.0))
            warpf,iwarpf,logdwarpf = get_warps(lb,ub,scale)
            param_dict[name] = {'dim':dim,'bounds':bounds,
                                'warpf':warpf,'iwarpf':iwarpf,'logdwarpf':logdwarpf}
        self.param_dict = param_dict
    
    def join_parameters(self,params):
        return torch.cat([params[name] for name in self.names],dim=-1)
    
    def join_and_warp_parameters(self,params):
        tocat = [self.warpf(name)(params[name]) for name in self.names]
        res = torch.cat(tocat,dim=-1)
        return res
    
    def split_parameters(self,x):
        return torch.split(x,[self.dim(name) for name in self.names],dim=-1)

    def split_and_unwarp_parameters(self,x):
        splits = torch.split(x,[self.dim(name) for name in self.names],dim=-1)
        res = dict([(name,self.iwarpf(name)(splits[i])) for i,name in enumerate(self.names)])
        return res
    
    @property
    def names(self):
        return list(self.param_dict.keys())
    
    def dim(self,key):
        return self.param_dict[key]['dim']
    
    def bounds(self,key):
        return self.param_dict[key]['bounds']

    def warpf(self,key):
        return self.param_dict[key]['warpf']

    def iwarpf(self,key):
        return self.param_dict[key]['iwarpf']

    def logdwarpf(self,key):
        return self.param_dict[key]['logdwarpf']



def get_warps(lb,ub,scale=1.0):
    #warpf : bounds -> R
    #iwarpf : R -> bounds
    #logdwarpf = log|warpf'\ : bounds -> R^+
    if lb == None and ub == None:
        warpf = lambda x : x/scale
        iwarpf = lambda x : scale*x
        logdwarpf = lambda x : torch.zeros_like(x) - torch.log(scale)
    elif lb != None and ub == None:
        bwf,biwf,bdwf = base_positive_warps()
        warpf = lambda x : bwf((x-lb)/scale)
        iwarpf = lambda x :scale*biwf(x) + lb
        logdwarpf = lambda x : bdwf((x-lb)/scale) - torch.log(scale)
    elif lb == None and ub != None:
        raise NotImplementedError
    elif lb != None and ub != None:
        raise NotImplementedError
    return warpf,iwarpf,logdwarpf


def base_positive_warps():
    warpf = utils.invsoftplus
    iwarpf = utils.softplus
    logdwarpf = lambda x : -torch.log(1.0-torch.exp(-x))
    return warpf,iwarpf,logdwarpf

def base_bounded_warps():
    #[0,1] -> R
    raise NotImplementedError
    return warpf,iwarpf,logdwarpf