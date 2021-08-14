# -*- coding: utf-8 -*-
# pylint: disable=E1101

import collections

import torch

from . import utils


class NamedDistribution(object):
    def __init__(self,
                 params_name,
                 params_dim,
                 params_bound,
                 basedistrib=None,
                 params_scale=None):
        self.set_param_distrib(params_name, params_dim,
                               params_bound, params_scale)
        self.set_basedistrib(basedistrib)

    def logprob(self, params):
        # logdensity_{X}(x) = logdensity_{warp(X)}(warp(X)) + logdwarp(X)
        # Ordering
        assert self.basedistrib is not None
        params_ = self.organize_params(params)
        joint_and_warped_x = self.join_and_warp_parameters(params_)
        uncorrected_res = self.basedistrib.logprob(joint_and_warped_x)
        corrections = [self.logdwarpf(key)(value)
                       for key, value in params_.items()]
        correction = torch.sum(torch.cat(corrections, dim=-1), dim=-1)
        res = uncorrected_res + correction
        return res

    def sample(self, n):
        assert self.basedistrib is not None
        warped_samples = self.basedistrib.sample(n)
        samples = self.split_and_unwarp_parameters(warped_samples)
        return samples

    def set_param_distrib(self, params_name, params_dim, params_bound, params_scale):
        if params_scale is None:
            params_scale = dict()
        param_dict = collections.OrderedDict()
        for _, (name, dim, bound) in enumerate(zip(params_name, params_dim, params_bound)):
            lb, ub = bound
            scale = utils.tensor_convert(params_scale.get(name, 1.0))
            warpf, iwarpf, logdwarpf = get_warps(lb, ub, scale)
            param_dict[name] = {'dim': dim, 'bound': bound, 'scale': scale,
                                'warpf': warpf, 'iwarpf': iwarpf, 'logdwarpf': logdwarpf}
        self.param_dict = param_dict

    def join_parameters(self, params):
        return torch.cat([params[name] for name in self.names], dim=-1)

    def join_and_warp_parameters(self, params):
        tocat = [self.warpf(name)(params[name]) for name in self.names]
        res = torch.cat(tocat, dim=-1)
        return res

    def split_parameters(self, x):
        return torch.split(x, [self.dim(name) for name in self.names], dim=-1)

    def split_and_unwarp_parameters(self, x):
        splits = torch.split(x, [self.dim(name)
                                 for name in self.names], dim=-1)
        res = dict([(name, self.iwarpf(name)(splits[i]))
                    for i, name in enumerate(self.names)])
        return res

    def organize_params(self, params):
        params_ = collections.OrderedDict([(key, utils.tensor_convert(params[key]))
                                           for key in self.names])
        return params_

    def set_basedistrib(self, basedistrib):
        if basedistrib is not None:
            assert basedistrib.ndim == self.total_dim
        self.basedistrib = basedistrib
        return self

    def dim(self, key):
        return self.param_dict[key]['dim']

    def bound(self, key):
        return self.param_dict[key]['bound']

    def scale(self, key):
        return self.param_dict[key]['scale']

    def warpf(self, key):
        return self.param_dict[key]['warpf']

    def iwarpf(self, key):
        return self.param_dict[key]['iwarpf']

    def logdwarpf(self, key):
        return self.param_dict[key]['logdwarpf']

    @property
    def names(self):
        return list(self.param_dict.keys())

    @property
    def dims(self):
        return utils.get_subdict(self.param_dict, 'dim')

    @property
    def bounds(self):
        return utils.get_subdict(self.param_dict, 'bound')

    @property
    def scales(self):
        return utils.get_subdict(self.param_dict, 'scacle')

    @property
    def total_dim(self):
        return sum(self.dims.values())


def get_warps(lb, ub, scale=1.0):
    # warpf : bounds -> R
    # iwarpf : R -> bounds
    # logdwarpf = log|warpf'\ : bounds -> R^+
    if lb is None and ub is None:
        warpf = lambda x: x/scale
        iwarpf = lambda x: scale*x
        logdwarpf = lambda x: torch.zeros_like(x) - torch.log(scale)
    elif lb is not None and ub is None:
        bwf, biwf, bdwf = base_positive_warps()
        warpf = lambda x: bwf((x-lb)/scale)
        iwarpf = lambda x: scale*biwf(x) + lb
        logdwarpf = lambda x: bdwf((x-lb)/scale) - torch.log(scale)
    elif lb is None and ub is not None:
        raise NotImplementedError
    elif lb is None and ub is not None:
        assert ub > lb
        warpf = lambda x: bwf((x-lb)/(ub-lb))
        iwarpf = lambda x: (ub-lb)*biwf(x) + lb
        logdwarpf = lambda x: bdwf((x-lb)/(ub-lb)) - \
            torch.log(torch.tensor(1.0)*(ub-lb))
    return warpf, iwarpf, logdwarpf


def base_positive_warps():
    warpf = utils.invsoftplus
    iwarpf = utils.softplus
    logdwarpf = lambda x: -torch.log(1.0-torch.exp(-x))
    return warpf, iwarpf, logdwarpf


def base_bounded_warps():
    # [0,1] -> R
    warpf = lambda x: torch.log(torch.sigmoid(2*x))
    iwarpf = lambda x: torch.log(0.5*torch.logit(x))
    logdwarpf = lambda x: torch.log(2*utils.dsigmoid(2*x))
    return warpf, iwarpf, logdwarpf
