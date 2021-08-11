# -*- coding: utf-8 -*-
import collections
import torch
import dict_minimize.torch_api

from . import acquisition_functions
from . import utils


def acquire_next_point_mixmvn(x,gp,distrib,name='PP',
                              method='L-BFGS-B',tol=1e-6,
                              options={}):
    x = x.detach().clone()
    acqf = map_name_acqfunction(name)
    acqf_wrapper = lambda params: \
        -torch.squeeze(acqf(params['x'],gp,distrib))
    params = collections.OrderedDict({'x':x})
    dwrapper = utils.dict_minimize_torch_wrapper(acqf_wrapper)
    res = dict_minimize.torch_api.minimize(dwrapper,
                                           params,
                                           method=method,
                                           tol=tol,
                                           options=options)
    xres = res['x'].detach()
    return xres


def map_name_acqfunction(name):
    if name in ['prospective_prediction','PP']:
        return acquisition_functions.prospective_prediction
    elif name in ['moment_matched_log_transform','MMLT']:
        return acquisition_functions.moment_matched_log_transform
    elif name in ['prospective_mmlt','PMMLT']:
        return acquisition_functions.prospective_mmlt