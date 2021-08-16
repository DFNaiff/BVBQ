# -*- coding: utf-8 -*-
# pylint: disable=E1101
"""
    Optimizers for getting next evaluation point for gp,
    based of maximization of acquisition functions
"""

import collections
import torch
import dict_minimize.torch_api

from . import acquisition_functions
from . import utils


def acquire_next_point_mixmvn(x, gp, distrib, name='PP',
                              method='L-BFGS-B', tol=1e-6,
                              options=None):
    """
        Optimizer method for selecting next evaluation point,
        that is, xnew = max_x f(x;gp,distrib)

        Parameters
        ----------
        x : torch.Tensor
            Initial guess for optimization
        gp : SimpleGP
            The associated gp class
        distrib : ProbabilityDistribution
            The associated distribution object
        name : str
            The name of the acquisition function to be used.
            'PP' - Prospective prediction
            'MMLT' - Moment matched log transform
            'PMMLT' - Prospective moment matched log transform
        method : str
            The optimization method to be used in dict_minimize
        tol : float
            Tolerance for optimizer method
        options: None or dict
            Options for the optimizer

        Returns
        -------
        torch.Tensor
            The proposed evaluation point
        
    """
    options = dict() if options is None else options
    x = x.detach().clone()
    acqf = _map_name_acqfunction(name)
    acqf_wrapper = lambda params: -torch.squeeze(acqf(params['x'], gp, distrib))
    params = collections.OrderedDict({'x': x})
    dwrapper = utils.dict_minimize_torch_wrapper(acqf_wrapper)
    res = dict_minimize.torch_api.minimize(dwrapper,
                                           params,
                                           method=method,
                                           tol=tol,
                                           options=options)
    xres = res['x'].detach()
    return xres


def _map_name_acqfunction(name):
    """A simple string -> acqfunction map"""
    if name in ['prospective_prediction', 'PP']:
        acqf = acquisition_functions.prospective_prediction
    elif name in ['moment_matched_log_transform', 'MMLT']:
        acqf = acquisition_functions.moment_matched_log_transform
    elif name in ['prospective_mmlt', 'PMMLT']:
        acqf = acquisition_functions.prospective_mmlt
    return acqf
