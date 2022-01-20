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


def acquire_next_point_mixmvn(x, gp, distrib,
                              name='PP',
                              unwarped=False,
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
    unwarped : bool
        If True, ajust mean and logprob to correspond to unwarped density
    name : str
        The name of the acquisition function to be used.
        'PP' - Prospective prediction
        'MMLT' - Moment matched log transform
        'PMMLT' - Prospective moment matched log transform
        'WE' - Warped entropy
    method : str
        The optimization method to be used in dict_minimize
    tol : float
        Tolerance for optimizer method
    options: None or dict
        Options for the optimizer

    Returns
    -------
    torch.Tensor, torch.Tensor
        The proposed evaluation point and the value
        
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
    yres = -acqf_wrapper(res).detach()
    return xres, yres


def wiggles_acquisition_point(x, gp, nsamples=100, lfactor=0.2):
    """
    Wiggles x in order to increase distance from gp points, 
    thus trying to increase stability
    
    Parameters
    ----------
    x : torch.Tensor
        Point to be wiggled
    gp : SimpleGP
        The associated gp class
    nsamples : int
        Number of samples for wiggling
    lfactor : float
        Factor to multiply GP lengthscale in wiggling ball
        
    Returns
    -------
    torch.Tensor
        Wiggled point

    """
    xdata = gp.X
    r = lfactor*gp.lengthscale
    x = utils.get_greater_distance_random(x,xdata,r,nsamples)
    return x

    
def _map_name_acqfunction(name):
    """A simple string -> acqfunction map"""
    if name in ['prospective_prediction', 'PP']:
        acqf = acquisition_functions.prospective_prediction
    elif name in ['moment_matched_log_transform', 'MMLT']:
        acqf = acquisition_functions.moment_matched_log_transform
    elif name in ['prospective_mmlt', 'PMMLT']:
        acqf = acquisition_functions.prospective_mmlt
    elif name in ['warped_entropy', 'WE']:
        acqf = acquisition_functions.warped_entropy
    return acqf
