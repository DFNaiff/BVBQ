# -*- coding: utf-8 -*-
import math

import numpy as np
import torch

from . import kernel_functions
from . import utils

LOCS_25,WEIGHTS_25 = [torch.tensor(x,dtype=torch.float32) for 
                      x in np.polynomial.hermite.hermgauss(25)]
LOCS_50,WEIGHTS_50 = [torch.tensor(x,dtype=torch.float32) for 
                      x in np.polynomial.hermite.hermgauss(50)]


def monte_carlo_bayesian_quadrature(gp,distrib,nsamples,return_var=True):
    samples1 = distrib.sample(nsamples)
    samples2 = distrib.sample(nsamples)
    samples3 = distrib.sample(nsamples)
    z = gp.kernel_function(gp.X,samples1).mean(dim=-1,keepdim=True) #(m,1)
    if return_var:
        gamma = gp.kernel_function(samples2,samples3,diagonal=True).mean()
    else:
        gamma = None
    mean,var = calculate_bq_mean_var(gp,z,gamma)
    if var is None:
        return mean
    else:
        return mean,var


def separable_dmvn_bq(gp,mean,var,nhermite=50,return_var=True):
    mean,var = utils.tensor_convert_(mean,var)
    locs,weights = _get_hermgauss(nhermite)
    locs_ = locs[:,None]*torch.sqrt(var)*math.sqrt(2) + mean
    weights_ = weights[:,None,None] #(k,1,1)
    weights__ = weights[None,:,None] #(1,k,1)
    
    xdata = gp.X
    kernel_tensor = kernel_functions.kernel_function_separated(locs_,
                                                              xdata,
                                                              gp.theta,
                                                              gp.lengthscale,
                                                              kind=gp.kind) #(k,n,d)
    kernel_tensor *= weights_ #(k,n,d)
    kernel_matrix = torch.sum(kernel_tensor,dim=0)*1/math.sqrt(math.pi) #(n,d)
    z = torch.prod(kernel_matrix,dim=-1,keepdim=True)
    if return_var:
        kernel_tensor_2 = kernel_functions.kernel_function_separated(locs_,
                                                                    locs_,
                                                                    gp.theta,
                                                                    gp.lengthscale,
                                                                    kind=gp.kind) #(k,k,d)
        kernel_tensor_2 *= weights_*weights__
        gamma = torch.prod(1/math.pi*torch.sum(torch.sum(kernel_tensor_2,dim=0),dim=0))
    else:
        gamma = None
    mean,var = calculate_bq_mean_var(gp,z,gamma)
    if var is None:
        return mean
    else:
        return mean,var


def separable_mixdmvn_bq(gp,mixmeans,mixvars,weights,nhermite=50,return_var=True):
    mixmeans,mixvars,weights = utils.tensor_convert_(mixmeans,mixvars,weights)
    hlocs,hweights = _get_hermgauss(nhermite)
    hlocs_ = hlocs[:,None,None]*torch.sqrt(mixvars)*math.sqrt(2) + mixmeans #(k,m,d)
    hweights_ = hweights[:,None,None,None] #(k,1,1,1)
    xdata = gp.X
    kernel_tensor = kernel_functions.kernel_function_separated(hlocs_,
                                                              xdata,
                                                              theta=gp.theta,
                                                              l=gp.lengthscale,
                                                              kind=gp.kind) #(k,m,n,d)
    kernel_tensor *= hweights_ #(k,m,n,d)
    kernel_matrices = torch.sum(kernel_tensor,dim=0)*1/math.sqrt(math.pi) #(m,n,d)
    z = torch.unsqueeze(
                    torch.sum(
                        torch.prod(kernel_matrices,dim=-1)*\
                        torch.unsqueeze(weights,-1),
                     dim=0),-1) #(m,n,d) -> (n,d) -> (d,) -> (d,1)
    if return_var:
        hweights__ = hweights[:,None,None] #(k,1,1)
        hweights___ = hweights[:,None,None,None,None] #(k,1,1,1,1)
        kernel_tensor_2 = kernel_functions.kernel_function_separated(hlocs_,
                                                                    hlocs_,
                                                                    theta=gp.theta,
                                                                    l=gp.lengthscale,
                                                                    kind=gp.kind) #(k,m,k,m,d)
        kernel_tensor_2 *= hweights__*hweights___
        gamma_matrix = torch.prod(1/math.pi*torch.sum(torch.sum(kernel_tensor_2,0),1),-1)
        gamma = torch.sum(gamma_matrix*weights*torch.unsqueeze(weights,-1))
    else:
        gamma = None
    mean,var = calculate_bq_mean_var(gp,z,gamma)
    if var is None:
        return mean
    else:
        return mean,var


def calculate_bq_mean_var(gp,z,gamma=None):
    y_,_ = torch.triangular_solve(gp.y-gp.mean,
                                  gp.upper_chol_matrix,
                                  upper=True,
                                  transpose=True) #(m,1)
    z_,_ = torch.triangular_solve(z,
                                  gp.upper_chol_matrix,
                                  upper=True,
                                  transpose=True) #(m,1)
    mean = (gp.mean + z_.transpose(-2,-1)@y_)[0][0] #(1,1) -> (,)
    if gamma is None:
        var = None
    else:
        var = (gamma - z_.transpose(-2,-1)@z_)[0][0] #(1,1) -> (,)
    return mean+gp.ymax,var

def _get_hermgauss(nhermite):
    if nhermite == 25:
        return LOCS_25,WEIGHTS_25
    elif nhermite == 50:
        return LOCS_50,WEIGHTS_50
    else:
        locs,weights = np.polynomial.hermite.hermgauss(nhermite)
        locs = torch.tensor(locs,dtype=torch.float32)
        weights = torch.tensor(weights,dtype=torch.float32)
        return locs,weights