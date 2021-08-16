# -*- coding: utf-8 -*-
# pylint: disable=E1101
"""Evaluation metrics for BVBQ algorithm"""

import torch

from . import bvbq_functions
from . import distributions
from . import bayesquad


def bq_mixmvn_elbo(logprobgp, mixmeans, mixvars, mixweights, nsamples):
    """Alias for bvbq_functions.bq_mixmvn_elbo"""
    return bvbq_functions.bq_mixmvn_elbo(logprobgp,
                                         mixmeans,
                                         mixvars,
                                         mixweights,
                                         nsamples)


def bq_mixmvn_elbo_with_var(logprobgp,
                            mixmeans,
                            mixvars,
                            mixweights,
                            nsamples):
    """
    Mean and variance of ELBO between GP estimation of logdensity
    and proposed mixture of diagonal gaussians

    Parameters
    ----------
    logprobgp : SimpleGP
        Gaussian Process object approximating logdensity
    mixmeans : torch.Tensor
        Mean matrix of current mixtures of
        diagonal normal distribution, of shape (nmixtures,dim)
    mixvars : torch.Tensor
        Variance matrix of current mixtures of
        diagonal normal distribution, of shape (nmixtures,dim)
    mixweights : torch.Tensor
        Weights vector of current mixture components
    nsamples : int
        Number of samples for Monte Carlo estimation of entropy

    Returns
    -------
    torch.Tensor,torch.Tensor
        mean and standard deviation of ELBO

    """
    term1, var = bayesquad.separable_mixdmvn_bq(logprobgp, mixmeans,
                                                mixvars, mixweights,
                                                return_var=True)
    samples = distributions.MixtureDiagonalNormalDistribution.sample_(
        nsamples, mixmeans, mixvars, mixweights)
    term2 = -distributions.MixtureDiagonalNormalDistribution.logprob_(
        samples, mixmeans, mixvars, mixweights).mean()
    mean = term1 + term2
    std = torch.sqrt(torch.clip(var, 0.0))
    return mean, std
