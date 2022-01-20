# -*- coding: utf-8 -*-
# pylint: disable=E1101
"""
    A collection of acquisition functions to be used

    All functions are of format f(q(x),mean(x),var(x))
    where q(x) is the current distribution of BVBQ,
    mean(x) and var(x) are the mean and variances
    of the associated Gaussian Process at x
"""

import torch


def prospective_prediction(x, gp, distrib, unwarped=False):
    """
    Prospective prediction acquisition function
    
    Parameters
    ----------
    x : torch.Tensor
        Evaluation point
    gp : gp.SimpleGP
        GP approximating (warped) log density
    distrib : named_distributions.NamedDistribution
        Named distribution approximating log density
    unwarped : bool
        If True, ajust mean and logprob to correspond to unwarped density
    
    Returns
    -------
    torch.Tensor
        The acquisition function value

    """
    #q(x)**2*k(x,x)*exp(m(x))
    mean, var = gp.predict(x, to_return='var')
    logprob = distrib.basedistrib.logprob(x)
    if unwarped:
        correction = _apply_unwarped_correction(x, distrib)
        mean = mean + correction
        logprob = logprob + correction
    res = torch.exp(mean+2*logprob)*var
    return res


def moment_matched_log_transform(x, gp, distrib, unwarped=False):
    """
    Moment matched log transform acquisition function
    
    Parameters
    ----------
    x : torch.Tensor
        Evaluation point
    gp : gp.SimpleGP
        GP approximating (warped) log density
    distrib : named_distributions.NamedDistribution
        Named distribution approximating log density
    unwarped : bool
        If True, ajust mean and logprob to correspond to unwarped density
    
    Returns
    -------
    torch.Tensor
        The acquisition function value

    """
    mean, var = gp.predict(x, to_return='var')
    if unwarped:
        correction = _apply_unwarped_correction(x, distrib)
        mean = mean + correction
    res = torch.exp(2*mean + var)*(torch.exp(var)-1)
    return res


def warped_entropy(x, gp, distrib, unwarped=False):
    """
    Warped entropy acquisition function
    
    Parameters
    ----------
    x : torch.Tensor
        Evaluation point
    gp : gp.SimpleGP
        GP approximating (warped) log density
    distrib : named_distributions.NamedDistribution
        Named distribution approximating log density
    unwarped : bool
        If True, ajust mean and logprob to correspond to unwarped density
    
    Returns
    -------
    torch.Tensor
        The acquisition function value

    """
    mean, var = gp.predict(x, to_return='var')
    if unwarped:
        correction = _apply_unwarped_correction(x, distrib)
        mean = mean + correction
    res = torch.log(var)/2.0 + mean
    return res


def prospective_mmlt(x, gp, distrib, unwarped=False):
    """
    Prospective moment matched log transform acquisition function
    
    Parameters
    ----------
    x : torch.Tensor
        Evaluation point
    gp : gp.SimpleGP
        GP approximating (warped) log density
    distrib : named_distributions.NamedDistribution
        Named distribution approximating log density
    unwarped : bool
        If True, ajust mean and logprob to correspond to unwarped density
    
    Returns
    -------
    torch.Tensor
        The acquisition function value

    """
    mean, var = gp.predict(x, to_return='var')
    logprob = distrib.basedistrib.logprob(x)
    if unwarped:
        correction = _apply_unwarped_correction(x, distrib)
        mean = mean + correction
        logprob = logprob + correction
    res = torch.exp(2*mean+2*logprob+var)*(torch.exp(var)-1)
    return res


def _apply_unwarped_correction(x, distrib):
    splits = torch.split(x, [distrib.dim(name)
                             for name in distrib.names], dim=-1)
    corrections = [distrib.logdiwarpf(name)(splits[i])
                   for i, name in enumerate(distrib.names)]
    correction = torch.sum(torch.cat(corrections, dim=-1), dim=-1)
    return correction
