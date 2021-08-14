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


def prospective_prediction(x, gp, distrib):
    """Prospective prediction acquisition function"""
    mean, var = gp.predict(x, return_cov=True, onlyvar=True)
    logprob = distrib.logprob(x)
    res = torch.exp(mean+2*logprob)*var
    return res


def moment_matched_log_transform(x, gp, distrib):
    """Moment matched log transform acquisition function"""
    mean, var = gp.predict(x, return_cov=True, onlyvar=True)
    res = torch.exp(2*mean + var)*(torch.exp(var)-1)
    return res


def prospective_mmlt(x, gp, distrib):
    """Prospective moment matched log transform acquisition function"""
    mean, var = gp.predict(x, return_cov=True, onlyvar=True)
    logprob = distrib.logprob(x)
    res = torch.exp(2*mean+2*logprob+var)*(torch.exp(var)-1)
    return res
