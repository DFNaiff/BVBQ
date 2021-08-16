# -*- coding: utf-8 -*-
# pylint: disable=E1101
"""
    BVBQ functions for distribution updating
"""

import collections
import random

import torch
import dict_minimize.torch_api

from . import distributions
from . import bvbq_functions
from . import utils


def propose_component_mvn_mixmvn_relbo(logprobgp,
                                       mixmeans, mixvars, mixweights,
                                       nsamples=100,
                                       maxiter=100,
                                       optim=torch.optim.Adam,
                                       lr=1e-1):
    """
        Propose new component for mixture of diagonal Gaussians
        using RELBO objective function, using stochastic gradient
        descent

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
            Number of samples for Monte Carlo estimation of cross-entropy
        maxiter : int
            Number of steps for SGD
        optim : torch.optim.Optimizer
            Optimizer to be used
        lr : float
            Learning rate of optimizer

        Returns
        -------
        torch.Tensor, torch.Tensor
            mean and variance vector of proposed component

    """
    ndim = mixmeans.shape[1]
    mean0 = distributions.MixtureDiagonalNormalDistribution.sample_(
        1, mixmeans, mixvars, mixweights)
    var0 = torch.distributions.HalfNormal(1.0).sample((ndim,))
    rawvar0 = torch.log(torch.exp(var0)-1)
    optimizer = optim([mean0, rawvar0], lr=lr)
    mean0.requires_grad = True
    rawvar0.requires_grad = True
    for _ in range(maxiter):
        optimizer.zero_grad()
        var0 = torch.log(torch.exp(rawvar0)+1)
        reg = torch.rand(1)
        relbo = bvbq_functions.mcbq_dmvn_relbo(logprobgp,
                                               mean0,
                                               var0,
                                               mixmeans,
                                               mixvars,
                                               mixweights,
                                               nsamples=nsamples,
                                               reg=reg)
        loss = -relbo
        loss.backward()
        optimizer.step()
    mean = mean0.detach()
    rawvar = rawvar0.detach()
    var = torch.log(torch.exp(rawvar)+1)
    return mean, var


def propose_component_mvn_mixmvn_lbrelbo(logprobgp,
                                         mixmeans, mixvars, mixweights,
                                         method='L-BFGS-B',
                                         tol=1e-6,
                                         options=None):
    """
        Propose new component for mixture of diagonal Gaussians
        using LRELBO objective function, using dict-minimize

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
        method : str
            The optimization method to be used in dict_minimize
        tol : float
            Tolerance for optimizer method
        options: None or dict
            Options for the optimizer

        Returns
        -------
        torch.Tensor, torch.Tensor
            mean and variance vector of proposed component

    """
    options = dict() if options is None else options
    ndim = mixmeans.shape[1]
    mean0 = distributions.MixtureDiagonalNormalDistribution.sample_(
        1, mixmeans, mixvars, mixweights)[0, :]
    var0 = torch.distributions.HalfNormal(1.0).sample((ndim,))
    rawvar0 = torch.log(torch.exp(var0)-1)

    reg = random.random()

    def lbrelbo_wrapper(params):
        return -bvbq_functions.mcbq_dmvn_lbrelbo(
            logprobgp,
            params['mean'],
            torch.log(torch.exp(params['rawvar'])-1),
            mixmeans,
            mixvars,
            mixweights,
            reg=reg)
    params = collections.OrderedDict({'mean': mean0, 'rawvar': rawvar0})
    dwrapper = utils.dict_minimize_torch_wrapper(lbrelbo_wrapper)
    res = dict_minimize.torch_api.minimize(dwrapper,
                                           params,
                                           method=method,
                                           tol=tol,
                                           options=options)

    mean = res['mean'].detach()
    rawvar = res['rawvar'].detach()
    var = torch.log(torch.exp(rawvar)+1)
    return mean, var


def update_distribution_mvn_mixmvn(logprobgp,
                                   mean, var,
                                   mixmeans, mixvars,
                                   mixweights,
                                   nsamples=1000,
                                   weight_delta=1e-8,
                                   lr=1e-1,
                                   maxiter=100,
                                   decaying_lr=True):
    """
        Update distribution for new component of
        mixture of diagonal Gaussians, adjusting
        the component weight. Weight to be in
        [weight_delta,1-weight_delta]

        Parameters
        ----------
        logprobgp : SimpleGP
            Gaussian Process object approximating logdensity
        mean : torch.Tensor
            Mean vector of proposed diagonal Gaussian distribution
        var : torch.Tensor
            Variance vector of proposed diagonal Gaussian distribution
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
        weight_delta : float
            Value for bounds of weight
        lr : float
            Learning rate of optimization
        maxiter : int
            Number of steps for SGD
        decaying_lr : bool
            If true, learning rate decays as lr/(nsteps + 1)

        Returns
        -------
        torch.Tensor,torch.Tensor,torch.Tensor
            Mean vector, variance vector and weights of updated distribution

    """
    weight = torch.tensor(weight_delta)
    grad_term_logprob = bvbq_functions.logprob_terms_mixdmvn_delbodw(logprobgp,
                                                                     mean, var,
                                                                     mixmeans,
                                                                     mixvars,
                                                                     mixweights)
    for j in range(maxiter):
        grad_term_entropy = bvbq_functions.entropy_terms_mixdmvn_delbodw(weight,
                                                                         mean,
                                                                         var,
                                                                         mixmeans,
                                                                         mixvars,
                                                                         mixweights,
                                                                         nsamples)
        grad = grad_term_logprob + grad_term_entropy
        if decaying_lr:
            lr_ = lr/(j+1)
        else:
            lr_ = lr
        dweight = lr_*grad
        weight += dweight
        weight = torch.clamp(weight, weight_delta, 1-weight_delta)
    mixmeans_up = torch.vstack([mixmeans, mean])
    mixvars_up = torch.vstack([mixvars, var])
    mixweights_up = torch.hstack([(1-weight)*mixweights, weight])
    return mixmeans_up, mixvars_up, mixweights_up


def fit_mixmvn_elbo(logprobgp,
                    mixmeans, mixvars, mixweights,
                    nsamples=100,
                    maxiter=100,
                    optim=torch.optim.Adam,
                    lr=1e-1,
                    weights_bound=1e-8):
    """
        Fit mixtures of diagonal Gaussians
        using ELBO objective function,
        using stochastic gradient descent

        Parameters
        ----------
        logprobgp : SimpleGP
            Gaussian Process object approximating logdensity
        mixmeans : torch.Tensor
            Initial mean matrix of current mixtures of
            diagonal normal distribution, of shape (nmixtures,dim)
        mixvars : torch.Tensor
            Initial variance matrix of current mixtures of
            diagonal normal distribution, of shape (nmixtures,dim)
        mixweights : torch.Tensor
            Initial weights vector of current mixture components
        nsamples : int
            Number of samples for Monte Carlo estimation of cross-entropy
        maxiter : int
            Number of steps for SGD
        optim : torch.optim.Optimizer
            Optimizer to be used
        lr : float
            Learning rate of optimizer
        weights_bound : float
            Minimum weight of each component (before normalization)

        Returns
        -------
        torch.Tensor,torch.Tensor,torch.Tensor
            Mean vector, variance vector and weights of updated distribution

    """
    rawmixmeans = mixmeans.detach().clone()
    rawmixvars = torch.log(torch.exp(mixvars.detach().clone())-1)
    rawmixweights = torch.log(torch.exp(mixweights.detach().clone())-1)
    optimizer = optim([rawmixmeans, rawmixvars, rawmixweights], lr=lr)
    rawmixmeans.requires_grad = True
    rawmixvars.requires_grad = True
    rawmixweights.requires_grad = True
    for _ in range(maxiter):
        optimizer.zero_grad()
        mixmeans = rawmixmeans
        mixvars = torch.log(torch.exp(rawmixvars)+1)
        mixweights = torch.log(torch.exp(rawmixweights)+1)+weights_bound
        mixweights = mixweights/torch.sum(mixweights)
        elbo = bvbq_functions.bq_mixmvn_elbo(logprobgp,
                                             mixmeans,
                                             mixvars,
                                             mixweights,
                                             nsamples)
        loss = -elbo
        loss.backward()
        optimizer.step()
    mixmeans = rawmixmeans.detach()
    mixvars = torch.log(torch.exp(rawmixvars.detach())+1)
    mixweights = torch.log(torch.exp(rawmixweights.detach())+1)+weights_bound
    mixweights = mixweights/torch.sum(mixweights.detach())
    return mixmeans, mixvars, mixweights


def reweight_mixmvn_elbo(logprobgp,
                         mixmeans, mixvars, mixweights,
                         nsamples=100,
                         maxiter=100,
                         optim=torch.optim.Adam,
                         lr=1e-1,
                         weights_bound=1e-8):
    """
        Fit weights mixtures of diagonal Gaussians
        using ELBO objective function,
        using stochastic gradient descent

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
            Initial weights vector of current mixture components
        nsamples : int
            Number of samples for Monte Carlo estimation of cross-entropy
        maxiter : int
            Number of steps for SGD
        optim : torch.optim.Optimizer
            Optimizer to be used
        lr : float
            Learning rate of optimizer
        weights_bound : float
            Minimum weight of each component (before normalization)

        Returns
        -------
        torch.Tensor,torch.Tensor,torch.Tensor
            Mean vector, variance vector and weights of updated distribution

    """
    mixmeans = mixmeans.detach().clone()
    mixvars = mixvars.detach().clone()
    rawmixweights = torch.log(torch.exp(mixweights.detach().clone())-1)
    optimizer = optim([rawmixweights], lr=lr)
    rawmixweights.requires_grad = True
    for _ in range(maxiter):
        optimizer.zero_grad()
        mixweights_ = torch.log(torch.exp(rawmixweights)+1)+weights_bound
        mixweights = mixweights_/torch.sum(mixweights_)
        elbo = bvbq_functions.bq_mixmvn_elbo(logprobgp,
                                             mixmeans,
                                             mixvars,
                                             mixweights,
                                             nsamples)
        loss = -elbo
        loss.backward()
        optimizer.step()
    mixweights = torch.log(torch.exp(rawmixweights.detach())+1)+weights_bound
    mixweights = mixweights/torch.sum(mixweights.detach())
    return mixmeans, mixvars, mixweights
