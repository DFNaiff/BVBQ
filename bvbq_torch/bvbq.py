# -*- coding: utf-8 -*-
import collections
import random

import torch
import dict_minimize.torch_api

from . import distributions
from . import bvbq_functions
from . import utils


def propose_component_mvn_mixmvn_relbo(logprobgp,
                                       mixmeans,mixvars,mixweights,
                                       nsamples=100,
                                       maxiter=100,
                                       optim=torch.optim.Adam,
                                       lr=1e-1):
    ndim = mixmeans.shape[1]
    mean0 = distributions.MixtureDiagonalNormalDistribution.sample_(
            1,mixmeans,mixvars,mixweights)[0,:]
    var0 = torch.distributions.HalfNormal(1.0).sample((ndim,))
    rawvar0 = torch.log(torch.exp(var0)-1)
    optimizer = optim([mean0,rawvar0],lr=lr)
    mean0.requires_grad = True
    rawvar0.requires_grad = True
    for i in range(maxiter):
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
    return mean,var


def propose_component_mvn_mixmvn_lbrelbo(logprobgp,
                                        mixmeans,mixvars,mixweights,
                                        method='L-BFGS-B',
                                        tol=1e-6,
                                        options={}):
    ndim = mixmeans.shape[1]
    mean0 = distributions.MixtureDiagonalNormalDistribution.sample_(
            1,mixmeans,mixvars,mixweights)[0,:]
    var0 = torch.distributions.HalfNormal(1.0).sample((ndim,))
    rawvar0 = torch.log(torch.exp(var0)-1)
    
    reg = random.random()
    lbrelbo_wrapper = lambda params: \
        -bvbq_functions.mcbq_dmvn_lbrelbo(
            logprobgp,
            params['mean'],
            torch.log(torch.exp(params['rawvar'])-1),
            mixmeans,
            mixvars,
            mixweights,
            reg=reg)
    params = collections.OrderedDict({'mean':mean0,'rawvar':rawvar0})
    dwrapper = utils.dict_minimize_torch_wrapper(lbrelbo_wrapper)
    res = dict_minimize.torch_api.minimize(dwrapper,
                                           params,
                                           method=method,
                                           tol=tol,
                                           options=options)

    mean = res['mean'].detach()
    rawvar = res['rawvar'].detach()
    var = torch.log(torch.exp(rawvar)+1)
    return mean,var


def update_distribution_mvn_mixmvn(logprobgp,
                                   mean,var,
                                   mixmeans,mixvars,
                                   mixweights,
                                   nsamples=1000,
                                   weight_delta=1e-8,
                                   lr=1e-1,
                                   maxiter=100,
                                   decaying_lr=True):
    weight = torch.tensor(weight_delta)
    grad_term_logprob = bvbq_functions.logprob_terms_mixdmvn_delbodw(weight,
                                                                     logprobgp,
                                                                     mean,var,
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
        weight = torch.clamp(weight,weight_delta,1-weight_delta)
    mixmeans_up = torch.vstack([mixmeans,mean])
    mixvars_up = torch.vstack([mixvars,var])
    mixweights_up = torch.hstack([(1-weight)*mixweights,weight])
    return mixmeans_up,mixvars_up,mixweights_up


def fit_mixmvn_elbo(logprobgp,
                    mixmeans,mixvars,mixweights,
                    nsamples=100,
                    maxiter=100,
                    optim=torch.optim.Adam,
                    lr=1e-1,
                    weights_bound=1e-12):
    rawmixmeans = mixmeans.detach().clone()
    rawmixvars = torch.log(torch.exp(mixvars.detach().clone())-1)
    rawmixweights = torch.log(torch.exp(mixweights.detach().clone())-1)
    optimizer = optim([rawmixmeans,rawmixvars,rawmixweights],lr=lr)
    rawmixmeans.requires_grad = True
    rawmixvars.requires_grad = True
    rawmixweights.requires_grad = True
    for i in range(maxiter):
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
    return mixmeans,mixvars,mixweights


def reweight_mixmvn_elbo(logprobgp,
                    mixmeans,mixvars,mixweights,
                    nsamples=100,
                    maxiter=100,
                    optim=torch.optim.Adam,
                    lr=1e-1,
                    weights_bound=1e-8):
    mixmeans = mixmeans.detach().clone()
    mixvars = mixvars.detach().clone()
    rawmixweights = torch.log(torch.exp(mixweights.detach().clone())-1)
    optimizer = optim([rawmixweights],lr=lr)
    rawmixweights.requires_grad = True
    for i in range(maxiter):
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
    return mixmeans,mixvars,mixweights
