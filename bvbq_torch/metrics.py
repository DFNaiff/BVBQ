# -*- coding: utf-8 -*-

import torch

from . import bvbq_functions
from . import distributions
from . import bayesquad


bq_mixmvn_elbo = bvbq_functions.bq_mixmvn_elbo


def bq_mixmvn_elbo_with_var(logprobgp,
                            mixmeans,
                            mixvars,
                            mixweights,
                            nsamples):
    term1,var = bayesquad.separable_mixdmvn_bq(logprobgp,mixmeans,
                                           mixvars,mixweights,
                                           return_var=True)
    samples = distributions.MixtureDiagonalNormalDistribution.sample_(
                        nsamples,mixmeans,mixvars,mixweights)
    term2 = -distributions.MixtureDiagonalNormalDistribution.logprob_(
                samples,mixmeans,mixvars,mixweights).mean()
    mean = term1 + term2
    std = torch.sqrt(torch.clip(var,0.0))
    return mean,std