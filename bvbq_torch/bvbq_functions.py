# -*- coding: utf-8 -*-
import math

import torch

from . import bayesquad
from . import distributions
from . import utils


def mcbq_dmvn_relbo(logprobgp,mean,var,mixmeans,mixvars,mixweights,
                    nsamples=100,logdelta=-20,reg=1.0):
    term1 = bayesquad.separable_dmvn_bq(logprobgp,mean,var,return_var=False) #Variance
    samples = distributions.DiagonalNormalDistribution.sample_(nsamples,mean,var)
    term2_ = distributions.MixtureDiagonalNormalDistribution.logprob_(
                samples,mixmeans,mixvars,mixweights)
    term2 = -utils.logbound(term2_,logdelta).mean() #Cross entropy
    term3 = 0.5*torch.sum(torch.log(2*math.pi*math.e*var)) #Entropy
    return term1 + term2 + reg*term3


def mcbq_dmvn_lbrelbo(logprobgp,mean,var,mixmeans,mixvars,mixweights,
                      logdelta=-20,reg=1.0):
    term1 = bayesquad.separable_dmvn_bq(logprobgp,mean,var,return_var=False) #Variance
    term2 = utils.lb_mvn_mixmvn_cross_entropy(
                mean,var,mixmeans,mixvars,mixweights,logdelta) #Cross entropy
    term3 = 0.5*torch.sum(torch.log(2*math.pi*math.e*var)) #Entropy
    return term1 + term2 + reg*term3


def mcbq_mixdmvn_delbodw(weight,logprobgp,mean,var,
                         mixmeans,mixvars,mixweights,
                         nsamples=1000):
    weight = utils.tensor_convert(weight)
    logprob_terms = logprob_terms_mixdmvn_delbodw(
                        weight,logprobgp,mean,var,
                        mixmeans,mixvars,mixweights)
    entropy_terms = entropy_terms_mixdmvn_delbodw(
                        weight,mean,var,
                        mixmeans,mixvars,mixweights,
                        nsamples)
    return logprob_terms + entropy_terms


def logprob_terms_mixdmvn_delbodw(weight,logprobgp,mean,var,
                                  mixmeans,mixvars,mixweights):
    weight = utils.tensor_convert(weight)
    term1 = bayesquad.separable_dmvn_bq(logprobgp,mean,var,return_var=False) 
    term2 = -bayesquad.separable_mixdmvn_bq(logprobgp,mixmeans,
                                          mixvars,mixweights,
                                          return_var=False)
    return term1 + term2


def entropy_terms_mixdmvn_delbodw(weight,mean,var,
                                  mixmeans,mixvars,mixweights,
                                  nsamples=1000):
    weight = utils.tensor_convert(weight)
    mixmeans_up = torch.vstack([mixmeans,mean])
    mixvars_up = torch.vstack([mixvars,var])
    mixweights_up = torch.hstack([(1-weight)*mixweights,weight])

    samplesprevious = distributions.MixtureDiagonalNormalDistribution.sample_(
                        nsamples,mixmeans,mixvars,mixweights)
    samplesproposal = distributions.DiagonalNormalDistribution.sample_(nsamples,mean,var)

    term3 = -distributions.MixtureDiagonalNormalDistribution.logprob_(
                samplesproposal,mixmeans_up,mixvars_up,mixweights_up).mean()
    term4 = distributions.MixtureDiagonalNormalDistribution.logprob_(
                samplesprevious,mixmeans_up,mixvars_up,mixweights_up).mean()
    return term3 + term4


def bq_mixmvn_elbo(logprobgp,mixmeans,mixvars,mixweights,nsamples):
    term1 = bayesquad.separable_mixdmvn_bq(logprobgp,mixmeans,
                                           mixvars,mixweights,
                                           return_var=False)
    samples = distributions.MixtureDiagonalNormalDistribution.sample_(
                        nsamples,mixmeans,mixvars,mixweights)
    term2 = -distributions.MixtureDiagonalNormalDistribution.logprob_(
                samples,mixmeans,mixvars,mixweights).mean()
    return term1 + term2