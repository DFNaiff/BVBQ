# -*- coding: utf-8 -*-
from . import utils

def elbo(logprob,distrib,nsamples):
    term1 = distrib.monte_carlo_entropy(nsamples)
    term2 = logprob(distrib.sample(nsamples))
    return term1 + term2


def boosted_elbo(logprob,distrib,newcomp,newweight,nsamples):
    samples_distrib = distrib.sample(nsamples)
    samples_newcomp = newcomp.sample(nsamples)
    joint_distrib = distrib.add_component(*newcomp.params,newweight,return_new=True)
    logprob_mean_1 = logprob(samples_distrib).mean()
    partial_entropy_1 = -joint_distrib.logprob(samples_distrib).mean()
    logprob_mean_2 = logprob(samples_newcomp).mean()
    partial_entropy_2 = -joint_distrib.logprob(samples_newcomp).mean()
    res = (1-newweight)*(logprob_mean_1 + partial_entropy_1) + \
          newweight*(logprob_mean_2 + partial_entropy_2)
    return res


def relbo(logprob,distrib,newcomp,nsamples,reg=1e-2,logdelta=-10):
    global logprobmean
    samples = newcomp.sample(nsamples)
#     entropy = -newcomp.logprob(samples).mean() #-E[log(q)]
    entropy = newcomp.analytical_entropy()
#     logprobmean = logbound(logprob(samples),logdelta).mean()
    logprobmean = logprob(samples).mean()
    negdistribmean = -utils.logbound(distrib.logprob(samples),logdelta).mean()
#     print(logprobmean.primal,
#           negdistribmean.primal,
#           (reg*entropy).primal)
    res = logprobmean + negdistribmean + reg*entropy
#     res = logprobmean + negdistribmean + reg*entropy
    return res