# -*- coding: utf-8 -*-
import functools

import jax
import jax.numpy as jnp

from . import distributions
from . import utils
from . import bq

def mcbq_dmvn_relbo(gp,mean,var,mixmeans,mixvars,mixweights,
                    zsamples,logdelta,reg):
    hlocs,hweights = bq.LOCS_25,bq.WEIGHTS_25
    res = mcbq_dmvn_relbo_aux(mean,var,hlocs,hweights,
                              gp.X,gp.y_,gp.mean,
                              gp.theta,gp.lengthscale,
                              gp.upper_chol_matrix,mixmeans,mixvars,mixweights,
                              zsamples,logdelta,reg,gp.kind)
    return res

def mcbq_mixdmvn_gradboost_delbodw(weight,logprobgp,mean,var,
                                   mixmeans,mixvars,mixweights,
                                   key,nsamples=100):
    mixmeans_up = jnp.vstack([mixmeans,mean])
    mixvars_up = jnp.vstack([mixvars,var])
    mixweights_up = jnp.append((1-weight)*mixweights,weight)

    subkeya,key = jax.random.split(key)
    subkeyb,key = jax.random.split(key)
    subkeyc,key = jax.random.split(key)

    nmixtures,ndim = mixmeans.shape
    catinds = jax.random.choice(subkeya,nmixtures,shape=(nsamples,),p=mixweights)
    zprevious = jax.random.normal(subkeyb,shape=(nsamples,ndim))
    zproposal = jax.random.normal(subkeyc,shape=(nsamples,ndim))
    samplesprevious = utils.mixdmvn_samples_from_zsamples_catinds(mixmeans,mixvars,zprevious,catinds)
    samplesproposal = utils.dmvn_samples_from_zsamples(mean,var,zproposal)

    term1 = bq.separable_dmvn_bq(logprobgp,mean,var,return_var=False) 
    term2 = -bq.separable_mixdmvn_bq(logprobgp,mixmeans,
                                          mixvars,mixweights,
                                          return_var=False)
    term3 = -distributions.MixtureDiagonalNormalDistribution.logprob_(
                samplesproposal,mixmeans_up,mixvars_up,mixweights_up).mean()
    term4 = distributions.MixtureDiagonalNormalDistribution.logprob_(
                samplesprevious,mixmeans_up,mixvars_up,mixweights_up).mean()
    return term1 + term2 + term3 + term4

# =============================================================================
# TO JIT
# =============================================================================
@functools.partial(jax.jit,static_argnums=(16,))
def mcbq_dmvn_relbo_aux(mean,var,hlocs,hweights,
                        xdata,y_,gpmean,theta,lengthscale,
                        upper_chol_matrix,
                        mixmeans,mixvars,mixweights,
                        zsamples,logdelta,reg,
                        kind):
    samples = jnp.sqrt(var)*zsamples + mean
    term1 = bq._separable_dmvn_bq_mean(mean,var,hlocs,hweights,
                                    xdata,y_,gpmean,theta,lengthscale,
                                    upper_chol_matrix,kind)
    term2_ = distributions.MixtureDiagonalNormalDistribution.logprob_(
                    samples,mixmeans,mixvars,mixweights)
    term2_ = utils.logbound(term2_,logdelta)
    term2 = -term2_.mean()
    term3 = 0.5*jnp.sum(jnp.log(2*jnp.pi*jnp.e*var)) #Entropy
    return term1 + term2 + reg*term3

def mcbq_mixdmvn_gradboost_delbodw_term34(weight,term1,term2,mean,var,
                                          mixmeans,mixvars,mixweights,
                                          catinds,zprevious,zproposal):
    mixmeans_up = jnp.vstack([mixmeans,mean])
    mixvars_up = jnp.vstack([mixvars,var])
    mixweights_up = jnp.append((1-weight)*mixweights,weight)

    samplesprevious = utils.mixdmvn_samples_from_zsamples_catinds(mixmeans,mixvars,zprevious,catinds)
    samplesproposal = utils.dmvn_samples_from_zsamples(mean,var,zproposal)

    term3 = -distributions.MixtureDiagonalNormalDistribution.logprob_(
                samplesproposal,mixmeans_up,mixvars_up,mixweights_up).mean()
    term4 = distributions.MixtureDiagonalNormalDistribution.logprob_(
                samplesprevious,mixmeans_up,mixvars_up,mixweights_up).mean()
    return term1 + term2 + term3 + term4

# =============================================================================
# AUX
# =============================================================================
def make_gradboost_samples(key,nsamples,mixweights,nmixtures,ndim):
    subkeya,key = jax.random.split(key)
    subkeyb,key = jax.random.split(key)
    subkeyc,key = jax.random.split(key)

    catinds = jax.random.choice(subkeya,nmixtures,shape=(nsamples,),p=mixweights)
    zprevious = jax.random.normal(subkeyb,shape=(nsamples,ndim))
    zproposal = jax.random.normal(subkeyc,shape=(nsamples,ndim))
    return catinds,zprevious,zproposal