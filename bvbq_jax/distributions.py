# -*- coding: utf-8 -*-
import abc
import random
import functools

import jax
import jax.numpy as jnp

from . import utils
from . import distributions_utils


class ProbabilityDistribution(abc.ABC):
    def __init__(self,seed):
        self.seed = seed
        self.key = jax.random.PRNGKey(seed)
        self.ismixture = False
        
    def split_key(self):
        key,subkey = jax.random.split(self.key)
        self.key = key
        return subkey
    
    @abc.abstractmethod
    def logprob(self,x):
        pass
    
    @abc.abstractmethod
    def sample(self,n):
        pass

    @property
    @abc.abstractmethod
    def params(self):
        pass
    
    def monte_carlo_entropy(self,n):
        return -jnp.mean(self.logprob(self.sample(n)))

    
class DiagonalNormalDistribution(ProbabilityDistribution):
    def __init__(self,mean,var,seed=random.randint(1,1000)):
        super().__init__(seed)
        self.ndim = mean.shape[0]
        self.mean = mean
        self.var = var
        assert(len(self.var) == self.ndim)
        assert(len(self.mean) == self.ndim)
        self.std = jnp.sqrt(self.var)
    
    def logprob(self,x):
        res = DiagonalNormalDistribution.logprob_(
                x,self.mean,self.var)
        return res
    
    @staticmethod
    @jax.jit
    def logprob_(x,mean,var):
        ndim = mean.shape[0]
        std = jnp.sqrt(var)
        res = -0.5*jnp.sum(((x-mean)/std)**2,axis=-1) \
              -jnp.sum(jnp.log(std)) - ndim/2*jnp.log(2*jnp.pi)
        return res
    
    def sample(self,n):
        subkey = self.split_key()
        res = DiagonalNormalDistribution.sample_(
                n,self.mean,self.var,subkey)
        return res
    
    @staticmethod
    def sample_(n,mean,var,subkey):
        ndim = mean.shape[0]
        z = jax.random.normal(subkey,shape=(n,ndim))
        res = utils.dmvn_samples_from_zsamples(mean,var,z)
        return res
    
    def make_mixture(self):
        means = jnp.expand_dims(self.mean,0)
        variances = jnp.expand_dims(self.var,0)
        weights = jnp.ones(1)
        return MixtureDiagonalNormalDistribution(means,variances,weights)
    
    @property
    def params(self):
        return self.mean,self.var

    def analytical_entropy(self):
        return 0.5*jnp.sum(jnp.log(2*jnp.pi*jnp.e*self.var))


class MixtureDiagonalNormalDistribution(ProbabilityDistribution):
    def __init__(self,means,variances,weights,seed=random.randint(1,1000)):
        super().__init__(seed)
        self.nmixtures = weights.shape[0]
        self.ndim = means.shape[1]
        self.means = means
        self.variances = variances
        self.weights = weights
        assert(self.variances.shape[1] == self.ndim)
        assert(self.means.shape[0] == self.nmixtures)
        assert(self.variances.shape[0] == self.nmixtures)
        self.stds = jnp.sqrt(variances)
        self.ismixture = True
        
    def logprob(self,x):
        res = MixtureDiagonalNormalDistribution.logprob_(
                x,self.means,self.variances,self.weights)
        return res

    @staticmethod
    @jax.jit
    def logprob_(x,means,variances,weights):
        stds = jnp.sqrt(variances)
        ndim = means.shape[1]
        x = jnp.expand_dims(x,-2) #(n,1,d)
        yi1 = -0.5*jnp.sum(((x-means)/stds)**2,axis=-1) #(n,m)
        yi2 = -0.5*jnp.sum(jnp.log(variances),axis=-1) #(m,)
        yi3 = -ndim/2*jnp.log(2*jnp.pi) #(,)
        yi = yi1 + yi2 + yi3 #(n,m)
        res = utils.logsumexp(yi,weights,axis=-1)
#        ymax = jnp.max(yi,axis=-1,keepdims=True) #(n,1)
#        sumexp = jnp.sum(weights*jnp.exp(yi-ymax),axis=-1)
#        res = jnp.squeeze(ymax,axis=-1) + jnp.log(sumexp) #(n,)
        return res
    
    def sample(self,n):
        subkey1 = self.split_key()
        subkey2 = self.split_key()
        res = self.sample_(n,self.means,self.variances,self.weights,subkey1,subkey2)
        return res
    
##     @functools.partial(jax.jit,static_argnums=(0,1))
#    def _sample(self,n,means,stds,weights):
#        subkey = self.split_key()
#        nmixtures,ndim = means.shape
#        catinds = jax.random.choice(subkey,nmixtures,shape=(n,),p=weights)
#        z = jax.random.normal(subkey,shape=(n,ndim))
#        res = means[catinds,:] + stds[catinds,:]*z
#        return res
    
    @staticmethod
    def sample_(n,means,variances,weights,subkey1,subkey2):
        nmixtures,ndim = means.shape
        catinds = jax.random.choice(subkey1,nmixtures,shape=(n,),p=weights)
        z = jax.random.normal(subkey2,shape=(n,ndim))
        res = utils.mixdmvn_samples_from_zsamples_catinds(means,variances,z,catinds)
        return res
    
    @property
    def params(self):
        return self.means,self.variances,self.weights
    
    def add_component(self,mean,var,weight,return_new=False):
        weights = jnp.append((1-weight)*self.weights,weight)
        means = jnp.vstack([self.means,mean])
        variances = jnp.vstack([self.variances,var])
        if return_new:
            return MixtureDiagonalNormalDistribution(means,variances,weights)
        else:
            self.means = means
            self.variances = variances
            self.weights = weights
            self.stds = jnp.sqrt(variances)
            self.nmixtures += 1