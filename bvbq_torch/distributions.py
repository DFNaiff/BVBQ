# -*- coding: utf-8 -*-
import abc
import random
import math

import torch

from . import utils


class ProbabilityDistribution(abc.ABC):
    def __init__(self,ismixture=False):
        self.ismixture = False
    
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
        return -torch.mean(self.logprob(self.sample(n)))

    
class DiagonalNormalDistribution(ProbabilityDistribution):
    def __init__(self,mean,var):
        super().__init__()
        self.ndim = mean.shape[0]
        mean = torch.tensor(mean,dtype=torch.float32)
        var = torch.tensor(var,dtype=torch.float32)
        self.mean = mean
        self.var = var
        assert(len(self.var) == self.ndim)
        assert(len(self.mean) == self.ndim)
        self.std = torch.sqrt(self.var)
    
    def logprob(self,x):
        res = DiagonalNormalDistribution.logprob_(
                x,self.mean,self.var)
        return res
    
    @staticmethod
    def logprob_(x,mean,var):
        ndim = mean.shape[0]
        std = torch.sqrt(var)
        res = -0.5*torch.sum(((x-mean)/std)**2,axis=-1) \
              -torch.sum(torch.log(std)) - \
               ndim/2*math.log(2*math.pi)
        return res
    
    def sample(self,n):
        subkey = self.split_key()
        res = DiagonalNormalDistribution.sample_(
                n,self.mean,self.var,subkey)
        return res
        
    @staticmethod
    def sample_(n,mean,var,subkey):
        std = torch.sqrt(var)
        ndim = mean.shape[0]
        z = torch.randn((n,ndim))
        res = mean + std*z
        return res
    
    def make_mixture(self):
        means = torch.unsqueeze(self.mean,0)
        variances = torch.unsqueeze(self.var,0)
        weights = torch.ones((1,))
        return MixtureDiagonalNormalDistribution(means,variances,weights)
    
    @property
    def params(self):
        return self.mean,self.var

    def analytical_entropy(self):
        return 0.5*torch.sum(torch.log(2*math.pi*math.e*self.var))
    
class MixtureDiagonalNormalDistribution(ProbabilityDistribution):
    def __init__(self,means,variances,weights,seed=random.randint(1,1000)):
        super().__init__(seed)
        self.nmixtures = weights.shape[0]
        self.ndim = means.shape[1]
        means = torch.tensor(means,dtype=torch.float32)
        variances = torch.tensor(variances,dtype=torch.float32)
        weights = torch.tensor(weights,dtype=torch.float32)
        self.means = means
        self.variances = variances
        self.weights = weights
        assert(self.variances.shape[1] == self.ndim)
        assert(self.means.shape[0] == self.nmixtures)
        assert(self.variances.shape[0] == self.nmixtures)
        self.stds = torch.sqrt(variances)
        self.ismixture = True
        
    def logprob(self,x):
        res = MixtureDiagonalNormalDistribution.logprob_(
                x,self.means,self.variances,self.weights)
        return res

    @staticmethod
    def logprob_(x,means,variances,weights):
        stds = torch.sqrt(variances)
        ndim = means.shape[1]
        x = torch.unsqueeze(x,-2) #(n,1,d)
        yi1 = -0.5*torch.sum(((x-means)/stds)**2,axis=-1) #(n,m)
        yi2 = -torch.sum(torch.log(stds),axis=-1) #(m,)
        yi3 = -ndim/2*math.log(2*math.pi) #(,)
        yi = yi1 + yi2 + yi3 #(n,m)
        res = utils.logsumexp(yi*torch.log(weights),axis=-1)
        return res
    
    def sample(self,n):
        res = self.sample_(n,self.means,self.variances,self.weights)
        return res
    
    @staticmethod
    def sample_(n,means,variances,weights):
        stds = torch.sqrt(variances)
        nmixtures,ndim = means.shape
        catinds = torch.multinomial(weights,n,replacement=True)
        z = torch.normal(shape=(n,ndim))
        res = means[catinds,:] + stds[catinds,:]*z
        return res
    
    @property
    def params(self):
        return self.means,self.variances,self.weights
    
    def add_component(self,mean,var,weight,return_new=False):
        weights = torch.hstack([(1-weight)*self.weights,weight])
        means = torch.vstack([self.means,mean])
        variances = torch.vstack([self.variances,var])
        if return_new:
            return MixtureDiagonalNormalDistribution(means,variances,weights)
        else:
            self.means = means
            self.variances = variances
            self.weights = weights
            self.stds = torch.sqrt(variances)
            self.nmixtures += 1