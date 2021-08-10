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
        mean,var = utils.tensor_convert_(mean,var)
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
#        x,mean,var = [utils.tensor_convert(x) for x in 
        x,mean,var = utils.tensor_convert_(x,mean,var)
        ndim = mean.shape[0]
        std = torch.sqrt(var)
        res = -0.5*torch.sum(((x-mean)/std)**2,dim=-1) \
              -torch.sum(torch.log(std)) - \
               ndim/2*math.log(2*math.pi)
        return res
    
    def sample(self,n):
        subkey = self.split_key()
        res = DiagonalNormalDistribution.sample_(
                n,self.mean,self.var,subkey)
        return res
        
    @staticmethod
    def sample_(n,mean,var):
        mean,var = utils.tensor_convert_(mean,var)
        std = torch.sqrt(var)
        ndim = mean.shape[0]
        z = torch.randn((n,ndim))
        res = mean + std*z
        return res
    
    def make_mixture(self):
        mixmeans = torch.unsqueeze(self.mean,0)
        mixvars = torch.unsqueeze(self.var,0)
        weights = torch.ones((1,))
        return MixtureDiagonalNormalDistribution(mixmeans,mixvars,weights)
    
    @property
    def params(self):
        return self.mean,self.var

    def analytical_entropy(self):
        return 0.5*torch.sum(torch.log(2*math.pi*math.e*self.var))
    
class MixtureDiagonalNormalDistribution(ProbabilityDistribution):
    def __init__(self,mixmeans,mixvars,weights,seed=random.randint(1,1000)):
        super().__init__(seed)
        self.nmixtures = weights.shape[0]
        self.ndim = mixmeans.shape[1]
        means,mixvars,weights = utils.tensor_convert_(mixmeans,mixvars,weights)
        self.mixmeans = mixmeans
        self.mixvars = mixvars
        self.weights = weights
        assert(self.mixvars.shape[1] == self.ndim)
        assert(self.mixmeans.shape[0] == self.nmixtures)
        assert(self.mixvars.shape[0] == self.nmixtures)
        self.stds = torch.sqrt(mixvars)
        self.ismixture = True
        
    def logprob(self,x):
        res = MixtureDiagonalNormalDistribution.logprob_(
                x,self.mixmeans,self.mixvars,self.weights)
        return res

    @staticmethod
    def logprob_(x,mixmeans,mixvars,weights):
        x,mixmeans,mixvars,weights = utils.tensor_convert_(x,mixmeans,mixvars,weights)
        stds = torch.sqrt(mixvars)
        ndim = mixmeans.shape[1]
        x = torch.unsqueeze(x,-2) #(n,1,d)
        yi1 = -0.5*torch.sum(((x-mixmeans)/stds)**2,dim=-1) #(n,m)
        yi2 = -torch.sum(torch.log(stds),dim=-1) #(m,)
        yi3 = -ndim/2*math.log(2*math.pi) #(,)
        yi = yi1 + yi2 + yi3 #(n,m)
        res = torch.logsumexp(yi+torch.log(weights),dim=-1)
        return res
    
    def sample(self,n):
        res = self.sample_(n,self.mixmeans,self.mixvars,self.weights)
        return res
    
    @staticmethod
    def sample_(n,mixmeans,mixvars,weights):
        mixmeans,mixvars,weights = utils.tensor_convert_(mixmeans,mixvars,weights)
        stds = torch.sqrt(mixvars)
        nmixtures,ndim = mixmeans.shape
        catinds = torch.multinomial(weights,n,replacement=True)
        z = torch.randn((n,ndim))
        res = mixmeans[catinds,:] + stds[catinds,:]*z
        return res
    
    @property
    def params(self):
        return self.mixmeans,self.mixvars,self.weights
    
    def add_component(self,mean,var,weight,return_new=False):
        mean,var,weights = utils.tensor_convert_(mean,var,weight)
        weights = torch.hstack([(1-weight)*self.weights,weight])
        mixmeans = torch.vstack([self.mixmeans,mean])
        mixvars = torch.vstack([self.mixvars,var])
        if return_new:
            return MixtureDiagonalNormalDistribution(mixmeans,mixvars,weights)
        else:
            self.mixmeans = mixmeans
            self.mixvars = mixvars
            self.weights = weights
            self.stds = torch.sqrt(mixvars)
            self.nmixtures += 1