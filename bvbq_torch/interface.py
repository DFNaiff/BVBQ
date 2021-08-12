# -*- coding: utf-8 -*-
import torch

from . import utils
from . import bvbq
from . import distributions
from . import gp
from . import acquisition
from . import metrics

class BVBQMixMVN(object):
    def __init__(self,eval_function,ndim):
        self.set_eval_function(eval_function)
        self.ndim = ndim
        
    def initialize_data(self,init_policy='data',kind='smatern52',
                        noise=0.0,mean=-30.0,empirical_params=False,
                        **kwargs):
        #TODO : Assertions, customizations and new policies
        assert init_policy in ['data']
        if init_policy == 'data':
            xdata = kwargs.get('xdata')
            ydata = kwargs.get('ydata')
            
        logprobgp = gp.SimpleGP(1,kind=kind,noise=noise,zeromax=True)
        logprobgp.mean = mean
        logprobgp.fix_mean()
        logprobgp.fix_noise()
        logprobgp.set_data(xdata,ydata,empirical_params=empirical_params)
        self.logprobgp = logprobgp
        
    def initialize_components(self,init_policy='manual',**kwargs):
        #TODO : Assertions, customization and new policies
        assert init_policy in ['manual','manual_mix']
        if init_policy == 'manual':
            mean = kwargs.get('mean')
            var = kwargs.get('var')
            mixmeans = torch.atleast_2d(utils.tensor_convert(mean))
            mixvars = torch.atleast_2d(utils.tensor_convert(var))
            mixweights = torch.ones(1)
            nmixtures = 1
        elif init_policy == 'manual_mix':
            nmixtures = mixmeans.shape[0]
        self.mixmeans = mixmeans
        self.mixvars = mixvars
        self.mixweights = mixweights
        self.nmixtures = nmixtures
        
    def update_distribution(self):
        #TODO : Customization
        mean,var = bvbq.propose_component_mvn_mixmvn_relbo(
                    self.logprobgp,
                    self.mixmeans,
                    self.mixvars,
                    self.mixweights)
        mixmeans,mixvars,mixweights = bvbq.update_distribution_mvn_mixmvn(
                                                self.logprobgp,
                                                mean,var,
                                                self.mixmeans,
                                                self.mixvars,
                                                self.mixweights)
        self.mixmeans = mixmeans
        self.mixvars = mixvars
        self.mixweights = mixweights
        
    def update_evaluations(self,name='PP'):
        x0 = self.distribution.sample(1)[0,:]
        x = acquisition.acquire_next_point_mixmvn(x0,
                                                  self.logprobgp,
                                                  self.distribution,
                                                  name='PP')
        y = self.evaluate_single(x)

        #FIXME: Fix this function
#        self.logprobgp.update(x,y)
        #FIXME : Substitute below lines for actual (fixed) efficient update above
        X = torch.vstack([self.eval_points,x])
        y = torch.vstack([self.eval_values,y])
        self.logprobgp.set_data(X,y)
        
    def evaluate_single(self,x):
        return torch.squeeze(self.eval_function(x))
    
    def fit_all_parameters(self):
        #TODO : Customization
        mixmeans,mixvars,mixweights = bvbq.fit_mixmvn_elbo(
            self.logprobgp,self.mixmeans,self.mixvars,self.mixweights)
        
    def fit_all_weights(self):
        #TODO : Customization
        mixmeans,mixvars,mixweights = bvbq.reweight_mixmvn_elbo(
                    self.logprobgp,self.mixmeans,self.mixvars,self.mixweights)
        
    def set_eval_function(self,eval_function):
        self._eval_function = eval_function
        self.eval_function = utils.numpy_to_torch_wrapper(eval_function)
        
    def elbo_metric(self,nsamples=1000):
        return metrics.bq_mixmvn_elbo_with_var(self.logprobgp,
                                               self.mixmeans,
                                               self.mixvars,
                                               self.mixweights,
                                               nsamples=1000)
    
    def optimize_gp_params(self,*args,**kwargs):
        baseopt = kwargs.get('baseopt','QN')
        kwargs.pop('baseopt',None)
        assert baseopt in ['QN','SGD']
        if baseopt == 'QN':
            return self.optimize_gp_params_qn(*args,**kwargs)
        elif baseopt == 'SGD':
            return self.optimize_gp_params_sgd(*args,**kwargs)
    
    @property
    def distribution(self):
        return distributions.MixtureDiagonalNormalDistribution(
                    self.mixmeans,self.mixvars,self.mixweights)
        
    #XXX: This actually performs computation
    @property
    def optimize_gp_params_qn(self):
        return self.logprobgp.optimize_params_qn
    
    @property
    def optimize_gp_params_sgd(self):
        return self.logprobgp.optimize_params_sgd
    
    @property
    def eval_points(self):
        return self.logprobgp.X
    
    @property
    def eval_values(self):
        return self.logprobgp.y