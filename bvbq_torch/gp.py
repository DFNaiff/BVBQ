# -*- coding: utf-8 -*-
import functools
import collections
import math

import numpy as np
import torch
import dict_minimize.torch_api

from . import kernelfunctions
from . import utils


class SimpleGP(object):
    def __init__(self,ndim,
                 theta=1.0,
                 lengthscale=1.0,
                 noise=1e-2,
                 mean=0.0,
                 ard=False,
                 min_jitter=1e-4,
                 kind='sqe',
                 fixed_params=[],
                 zeromax=None):
        self.ndim = ndim
        self.kind = kind
        self.mean = mean #THIS IS THE MEAN AFTER ZEROMAX TRANSFORMATION.
                         #IF NOT USING ZEROMAX TRANSFORMATION, IGNORE
                         #THIS WARNING
        self.theta = utils.tensor_convert(theta)
        self.min_jitter = min_jitter
        if noise <= 1e-20:
            noise = 1e-20 #Just for not getting any infs, jitter takes care of the rest
        if not np.isscalar(lengthscale):
            assert(lengthscale.ndim == 1)
            assert(lengthscale.shape[0] == ndim)
            self.ard = True
        else:
            self.ard = ard
            if self.ard:
                lengthscale = torch.ones(self.ndim)*lengthscale
        self.lengthscale = utils.tensor_convert(lengthscale)
        self.noise = utils.tensor_convert(noise)
        self.fixed_params = set(fixed_params)
        self.zeromax = zeromax
        self.ymax = 0.0 #Neutral element in sum
        
    def set_data(self,X,y,empirical_params=False):
        ndata = X.shape[0]
        X,y = utils.tensor_convert_(X,y)
        if len(y.shape) == 1:
            y = torch.unsqueeze(y,-1) #(d,1)
        if self.zeromax:
            self.ymax = torch.max(y)
        assert(y.shape[0] == ndata)
        assert(y.shape[1] == 1)
        assert(X.shape[1] == self.ndim)
        if empirical_params:
            raise NotImplementedError
            self.set_params_empirical(X,y)
        kernel_matrix_ = self.make_kernel_matrix(X,X,self.theta,self.lengthscale)
        kernel_matrix = self.noisify_kernel_matrix(kernel_matrix_,self.noise)
        upper_chol_matrix = self.make_cholesky(kernel_matrix)
        self.ndata = ndata
        self.X = X
        self.y_ = y - self.ymax
        self.kernel_matrix = kernel_matrix #Noised already
        self.upper_chol_matrix = upper_chol_matrix
    
    def predict(self,xpred,return_cov=True,onlyvar=False,
                return_tensor=True):
        xpred = utils.tensor_convert(xpred)
        s = xpred.shape
        d = len(s)
        if d == 1:
            xpred = torch.unsqueeze(xpred,0)
            res = self._predict(xpred,return_cov=return_cov,onlyvar=onlyvar) #No difference for one item
        elif d == 2:
            res = self._predict(xpred,return_cov=return_cov,onlyvar=onlyvar)
        else: #some reshaping trick in order
            #[...,d] -> [n,d]
            print("If tensor has more than 2 dimensions, only diagonal of covariance is returned")
            onlyvar = True
            n = int(np.prod((s[:-1])))
            xpred_r = xpred.reshape(n,s[-1])
            print(xpred_r.shape)
            res_r = self._predict(xpred_r,return_cov=return_cov,onlyvar=onlyvar)
            if not return_cov:
                mean_r = res_r
                mean = mean_r.reshape(*(s[:-1]))
                res = mean
            else:
                mean_r,var_r = res_r
                mean = mean_r.reshape(*(s[:-1]))
                var = var_r.reshape(*(s[:-1]))
                res = mean,var
        if not return_cov:
            mean = res
            if not return_tensor:
                mean = np.array(mean)
            return mean
        else:
            mean,cov = res
            if not return_tensor:
                mean,cov = np.array(mean),np.array(cov)
            return mean,cov
        
    def _predict(self,xpred,return_cov=True,onlyvar=False):
        #a^T K^-1 b = a^T (U^T U)^-1 b= (U^-T a)^T (U^-T b)
        if len(xpred.shape) == 1:
            xpred = torch.unsqueeze(xpred,0) #(n,d)
        kxpred = self.make_kernel_matrix(self.X,xpred,self.theta,self.lengthscale) #(m,n)
        #,lower=False,trans='T'
        y_,_ = torch.triangular_solve(self.y_-self.mean,
                                      self.upper_chol_matrix,
                                      upper=True,
                                      transpose=True) #(m,1)
        kxpred_,_ = torch.triangular_solve(kxpred,
                                           self.upper_chol_matrix,
                                           upper=True,
                                           transpose=True) #(m,n)
        pred_mean = (kxpred_.transpose(-2,-1)@y_) + self.mean + self.ymax
        pred_mean = torch.squeeze(pred_mean,-1)
        if not return_cov:
            return pred_mean
        else:
            if not onlyvar:
                Kxpxp = self.make_kernel_matrix(xpred,xpred,self.theta,self.lengthscale)
                Kxpxp = self.noisify_kernel_matrix(Kxpxp,self.noise)
                pred_cov = Kxpxp - \
                           kxpred_.transpose(-2,-1)@kxpred_
                return pred_mean,pred_cov
            else:
                Kxpxp = self.make_kernel_matrix(xpred,xpred,self.theta,self.lengthscale,
                                                diagonal=True)
                Kxpxp += self.noise**2 + self.min_jitter
                pred_var = Kxpxp - (kxpred_.transpose(-2,-1)**2).sum(dim=-1)
                return pred_mean,pred_var
    
    def loo_mean_prediction(self,xpred): #Only return mean
        raise NotImplementedError
        
    def optimize_params(self,fixed_params=[],
                        method='L-BFGS-B',
                        tol=1e-4,options={'disp':True}):
        params = {'raw_theta':self._raw_theta,
                  'mean':self.mean,
                  'raw_lengthscale':self._raw_lengthscale,
                  'raw_noise':self._raw_noise}
        params_list = set(params.keys())
        for param in params_list:
            if param[:4] == 'raw_':
                param = param[4:]
            if param in fixed_params or param in self.fixed_params:
                params.pop(param,None)
                params.pop('raw_'+param,None)
        params = collections.OrderedDict(params)
        dwrapper = utils.dict_minimize_torch_wrapper(self.loglikelihood_wrapper)
        res = dict_minimize.torch_api.minimize(dwrapper,
                                               params,
                                               method=method,
                                               tol=tol,
                                               options=options)
        res = dict([(key,value.detach()) for key,value in res.items()])
        self.theta = self.derawfy(res.get('raw_theta',self._raw_theta))
        self.noise = self.derawfy(res.get('raw_noise',self._raw_noise))
        self.mean = res.get('mean',self.mean)
        self.lengthscale = self.derawfy(res.get('raw_lengthscale',self._raw_lengthscale))
        self.set_data(self.X,self.y)
        return res
    
    def gradient_step_params(self,fixed_params=[],
                             alpha=1e-1,niter=1):
        raise NotImplementedError
#        func_and_grad = jax.value_and_grad(self.loglikelihood_wrapper)
#        params = {'raw_theta':self._raw_theta,
#                  'mean':self.mean,
#                  'raw_lengthscale':self._raw_lengthscale,
#                  'raw_noise':self._raw_noise}
#        for param in params:
#            if param in fixed_params or param in self.fixed_params:
#                params.pop(param,None)
#                params.pop('raw_'+param,None)
#                
#        params = collections.OrderedDict(params)
#        for i in range(niter):
#            _,grads = func_and_grad(params)
#            for key,value in grads.items():
#                params[key] -= alpha*value
#        self.theta = self.derawfy(params.get('raw_theta',self._raw_theta))
#        self.noise = self.derawfy(params.get('raw_noise',self._raw_noise))
#        self.mean = params.get('mean',self.mean)
#        self.lengthscale = self.derawfy(params.get('raw_lengthscale',self._raw_lengthscale))
#        self.set_data(self.X,self.y)
    
    def loglikelihood_wrapper(self,params):
        theta = self.derawfy(params.get('raw_theta',self._raw_theta))
        noise = self.derawfy(params.get('raw_noise',self._raw_noise))
        mean = params.get('mean',self.mean)
        lengthscale = self.derawfy(params.get('raw_lengthscale',self._raw_lengthscale))
        print(theta,lengthscale)
        return -self.loglikelihood(theta,lengthscale,noise,mean) #Used for maximization in minimizer
    
    def loglikelihood(self,theta,lengthscale,noise,mean):
        kernel_matrix_ = self.make_kernel_matrix(self.X,self.X,theta,lengthscale)
        kernel_matrix = self.noisify_kernel_matrix(kernel_matrix_,noise)
        upper_chol_matrix = self.make_cholesky(kernel_matrix)
        y_,_ = torch.triangular_solve(self.y_-mean,
                                    upper_chol_matrix,
                                    upper=True,
                                    transpose=True) #(m,1)
        term1 = -0.5*torch.sum(y_**2)
        term2 = -torch.sum(torch.log(torch.diagonal(upper_chol_matrix)))
        term3 = -0.5*self.ndata*math.log(2*np.pi)
        return term1 + term2 + term3
        
    def current_loglikelihood(self):
        return self.loglikelihood(self.theta,self.lengthscale,self.noise,self.mean)
    
    def update(self,Xnew,ynew):
        nnew = Xnew.shape[0]
        Xnew,ynew = utils.tensor_convert_(Xnew,ynew)
        if len(ynew.shape) == 1:
            ynew = torch.unsqueeze(ynew,-1) #(d,1)
        if self.zeromax:
            self.ymax = max(torch.max(ynew),self.ymax)
        assert(ynew.shape[0] == nnew)
        assert(ynew.shape[1] == 1)
        assert(Xnew.shape[1] == self.ndim)
        Xup = torch.vstack([self.X,Xnew])
        yup = torch.vstack([self.y_,ynew])
        K11 = self.kernel_matrix
        K12 = self.make_kernel_matrix(self.X,Xnew,
                                      self.theta,
                                      self.lengthscale)
        K21 = K12.transpose(-2,-1)
        K22_ = self.make_kernel_matrix(Xnew,Xnew,
                                      self.theta,
                                      self.lengthscale)
        K22 = self.noisify_kernel_matrix(K22_,self.noise)
        K = torch.vstack([torch.hstack([K11,K12]),
                          torch.hstack([K21,K22])])
        U11 = self.upper_chol_matrix
        U12,_ = torch.triangular_solve(K12,U11,
                                       upper=True,
                                       transpose=True) #(m,1)
        U21 = torch.zeros(K21.shape)
        U22 = torch.linalg.cholesky(K22 - U12.transpose(-2,-1)@U12).transpose(-2,-1)
        U = torch.vstack([torch.hstack([U11,U12]),
                          torch.hstack([U21,U22])])
        self.ndata += nnew
        self.X = Xup
        self.y_ = yup - self.ymax
        self.kernel_matrix = K
        self.upper_chol_matrix = U
        
    def downdate(self,drop_inds):
        raise NotImplementedError
    
    def kernel_function(self,X1,X2,diagonal=False):
        return self.make_kernel_matrix(X1,X2,
                                       self.theta,
                                       self.lengthscale,
                                       diagonal=diagonal)
        
    def make_kernel_matrix(self,X1,X2,theta,lengthscale,
                           diagonal=False):
        output = 'pairwise' if not diagonal else 'diagonal'
        K = kernelfunctions.kernel_function(X1,X2,
                                            theta=theta,
                                            l=lengthscale,
                                            kind=self.kind,
                                            output=output)
        return K
    
    def noisify_kernel_matrix(self,kernel_matrix,noise):
        K_ = utils.jittering(kernel_matrix,noise**2+self.min_jitter)
        return K_
        
    def make_cholesky(self,K):
        U = torch.linalg.cholesky(K).transpose(-2,-1) #Lower to upper
        return U
    
#    def make_inverse(self,K):
#        return jax.scipy.linalg.inv(K)
    
    def set_params_empirical(self,X,y):
        mean = torch.mean(y)
        theta = torch.std(y)
        y_ = 2*(y - torch.min(y))/(torch.max(y) - torch.min(y)) - 1
        ay_ = torch.arccos(y_)
        omega = torch.linalg.lstsq(np.hstack([np.ones((X.shape[0],1)),X]),ay_,rcond=None)[0][1:]
        l = omega/(2*math.pi)
        if 'mean' not in self.fixed_params:
            self.mean = mean
        if 'theta' not in self.fixed_params:
            self.theta = theta
        if 'lengthscale' not in self.fixed_params:
            self.lengthscale = l
        
    def rawfy(self,x):
        return torch.log(torch.exp(x)-1) #invsoftmax
    
    def derawfy(self,y):
        return torch.log(torch.exp(y)+1) #softmax
    
    def fix_noise(self):
        self.fixed_params.add('noise')
    
    def fix_mean(self):
        self.fixed_params.add('mean')
    
    def unfix_noise(self):
        self.fixed_params.discard('noise')
        
    def unfix_mean(self):
        self.fixed_params.discard('mean')
    
    @property
    def mean(self):
        return self._mean
        
    @property
    def theta(self):
        return self.derawfy(self._raw_theta)
    
    @property
    def lengthscale(self):
        return self.derawfy(self._raw_lengthscale)
    
    @property
    def noise(self):
        return self.derawfy(self._raw_noise)
        
    @theta.setter
    def theta(self,x):
        self._raw_theta = self.rawfy(x)

    @mean.setter
    def mean(self,x):
        self._mean = utils.tensor_convert(x)
        
    @lengthscale.setter
    def lengthscale(self,x):
        self._raw_lengthscale = self.rawfy(x)
    
    @noise.setter
    def noise(self,x):
        self._raw_noise = self.rawfy(x)
        
    @property
    def y(self):
        return self.y_ + self.ymax