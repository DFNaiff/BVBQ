# -*- coding: utf-8 -*-
import functools
import collections

import numpy as np
import torch
import dict_minimize.jax_api

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
        self.mean = torch.tensor(mean) #THIS IS THE MEAN AFTER ZEROMAX TRANSFORMATION.
                                    #IF NOT USING ZEROMAX TRANSFORMATION, IGNORE
                                    #THIS WARNING
        self.theta = torch.tensor(theta)
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
        self.lengthscale = torch.tensor(lengthscale)
        self.noise = torch.tensor(noise)
        self.fixed_params = set(fixed_params)
        self.zeromax = zeromax
        self.ymax = 0.0 #Neutral element in sum
        
    def set_data(self,X,y,empirical_params=False):
        ndata = X.shape[0]
        X = torch.tensor(X)
        y = torch.tensor(y)
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
        self.make_mean_grad()
    
    def predict(self,xpred,return_cov=True,onlyvar=False):
        s = xpred.shape
        d = len(s)
        if d == 1:
            xpred = jnp.expand_dims(xpred,0)
            res = self._predict(xpred,return_cov=return_cov,onlyvar=onlyvar) #No difference for one item
            if not return_cov:
                mean = res
                return mean[0][0] #For gradient taking
            else:
                mean,var = res
                return mean.item(),var.item()
        elif d == 2:
            return self._predict(xpred,return_cov=return_cov,onlyvar=onlyvar)
        else: #some reshaping trick in order
            #[...,d] -> [n,d]
            print("If tensor has more than 2 dimensions, only diagonal of covariance is returned")
            onlyvar = True
            n = int(np.prod((s[:-1])))
            xpred_r = xpred.reshape(n,s[-1])
            res_r = self._predict(xpred_r,return_cov=return_cov,onlyvar=onlyvar)
            if not return_cov:
                mean_r = res_r
                mean = mean_r.reshape(*(s[:-1]))
                return mean
            else:
                mean_r,var_r = res_r
                mean = mean_r.reshape(*(s[:-1]))
                var = var_r.reshape(*(s[:-1]))
                return mean,var
        
    def _predict(self,xpred,return_cov=True,onlyvar=False):
        #a^T K^-1 b = a^T (U^T U)^-1 b= (U^-T a)^T (U^-T b)
        if len(xpred.shape) == 1:
            xpred = jnp.expand_dims(xpred,0) #(n,d)
        kxpred = self.make_kernel_matrix(self.X,xpred,self.theta,self.lengthscale) #(m,n)
        #,lower=False,trans='T'
        y_ = jax.scipy.linalg.solve_triangular(self.upper_chol_matrix,
                                               self.y_-self.mean,
                                               trans='T') #(m,1)
        kxpred_ = jax.scipy.linalg.solve_triangular(self.upper_chol_matrix,
                                                    kxpred,
                                                    trans='T') #(m,n)
        pred_mean = (kxpred_.transpose()@y_) + self.mean + self.ymax
        if not return_cov:
            return pred_mean
        else:
            Kxpxp = self.make_kernel_matrix(xpred,xpred,self.theta,self.lengthscale)
            Kxpxp = self.noisify_kernel_matrix(Kxpxp,self.noise)
            pred_cov = Kxpxp - \
                       kxpred.transpose()@kxpred_
            if not onlyvar:
                return pred_mean,pred_cov
            else:
                pred_var = jnp.diag(pred_cov)
                return pred_mean,pred_var
    
    def loo_mean_prediction(self,xpred): #Only return mean
        if len(xpred.shape) == 1:
            flatten_at_end = True
            xpred = jnp.expand_dims(xpred,0) #(n,d)
        else:
            flatten_at_end = False
        kxpred = self.make_kernel_matrix(self.X,xpred,self.theta,self.lengthscale) #(m,n)
        means = []
        for i in range(self.ndata):
            drop_inds = jnp.array([i])
            kxpreddown = jnp.delete(kxpred,drop_inds,axis=0) #(m-1,n)
            Uaux = np.array(utils.delete_submatrix(self.upper_chol_matrix,drop_inds))
            Uvec = np.array(jnp.delete(self.upper_chol_matrix[drop_inds,:],drop_inds,axis=1))
            Udown = jnp.array(utils.rankn_update_upper(Uaux,Uvec.transpose())) #(m-1,m-1)
            ydown = jnp.delete(self.y_,drop_inds,axis=0) #(m-1,1)
            ydown_ = jax.scipy.linalg.solve_triangular(Udown,
                                                   ydown-self.mean,
                                                   trans='T') #(m-1,1)
            kxpreddown_ = jax.scipy.linalg.solve_triangular(Udown,
                                                        kxpreddown,
                                                        trans='T') #(m-1,n)
            pred_mean_down = (kxpreddown_.transpose()@ydown_) + self.mean #(n,1)
            means.append(pred_mean_down)
        means = jnp.hstack(means)
        if flatten_at_end:
            means = means.flatten()
        return means
            
    def make_mean_grad(self):
        f = functools.partial(self.predict,return_cov=False)
        mean_grad = jax.grad(f)
        self.mean_grad = mean_grad
        
    def optimize_params(self,fixed_params=[],
                        method='L-BFGS-B',
                        tol=1e-4,options={'disp':True}):
        func_and_grad = jax.value_and_grad(self.loglikelihood_wrapper)
        params = {'raw_theta':self._raw_theta,
                  'mean':self.mean,
                  'raw_lengthscale':self._raw_lengthscale,
                  'raw_noise':self._raw_noise}
        params_list = set(params.keys())
        print(params_list)
        print(fixed_params)
        print(self.fixed_params)
        for param in params_list:
            if param in fixed_params or param in self.fixed_params:
                params.pop(param,None)
                params.pop('raw_'+param,None)
        params = collections.OrderedDict(params)
        res = dict_minimize.jax_api.minimize(func_and_grad, params,
                                             method=method,
                                             tol=tol,
                                             options=options)
        self.theta = self.derawfy(res.get('raw_theta',self._raw_theta))
        self.noise = self.derawfy(res.get('raw_noise',self._raw_noise))
        self.mean = res.get('mean',self.mean)
        self.lengthscale = self.derawfy(res.get('raw_lengthscale',self._raw_lengthscale))
        self.set_data(self.X,self.y)
        return res
    
    def gradient_step_params(self,fixed_params=[],
                             alpha=1e-1,niter=1):
        func_and_grad = jax.value_and_grad(self.loglikelihood_wrapper)
        params = {'raw_theta':self._raw_theta,
                  'mean':self.mean,
                  'raw_lengthscale':self._raw_lengthscale,
                  'raw_noise':self._raw_noise}
        for param in params:
            if param in fixed_params or param in self.fixed_params:
                params.pop(param,None)
                params.pop('raw_'+param,None)
        params = collections.OrderedDict(params)
        for i in range(niter):
            _,grads = func_and_grad(params)
            for key,value in grads.items():
                params[key] -= alpha*value
        self.theta = self.derawfy(params.get('raw_theta',self._raw_theta))
        self.noise = self.derawfy(params.get('raw_noise',self._raw_noise))
        self.mean = params.get('mean',self.mean)
        self.lengthscale = self.derawfy(params.get('raw_lengthscale',self._raw_lengthscale))
        self.set_data(self.X,self.y)
    
    def loglikelihood_wrapper(self,params):
        theta = self.derawfy(params.get('raw_theta',self._raw_theta))
        noise = self.derawfy(params.get('raw_noise',self._raw_noise))
        mean = params.get('mean',self.mean)
        lengthscale = self.derawfy(params.get('raw_lengthscale',self._raw_lengthscale))
        return -self.loglikelihood(theta,lengthscale,noise,mean) #Used for maximization in minimizer
    
    def loglikelihood(self,theta,lengthscale,noise,mean):
        kernel_matrix_ = self.make_kernel_matrix(self.X,self.X,theta,lengthscale)
        kernel_matrix = self.noisify_kernel_matrix(kernel_matrix_,noise)
        upper_chol_matrix = self.make_cholesky(kernel_matrix)
        y_ = jax.scipy.linalg.solve_triangular(upper_chol_matrix,
                                               self.y_-mean,
                                               trans='T') #(m,1)
        term1 = -0.5*jnp.sum(y_**2)
        term2 = -jnp.sum(jnp.log(jnp.diag(upper_chol_matrix)))
        term3 = -0.5*self.ndata*jnp.log(2*np.pi)
        return term1 + term2 + term3
        
    def update(self,Xnew,ynew):
        nnew = Xnew.shape[0]
        if len(ynew.shape) == 1:
            ynew = jnp.expand_dims(ynew,-1) #(d,1)
        if self.zeromax:
            self.ymax = max(jnp.max(ynew),self.ymax)
        assert(ynew.shape[0] == nnew)
        assert(ynew.shape[1] == 1)
        assert(Xnew.shape[1] == self.ndim)
        Xup = jnp.vstack([self.X,Xnew])
        yup = jnp.vstack([self.y_,ynew])
        K11 = self.kernel_matrix
        K12 = self.make_kernel_matrix(self.X,Xnew,
                                      self.theta,
                                      self.lengthscale)
        K21 = K12.transpose()
        K22_ = self.make_kernel_matrix(Xnew,Xnew,
                                      self.theta,
                                      self.lengthscale)
        K22 = self.noisify_kernel_matrix(K22_,self.noise)
        K = jnp.block([[K11,K12],[K21,K22]])
        U11 = self.upper_chol_matrix
        U12 = jax.scipy.linalg.solve_triangular(U11,
                                                K12,
                                                trans='T') #(m,1)
        U21 = jnp.zeros(K21.shape)
        U22 = jax.scipy.linalg.cholesky(K22 - U12.transpose()@U12)
        U = jnp.block([[U11,U12],[U21,U22]])
        self.ndata += nnew
        self.X = Xup
        self.y_ = yup - self.ymax
        self.kernel_matrix = K
        self.upper_chol_matrix = U
        
    def downdate(self,drop_inds):
        drop_inds = jnp.array(drop_inds)
        ndrop = len(drop_inds)
        ndata = self.ndata
        ndown = ndata - ndrop
        swap_inds = jnp.hstack([jnp.delete(jnp.arange(ndata),drop_inds),drop_inds])
        dropped_inds = swap_inds[:ndown]
        Xdown = self.X[dropped_inds,:]
        ydown = self.y_[dropped_inds,:]
        if self.zeromax:
            self.ymax = jnp.max(ydown)
        Kdown = self.kernel_matrix[dropped_inds,:][:,dropped_inds]
        U = self.upper_chol_matrix
        Uaux = np.array(utils.delete_submatrix(U,drop_inds))
        Uvec = np.array(jnp.delete(U[drop_inds,:],drop_inds,axis=1))
        Udown = jnp.array(utils.rankn_update_upper(Uaux,Uvec.transpose()))
        self.ndata = ndown
        self.X = Xdown
        self.y_ = ydown - self.ymax
        self.kernel_matrix = Kdown
        self.upper_chol_matrix = Udown
        return dropped_inds
    
    def kernel_function(self,X1,X2,diagonal=False):
        return self.make_kernel_matrix(X1,X2,
                                       self.theta,
                                       self.lengthscale,
                                       diagonal=diagonal)
        
    def make_kernel_matrix(self,X1,X2,theta,lengthscale,
                           diagonal=False):
        output = 'pairwise' if not diagonal else 'diagonal'
        K = kernelfunctions.kernel_function(X1,X2,kind=self.kind,
                                            output=output,
                                            theta=theta,
                                            l=lengthscale)
        return K
    
    def noisify_kernel_matrix(self,kernel_matrix,noise):
        K_ = utils.jittering(kernel_matrix,noise**2,self.min_jitter)
        return K_
        
    def make_cholesky(self,K):
        U = jax.scipy.linalg.cholesky(K)
        return U
    
    def make_inverse(self,K):
        return jax.scipy.linalg.inv(K)
    
    def set_params_empirical(self,X,y):
        mean = jnp.mean(y)
        theta = jnp.std(y)
        y_ = 2*(y - jnp.min(y))/(jnp.max(y) - jnp.min(y)) - 1
        ay_ = jnp.arccos(y_)
        omega = jnp.linalg.lstsq(np.hstack([np.ones((X.shape[0],1)),X]),ay_,rcond=None)[0][1:]
        l = omega/(2*jnp.pi)
        self.mean = jnp.array(mean)
        self.theta = jnp.array(theta)
        self.l = jnp.array(l)
        
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
    
    @lengthscale.setter
    def lengthscale(self,x):
        self._raw_lengthscale = self.rawfy(x)
    
    @noise.setter
    def noise(self,x):
        self._raw_noise = self.rawfy(x)
        
    @property
    def y(self):
        return self.y_ + self.ymax