# -*- coding: utf-8 -*-
# pylint: disable=E1101

import collections
import math

import numpy as np
import torch
import dict_minimize.torch_api

from . import kernel_functions
from . import utils


class SimpleGP(object):
    def __init__(self, dim,
                 theta=1.0,
                 lengthscale=1.0,
                 noise=1e-2,
                 mean=0.0,
                 ard=False,
                 min_jitter=1e-4,
                 kind='sqe',
                 fixed_params=tuple(),
                 zeromax=None):
        self.dim = dim
        self.kind = kind
        self.mean = mean  # THIS IS THE MEAN AFTER ZEROMAX TRANSFORMATION.
        # IF NOT USING ZEROMAX TRANSFORMATION, IGNORE
        # THIS WARNING
        self.theta = utils.tensor_convert(theta)
        self.min_jitter = min_jitter
        self.lengthscale = self._set_lengthscalengthscale(lengthscale, ard)
        self.noise = utils.tensor_convert(noise)
        self.fixed_params = set(fixed_params)
        self.zeromax = zeromax
        self.ymax = 0.0  # Neutral element in sum
        self.ndata = 0
        self.X = None
        self.y_ = None
        self.kernel_matrix = None
        self.upper_chol_matrix = None

    def set_data(self, X, y, empirical_params=False):
        ndata = X.shape[0]
        X, y = utils.tensor_convert_(X, y)
        if len(y.shape) == 1:
            y = torch.unsqueeze(y, -1)  # (d,1)
        assert y.shape[0] == ndata
        assert y.shape[1] == 1
        assert X.shape[1] == self.dim
        if self.zeromax:
            self.ymax = torch.max(y)
        y_ = y - self.ymax  # Output after output rescaling
        if empirical_params:
            self.set_params_empirical_(X, y)
        kernel_matrix_ = self._make_kernel_matrix(
            X, X, self.theta, self.lengthscale)
        kernel_matrix = self._noisify_kernel_matrix(kernel_matrix_, self.noise)
        upper_chol_matrix = self._make_cholesky(kernel_matrix)
        self.ndata = ndata
        self.X = X
        self.y_ = y_
        self.kernel_matrix = kernel_matrix  # Noised already
        self.upper_chol_matrix = upper_chol_matrix

    def predict(self, xpred, return_cov=True, onlyvar=False,
                return_tensor=True):
        xpred = utils.tensor_convert(xpred)
        s = xpred.shape
        d = len(s)
        if d == 1:
            xpred = torch.unsqueeze(xpred, 0)
            # No difference for one item
            res = self._predict(xpred, return_cov=return_cov, onlyvar=onlyvar)
        elif d == 2:
            res = self._predict(xpred, return_cov=return_cov, onlyvar=onlyvar)
        else:  # some reshaping trick in order
            # [...,d] -> [n,d]
            print(
                "If tensor has more than 2 dimensions, only diagonal of covariance is returned")
            onlyvar = True
            n = int(np.prod((s[:-1])))
            xpred_r = xpred.reshape(n, s[-1])
            res_r = self._predict(
                xpred_r, return_cov=return_cov, onlyvar=onlyvar)
            if not return_cov:
                mean_r = res_r
                mean = mean_r.reshape(*(s[:-1]))
                res = mean
            else:
                mean_r, var_r = res_r
                mean = mean_r.reshape(*(s[:-1]))
                var = var_r.reshape(*(s[:-1]))
                res = mean, var
        if not return_cov:
            mean = res
            if not return_tensor:
                mean = np.array(mean)
            return mean
        else:
            mean, cov = res
            if not return_tensor:
                mean, cov = np.array(mean), np.array(cov)
            return mean, cov

    def loo_mean_prediction(self, xpred):  # Only return mean
        raise NotImplementedError

    def optimize_params_qn(self, fixed_params=tuple(),
                           method='L-BFGS-B',
                           tol=1e-1, options=None):
        params = {'raw_theta': self._raw_theta,
                  'mean': self.mean,
                  'raw_lengthscale': self._raw_lengthscale,
                  'raw_noise': self._raw_noise}
        params = utils.crop_fixed_params_gp(params,
                                            self.fixed_params.union(fixed_params))
        params = collections.OrderedDict(params)
        dwrapper = utils.dict_minimize_torch_wrapper(
            self._loglikelihood_wrapper)
        options = dict() if options is None else options
        res = dict_minimize.torch_api.minimize(dwrapper,
                                               params,
                                               method=method,
                                               tol=tol,
                                               options=options)
        res = dict([(key, value.detach()) for key, value in res.items()])
        self.theta = self._derawfy(res.get('raw_theta', self._raw_theta))
        self.noise = self._derawfy(res.get('raw_noise', self._raw_noise))
        self.mean = res.get('mean', self.mean)
        self.lengthscale = self._derawfy(
            res.get('raw_lengthscale', self._raw_lengthscale))
        self.set_data(self.X, self.y)
        return res

    def optimize_params_sgd(self, fixed_params=tuple(),
                            maxiter=100,
                            optim=torch.optim.Adam,
                            lr=1e-1,
                            verbose=False):
        params = {'raw_theta': self._raw_theta,
                  'mean': self.mean,
                  'raw_lengthscale': self._raw_lengthscale,
                  'raw_noise': self._raw_noise}
        params = utils.crop_fixed_params_gp(params,
                                            self.fixed_params.union(fixed_params))
        for _, tensor in params.items():
            tensor.requires_grad = True
        optimizer = optim(list(params.values()), lr=lr)
        for i in range(maxiter):
            optimizer.zero_grad()
            loss = self._loglikelihood_wrapper(params)
            loss.backward()
            if verbose:
                print(i, loss.detach().numpy().item())
                print([(p, v.detach().numpy()) for p, v in params.items()])
                print('-'*5)
            optimizer.step()
        res = dict([(key, value.detach().clone())
                    for key, value in params.items()])
        self.theta = self._derawfy(res.get('raw_theta', self._raw_theta))
        self.noise = self._derawfy(res.get('raw_noise', self._raw_noise))
        self.mean = res.get('mean', self.mean)
        self.lengthscale = self._derawfy(
            res.get('raw_lengthscale', self._raw_lengthscale))
        self.set_data(self.X, self.y)
        return res

    def gradient_step_params(self, fixed_params=tuple(),
                             alpha=1e-1, niter=1):
        raise NotImplementedError

    def current_loglikelihood(self):
        return self._loglikelihood(self.theta, self.lengthscale, self.noise, self.mean)

    def update(self, Xnew, ynew):
        nnew = Xnew.shape[0]
        Xnew, ynew = utils.tensor_convert_(Xnew, ynew)
        Xnew = torch.atleast_2d(Xnew)
        ynew = torch.atleast_2d(ynew)
        if self.zeromax:
            self.ymax = max(torch.max(ynew), self.ymax)
        assert ynew.shape[0] == nnew
        assert ynew.shape[1] == 1
        assert Xnew.shape[1] == self.dim
        Xup = torch.vstack([self.X, Xnew])
        yup = torch.vstack([self.y, ynew])
        K11 = self.kernel_matrix
        K12 = self._make_kernel_matrix(self.X, Xnew,
                                      self.theta,
                                      self.lengthscale)
        K21 = K12.transpose(-2, -1)
        K22_ = self._make_kernel_matrix(Xnew, Xnew,
                                       self.theta,
                                       self.lengthscale)
        K22 = self._noisify_kernel_matrix(K22_, self.noise)
        K = torch.vstack([torch.hstack([K11, K12]),
                          torch.hstack([K21, K22])])
        U11 = self.upper_chol_matrix
        U12, _ = torch.triangular_solve(K12, U11,
                                        upper=True,
                                        transpose=True)  # (m,1)
        U21 = torch.zeros(K21.shape)
        U22 = torch.linalg.cholesky(
            K22 - U12.transpose(-2, -1)@U12).transpose(-2, -1)
        U = torch.vstack([torch.hstack([U11, U12]),
                          torch.hstack([U21, U22])])
        self.ndata += nnew
        self.X = Xup
        self.y_ = yup - self.ymax
        self.kernel_matrix = K
        self.upper_chol_matrix = U

    def downdate(self, drop_inds):
        raise NotImplementedError

    def kernel_function(self, X1, X2, diagonal=False):
        return self._make_kernel_matrix(X1, X2,
                                       self.theta,
                                       self.lengthscale,
                                       diagonal=diagonal)

    def set_params_empirical_(self, X, y):
        mean = torch.mean(y) if 'mean' not in self.fixed_params else self.mean
        theta = torch.sqrt(torch.mean((y-mean)**2))  # Biased, but whatever
        horizontal_scale = (torch.max(X, dim=0).values -
                            torch.min(X, dim=0).values)
        lengthscale = horizontal_scale/3.0
        if 'mean' not in self.fixed_params:
            self.mean = mean
        if 'theta' not in self.fixed_params:
            self.theta = theta
        if 'lengthscale' not in self.fixed_params:
            self.lengthscale = lengthscale

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
        return self._derawfy(self._raw_theta)

    @property
    def lengthscale(self):
        return self._derawfy(self._raw_lengthscale)

    @property
    def noise(self):
        return self._derawfy(self._raw_noise)

    @theta.setter
    def theta(self, x):
        self._raw_theta = self._rawfy(x)

    @mean.setter
    def mean(self, x):
        self._mean = utils.tensor_convert(x)

    @lengthscale.setter
    def lengthscale(self, x):
        self._raw_lengthscale = self._rawfy(x)

    @noise.setter
    def noise(self, x):
        x = torch.clamp(x, 1e-20, None) #In order to avoid -infs for rawfy
        self._raw_noise = self._rawfy(x)

    @property
    def y(self):
        return self.y_ + self.ymax

    def _predict(self, xpred, return_cov=True, onlyvar=False):
        # a^T K^-1 b = a^T (U^T U)^-1 b= (U^-T a)^T (U^-T b)
        if len(xpred.shape) == 1:
            xpred = torch.unsqueeze(xpred, 0)  # (n,d)
        kxpred = self._make_kernel_matrix(
            self.X, xpred, self.theta, self.lengthscale)  # (m,n)
        y_, _ = torch.triangular_solve(self.y_-self.mean,
                                       self.upper_chol_matrix,
                                       upper=True,
                                       transpose=True)  # (m,1)
        kxpred_, _ = torch.triangular_solve(kxpred,
                                            self.upper_chol_matrix,
                                            upper=True,
                                            transpose=True)  # (m,n)
        pred_mean = (kxpred_.transpose(-2, -1)@y_) + self.mean + self.ymax
        pred_mean = torch.squeeze(pred_mean, -1)
        if not return_cov:
            return pred_mean
        else:
            if not onlyvar:
                Kxpxp = self._make_kernel_matrix(
                    xpred, xpred, self.theta, self.lengthscale)
                Kxpxp = self._noisify_kernel_matrix(Kxpxp, self.noise)
                pred_cov = Kxpxp - \
                    kxpred_.transpose(-2, -1)@kxpred_
                return pred_mean, pred_cov
            else:
                Kxpxp = self._make_kernel_matrix(xpred, xpred, self.theta, self.lengthscale,
                                                diagonal=True)
                Kxpxp += self.noise**2 + self.min_jitter
                pred_var = Kxpxp - (kxpred_.transpose(-2, -1)**2).sum(dim=-1)
                return pred_mean, pred_var

    def _rawfy(self, x):
        return utils.invsoftplus(x)  # invsoftmax

    def _derawfy(self, y):
        return utils.softplus(y)  # softmax

    def _make_kernel_matrix(self, X1, X2, theta, lengthscale,
                           diagonal=False):
        output = 'pairwise' if not diagonal else 'diagonal'
        K = kernel_functions.kernel_function(X1, X2,
                                             theta=theta,
                                             l=lengthscale,
                                             kind=self.kind,
                                             output=output)
        return K

    def _noisify_kernel_matrix(self, kernel_matrix, noise):
        K_ = utils.jittering(kernel_matrix, noise**2+self.min_jitter)
        return K_

    def _make_cholesky(self, K):
        U = torch.linalg.cholesky(K).transpose(-2, -1)  # Lower to upper
        return U

    def _loglikelihood_wrapper(self, params):
        theta = self._derawfy(params.get('raw_theta', self._raw_theta))
        noise = self._derawfy(params.get('raw_noise', self._raw_noise))
        mean = params.get('mean', self.mean)
        lengthscale = self._derawfy(params.get(
            'raw_lengthscale', self._raw_lengthscale))
        # Used for maximization in minimizer
        res = -self._loglikelihood(theta, lengthscale, noise, mean)
        return res

    def _loglikelihood(self, theta, lengthscale, noise, mean):
        kernel_matrix_ = self._make_kernel_matrix(
            self.X, self.X, theta, lengthscale)
        kernel_matrix = self._noisify_kernel_matrix(kernel_matrix_, noise)
        upper_chol_matrix = self._make_cholesky(kernel_matrix)
        y_, _ = torch.triangular_solve(self.y_-mean,
                                       upper_chol_matrix,
                                       upper=True,
                                       transpose=True)  # (m,1)
        term1 = -0.5*torch.sum(y_**2)
        term2 = -torch.sum(torch.log(torch.diagonal(upper_chol_matrix)))
        term3 = -0.5*self.ndata*math.log(2*math.pi)
        return term1 + term2 + term3

    def _set_lengthscale(self, lengthscale, ard):
        lengthscale = utils.tensor_convert(lengthscale)
        if lengthscale.ndim > 0:
            lengthscale = torch.squeeze(lengthscale)
            assert lengthscale.ndim == 1
            assert lengthscale.shape[0] == self.dim
            self.ard = True
        else:
            self.ard = ard
            if self.ard:
                lengthscale = torch.ones(self.dim)*lengthscale
        return lengthscale