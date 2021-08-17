# -*- coding: utf-8 -*-
# pylint: disable=E1101
"""
    BVBQ main interface
"""
import torch

from . import utils
from . import bvbq
from . import distributions
from . import gp
from . import acquisition
from . import metrics
from . import named_distributions


class BVBQMixMVN(object):
    """
    Test docstring
    """
    def __init__(self, params_name, params_dim, params_bound, params_scale=None):
        self.logprobgp = None
        self.mixmeans = None
        self.mixvars = None
        self.mixweights = None
        self.eval_values = None
        self.eval_params = None
        self._named_distribution = named_distributions.NamedDistribution(
            params_name,
            params_dim,
            params_bound,
            self.base_distribution,
            params_scale)
        self.nmixtures = 0

    def initialize_data(self, eval_params, eval_values, kind='smatern32',
                        noise=0.0, mean=-30.0, empirical_params=False,
                        **kwargs):
        # TODO : Assertions, customizations and new policies
        params_ = self._named_distribution.organize_params(eval_params)
        values_ = utils.tensor_convert(eval_values)
        xdata_gp, ydata_gp = self.warp_data(params_, values_)
        logprobgp = gp.SimpleGP(self.total_dim, kind=kind,
                                noise=noise, zeromax=True)
        logprobgp.mean = mean
        logprobgp.fix_mean()
        logprobgp.fix_noise()
        logprobgp.set_data(xdata_gp, ydata_gp,
                           empirical_params=empirical_params)
        self.logprobgp = logprobgp
        self.eval_params = params_
        self.eval_values = values_

    def initialize_components(self, init_policy='manual', **kwargs):
        # TODO : Assertions, customization and new policies
        assert init_policy in ['manual', 'manual_mix']
        if init_policy == 'manual':
            mean = kwargs.get('mean', torch.zeros(self.total_dim))
            var = kwargs.get('var', 20*torch.ones(self.total_dim))
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
        mean, var = bvbq.propose_component_mvn_mixmvn_relbo(
            self.logprobgp,
            self.mixmeans,
            self.mixvars,
            self.mixweights)
        mixmeans, mixvars, mixweights = bvbq.update_distribution_mvn_mixmvn(
            self.logprobgp,
            mean, var,
            self.mixmeans,
            self.mixvars,
            self.mixweights)
        self.mixmeans = mixmeans
        self.mixvars = mixvars
        self.mixweights = mixweights

    def new_evaluation_point(self, name='PP', unwarped=False, numpy=True):
        x0 = self.base_distribution.sample(1)[0, :]
        x = acquisition.acquire_next_point_mixmvn(x0,
                                                  self.logprobgp,
                                                  self.distribution,
                                                  name=name,
                                                  unwarped=unwarped)
        params = self._named_distribution.split_and_unwarp_parameters(x)
        if numpy:
            params = {k:v.detach().numpy() for k,v in params.items()}
        return params

    def insert_new_evaluations(self, new_eval_params, new_eval_values,
                               empirical_params = True):
        params_ = self._named_distribution.organize_params(new_eval_params)
        values_ = utils.tensor_convert(new_eval_values)
        x, y = self.warp_data(params_, values_)
        # FIXME: Fix this function
#        self.logprobgp.update(x,y)
        # FIXME : Substitute below lines for actual (fixed) efficient update above
        X = torch.vstack([self.warped_eval_params, x])
        y = torch.vstack([self.warped_eval_values, y])
        self.logprobgp.set_data(X, y)
        self.eval_params = utils.vstack_params(self.eval_params, params_)
        #self.eval_values = torch

    def fit_all_parameters(self):
        #TODO : Customization
        mixmeans, mixvars, mixweights = bvbq.fit_mixmvn_elbo(
            self.logprobgp, self.mixmeans, self.mixvars, self.mixweights)
        self.mixmeans = mixmeans
        self.mixvars = mixvars
        self.mixweights = mixweights

    def fit_all_weights(self):
        #TODO : Customization
        mixmeans, mixvars, mixweights = bvbq.reweight_mixmvn_elbo(
            self.logprobgp, self.mixmeans, self.mixvars, self.mixweights)
        self.mixmeans = mixmeans
        self.mixvars = mixvars
        self.mixweights = mixweights

    def elbo_metric(self, nsamples=1000):
        return metrics.bq_mixmvn_elbo_with_var(self.logprobgp,
                                               self.mixmeans,
                                               self.mixvars,
                                               self.mixweights,
                                               nsamples=nsamples)

    def optimize_gp_params(self, *args, **kwargs):
        baseopt = kwargs.get('baseopt', 'QN')
        kwargs.pop('baseopt', None)
        assert baseopt in ['QN', 'SGD']
        if baseopt == 'QN':
            return self.optimize_gp_params_qn(*args, **kwargs)
        elif baseopt == 'SGD':
            return self.optimize_gp_params_sgd(*args, **kwargs)

    def suggest_initialization_points(self, n):
        # TODO : Algorithms for suggesting initialization points
        raise NotImplementedError
        # return xdata

    def warp_data(self, params, evals):
        xdata = self._named_distribution.join_and_warp_parameters(params)
        # Minus sign in correction
        corrections = [-self._named_distribution.logdwarpf(key)(value)
                       for key, value in params.items()]
        correction = torch.sum(torch.cat(corrections, dim=-1), dim=-1)
        correction = correction.reshape(*evals.shape)
        ydata = evals - correction
        return xdata, ydata

    def surrogate_prediction(self, params):
        params_ = self._named_distribution.organize_params(params)
        xpred = self._named_distribution.join_and_warp_parameters(params_)
        ypred = self.logprobgp.predict(xpred, to_return='mean')
        corrections = [self._named_distribution.logdwarpf(key)(value)
                       for key, value in params_.items()]
        correction = torch.sum(torch.cat(corrections, dim=-1), dim=-1)
        corrected_pred = ypred + correction
        res = corrected_pred
        return res

    @property
    def base_distribution(self):
        if self.mixmeans is None:
            return None
        else:
            return distributions.MixtureDiagonalNormalDistribution(
                self.mixmeans, self.mixvars, self.mixweights)

    @property
    def distribution(self):
        return self._named_distribution.set_basedistrib(self.base_distribution)

    # XXX: This actually performs computation
    @property
    def optimize_gp_params_qn(self):
        return self.logprobgp.optimize_params_qn

    @property
    def optimize_gp_params_sgd(self):
        return self.logprobgp.optimize_params_sgd

    @property
    def warped_eval_params(self):
        return self.logprobgp.X

    @property
    def warped_eval_values(self):
        return self.logprobgp.y

    @property
    def dim(self):
        return self._named_distribution.dim

    @property
    def bound(self):
        return self._named_distribution.bound

    @property
    def scale(self):
        return self._named_distribution.scale

    @property
    def names(self):
        return self._named_distribution.names

    @property
    def dims(self):
        return self._named_distribution.dims

    @property
    def bounds(self):
        return self._named_distribution.bounds

    @property
    def scales(self):
        return self._named_distribution.scales

    @property
    def total_dim(self):
        return self._named_distribution.total_dim
