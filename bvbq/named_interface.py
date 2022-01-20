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
    Main inference class for the BVBQ method using mixtures of 
    diagonal-covariance multivariate normal distribution.

    The method approximates a continuous probability distribution 
    available from expensive evaluations of a unnormalized log-probabilities 
    f_X(x), via approximating the f_{w(X)}(w(x)) via a mixture of
    diagonal-covariance multivariate normal distribution q_{w(X)}(w(x)), 
    and warping this mixture back to q_{X}(X), resulting in a approximate 
    distribution whose sampling and evaluation is done trivially. This is done 
    by using Gaussian Processes (GP) to approximate f_{w(X)}, and using generic 
    variational inference on this GP via Bayesian quadrature to calculate 
    surrogate integrals. The variatonal inference itself is done by using 
    boosting to calculate a arbitrary number of mixtures.

    More informations on the method can be found in 
    REFERENCE

    Attributes
    ----------
    logprobgp: gp.SimpleGP or None
        If not None, the GP surrogate model
    mixmeans: torch.Tensor or None
        If not None, means of approximate distribution mixtures
    mixvars: torch.Tensor or None
        If not None, variances of approximate distribution mixtures
    mixweights: torch.Tensor or None
        If not None, weights of approximate distribution mixtures
    eval_params: dict[str:torch.Tensor] or None
        If not None, evaluations points of unnormalized log-density
    eval_values: torch.Tensor or None
        If not None, evaluations of unnormalized log-density

    """

    def __init__(self, params_name, params_dim, params_bound, params_scale=None):
        """
        Parameters
        ----------
        params_name : List[str]
            Name of parameters
        params_dim : List[int]
            Dimension of each parameter in params_name
        params_bound : List[(float,float)]
            Lower and upper bound for each parameter in params_name
        params_scale : None or dict[str,List[float]} dict
            If not None, the scale factor of each parameter

        """
        self.logprobgp = None
        self.mixmeans = None
        self.mixvars = None
        self.mixweights = None
        self.eval_params = None
        self.eval_values = None
        self._named_distribution = named_distributions.NamedDistribution(
            params_name,
            params_dim,
            params_bound,
            self.base_distribution,
            params_scale)

    def initialize_data(self, eval_params, eval_values, kind='smatern32',
                        noise=0.0, mean=-30.0, empirical_params=False):
        """
        Initialize data of BVBQ

        Parameters
        ----------
        eval_params: dict[str:List[float]]
            Evaluations points of unnormalized log-density
        eval_values: List[float]
            Evaluations of unnormalized log-density
        kind : str
            Kind of kernel used in GP
        noise : 0.0
            Noise of GP (if less than 1e-2, jittering in set on GP)
        mean : -30.0
            Base mean of GP (if less than 1e-2, jittering in set on GP)
        empirical_params : bool
            Whether kernel parameters are initialized empirically

        """
        # TODO : Assertions, customizations and new policies
        # Set GP
        # HACK : Has to define GP before setting data,
        #        due to need for possibly capping data
        logprobgp = gp.SimpleGP(self.total_dim, kind=kind,
                                noise=noise, zeromax=True)
        logprobgp.mean = mean
        logprobgp.fix_mean()
        logprobgp.fix_noise()
        self.logprobgp = logprobgp
        params_ = self._named_distribution.organize_params(eval_params)
        values_ = utils.tensor_convert(eval_values)
        xdata_gp, ydata_gp = self._warp_data(params_, values_)
        self.logprobgp.set_data(xdata_gp, ydata_gp,
                                empirical_params=empirical_params)
        self.eval_params = params_
        self.eval_values = values_

    def initialize_components(self, init_policy='manual', **kwargs):
        """
        Initialize components of variational distribution

        Parameters
        ----------
        init_policy : str
            If 'manual', set initial mean and var of first component
            If 'manual_mix', set initial mixture distribution
        'mean' : List[float]
            If init_policy == 'manual', initial mean of first component
        'var' : List[float]
            If init_policy == 'manual', initial var of first component
        'mixmeans': List[float]:
            If init_policy == 'manual_mix', means of initial mixture
        'mixvars': List[float]:
            If init_policy == 'manual_mix', variances of initial mixture
        'mixweights': List[float]:
            If init_policy == 'manual_mix', weights of initial mixture

        """
        # TODO : Assertions, customization and new policies
        assert init_policy in ['manual', 'manual_mix']
        if init_policy == 'manual':
            mean = kwargs.get('mean', torch.zeros(self.total_dim))
            var = kwargs.get('var', 20*torch.ones(self.total_dim))
            mixmeans = torch.atleast_2d(utils.tensor_convert(mean))
            mixvars = torch.atleast_2d(utils.tensor_convert(var))
            mixweights = torch.ones(1)
        elif init_policy == 'manual_mix':
            mixmeans = kwargs.get('mixmeans')
            mixvars = kwargs.get('mixvars')
            mixweights = kwargs.get('mixweights')
            mixmeans, mixvars, mixweights = \
                utils.tensor_convert_(mixmeans, mixvars, mixweights)
        self.mixmeans = mixmeans
        self.mixvars = mixvars
        self.mixweights = mixweights

    def update_distribution(self):
        """
        Update distributioh through gradient boosting

        """
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

    def new_evaluation_point(self, name='WE', unwarped=False, numpy=True,
                             take_distance=False, nsamples=100, lfactor=0.2):
        """
        Propose new evaluation point for unnormalized log-density

        Parameters
        ----------
        name : str
            The name of the acquisition function to be used.
            'PP' - Prospective prediction
            'MMLT' - Moment matched log transform
            'PMMLT' - Prospective moment matched log transform
            'WE' - Warped entropy
        unwarped : bool
            If True, ajust mean and logprob of acquisition function 
            to correspond to unwarped density
        numpy : bool
            If True, return parameter values as numpy.array
            If False, return parameter values as torch.Tensor
        take_distance : bool
            If True, after getting acquisition function, wiggles it 
            to get some distance from other points, 
            in order to avoid instability in GP
        nsamples : int
            If take_distance is True, number of samples for wiggling
        lfactor : float
            If take_distance is True, factor to multiply GP lengthscale 
            in wiggling ball

        Returns
        -------
        params : dict[str,torch.Tensor] or dict[str,numpy.array]
            Proposed evaluation point

        """
        if self.nmixtures == 0 and name in ['PP', 'MMLT']:
            raise ValueError("%s can't be used with no distribution"%name)
        x0 = self.base_distribution.sample(1)[0, :]
        x, y = acquisition.acquire_next_point_mixmvn(x0,
                                                  self.logprobgp,
                                                  self.distribution,
                                                  name=name,
                                                  unwarped=unwarped)
        if take_distance:
            x = acquisition.wiggles_acquisition_point(x,
                                                      self.logprobgp,
                                                      nsamples,
                                                      lfactor)
        params = self._named_distribution.split_and_unwarp_parameters(x)
        if numpy:
            params = {k: v.detach().numpy() for k, v in params.items()}
            y = y.detach().numpy().item()
        return params, y

    def insert_new_evaluations(self, new_eval_params, new_eval_values,
                               empirical_params=True):
        """
        Insert new evaluations on GP

        Parameters
        ----------
        eval_params: dict[str:List[float]]
            Evaluations points of unnormalized log-density
        eval_values: List[float]
            Evaluations of unnormalized log-density
        empirical_params : bool
            Whether kernel parameters are initialized empirically

        """
        params_ = self._named_distribution.organize_params(new_eval_params)
        values_ = utils.tensor_convert(new_eval_values)
        x, y = self._warp_data(params_, values_)
        # FIXME: Fix this function
#        self.logprobgp.update(x,y)
        # FIXME : Substitute below lines for actual (fixed) efficient update above
        X = torch.vstack([self.warped_eval_params, x])
        y = torch.vstack([self.warped_eval_values, y])
        self.logprobgp.set_data(X, y)
        self.eval_params = utils.vstack_params(self.eval_params, params_)
        #self.eval_values = torch

    def fit_all_variational_parameters(self):
        """Fit all variational mixture parameters through ELBO maximization"""
        #TODO : Customization
        mixmeans, mixvars, mixweights = bvbq.fit_mixmvn_elbo(
            self.logprobgp, self.mixmeans, self.mixvars, self.mixweights)
        self.mixmeans = mixmeans
        self.mixvars = mixvars
        self.mixweights = mixweights

    def fit_all_variational_weights(self):
        """Fit all variational weights parameters through ELBO maximization"""
        #TODO : Customization
        mixmeans, mixvars, mixweights = bvbq.reweight_mixmvn_elbo(
            self.logprobgp, self.mixmeans, self.mixvars, self.mixweights)
        self.mixmeans = mixmeans
        self.mixvars = mixvars
        self.mixweights = mixweights

    def cut_components(self, cutoff_limit=1e-6):
        """
        Remove components of mixture whose weights 
        are below some cutoff point
    
        Parameters
        ----------
        cutoff_limit : float
            Cutoff point for weights
        """
        mixmeans,mixvars,mixweights = \
            utils.cut_components_mixmvn(self.mixmeans,
                                        self.mixvars,
                                        self.mixweights,
                                        cutoff_limit=cutoff_limit)
        self.mixmeans = mixmeans
        self.mixvars = mixvars
        self.mixweights = mixweights
        
    def elbo_metric(self, nsamples=1000):
        """Get mean and var of current ELBO on GP"""
        return metrics.bq_mixmvn_elbo_with_var(self.logprobgp,
                                               self.mixmeans,
                                               self.mixvars,
                                               self.mixweights,
                                               nsamples=nsamples)

    def optimize_gp_params(self, *args, **kwargs):
        """
        Optimize parameters of GP

        Parameters
        ----------
        baseopt : str
            If 'QN', uses quasi-newton method for optimization (dict-minimize)
            If 'SGD', uses gradient descent for optimization (torch.optim)
        fixed_params : List[str]
            Parameters to be fixed in this optimization
        method : str
            If baseopt == 'QN', optimization method for dict-minimize
        tol : float
            If baseopt == 'QN', optimization tolerance for dict-minimize
        options : dict or None
            If baseopt == 'QN', options for dict-minimize
        maxiter : int
            If baseopt == 'SGD', number of steps for SGD
        optim : torch.optim.Optimizer
            If baseopt == 'SGD', optimizer to be used
        lr : float
            If baseopt == 'SGD', learning rate of optimizer
        verbose : bool
            If baseopt == 'SGD', whether to show maxiter steps

        Returns
        -------
        dict[str,torch.Tensor]
            A copy of optimal values found

        """
        baseopt = kwargs.get('baseopt', 'QN')
        kwargs.pop('baseopt', None)
        assert baseopt in ['QN', 'SGD']
        if baseopt == 'QN':
            return self.logprobgp.optimize_params_qn(*args, **kwargs)
        elif baseopt == 'SGD':
            return self.logprobgp.optimize_params_sgd(*args, **kwargs)

    def suggest_initialization_points(self, n):
        # TODO : Algorithms for suggesting initialization points
        raise NotImplementedError
        # return xdata

    def surrogate_prediction(self, params):
        """
        Mean of GP surrogate

        Parameters
        ----------
        params: dict[str:List[float]]
            Prediction points

        Returns
        -------
        torch.Tensor
            Mean of GP surrogate

        """

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
        """
        distributions.BaseDistribution : 
        base distribution of variational distribution
        """
        if self.mixmeans is None:
            return None
        else:
            return distributions.MixtureDiagonalNormalDistribution(
                self.mixmeans, self.mixvars, self.mixweights)

    @property
    def distribution(self):
        """
        named_distributions.NamedDistribution: 
        Variational distribution
        """

        return self._named_distribution.set_basedistrib(self.base_distribution)

    @property
    def warped_eval_params(self):
        """torch.Tensor : warped evaluation parameters on surrogate GP"""
        return self.logprobgp.X

    @property
    def warped_eval_values(self):
        """torch.Tensor : warped evaluation values on surrogate GP"""
        return self.logprobgp.y

    @property
    def eval_params_numpy(self):
        """dict[str:numpy.array] : evaluation parameters as numpy array"""
        return {k: v.numpy() for k, v in self.eval_params.items()}

    @property
    def eval_values_numpy(self):
        """numpy.array : evaluation values as numpy array"""
        return self.eval_values.numpy()

    @property
    def dim(self):
        """Alias for self.distribution.dim"""
        return self._named_distribution.dim

    @property
    def bound(self):
        """Alias for self.distribution.bound"""
        return self._named_distribution.bound

    @property
    def scale(self):
        """Alias for self.distribution.scale"""
        return self._named_distribution.scale

    @property
    def names(self):
        """Alias for self.distribution.names"""
        return self._named_distribution.names

    @property
    def dims(self):
        """Alias for self.distribution.dims"""
        return self._named_distribution.dims

    @property
    def bounds(self):
        """Alias for self.distribution.bounds"""
        return self._named_distribution.bounds

    @property
    def scales(self):
        """Alias for self.distribution.scales"""
        return self._named_distribution.scales

    @property
    def total_dim(self):
        """Alias for self.distribution.total_dim"""
        return self._named_distribution.total_dim

    @property
    def nmixtures(self):
        """ int: number of mixtures of approximate distribution mixtures"""
        return len(self.mixweights)

    @property
    def gpmean(self):
        """Base mean of GP"""
        return self.logprobgp.mean

    @property
    def initialized(self):
        """bool : Whether GP is initialized (hence model can be run)"""
        return self.logprobgp is not None

    def _warp_data(self, params, evals, clamp_evals=True):
        xdata = self._named_distribution.join_and_warp_parameters(params)
        # Minus sign in correction
        corrections = [-self._named_distribution.logdwarpf(key)(value)
                       for key, value in params.items()]
        correction = torch.sum(torch.cat(corrections, dim=-1), dim=-1)
        correction = correction.reshape(shape=evals.shape)
        ydata = evals - correction
        if clamp_evals:
            #ydata = torch.clamp(ydata, min=self.gpmean)
            ydata = utils.logbound(ydata, self.gpmean)
        return xdata, ydata
