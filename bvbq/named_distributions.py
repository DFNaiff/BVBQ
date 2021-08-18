# -*- coding: utf-8 -*-
# pylint: disable=E1101
"""
Wrapper and warper for distribution in
arbitrary (connected) product subsets of R^n
"""

import collections
import math

import torch

from . import utils


class NamedDistribution(object):
    """
    A distribution wrapper for distributions of named random 
    variables in product subsets of R^n, with each variable 
    being either defined on the real line, bounded from below, 
    or bounded by an interval.
    
    The random variables X1,...,Xn are defined such as, for each 
    warping function w1,...,wn from the domain of X to R^n, we have 
    that w1(X1),...,wn(X1) ~ basedistrib, where basedistrib in 
    a distributions.ProbabilityDistribution defined on R^n.

    Attributes
    ---------
    basedistrib : distributions.ProbabilityDistribution
        Base distribution in R^n
    params_dict : dict
        Dictionary with keys containing parameter names,
        and values being a dict consisting of
        
        dim : int
            Dimension of parameter    
        bound : (float,float)
            Bounds of parameter
        scale : torch.Tensor
            Scaling factor of parameter
        warpf : Callable[torch.Tensor,torch.Tensor]
            Warping function from bounds to R
        iwarpf : Callable[torch.Tensor,torch.Tensor]
            Inverse warping function from bounds to R
        logdwarpf : Callable[torch.Tensor,torch.Tensor]
            Log of derivative of warping

    """
    def __init__(self,
                 params_name,
                 params_dim,
                 params_bound,
                 basedistrib=None,
                 params_scale=None):
        """
        Parameters
        ----------
        params_name : List[str]
            Name of parameters
        params_dim : List[int]
            Dimension of each parameter in params_name
        params_bound : List[(float,float)]
            Lower and upper bound for each parameter in params_name
        basedistrib : None or distributions.ProbabilityDistribution
            If None, basedistribution is to be set later
            else, the base ProbabilityDistribution in R^n
        params_scale : None or dict[str,List[float]} dict
            If not None, the scale factor of each parameter

        """
        self._set_param_distrib(params_name, params_dim,
                                params_bound, params_scale)
        self.set_basedistrib(basedistrib)

    def logprob(self, params, numpy=False):
        """
        Log probability of unwarped distribution
        logdensity_{X}(x) = logdensity_{warp(X)}(warp(X)) + sum_i logdwarp_i(X_i)

        Parameters
        ----------
        params : dict[str,List[float]]
            The parameter values to be calculated log density
        numpy : bool
            If False, return torch.Tensor
            If True, return np.array

        Returns
        -------
        torch.Tensor or np.array
            Values of log density

        """
        # logdensity_{X}(x) = logdensity_{warp(X)}(warp(X)) + logdwarp(X)
        # Ordering
        assert self.basedistrib is not None
        params_ = self.organize_params(params)
        joint_and_warped_x = self.join_and_warp_parameters(params_)
        uncorrected_res = self.basedistrib.logprob(joint_and_warped_x)
        corrections = [self.logdwarpf(key)(value)
                       for key, value in params_.items()]
        correction = torch.sum(torch.cat(corrections, dim=-1), dim=-1)
        res = uncorrected_res + correction
        if numpy:
            res = res.detach().numpy()
        return res

    def sample(self, n, numpy=False):
        """
        Sample from unwarped distribution

        Parameters
        ----------
        n : int
            Number of samples
        numpy : bool
            If False, return torch.Tensor as values
            If True, return np.array as values
        Returns
        -------
        params : dict[str,torch.Tensor] or dict[str,numpy.array]
            Samples from distribution

        """

        assert self.basedistrib is not None
        warped_samples = self.basedistrib.sample(n)
        samples = self.split_and_unwarp_parameters(warped_samples)
        if numpy:
            samples = {k:v.detach().numpy() for k,v in samples.items()}
        return samples

    def join_parameters(self, params):
        """
        Join the parameters in a unique tensor

        Parameters
        ----------
        params : dict[str,torch.Tensor]
            The parameter values to be joined

        Returns
        -------
        torch.Tensor
            The joint parameter matrix
        """

        res = torch.cat([params[name] for name in self.names], dim=-1)
        return res

    def join_and_warp_parameters(self, params):
        """
        Join the parameters in a unique tensor, and 
        warp then to R^n

        Parameters
        ----------
        params : dict[str,torch.Tensor]
            The parameter values to be joined and warped

        Returns
        -------
        torch.Tensor
            The joint and warped parameter matrix
        """

        tocat = [self.warpf(name)(params[name]) for name in self.names]
        res = torch.cat(tocat, dim=-1)
        return res

    def split_parameters(self, x):
        """
        Split the tensor x into each corresponding parameter

        Parameters
        ----------
        torch.Tensor
            The joint parameter tensor

        Returns
        -------
        params : dict[str,torch.Tensor]
            The splitted parameter values

        """

        return torch.split(x, [self.dim(name) for name in self.names], dim=-1)

    def split_and_unwarp_parameters(self, x):
        """
        Split the tensor x into each corresponding parameter, 
        and warp then from R^n to the original domain

        Parameters
        ----------
        torch.Tensor
            The joint and warped parameter tensor

        Returns
        -------
        params : dict[str,torch.Tensor]
            The splitted and unwarped parameter values

        """
        splits = torch.split(x, [self.dim(name)
                                 for name in self.names], dim=-1)
        res = dict([(name, self.iwarpf(name)(splits[i]))
                    for i, name in enumerate(self.names)])
        return res
        
    def set_basedistrib(self, basedistrib):
        """
        Set base distribution

        Parameters
        ----------
        basedistrib : None or distributions.ProbabilityDistribution
            If None, basedistribution is to be set later
            else, the base ProbabilityDistribution in R^n

        """
        if basedistrib is not None:
            assert basedistrib.dim == self.total_dim
        self.basedistrib = basedistrib
        return self

    def dim(self, key):
        """int : returns the dimension of 'key' parameter"""
        return self.param_dict[key]['dim']

    def bound(self, key):
        """tuple[float] : Returns the bound of 'key' parameter"""
        return self.param_dict[key]['bound']

    def scale(self, key):
        """tuple[float] : Returns the scale of 'key' parameter"""
        return self.param_dict[key]['scale']

    def warpf(self, key):
        """
        Callable[torch.Tensor,torch.Tensor]: 
        Returns the warping function of 'key' parameter
        """
        return self.param_dict[key]['warpf']

    def iwarpf(self, key):
        """
        Callable[torch.Tensor,torch.Tensor]
        Returns the inverse warping function of 'key' parameter
        """
        return self.param_dict[key]['iwarpf']

    def logdwarpf(self, key):
        """
        Callable[torch.Tensor,torch.Tensor]
        Returns the log of derivative of warping of 'key' parameter
        """
        return self.param_dict[key]['logdwarpf']

    def logdiwarpf(self, key):
        """
        Callable[torch.Tensor,torch.Tensor]
        Returns the log of derivative of inverse warping of 'key' parameter
        """
        #D(w^{-1})(y) = Dw(w^{-1}(y))
        def f(x):
            return self.logdwarpf(key)(self.iwarpf(key)(x))
        return f

    @property
    def names(self):
        """list[str] : names of parameters"""
        return list(self.param_dict.keys())

    @property
    def dims(self):
        """dict[str,int] : dimension dictionary of parameters"""
        return utils.get_subdict(self.param_dict, 'dim')

    @property
    def bounds(self):
        """dict[str:(float,float)] : sounds dictionary of parameters"""
        return utils.get_subdict(self.param_dict, 'bound')

    @property
    def scales(self):
        """dict[str:List[float]] : scales dictionary of parameters"""
        return utils.get_subdict(self.param_dict, 'scacle')

    @property
    def total_dim(self):
        """int : total dimension of underlying domain"""
        return sum(self.dims.values())

    def organize_params(self, params):
        """Convert to torch.Tensor and organize in the order of param_dict"""
        params_ = collections.OrderedDict([(key, utils.tensor_convert(params[key]))
                                           for key in self.names])
        return params_

    def _set_param_distrib(self, params_name, params_dim, params_bound, params_scale):
        if params_scale is None:
            params_scale = dict()
        param_dict = collections.OrderedDict()
        for _, (name, dim, bound) in enumerate(zip(params_name, params_dim, params_bound)):
            lb, ub = bound
            scale = utils.tensor_convert(params_scale.get(name, 1.0))
            warpf, iwarpf, logdwarpf = get_warps(lb, ub, scale)
            param_dict[name] = {'dim': dim, 'bound': bound, 'scale': scale,
                                'warpf': warpf, 'iwarpf': iwarpf, 'logdwarpf': logdwarpf}
        self.param_dict = param_dict


def get_warps(lb, ub, scale=1.0):
    """
    Get warp functions associated with domain (lb,ub), and scaled

    Parameters
    ----------
    lb : float or None
        Lower bound of domain. If None, considered to be unbounded
    ub : float or None
        Upper bound of domain. If None, considered to be unbounded
    scale : float
        Scale factor of random variable. Not used if ub and lb are not None

    Returns
    -------
    (torch.Tensor -> torch.Tensor),
    (torch.Tensor -> torch.Tensor),
    (torch.Tensor -> torch.Tensor)
        Function from domain to R, from R to domain,
        and log of derivative of function from (0,inf) to R
    """
    # warpf : bounds -> R
    # iwarpf : R -> bounds
    # logdwarpf = log|warpf'\ : bounds -> R^+
    if lb is None and ub is None:
        warpf = lambda x: x/scale
        iwarpf = lambda x: scale*x
        logdwarpf = lambda x: torch.zeros_like(x) - torch.log(scale)
    elif lb is not None and ub is None: #(lb,inf) -> (0,inf)
        bwf, biwf, bldwf = base_positive_warps()
        warpf = lambda x: bwf((x-lb)/scale)
        iwarpf = lambda x: scale*biwf(x) + lb
        logdwarpf = lambda x: bldwf((x-lb)/scale) - torch.log(scale)
    elif lb is None and ub is not None:
        raise NotImplementedError
    elif lb is not None and ub is not None: #(lb,ub) -> (-1,1)
        bwf, biwf, bldwf = base_bounded_warps()
        assert ub > lb
        a = (ub - lb)/2
        b = (ub + lb)/2
        warpf = lambda x: bwf((x-b)/a)
        iwarpf = lambda x: a*biwf(x) + b
        logdwarpf = lambda x: bldwf((x-b)/a) - \
                              torch.log(torch.tensor(1.0)*a)
    return warpf, iwarpf, logdwarpf


def base_positive_warps():
    """
    Get warp functions associated with domain (0,inf), scale 1.0
    Warp function is defined as f(x) = log(exp(x)-1)
    
    Returns
    -------
    Callable[torch.Tensor,torch.Tensor],
    Callable[torch.Tensor,torch.Tensor],
    Callable[torch.Tensor,torch.Tensor]
        Function from (0,inf) to R, from R to (0,inf),
        and log of derivative of function from (0,inf) to R
    """
    warpf = utils.invsoftplus
    iwarpf = utils.softplus
    logdwarpf = lambda x: x - utils.invsoftplus(x)
    return warpf, iwarpf, logdwarpf


#def base_bounded_warps():
#    """
#    Get warp functions associated with domain (-1,1), scale 1.0.
#    Warp function is defined as f(x) = 2/pi*tan(pi/2*x)
#
#    Returns
#    -------
#    Callable[torch.Tensor,torch.Tensor],
#    Callable[torch.Tensor,torch.Tensor],
#    Callable[torch.Tensor,torch.Tensor]
#        Function from (-1,1) to R, from R to (-1,1),
#        and log of derivative of function from (-1,1) to R
#    """
#    # [-1,1] -> R
#    warpf = lambda x: 2/math.pi*torch.tan(math.pi/2*x)
#    iwarpf = lambda x :2/math.pi*torch.atan(math.pi/2*x)
#    logdwarpf = lambda x: -2*torch.log(torch.cos(math.pi/2*x))
#    return warpf, iwarpf, logdwarpf


def base_bounded_warps():
    """
    Get warp functions associated with domain (-1,1), scale 1.0.
    Warp function is defined as f(x) = atanh(x)

    Returns
    -------
    Callable[torch.Tensor,torch.Tensor],
    Callable[torch.Tensor,torch.Tensor],
    Callable[torch.Tensor,torch.Tensor]
        Function from (-1,1) to R, from R to (-1,1),
        and log of derivative of function from (-1,1) to R
    """
    # [-1,1] -> R
    warpf = torch.atanh
    iwarpf = torch.tanh
    logdwarpf = lambda x: -torch.log(1.0-x**2)
    return warpf, iwarpf, logdwarpf
