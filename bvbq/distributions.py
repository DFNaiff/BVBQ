# -*- coding: utf-8 -*-
# pylint: disable=E1101
"""Collection of distribution classes"""

import abc
import math

import torch

from . import utils


class ProbabilityDistribution(abc.ABC):
    """
    Base class for probability distributions. See individual
    items for reference
    """
    @abc.abstractmethod
    def logprob(self, x):
        pass

    @abc.abstractmethod
    def sample(self, n):
        pass

    @property
    @abc.abstractmethod
    def params(self):
        pass

    def monte_carlo_entropy(self, n):
        """Calculate distribution entropy via Monte Carlo sampling"""
        return -torch.mean(self.logprob(self.sample(n)))


class DiagonalNormalDistribution(ProbabilityDistribution):
    """
    Diagonal covariance normal distribution class

    Attributes
    ----------
    dim : int
        Dimension of distribution space
    mean : torch.Tensor
        Mean vector of distribution
    var : torch.Tensor
        Diagonal of covariance matrix
    std : torch.Tensor
        Standard deviation of each dimension

    """

    def __init__(self, mean, var):
        """
        Parameters
        ----------
        mean : [float]
            Mean vector of distribution
        var : [float]
            Diagonal of covariance matrix

        """

        self.dim = mean.shape[0]
        mean, var = utils.tensor_convert_(mean, var)
        self.mean = mean
        self.var = var
        assert len(self.var) == self.dim
        assert len(self.mean) == self.dim
        self.std = torch.sqrt(self.var)

    def logprob(self, x):
        """
        Calculates log density

        Parameters
        ----------
        x : [float]
            Values to be calculated log density

        Returns
        -------
        torch.Tensor
            Values of log density

        """

        res = DiagonalNormalDistribution.logprob_(
            x, self.mean, self.var)
        return res

    @staticmethod
    def logprob_(x, mean, var):
        """
        Calculates log density of DiagonalNormalDistribution

        Parameters
        ----------
        x : [float]
            Values to be calculated log density
        mean : [float]
            Mean vector of distribution
        var : [float]
            Diagonal of covariance matrix

        Returns
        -------
        torch.Tensor
            Values of log density

        """

        x, mean, var = utils.tensor_convert_(x, mean, var)
        ndim = mean.shape[0]
        std = torch.sqrt(var)
        res = -0.5*torch.sum(((x-mean)/std)**2, dim=-1) \
              - torch.sum(torch.log(std)) - \
            ndim/2*math.log(2*math.pi)
        return res

    def sample(self, n):
        """
        Sample from distribution

        Parameters
        ----------
        n : int
            Number of samples

        Returns
        -------
        torch.Tensor
            Samples from distribution

        """

        res = DiagonalNormalDistribution.sample_(
            n, self.mean, self.var)
        return res

    @staticmethod
    def sample_(n, mean, var):
        """
        Sample from DiagonalNormalDistribution

        Parameters
        ----------
        n : int
            Number of samples
        mean : [float]
            Mean vector of distribution
        var : [float]
            Diagonal of covariance matrix

        Returns
        -------
        torch.Tensor
            Samples from distribution

        """

        mean, var = utils.tensor_convert_(mean, var)
        std = torch.sqrt(var)
        ndim = mean.shape[0]
        z = torch.randn((n, ndim))
        res = mean + std*z
        return res

    def make_mixture(self):
        """
        Turns distribution into a MixtureDiagonalNormalDistribution
        of one component

        Returns
        -------
        MixtureDiagonalNormalDistribution
            Mixture distribution with one component

        """

        mixmeans = torch.unsqueeze(self.mean, 0)
        mixvars = torch.unsqueeze(self.var, 0)
        weights = torch.ones((1,))
        return MixtureDiagonalNormalDistribution(mixmeans, mixvars, weights)

    @property
    def params(self):
        """{str:torch.Tensor} : Parameters of distribution"""
        p = {'mean': self.mean, 'var': self.var}
        return p

    def analytical_entropy(self):
        """Analytical entropy of distribution"""
        return 0.5*torch.sum(torch.log(2*math.pi*math.e*self.var))


class MixtureDiagonalNormalDistribution(ProbabilityDistribution):
    """
    Mixture of diagonal covariance normal distribution class

    Attributes
    ----------
    dim : int
        Dimension of distribution space
    nmixtures : int
        Number of mixture components
    mixmeans : torch.Tensor
        Mean matrix of mixtures
    mixvars : torch.Tensor
        Variance matrix of mixtures
    weights : torch.Tensor
        Weights vector of current mixture components
    std : torch.Tensor
        Standard deviation of mixtures

    """

    def __init__(self, mixmeans, mixvars, weights):
        """
        Parameters
        ----------
        mixmeans : [float]
            Mean matrix of mixtures
        mixvars : [float]
            Variance matrix of mixtures
        weights : [float]
            Weights vector of current mixture components

        """

        self.nmixtures = weights.shape[0]
        self.dim = mixmeans.shape[1]
        mixmeans, mixvars, weights = utils.tensor_convert_(
            mixmeans, mixvars, weights)
        self.mixmeans = mixmeans
        self.mixvars = mixvars
        self.weights = weights
        assert self.mixvars.shape[1] == self.dim
        assert self.mixmeans.shape[0] == self.nmixtures
        assert self.mixvars.shape[0] == self.nmixtures
        self.stds = torch.sqrt(mixvars)

    def logprob(self, x):
        """
        Calculates log density

        Parameters
        ----------
        x : [float]
            Values to be calculated log density

        Returns
        -------
        torch.Tensor
            Values of log density

        """

        res = MixtureDiagonalNormalDistribution.logprob_(
            x, self.mixmeans, self.mixvars, self.weights)
        return res

    @staticmethod
    def logprob_(x, mixmeans, mixvars, weights):
        """
        Calculates log density

        Parameters
        ----------
        x : [float]
            Values to be calculated log density
        mixmeans : [float]
            Mean matrix of mixtures
        mixvars : [float]
            Variance matrix of mixtures
        weights : [float]
            Weights vector of current mixture components

        Returns
        -------
        torch.Tensor
            Values of log density

        """

        x, mixmeans, mixvars, weights = utils.tensor_convert_(
            x, mixmeans, mixvars, weights)
        stds = torch.sqrt(mixvars)
        ndim = mixmeans.shape[1]
        x = torch.unsqueeze(x, -2)  # (n,1,d)
        yi1 = -0.5*torch.sum(((x-mixmeans)/stds)**2, dim=-1)  # (n,m)
        yi2 = -torch.sum(torch.log(stds), dim=-1)  # (m,)
        yi3 = -ndim/2*math.log(2*math.pi)  # (,)
        yi = yi1 + yi2 + yi3  # (n,m)
        res = torch.logsumexp(yi+torch.log(weights), dim=-1)
        return res

    def sample(self, n):
        """
        Sample from distribution

        Parameters
        ----------
        n : int
            Number of samples

        Returns
        -------
        torch.Tensor
            Samples from distribution

        """

        res = self.sample_(n, self.mixmeans, self.mixvars, self.weights)
        return res

    @staticmethod
    def sample_(n, mixmeans, mixvars, weights):
        """
        Sample from distribution

        Parameters
        ----------
        n : int
            Number of samples
        mixmeans : [float]
            Mean matrix of mixtures
        mixvars : [float]
            Variance matrix of mixtures
        weights : [float]
            Weights vector of current mixture components

        Returns
        -------
        torch.Tensor
            Samples from distribution

        """

        mixmeans, mixvars, weights = utils.tensor_convert_(
            mixmeans, mixvars, weights)
        stds = torch.sqrt(mixvars)
        _, ndim = mixmeans.shape
        catinds = torch.multinomial(weights, n, replacement=True)
        z = torch.randn((n, ndim))
        res = mixmeans[catinds, :] + stds[catinds, :]*z
        return res

    @property
    def params(self):
        """{str:torch.Tensor} : Parameters of distribution"""
        p = {'mixmeans': self.mixmeans,
             'mixvars': self.mixvars,
             'weights': self.weights}
        return p

    def add_component(self, mean, var, weight, return_new=False):
        """
        Add new component to mixture

        Parameters
        ----------
        mean : [float]
            Mean vector of new component
        var : [float]
            Variance vector of new component
        weight : [float]
            Weight of new component
        return_new : bool
            If True, returns a copy of the distribution,
            without modyfing original.
            If False, modifies original distribution

        Returns
        -------
        MixtureDiagonalNormalDistribution
            Mixture of diagonal normal distributions with new component

        """
        mean, var, weights = utils.tensor_convert_(mean, var, weight)
        weights = torch.hstack([(1-weight)*self.weights, weight])
        mixmeans = torch.vstack([self.mixmeans, mean])
        mixvars = torch.vstack([self.mixvars, var])
        if return_new:
            mixture = MixtureDiagonalNormalDistribution(
                mixmeans, mixvars, weights)
        else:
            self.mixmeans = mixmeans
            self.mixvars = mixvars
            self.weights = weights
            self.stds = torch.sqrt(mixvars)
            self.nmixtures += 1
            mixture = self
        return mixture
