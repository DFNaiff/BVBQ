# -*- coding: utf-8 -*-
"""
BVBQ (Boosted Variational inference via Bayesian Quadrature) if a package
for generic variational inference in hard-to-evaluate
unnormalized posterior distributions, using a mixture of boosted variational
inference and Bayesian quadrature.

Currently BVBQ supports inference in multiple parameters
supported on the real line, positive real line, or interval domains.
This is done via warping to the corresponding Euclidean space,
and doing variational inference via mixtures of diagonal normal distributions.
Future developments may include non-product domains such as spheres.

Further reference on the method can be found in REFERENCE

Example
--------
>>> from bvbq import BVBQMixMVN #Main inference object
>>>
>>> #Defines log of density to be approximated
>>> def logprob(params):
>>>     #joint density of sigma ~ Gamma(5,1), mu ~ Normal(0,sigma**2)
>>>     mu,sigma = params['mu'],params['sigma']
>>>     term1 = np.log(scipy.stats.gamma(5.0).pdf(sigma))
>>>     term2 =  -0.5*(x/sigma)**2 - np.log(np.sqrt(2*np.pi)*sigma)
>>>     res = term1 + term2 + 3.0 #No need to be normalized
>>>     return res
>>>
>>> #Get some initial data points
>>> ndata = 20
>>> sigma = scipy.stats.gamma(5).rvs(size=(ndata, 1))
>>> mu = np.random.randn(ndata, 1)*sigma
>>> params_init = {'mu':mu, 'sigma':sigma}
>>> params_evals = baselogprob(params_init)
>>>
>>> #Defines inferencer. 'mu' and 'sigma' have one dimension each,
>>> #and 'mu' and 'sigma' are defined on the real line
>>> #and positive real line, respectively
>>> bvbq = BVBQMixMVN(['mu', 'sigma'], #Names
>>>                   [1, 1], #Dimensions
>>>                   [(None, None), (0.0, None)]) #Bounds
>>> #Initialize inferencer
>>> bvbq.initialize_data(params_init,params_evals,empirical_params=True)
>>> bvbq.initialize_components(mean=torch.zeros(2),var=20*torch.ones(2))
>>>
>>> #Make 30 boosts
>>> for i in range(30):
>>>     bvbq.update_distribution()
>>>     if i%5 == 0: #With some regularity, evaluate again
>>>         params = bvbq.new_evaluation_point() #Choose evaluation point
>>>         evals = baselogprob(params) #Evaluation is on the user
>>>         bvbq.insert_new_evaluations(params,evals) #Insert evaluation
>>>
>>> #Get distribution
>>> distrib = bvbq.distribution
>>>
>>> #Sample from distribution
>>> distrib.sample(1, numpy=True)
{'mu': array([[0.06106102]], dtype=float32),
 'sigma': array([[5.395544]], dtype=float32)}
>>> #Calculate log-density
>>> distrib.logprob({'mu': [1.0],'sigma': [1.0]}, numpy=True)
array(-8.361809, dtype=float32)

"""

from . import kernel_functions, distributions, bayesquad, utils,\
    bvbq_functions, bvbq, acquisition_functions, metrics

from .named_interface import BVBQMixMVN
from .gp import SimpleGP
from .named_distributions import NamedDistribution

__all__ = ['BVBQMixMVN','NamedDistribution','SimpleGP']
#__all__ = ['BVBQMixMVN', 'gp', 'kernel_functions',
#           'distributions', 'bayesquad', 'bvbq_functions',
#           'bvbq', 'acquisition_functions', 'metrics',
#           'utils']
