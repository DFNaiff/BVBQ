# -*- coding: utf-8 -*-
"""
    This will be the main documentation
"""

__all__ = ['BVBQMixMVN', 'gp', 'kernel_functions',
           'distributions', 'bayesquad', 'bvbq_functions',
           'bvbq', 'acquisition_functions', 'metrics',
           'utils']

from . import gp, kernel_functions, distributions, bayesquad, utils,\
    bvbq_functions, bvbq, acquisition_functions, metrics
from .named_interface import BVBQMixMVN
