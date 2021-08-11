# -*- coding: utf-8 -*-

import torch


def prospective_prediction(x,gp,distrib):
    mean,var = gp.predict(x,return_cov=True,onlyvar=True)
    logprob = distrib.logprob(x)
    res = torch.exp(mean+2*logprob)*var
    return res


def moment_matched_log_transform(x,gp,distrib):
    mean,var = gp.predict(x,return_cov=True,onlyvar=True)
    res = torch.exp(2*mean + var)*(torch.exp(var)-1)
    return res


def prospective_mmlt(x,gp,distrib):
    mean,var = gp.predict(x,return_cov=True,onlyvar=True)
    logprob = distrib.logprob(x)
    res = torch.exp(2*mean+2*logprob+var)*(torch.exp(var)-1)
    return res