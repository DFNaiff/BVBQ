# -*- coding: utf-8 -*-
import jax.experimental.optimizers


class JaxOptWrapper(object):
    def __init__(self,optimizer,*args,**kwargs):
        if isinstance(optimizer,str):
            optimizer_ = get_optimizer(optimizer)
        else:
            assert callable(optimizer)
            optimizer_ = optimizer
        self.opt_init,self.opt_update,self.get_params = \
            optimizer_(*args,**kwargs)
        self.opt_state = None
        
    def init(self,params,func_and_grad):
        self.i = 0
        self.opt_state = self.opt_init(params)
        self.func_and_grad = func_and_grad
        
    def step(self,*fargs,**fkwargs):
        value, grads = self.func_and_grad(self.get_params(self.opt_state),*fargs,**fkwargs)
        self.opt_state = self.opt_update(self.i,grads,self.opt_state)
        self.i += 1
        return value,grads
    
    @property
    def params(self):
        return self.get_params(self.opt_state)


def get_optimizer(optimizer):
    if optimizer == "adagrad":
        optimizer_ = jax.experimental.optimizers.adagrad
    elif optimizer == "adamax":
        optimizer_ = jax.experimental.optimizers.adamax
    elif optimizer == "adam":
        optimizer_ = jax.experimental.optimizers.adam
    elif optimizer == "momentum":
        optimizer_ = jax.experimental.optimizers.momentum
    elif optimizer == "nesterov":
        optimizer_ = jax.experimental.optimizers.nesterov
    elif optimizer == "rmsprop":
        optimizer_ = jax.experimental.optimizers.rmsprop
    elif optimizer == "rmsprop_momentum":
        optimizer_ = jax.experimental.optimizers.rmsprop_momentum
    elif optimizer == "sgd":
        optimizer_ = jax.experimental.optimizers.sgd
    elif optimizer == "sm3":
        optimizer_ = jax.experimental.optimizers.sm3
    else:
        raise ValueError("Optimizer name not recognized")
    return optimizer_