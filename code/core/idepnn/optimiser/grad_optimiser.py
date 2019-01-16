
'''

Created on 25.07.2015

_author_ = Pankaj Gupta
_credits_ =
_version_ = 1.0
_maintainer_ =
_email_ = pankaj.gupta@siemens.com, pankaj_gupta96@yahoo.com, pankaj.gupta@tum.de
_status_ = PROGRESS

  DATE            STATUS                        DESCRIPTION
25.07.2015    Initial File Creation       Initial File Creation
25.07.2015      Addition                  Grad optimization updates: 'sgd', 'adagrad', 'adadelta'


_filename_ = grad_optimiser.py

Description:
    Gradient optimizers: 'sgd', 'adagrad' and 'adadelta'

'''


from collections import OrderedDict
import theano
from theano import tensor as T
import numpy as np

def create_optimization_updates(cost, params, names, lr, rho, updates=None, max_norm=5.0,
                                eps=1e-6,
                                method = 'adadelta', gradients = None,
                                mom = 1.0):
    """
    Get the updates for a gradient descent optimizer using
    SGD, AdaDelta, or AdaGrad.
    Returns the shared variables for the gradient caches,
    and the updates dictionary for compilation by a
    theano function.
    Inputs
    ------
    cost     theano variable : what to minimize
    params   list            : list of theano variables
                               with respect to which
                               the gradient is taken.
    max_norm float           : cap on excess gradients
    lr       float           : base learning rate for
                               adagrad and SGD
    eps      float           : numerical stability value
                               to not divide by zero
                               sometimes
    rho      float           : adadelta hyperparameter.
    method   str             : 'adagrad', 'adadelta', or 'sgd'.
    Outputs:
    --------
    updates  OrderedDict   : the updates to pass to a
                             theano function
    gsums    list          : gradient caches for Adagrad
                             and Adadelta
    xsums    list          : gradient caches for AdaDelta only
    lr       theano shared : learning rate
    max_norm theano_shared : normalizing clipping value for
                             excessive gradients (exploding).
    mom      theano shared : mometum for sgd
    """
    #lr = theano.shared(np.float64(lr).astype(theano.config.floatX))
    eps = np.float64(eps).astype(theano.config.floatX)
    #rho = theano.shared(np.float64(rho).astype(theano.config.floatX))
    #mom = theano.shared(np.float64(mom).astype(theano.config.floatX))

    if max_norm is not None and max_norm is not False:
        max_norm = theano.shared(np.float64(max_norm).astype(theano.config.floatX))

    gsums   = [theano.shared(np.zeros_like(param.get_value(borrow=True))) if (method == 'adadelta' or method == 'adagrad') else None for param in params]
    xsums   = [theano.shared(np.zeros_like(param.get_value(borrow=True))) if method == 'adadelta' else None for param in params]

    gparams = theano.grad(cost, params) if gradients is None else gradients

    if updates is None:
        updates = OrderedDict()

    for name, gparam, param, gsum, xsum in zip(names, gparams, params, gsums, xsums):
        # clip gradients if they get too big
        if max_norm is not None and max_norm is not False:
            grad_norm = gparam.norm(L=2)
            gparam = (T.minimum(max_norm, grad_norm)/ (grad_norm + eps)) * gparam

        if method == 'adadelta':
            updates[gsum] = T.cast(rho * gsum + (1. - rho) * (gparam **2), theano.config.floatX)
            dparam = -T.sqrt((xsum + eps) / (updates[gsum] + eps)) * gparam
            updates[xsum] = T.cast(rho * xsum + (1. - rho) * (dparam **2), theano.config.floatX)
            updates[param] = T.cast(param + dparam, theano.config.floatX)
        elif method == 'adagrad':
            updates[gsum] =  T.cast(gsum + (gparam ** 2), theano.config.floatX)
            updates[param] =  T.cast(param - lr * (gparam / (T.sqrt(updates[gsum] + eps))), theano.config.floatX)
        elif method == 'sgd':
            if name == 'embeddings':
               updates[param] = mom * param - gparam * 0.25 * lr
            else:
               updates[param] = mom * param - gparam * lr
        else:
            raise('invalid optimiser')

    if method == 'adadelta':
        lr = rho

    return updates, gsums, xsums, lr, max_norm
