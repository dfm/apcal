#!/usr/bin/env python
# encoding: utf-8
"""
Unit conversions for mags/fluxes/probabilities

"""

__all__ = ['nmgy2mag', 'mag2nmgy', 'odds2prob', 'prob2odds',
                'logodds2prob', 'prob2logodds']

import numpy as np

def nmgy2mag(nmgy):
    return 22.5-2.5*np.log10(nmgy)

def mag2nmgy(mag):
    return 10**(-0.4*(mag-22.5))

def odds2prob(odds):
    return odds / (1. + odds)

def prob2odds(prob):
    return prob / (1. - prob)

def logodds2prob(logodds):
    logoddsflat = logodds.flatten()
    prob = odds2prob(np.exp(logoddsflat))
    # fix overflow
    isinfornan = np.isinf(prob) + np.isnan(prob)
    if np.any(isinfornan):
        prob[isinfornan] = 1.0
    return prob.reshape(np.shape(logodds))

def prob2logodds(prob):
    return np.log(prob2odds(prob))

