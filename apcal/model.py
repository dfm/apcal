#!/usr/bin/env python
# encoding: utf-8
"""
This is some _insane_ calibration action!

"""

from __future__ import division

__all__ = ['APCalModel']

import numpy as np
import scipy.optimize as op

import _likelihood

# crazy-ass conversion function magic
def _prob_conv(x):
    return 1/(1+np.exp(x))

def _inv_prob_conv(x):
    return np.log((1-x)/x)

class APCalModel(object):
    """
    This is the base calibration model

    Shapes: N stars, M runs

    Parameters
    ----------
    fobs : numpy.ndarray (N, M)
        The observed data in counts/flux units

    ivar : numpy.ndarray (N, M)
        The inverse variance matrix in the same units as fobs

    prior : numpy.ndarray (N)
        The cataloged values of the flux in the same units as fobs

    minobs : int, optional
        The minimum number of runs that must hit any given star. If the number
        of observations of any of the stars is less than this, an AssertionError
        is raised.

    minstars : int, optional
        The minimum number of stars needed per run. If this condition is not met,
        an AssertionError is raised.

    **kwargs : dict, optional
        Any other keyword arguments are passed to init_params as values for the
        nuisance parameters.

    """
    def __init__(self, fobs, ivar, prior, minobs=3, minstars=6, **kwargs):
        self.fobs = fobs
        self.ivar = ivar
        self.prior = prior

        # ignore zeroed entries of ivar
        self.ivarmask = ivar > 0

        # check to make sure that the stars have a minimum of minobs observations
        # and that the runs have at least minstars stars
        assert np.all(np.sum(self.ivarmask, axis=0) > minobs), \
                "All stars must have >%d observations"%(minobs)
        assert np.all(np.sum(self.ivarmask, axis=1) > minstars), \
                "All runs must have >%d stars"%(minstars)

        # dimensions
        self.N, self.M = self.fobs.shape

        self.init_params(**kwargs)

    def init_params(self, **kwargs):
        """
        Initialize the model parameters in some (hopefully, non-dumb) way

        """
        # initialize the fluxes as the cataloged values
        self.fs = np.array(self.prior)

        # use the observed data and the cataloged flux values, find the inverse
        # variance weighted mean zero point implied for each run
        self.f0 = np.sum(self.ivar*self.fobs/self.prior[:,None], axis=0)
        self.f0 /= np.sum(self.ivar, axis=0)

        # Default values for the nuisance parameters
        var_alpha = np.var(self.fobs, axis=1)
        self.Qbad  = kwargs.pop('Qbad', 0.01)
        self.Sbad2 = kwargs.pop('Sbad2', np.max(var_alpha))
        self.Qvar  = kwargs.pop('Qvar', 0.01)
        self.Svar2 = kwargs.pop('Svar2', np.mean(var_alpha))
        self.jitterabs2 = kwargs.pop('jitterabs2', 0.0)
        self.jitterrel2 = kwargs.pop('jitterrel2', 0.0)

    @property
    def nuisance(self):
        return np.array([self.Qbad, self.Sbad2, self.Qvar, self.Svar2,
            self.jitterabs2, self.jitterrel2])

    @property
    def nuisance_vec(self):
        return np.array([_inv_prob_conv(self.Qbad), np.sqrt(self.Sbad2),
            _inv_prob_conv(self.Qvar), np.sqrt(self.Svar2),
            np.sqrt(self.jitterabs2), np.sqrt(self.jitterrel2)])

    @nuisance_vec.setter
    def nuisance_vec(self, v):
        self.Qbad       = _prob_conv(v[0])
        self.Sbad2      = v[1]**2
        self.Qvar       = _prob_conv(v[2])
        self.Svar2, self.jitterabs2, self.jitterrel2 = v[3:]**2

    @property
    def vector(self):
        return np.concatenate([self.f0, self.fs])

    @vector.setter
    def vector(self, v):
        self.f0, self.fs = v[:self.M], v[self.M:]

    def iterative_means(self, miniter=5, maxiter=500, tol=1.25e-8, verbose=True):
        """
        Estimate the zero points and stellar fluxes by iterative weighted mean

        Parameters
        ----------
        miniter : int, optional
            The minimum number of iterations required so that convergence isn't
            prematurely reached.

        maxiter : int, optional
            The maximum number of iterations.

        tol : float, optional
            The convergence tolerance for the mean relative change of the zero
            points.

        verbose : bool, optional
            Output messages about convergence?

        Returns
        -------
        info : bool
            True if the zero points converged in <maxiter iterations.

        """
        # pre-compute the sums of the inverse variance
        iv0 = np.sum(self.ivar, axis=0)
        iv1 = np.sum(self.ivar, axis=1)
        for i in xrange(maxiter):
            f0 = np.sum(self.ivar*self.fobs/self.fs[:,None], axis=0)/iv0
            if i > miniter and np.mean(np.abs(f0-self.f0)/f0) < tol:
                self.f0 = f0
                break
            self.f0 = f0
            self.fs = np.sum(self.ivar*self.fobs/self.f0[None,:], axis=1)/iv1

        if i < maxiter-1:
            if verbose:
                print "Filtering model converged after %d iterations"%(i)
            return True

        if verbose:
            print "Warning: Filtering model didn't converge after %d iterations"%(i)
        return False

    def calibrate(self, miniter=2, maxiter=10, tol=1.25e-8, nuisance=True, verbose=False):
        """
        Optimize the full probabilistic model

        Parameters
        ----------
        miniter : int, optional
            The minimum number of iterations required so that convergence isn't
            prematurely reached.

        maxiter : int, optional
            The maximum number of iterations.

        tol : float, optional
            The convergence tolerance for the mean relative change of the zero
            points.

        nuisance : bool, optional
            Iteratively optimize the nuisance parameters too?

        verbose : bool, optional
            Output messages about convergence?

        """
        miniter = max(miniter, 1)
        lnprob0 = 0
        for i in xrange(maxiter):
            if nuisance:
                lnprob = self.optimize_nuisance(verbose=verbose)
            lnprob = self.optimize_physical(verbose=verbose)
            if i > miniter and np.abs((lnprob-lnprob0)/lnprob) < tol:
                break
            lnprob0 = lnprob

    def optimize_physical(self, verbose=True):
        def chi2(p):
            self.vector = p
            return -self.lnprob()
        p0 = self.vector
        res = op.fmin_bfgs(chi2, p0, full_output=True, disp=verbose)
        self.vector = res[0]
        return res[1]

    def optimize_nuisance(self, verbose=True):
        def chi2(p):
            self.nuisance_vec = p
            return -self.lnprob()
        p0 = self.nuisance_vec
        res = op.fmin_bfgs(chi2, p0, full_output=True, disp=verbose)
        self.nuisance_vec = res[0]
        return res[1]

    def lnprob(self):
        prior = self.lnprior()
        if np.isinf(prior):
            return -np.inf
        lnlike = _likelihood.lnlikelihood(self)
        lnpost = prior + lnlike
        return lnpost

    def lnprior(self):
        if not (0 <= self.Qbad <= 1 and 0 <= self.Qvar <= 1):
            return -np.inf
        var = 0.05
        return -0.5*np.sum((self.fs-self.prior)**2/var+np.log(var))

    def lnodds_bad(self):
        oarr = np.zeros(np.shape(self.fobs))
        _likelihood.lnoddsbad(self, oarr)
        return oarr

    def lnodds_var(self):
        oarr = np.zeros(self.N)
        _likelihood.lnoddsvar(self, oarr)
        return oarr

    def lnlikeratio_bad(self):
        oarr = np.zeros(np.shape(self.fobs))
        _likelihood.lnlikeratiobad(self, oarr)
        return oarr

