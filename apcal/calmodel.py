#!/usr/bin/env python
# encoding: utf-8
"""
The probabilistic patch model

"""

__all__ = ['PatchProbModel', 'PatchMedianModel']

import numpy as np
import numpy.ma as ma

import scipy.optimize as op

import _likelihood

# options
from conversions import mag2nmgy, nmgy2mag

class PatchProbModel(object):
    """
    Provide a probabilistic model for the calibration of lightcurves

    N stars and M runs.

    Parameters
    ----------
    obs_flux : numpy.ndarray (N, M)
        The observed lightcurves

    obs_ivar : numpy.ndarray (N, M)
        The inverse variances of obs_flux

    calib_mag : numpy.ndarray (N,)
        The cataloged values for the magnitude of the star

    """
    model_id = 1

    def __init__(self, obs_flux, obs_ivar, flux_prior):
        self._f         = obs_flux
        self._ivar_f    = obs_ivar
        self._f_prior   = flux_prior

        self._nstars = self._f.shape[0]
        self._nruns  = self._f.shape[1]

        self._freeze_nuisance = False

        self.init_params()

    def init_params(self):
        tmp = np.mean(self._f/self._f_prior[:,None], axis=0)
        vector = ma.concatenate([tmp,self._f_prior])

        self._f0    = vector[:self._nruns]
        self._mstar = nmgy2mag(vector[self._nruns:])
        self._fstar = self._f_prior

        # nuisance parameters
        self._pbad = 0.01
        self._pvar = 0.01
        self._sigbad2 = np.median(np.var(self._f, axis=1)) #np.max(np.abs(self._f - np.median(self._f, axis=1)[:,None]))**2 #20. #np.exp(17)
        print np.var(self._f, axis=1), np.var(self._f, axis=1).shape
        self._Q2 = 0.01 #0.3**2
        self._jitterabs2 = 0.0
        self._jitterrel2 = 1e-3

        # self.Qvar = 0.1
        # self.Qbad = 0.05
        # self.Sigvar2 = 2500.0**2
        # self.Sigbad2 = 1200.0**2
        # self.Sig2 = 10000


    def calibrate(self):
        p0 = self.vector
        chi2 = lambda p: -self(p)
        p1 = op.fmin_bfgs(chi2,p0)
        self.set_vector(p1)

    @property
    def vector(self):
        if self._freeze_nuisance:
            return np.concatenate((self._f0, self._fstar))
        return np.concatenate((self._f0, self._fstar,
            [np.sqrt(self._sigbad2), np.sqrt(self._Q2),
                np.sqrt(self._jitterabs2), np.sqrt(self._jitterrel2)]))

    def set_vector(self, vector):
        self._f0    = vector[:self._nruns]
        self._fstar = vector[self._nruns:self._nruns+self._nstars]
        # self._mstar = nmgy2mag(self._mstar)
        if not self._freeze_nuisance:
            self._sigbad2, self._Q2, self._jitterabs2, self._jitterrel2 =\
                vector[self._nruns+self._nstars:]
            self._sigbad2, self._Q2, self._jitterabs2, self._jitterrel2 =\
                    self._sigbad2**2, self._Q2**2, self._jitterabs2**2, self._jitterrel2**2
            self._jitterrel2 = 0.0

    @property
    def zeros(self):
        return self._f0

    @property
    def star_flux(self):
        return self._fstar

    @property
    def star_mag(self):
        return self._mstar

    @property
    def nuisance(self):
        return [self._pbad, self._pvar, self._sigbad2, self._Q2,
                self._jitterabs2, self._jitterrel2]

    def set_nuisance(self, **kwargs):
        self._pbad = kwargs.pop('pbad', self._pbad)
        self._pvar = kwargs.pop('pvar', self._pvar)
        self._sigbad2 = kwargs.pop('sigbad2', self._sigbad2)
        self._Q2 = kwargs.pop('Q2', self._Q2)
        self._jitterabs2 = kwargs.pop('jitterabs2', self._jitterabs2)
        self._jitterrel2 = kwargs.pop('jitterrel2', self._jitterrel2)

    def freeze_nuisance(self):
        self._freeze_nuisance = True

    def unfreeze_nuisance(self):
        self._freeze_nuisance = False

    def __unicode__(self):
        st = u"\nfluxes of stars\n"
        st +=   "---------------\n"
        st += repr(self._fstar)
        st += "\nzero points of runs [ADU/nMgy]\n"
        st +=   "------------------------------\n"
        st += repr(self._f0)
        st += "\n"
        for k in ['_jitterabs2','_jitterrel2','_pvar','_Q2',
                '_pbad','_sigbad2']:
            st += "%10s\t"%k
            st += "%e\n"%getattr(self,k)
        return st

    def __str__(self):
        return unicode(self)

    def __call__(self, p):
        self.set_vector(p)
        prior = self.lnprior()
        print prior
        if np.isinf(prior):
            return -1e10
        lnlike = _likelihood.lnlikelihood(self)
        lnpost = prior + lnlike
        print self._sigbad2, self._Q2, lnpost
        return lnpost

    def lnprior(self):
        if not (0 <= self._pbad <= 1 and 0 <= self._pvar <= 1):
            return -np.inf
        # g-band magnitude prior
        var = 0.005
        return -0.5*np.sum((self._fstar-self._f_prior)**2/var+np.log(var))

    def lnodds_bad(self):
        oarr = np.zeros(np.shape(self._f))
        _likelihood.lnoddsbad(self, oarr)
        return oarr

    def lnodds_var(self):
        oarr = np.zeros(self._nstars)
        _likelihood.lnoddsvar(self, oarr)
        return oarr

    def lnlikeratio_bad(self, p,data):
        oarr = np.zeros(np.shape(data._f))
        _likelihood.lnlikeratiobad(self, oarr)
        return oarr

class PatchMedianModel(PatchProbModel):
    model_id = 0

    def init_params(self):
        self._f0 = np.sum(self._ivar_f*self._f/self._f_prior[:,None], axis=0)/np.sum(self._ivar_f, axis=0)
        self._fstar = np.sum(self._ivar_f*self._f/self._f0[None,:], axis=1)/np.sum(self._ivar_f, axis=1)

        # nuisance parameters
        self._pbad = 0.01
        self._pvar = 0.01
        self._sigbad2 = np.exp(17)
        self._Q2 = 0.3**2
        self._jitterabs2 = 0.0
        self._jitterrel2 = 1e-3

    def calibrate(self, maxiter=1000, tol=1.25e-8):
        iv0 = np.sum(self._ivar_f, axis=0)
        iv1 = np.sum(self._ivar_f, axis=1)
        for i in xrange(maxiter):
            f0 = np.sum(self._ivar_f * self._f/self._fstar[:,None], axis=0)/iv0
            # f0[f0 < 0] = np.abs(np.median(f0))
            self._fstar = np.sum(self._ivar_f * self._f/f0[None,:], axis=1)/iv1
            # self._fstar[self._fstar == 0.0] = self._f_prior[self._fstar == 0.0]
            if i > 5 and np.sum(np.fabs(f0-self._f0)/f0) < tol:
                self._f0 = f0
                break
            self._f0 = f0

        if i < maxiter-1:
            print "Median model converged in %d iterations"%(i)
        else:
            print "Warning: Median model didn't converge after %d iterations"%(i)

        # self._mstar = nmgy2mag(self._fstar)

