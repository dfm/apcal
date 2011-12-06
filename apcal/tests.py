#!/usr/bin/env python
# encoding: utf-8
"""
Test the patch calibration model

"""

__all__ = ['FakeDataset']

import numpy as np

import calmodel

import conversions

class FakeDataset(object):
    def __init__(self, Nstars, Nobs):
        self.true_f0 = 1500*np.random.rand(Nobs)+1000
        self.true_fs = (5*np.random.rand(Nstars)+20)**2
        self.Qvar = 0.1
        self.Qbad = 0.01
        self.Qvar2 = 0.2**2
        self.Sigbad2 = 100000.0**2
        self.Sig2 = 5000.0**2

        # data
        self.f = np.outer(self.true_fs, self.true_f0)
        self.f += np.sqrt(self.Sig2) * np.random.randn(np.prod(self.f.shape)).reshape(*self.f.shape)

        # variables
        self.var_inds = np.random.randint(Nstars, size=self.Qvar*Nstars)
        self.f[self.var_inds,:] += self.f[self.var_inds,:]*np.sqrt(self.Qvar2)*np.random.randn(Nobs*len(self.var_inds))

        # bad observations
        self.bad_inds = np.random.randint(Nstars*Nobs, size=self.Qbad*Nstars*Nobs)
        self.f.flat[self.bad_inds] += np.sqrt(self.Sigbad2) * np.random.randn(len(self.bad_inds))

if __name__ == '__main__':
    import matplotlib.pyplot as pl

    nstars, nobs = 10, 50
    data = FakeDataset(nstars, nobs)

    model = calmodel.PatchMedianModel(data.f, np.ones(data.f.shape)/data.Sig2, data.true_fs+5*np.random.randn(len(data.true_fs)))
    model.calibrate()

    for i in range(nstars):
        pl.figure(i)
        pl.plot(np.arange(nobs), data.f[i,:]/model._f0, 'og', alpha=0.3)
        pl.gca().axhline(model._fstar[i], color='g', lw=2, alpha=0.5)

    model = calmodel.PatchProbModel(data.f, np.ones(data.f.shape)/data.Sig2, data.true_fs+5*np.random.randn(len(data.true_fs)))
    model.calibrate()
    print "Final Model"
    print "===== ====="
    print model

    lnprobbad = conversions.logodds2prob(model.lnodds_bad())
    lnoddsvar = model.lnodds_var()
    print lnoddsvar

    for i in range(nstars):
        pl.figure(i)
        pl.scatter(np.arange(nobs), data.f[i,:]/model._f0, c=lnprobbad[i,:])
        pl.gca().axhline(model._fstar[i], color='k')
        pl.gca().axhline((1+0.05)*model._fstar[i], color='k', ls='--')
        pl.gca().axhline((1-0.05)*model._fstar[i], color='k', ls='--')

    pl.show()

