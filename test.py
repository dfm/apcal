# coding: utf-8
import pylab as pl
import pyfits
import apcal.model as model
import numpy as np
import apcal.conversions as conversions

f = pyfits.open("data/97-0295246-gals-flux_matrix.fits")[0].data[15:40]
ivar = pyfits.open("data/97-0295246-gals-ivar_matrix.fits")[0].data[15:40]
f0 = np.sum(ivar*f, axis=1)/np.sum(ivar, axis=1)
inds = np.sum(ivar, axis=0) > 0
f = f[:,inds]
ivar = ivar[:,inds]
model = model.APCalModel(f,ivar, f0)

# model.freeze_nuisance()
model.calibrate()

lnprobbad = conversions.logodds2prob(model.lnodds_bad())
lnoddsvar = model.lnodds_var()
print lnoddsvar
print lnprobbad

for i in range(10):
    pl.figure()
    inds = ivar[i,:] > 0

    pl.plot(np.arange(f.shape[1])[inds], f[i,inds], 'og', alpha=0.3)

    pl.scatter(np.arange(f.shape[1])[inds], f[i,inds]/model._f0[inds], c=lnprobbad[i,:])
    pl.gca().axhline(model._fstar[i], color='k')
    pl.gca().axhline((1+0.05)*model._fstar[i], color='k', ls='--')
    pl.gca().axhline((1-0.05)*model._fstar[i], color='k', ls='--')

pl.show()

