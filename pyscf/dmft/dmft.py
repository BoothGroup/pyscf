import numpy as np


import pyscf
from pyscf.lib import logger

from helper import *
#from .helper import *
#from . import hybrid_fit
import hybrid_fit



class DMFT:

    def __init__(self, h1e, eri, nimp, freq=None, verbose=logger.DEBUG4):
        self.log = logger.Logger(verbose=verbose)
        self.h1e = h1e
        self.eri = eri
        self.nimp = nimp

        # Automatic Mastsubara grid
        if freq is None: freq = {}
        if isinstance(freq, dict):
            beta = freq.get("beta", 100.0)
            nfreq = freq.get("nfreq", 1000)
            freq = (2*np.arange(nfreq)+1) * np.pi/beta
        self.freq = freq

    @property
    def norb(self):
        return self.h1e.shape[-1]

    @property
    def nfreq(self):
        return len(self.freq)

    def kernel(self):

        imp = np.s_[:self.nimp]
        t0 = timer()
        # Make inverse G0
        e, v = np.linalg.eigh(h1e)
        g0 = einsum("ai,wi,bi->wab", v[imp], np.add.outer(1j*self.freq, -e), v[imp])
        # Make inverse G0imp
        eimp, vimp = np.linalg.eigh(h1e[imp,imp])
        g0imp = einsum("ai,wi,bi->wab", vimp, np.add.outer(1j*self.freq, -eimp), vimp)
        # Make hybridization
        hyb = g0imp - g0
        self.log.debug("Time for hybridization: %f s", (timer()-t0))


        # Fit hybridization
        hybe, hybc = hybrid_fit.kernel(self, self.freq, hyb, init_guess=3)


        self.log.info("Fitted parameters")
        self.log.info("*****************")
        fmtstr = "  * E= %16.8g : " + hybc[:,0].size*" %16.8g"
        for i, e in enumerate(hybe):
            self.log.info(fmtstr, e, *hybc[:,i].flatten())




if __name__ == "__main__":


    from addons import make_1D_hubbard_model

    h1e = make_1D_hubbard_model(12)
    u = 2.0
    nimp = 1

    dmft = DMFT(h1e, u, nimp)
    dmft.kernel()
