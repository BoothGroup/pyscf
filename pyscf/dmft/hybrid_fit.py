"""ModulX for the fitting of the hybridization function"""

import numpy as np
import scipy
import scipy.optimize

from helper import *

def get_init_guess(freq, hyb, npoles, multiplicity=1):
    energy = np.linspace(-freq.max()/2, freq.max()/2, npoles)
    nimp = hyb.shape[-1]
    coupling = np.full((nimp, npoles, multiplicity), 1e-2)

    return energy, coupling


def kernel(dmft, freq, hyb, init_guess=None, multiplicity=1, freq_weight=-1):

    if init_guess is None:
        init_guess = 5
    if np.isscalar(init_guess):
        e0, c0 = get_init_guess(freq, hyb, init_guess, multiplicity)
    else:
        e0, c0 = init_guess

    # Check if frequencies are uniform
    #uniform = np.allclose(np.diff(freq), freq[1]-freq[0])

    # Calculate frequency grid weights
    bounds = (freq[:-1] + np.diff(freq)/2)
    bounds = np.hstack((0, bounds, bounds[-1] + (bounds[-1]-bounds[-2])/2))
    grid_weight = np.diff(bounds)

    # Frequency weighting
    if np.isscalar(freq_weight):
        freq_weight = abs(np.power(freq, freq_weight))
    elif callable(freq_weight):
        freq_weight = freq_weight(freq)
    else:
        freq_weight = np.asarray(freq_weight)

    # Combine grid_weight and freq_weight
    wgt = grid_weight * freq_weight

    # Number of fit parameter
    nimp = hyb.shape[-1]
    npoles = len(e0)
    nparam = npoles * c0.size

    def pack_vec(e, c):
        vec = np.hstack((e, c.flatten()))
        return vec

    def unpack_vec(vec):
        e, c = np.hsplit(vec, [npoles])
        c = c.reshape(nimp, npoles*multiplicity)
        return e, c

    def make_fit(e, c):
        fit = einsum("ai,wi,bi->wab", c, 1/np.add.outer(1j*freq, -e), c)
        return fit


    def objective_func(vec):

        e, c = unpack_vec(vec)
        fit = make_fit(e, c)
        diff = fit-hyb
        funcval = (einsum("w,wab,wab->", wgt, diff.real, diff.real)
                 + einsum("w,wab,wab->", wgt, diff.imag, diff.imag))
        dmft.log.debug("e= %r c= %r f= %.8e", e, c, funcval)

        # Gradient
        #r = 1/np.add.outer(1j*freq, -e)
        #de = einsum("

        return funcval


    vec0 = pack_vec(e0, c0)

    t0 = timer()
    res = scipy.optimize.minimize(objective_func, vec0)

    energy, coupling = unpack_vec(res.x)

    return energy, coupling

