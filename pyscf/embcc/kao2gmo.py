"""AO to MO transformation from k-space (AOs) to supercell real space (MOs)

Author: Max Nusspickel
Email:  max.nusspickel@gmail.com
"""

# Standard
from timeit import default_timer as timer
import logging
# External
import numpy as np
# PySCF
import pyscf
from pyscf.mp.mp2 import _mo_without_core
import pyscf.pbc
import pyscf.pbc.tools
from pyscf.pbc.lib import kpts_helper
# Package
from .util import einsum, memory_string

log = logging.getLogger(__name__)

def gdf_to_pyscf_eris(mf, gdf, cm, fock=None):
    """Get supercell MO eris from k-point sampled GDF.

    This folds the MO back into k-space
    and contracts with the k-space three-center integrals..

    Arguments
    ---------
    mf: pyscf.scf.hf.RHF
        Supercell mean-field object.
    gdf: pyscf.pbc.df.GDF
        Gaussian density-fit object of primitive cell (with k-points)
    cm: pyscf.mp.mp2.MP2 or pyscf.cc.ccsd.CCSD
        Correlated method, must have mo_coeff set.
    fock: (N,N) array, optional
        Fock matrix. If None, calculated via mf.get_fock().

    Returns
    -------
    eris: pyscf.mp.mp2._ChemistsERIs or pyscf.cc.rccsd._ChemistsERIs
        ERIs which can be used for the respective correlated method.
    """
    log.debug("Correlated method in eris_kao2gmo= %s", type(cm))

    if fock is None: fock = mf.get_fock()

    # MP2 ERIS
    if isinstance(cm, pyscf.mp.mp2.MP2):
        from pyscf.mp.mp2 import _ChemistsERIs
        eris = _ChemistsERIs()
        only_ovov = True
    # Coupled-cluster ERIS
    elif isinstance(cm, pyscf.cc.ccsd.CCSD):
        from pyscf.cc.rccsd import _ChemistsERIs
        eris = _ChemistsERIs()
        only_ovov = False
    else:
        raise NotImplementedError("Unknown correlated method= %s" % type(cm))

    mo_coeff = _mo_without_core(cm, cm.mo_coeff)
    eris.mo_coeff = mo_coeff
    eris.nocc = cm.nocc
    eris.e_hf = cm._scf.e_tot
    eris.fock = np.linalg.multi_dot((mo_coeff.T, fock, mo_coeff))
    eris.mo_energy = eris.fock.diagonal().copy()

    # Remove EXXDIV correction from Fock matrix (necessary for CCSD)
    if mf.exxdiv and isinstance(cm, pyscf.cc.ccsd.CCSD):
        madelung = pyscf.pbc.tools.madelung(mf.mol, mf.kpt)
        for i in range(eris.nocc):
            eris.fock[i,i] += madelung

    g = gdf_to_eris(gdf, mo_coeff, cm.nocc, only_ovov=only_ovov)
    for key, val in g.items():
        setattr(eris, key, val)

    return eris


def gdf_to_eris(gdf, mo_coeff, nocc, only_ovov=False):
    """Make supercell ERIs from k-point sampled, density-fitted three-center integrals.

    Arguments
    ---------
    cell : pyscf.pbc.gto.Cell
        Primitive unit cell, not supercell!
    gdf : pyscf.pbc.df.GDF
        Density fitting object of primitive cell, with k-points.
    mo_coeff : (Nr*Naux, Nmo) array
        MO coefficients in supercell. The AOs in the supercell must be ordered
        in the same way as the k-points in the primitive cell!
    nocc : int
        Number of occupied orbitals.
    kpts : (Nk, 3)
        k-point coordinates.
    only_ovov : bool, optional
        Only calculate (occ,vir|occ,vir)-type ERIs (for MP2). Default=False.

    Returns
    -------
    eris : dict
        Dict of supercell ERIs. Has elements "ovov" `if only_ovov == True`
        and "oooo", "ovoo", "oovv", "ovov", "ovvo", "ovvv", and "vvvv" otherwise.
    """
    cell = gdf.cell
    kpts = gdf.kpts
    phase = pyscf.pbc.tools.k2gamma.get_phase(cell, kpts)[1]
    nr, nk = phase.shape
    assert len(kpts) == nk == nr
    nmo = mo_coeff.shape[-1]
    nao = cell.nao_nr()         # Atomic orbitals in primitive cell
    naux = gdf.auxcell.nao_nr()  # Auxiliary size in primitive cell
    nvir = nmo - nocc
    o, v = np.s_[:nocc], np.s_[nocc:]

    if only_ovov:
        mem_j3c = nk*naux*nocc*nvir * 16
        mem_eris = nocc**2*nvir**2 * 8
    else:
        mem_j3c = nk*naux*(nocc*nvir + nocc*nocc + nvir*nvir) * 16
        mem_eris = (nocc**4 + nocc**3*nvir + 3*nocc**2*nvir**2 + nocc*nvir**3 + nvir**4)*8
    log.debug("Memory needed for kAO->GMO: temporary= %s final= %s",
            memory_string(max(mem_j3c, mem_j3c/2+mem_eris)), memory_string(mem_eris))

    mo_coeff = mo_coeff.reshape(nr, nao, nmo)
    ck_o = einsum("Rk,Rai->kai", phase.conj(), mo_coeff[:,:,o])
    ck_v = einsum("Rk,Rai->kai", phase.conj(), mo_coeff[:,:,v])

    kconserv = kpts_helper.get_kconserv(cell, kpts)[:,:,0].copy()

    t0 = timer()
    j3c_ov = np.zeros((nk, naux, nocc, nvir), dtype=np.complex)
    if not only_ovov:
        j3c_oo = np.zeros((nk, naux, nocc, nocc), dtype=np.complex)
        j3c_vv = np.zeros((nk, naux, nvir, nvir), dtype=np.complex)
    for ki in range(nk):
        for kj in range(nk):
            kij = (kpts[ki], kpts[kj])
            kk = kconserv[ki,kj]
            blk0 = 0
            for lr, li, sign in gdf.sr_loop(kij, compact=False):
                assert (sign == 1)
                blksize = lr.shape[0]
                blk = np.s_[blk0:blk0+blksize]
                blk0 += blksize

                j3c_kij = (lr+1j*li).reshape(blksize, nao, nao) / np.sqrt(nk)

                j3c_ov[kk,blk] += einsum("Lab,ai,bj->Lij", j3c_kij, ck_o[ki].conj(), ck_v[kj])      # O(Nk^2 * Nocc * Nvir)
                if only_ovov: continue
                j3c_oo[kk,blk] += einsum("Lab,ai,bj->Lij", j3c_kij, ck_o[ki].conj(), ck_o[kj])      # O(Nk^2 * Nocc * Nocc)
                j3c_vv[kk,blk] += einsum("Lab,ai,bj->Lij", j3c_kij, ck_v[ki].conj(), ck_v[kj])      # O(Nk^2 * Nvir * Nvir)
    t_trafo = (timer()-t0)

    t0 = timer()
    eris = {}
    # FT auxiliary index : kR * k -> R
    j3c_ov = np.tensordot(phase, j3c_ov, axes=1)
    imag = abs(j3c_ov.imag).max()
    if imag > 1e-5:
        log.warning("WARNING: max|Im(j3c_ov)|= %.2e", imag)
    else:
        log.debug("max|Im(j3c_ov)|= %.2e", imag)
    j3c_ov = j3c_ov.real
    if not only_ovov:
        j3c_oo = np.tensordot(phase, j3c_oo, axes=1)
        j3c_vv = np.tensordot(phase, j3c_vv, axes=1)
        imag = abs(j3c_oo.imag).max()
        if imag > 1e-5:
            log.warning("WARNING: max|Im(j3c_oo)|= %.2e", imag)
        else:
            log.debug("max|Im(j3c_oo)|= %.2e", imag)
        imag = abs(j3c_vv.imag).max()
        if imag > 1e-5:
            log.warning("WARNING: max|Im(j3c_vv)|= %.2e", imag)
        else:
            log.debug("max|Im(j3c_vv)|= %.2e", imag)
        j3c_oo = j3c_oo.real
        j3c_vv = j3c_vv.real


    j3c_ov2, j3c_oo2, j3c_vv2 = j3c_kao2gmo(gdf, ck_o, ck_v, only_ov=only_ovov)
    assert np.allclose(j3c_ov, j3c_ov2)
    if j3c_oo2 is not None:
        assert np.allclose(j3c_oo, j3c_oo2)
        assert np.allclose(j3c_vv, j3c_vv2)

    j3c_ov = j3c_ov.reshape(nr*naux, nocc, nvir)
    if not only_ovov:
        j3c_oo = j3c_oo.reshape(nr*naux, nocc, nocc)
        j3c_vv = j3c_vv.reshape(nr*naux, nvir, nvir)

    # Contract Lij,Lkl->ijkl
    eris["ovov"] = np.tensordot(j3c_ov, j3c_ov, axes=(0, 0))
    if not only_ovov:
        eris["oooo"] = np.tensordot(j3c_oo, j3c_oo, axes=(0, 0))
        eris["ovoo"] = np.tensordot(j3c_ov, j3c_oo, axes=(0, 0))
        eris["oovv"] = np.tensordot(j3c_oo, j3c_vv, axes=(0, 0))
        eris["ovvo"] = np.tensordot(j3c_ov, j3c_ov.transpose(0, 2, 1), axes=(0, 0))
        eris["ovvv"] = np.tensordot(j3c_ov, j3c_vv, axes=(0, 0))
        eris["vvvv"] = np.tensordot(j3c_vv, j3c_vv, axes=(0, 0))

    t_contract = (timer()-t0)
    log.debug("Timings for kAO->GMO [s]: trafo= %.2f contract= %.2f", t_trafo, t_contract)

    return eris


def j3c_kao2gmo(gdf, cocc, cvir, only_ov=False, make_real=True):
    cell = gdf.cell
    kpts = gdf.kpts
    nk = len(kpts)
    nocc = cocc.shape[-1]
    nvir = cvir.shape[-1]
    naux = gdf.auxcell.nao_nr()
    nao = cell.nao_nr()
    kconserv = kpts_helper.get_kconserv(cell, kpts)[:,:,0].copy()

    j3c_ov = np.zeros((nk, naux, nocc, nvir), dtype=np.complex)
    if not only_ov:
        j3c_oo = np.zeros((nk, naux, nocc, nocc), dtype=np.complex)
        j3c_vv = np.zeros((nk, naux, nvir, nvir), dtype=np.complex)
    else:
        j3c_oo = j3c_vv = None
    for ki in range(nk):
        for kj in range(nk):
            kij = (kpts[ki], kpts[kj])
            kk = kconserv[ki,kj]
            blk0 = 0
            for lr, li, sign in gdf.sr_loop(kij, compact=False):
                assert (sign == 1)
                blksize = lr.shape[0]
                blk = np.s_[blk0:blk0+blksize]
                blk0 += blksize

                j3c_kij = (lr+1j*li).reshape(blksize, nao, nao) / np.sqrt(nk)

                j3c_ov[kk,blk] += einsum("Lab,ai,bj->Lij", j3c_kij, cocc[ki].conj(), cvir[kj])      # O(Nk^2 * Nocc * Nvir)
                if only_ov: continue
                j3c_oo[kk,blk] += einsum("Lab,ai,bj->Lij", j3c_kij, cocc[ki].conj(), cocc[kj])      # O(Nk^2 * Nocc * Nocc)
                j3c_vv[kk,blk] += einsum("Lab,ai,bj->Lij", j3c_kij, cvir[ki].conj(), cvir[kj])      # O(Nk^2 * Nvir * Nvir)

    if make_real:
        phase = pyscf.pbc.tools.k2gamma.get_phase(cell, kpts)[1]
        j3c_ov = np.tensordot(phase, j3c_ov, axes=1)
        imag = abs(j3c_ov.imag).max()
        if imag > 1e-5:
            log.warning("WARNING: max|Im(j3c_ov)|= %.2e", imag)
        else:
            log.debug("max|Im(j3c_ov)|= %.2e", imag)
        j3c_ov = j3c_ov.real
        if not only_ov:
            j3c_oo = np.tensordot(phase, j3c_oo, axes=1)
            j3c_vv = np.tensordot(phase, j3c_vv, axes=1)
            imag = abs(j3c_oo.imag).max()
            if imag > 1e-5:
                log.warning("WARNING: max|Im(j3c_oo)|= %.2e", imag)
            else:
                log.debug("max|Im(j3c_oo)|= %.2e", imag)
            imag = abs(j3c_vv.imag).max()
            if imag > 1e-5:
                log.warning("WARNING: max|Im(j3c_vv)|= %.2e", imag)
            else:
                log.debug("max|Im(j3c_vv)|= %.2e", imag)
            j3c_oo = j3c_oo.real
            j3c_vv = j3c_vv.real

    return j3c_ov, j3c_oo, j3c_vv
