# TODO
# Make this a unittest module

from timeit import default_timer as timer
import numpy as np

import pyscf
import pyscf.pbc
import pyscf.pbc.cc
import pyscf.pbc.tools

from pyscf.pbc.df.df_incore import IncoreGDF

import pyscf.embcc
import pyscf.embcc.k2gamma_gdf

def make_cubic(a, atom="He", supercell=False):
    amat = a * np.eye(3)
    atom = "%s %f %f %f" % (atom, a/2, a/2, a/2)

    cell = pyscf.pbc.gto.Cell()
    cell.a = amat
    cell.atom = atom
    cell.basis = "gth-dzv"
    cell.pseudo = "gth-pade"
    cell.precision = 1e-5
    cell.verbose = 0
    cell.build()
    if supercell:
        cell = pyscf.pbc.tools.super_cell(cell, supercell)
    return cell

def make_diamond(a, atoms=["C1", "C2"], supercell=False):
    amat = a * np.asarray([
        [0.5, 0.5, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5]])
    coords = a * np.asarray([[0, 0, 0], [1, 1, 1]])/4
    atom = [(atoms[0], coords[0]), (atoms[1], coords[1])]

    cell = pyscf.pbc.gto.Cell()
    cell.a = amat
    cell.atom = atom
    cell.basis = "gth-dzv"
    cell.pseudo = "gth-pade"
    cell.precision = 1e-5
    cell.verbose = 0
    cell.build()
    if supercell:
        cell = pyscf.pbc.tools.super_cell(cell, supercell)
    return cell

def test_helium():

    a = 2.0
    kmesh = [2, 2, 2]
    cell = make_cubic(a, "He")
    kpts = cell.make_kpts(kmesh)
    kmf = pyscf.pbc.scf.KRHF(cell, kpts)
    kmf = kmf.density_fit()
    kmf.kernel()

    # Unfold
    kmf = pyscf.embcc.k2gamma_gdf.k2gamma_gdf(kmf, kmesh)

    kcc = pyscf.embcc.EmbCC(kmf, bath_tol=1e-4)
    kcc.opts.popfile = None
    kcc.opts.eom_ccsd = True
    kcc.opts.prim_mp2_bath_tol_occ = 1e-3
    kcc.opts.prim_mp2_bath_tol_vir = 1e-3
    kcc.make_atom_cluster(0, symmetry_factor=2)
    kcc.kernel()
    print("K-CCSD E= %16.8g" % kcc.e_tot)
    assert np.isclose(kcc.e_tot, -22.8671)


    scell = make_cubic(a, "He", supercell=kmesh)
    smf = pyscf.pbc.scf.RHF(scell)
    smf = smf.density_fit()
    smf.kernel()

    scc = pyscf.embcc.EmbCC(smf, bath_tol=1e-4)
    scc.opts.popfile = None
    scc.opts.eom_ccsd = True
    scc.opts.prim_mp2_bath_tol_occ = 1e-3
    scc.opts.prim_mp2_bath_tol_vir = 1e-3
    scc.make_atom_cluster(0, symmetry_factor=2)
    scc.kernel()
    print("SC-CCSD E= %16.8g" % scc.e_tot)
    assert np.isclose(scc.e_tot, -22.8671)


def test_diamond(EXPECTED=None, kmesh=[2, 2, 2], bath_tol=1e-4):

    a = 3.5
    ncells = np.product(kmesh)

    # k-point calculation
    cell = make_diamond(a)
    kpts = cell.make_kpts(kmesh)
    kmf = pyscf.pbc.scf.KRHF(cell, kpts)
    kmf = kmf.density_fit()
    kmf.kernel()

    kcc = pyscf.embcc.EmbCC(kmf, bath_tol=bath_tol)
    kcc.opts.popfile = None
    kcc.opts.orbfile = None
    kcc.make_atom_cluster(0, symmetry_factor=2)
    t0 = timer()
    kcc.kernel()
    print("Time for k-EmbCC= %.3f" % (timer()-t0))
    print("k-EmbCC E= %16.8g" % kcc.e_tot)
    if EXPECTED:
        assert np.isclose(kcc.e_tot, EXPECTED)
    else:
        EXPECTED = kcc.e_tot

    # Supercell calculations
    scell = make_diamond(a, supercell=kmesh)
    smf = pyscf.pbc.scf.RHF(scell)
    smf = smf.density_fit()
    smf.kernel()

    scc = pyscf.embcc.EmbCC(smf, bath_tol=bath_tol)
    scc.opts.popfile = None
    scc.opts.orbfile = None
    scc.make_atom_cluster(0, symmetry_factor=2*ncells)
    t0 = timer()
    scc.kernel()
    print("Time for supercell-EmbCC= %.3f" % (timer()-t0))
    print("SC-CCSD E= %16.8g" % (scc.e_tot/ncells))
    if EXPECTED:
        assert np.isclose(scc.e_tot/ncells, EXPECTED)

    # k-point calculation + incore GDF
    cell = make_diamond(a)
    kpts = cell.make_kpts(kmesh)
    kmf = pyscf.pbc.scf.KRHF(cell, kpts)
    kmf.with_df = IncoreGDF(cell, kpts)
    kmf.kernel()

    kcc = pyscf.embcc.EmbCC(kmf, bath_tol=bath_tol)
    kcc.opts.popfile = None
    kcc.opts.orbfile = None
    kcc.make_atom_cluster(0, symmetry_factor=2)
    t0 = timer()
    kcc.kernel()
    print("Time for k-EmbCC (incore GDF)= %.3f" % (timer()-t0))
    print("k-EmbCC (incore GDF) E= %16.8g" % kcc.e_tot)
    if EXPECTED:
        assert np.isclose(kcc.e_tot, EXPECTED)

def test_full_ccsd_limit(EXPECTED, kmesh=[2, 2, 2]):

    a = 3.5
    ncells = np.product(kmesh)

    # k-point calculation
    cell = make_diamond(a)
    kpts = cell.make_kpts(kmesh)
    kmf = pyscf.pbc.scf.KRHF(cell, kpts)
    kmf = kmf.density_fit()
    kmf.kernel()

    # canonical CCSD
    ccsd = pyscf.pbc.cc.KCCSD(kmf)
    ccsd.kernel()
    print("Exact CCSD= %16.8g" % ccsd.e_tot)
    assert np.allclose(ccsd.e_tot, EXPECTED)

    # bath_tol=-1 -> complete environment as bath
    kcc = pyscf.embcc.EmbCC(kmf, bath_tol=-1)
    kcc.opts.popfile = None
    kcc.opts.orbfile = None
    kcc.make_atom_cluster(0, symmetry_factor=2)
    kcc.kernel()
    print("k-EmbCC E= %16.8g" % kcc.e_tot)
    assert np.allclose(ccsd.e_tot, EXPECTED)

def run_test():
    #test_helium()
    test_diamond(-11.138309)
    test_diamond(-11.164555, bath_tol=1e-6)
    test_full_ccsd_limit(-11.170842)

if __name__ == "__main__":
    run_test()
