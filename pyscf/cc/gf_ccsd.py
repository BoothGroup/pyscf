"""Perform Green's function coupled cluster truncated to singles and
doubles using a modified block Lanczos solver to conserve spectral
moments of the similarity-transformed Hamiltonian.
"""

import numpy as np
import scipy.linalg
from collections import defaultdict

from pyscf import lib, cc, ao2mo
from pyscf.lib import logger
from pyscf.agf2 import mpi_helper


def kernel(gfccsd, th=None, tp=None, eris=None):
    """Run GF-CCSD for the IP and EA excitations up to a given
    number of moments of the Green's function.
    """

    if th is None or tp is None:
        imds = gfccsd.make_imds(eris=eris)

    if th is None:
        th = gfccsd.get_ip_moments(imds=imds)
    e_ip, uh, bh = gfccsd.eigh_moments(th, gfccsd.nmom[0])
    v_left_ip = np.dot(bh, uh[:gfccsd.nmo])
    v_right_ip = np.dot(np.linalg.inv(uh)[:, :gfccsd.nmo], bh).T.conj()

    for n in range(2*gfccsd.nmom[0]+2):
        t1 = np.einsum("xk,yk,k->xy", v_left_ip, v_right_ip.conj(), e_ip**n)
        t1_scaled = t1 / np.max(np.abs(th[n]))
        t0_scaled = th[n] / np.max(np.abs(th[n]))

        imag = np.max(np.abs(t1_scaled.imag))
        if imag > 1e-8:
            logger.info(gfccsd, "Large imaginary part in recovered hole "
                                "moment %d: %10.6g", n, imag)

        err = np.max(np.abs(t0_scaled - t1_scaled))
        logger.info(gfccsd, "Scaled error in hole moment %d: %10.6g", n, err)

    # We have actually solved for occupied quasiparticle states, flip sign:
    e_ip *= -1

    if tp is None:
        tp = gfccsd.get_ea_moments(imds=imds)
    e_ea, up, bp = gfccsd.eigh_moments(tp, gfccsd.nmom[1])
    v_left_ea = np.dot(bp, up[:gfccsd.nmo])
    v_right_ea = np.dot(np.linalg.inv(up)[:, :gfccsd.nmo], bp).T.conj()

    for n in range(2*gfccsd.nmom[1]+2):
        t1 = np.einsum("xk,yk,k->xy", v_left_ea, v_right_ea.conj(), e_ea**n)
        t1_scaled = t1 / np.max(np.abs(t1))
        t0_scaled = tp[n] / np.max(np.abs(tp[n]))

        imag = np.max(np.abs(t1_scaled.imag))
        if imag > 1e-8:
            logger.info(gfccsd, "Large imaginary part in recovered particle "
                                "moment %d: %10.6g", n, imag)

        err = np.max(np.abs(t0_scaled - t1_scaled))
        logger.info(gfccsd, "Scaled error in particle moment %d: %10.6g", n, err)

    # Eigenvalues are complex, sort them by their real part:
    mask = np.argsort(e_ip.real)
    e_ip, v_left_ip, v_right_ip = e_ip[mask], v_left_ip[:, mask], v_right_ip[:, mask]
    mask = np.argsort(e_ea.real)
    e_ea, v_left_ea, v_right_ea = e_ea[mask], v_left_ea[:, mask], v_right_ea[:, mask]

    return e_ip, (v_left_ip, v_right_ip), e_ea, (v_left_ea, v_right_ea)


def _kernel_dynamic(gfccsd, grid, eta=1e-2, eris=None, conv_tol=1e-8):
    """Run a more traditional GF-CCSD calculation for a series of
    frequencies using vector correction.
    """

    from scipy.sparse.linalg import LinearOperator, gcrotmk

    ccsd = gfccsd._cc
    grid = np.array(grid)
    calls = defaultdict(int)

    def _part(diag, matvec, get_b, get_e):
        gf = np.zeros((grid.size, ccsd.nmo, ccsd.nmo), dtype=np.complex128)

        def matvec_dynamic(freq, vec, out=None):
            # Compute (freq - H - i\eta) vec
            if out is None:
                out = np.zeros((diag.size,), dtype=np.complex128)
            out = (freq - 1.0j * eta) * vec
            out -= matvec(vec)
            matvec_dynamic.count += 1
            return out
        matvec_dynamic.count = 0

        def matdiv_dynamic(freq, vec, out=None):
            # Apprioximate vec / (freq - H - i\eta) using the diagonal
            if out is None:
                out = np.zeros((diag.size,), dtype=np.complex128)
            out = vec / (freq - diag - 1.0j * eta)
            return out

        es = []
        for p in range(ccsd.nmo):
            es.append(eom.amplitudes_to_vector(*get_e(ccsd, p)))

        for p in range(ccsd.nmo):
            b = eom.amplitudes_to_vector(*get_b(ccsd, p))

            for w in mpi_helper.nrange(grid.size):
                freq = grid[w]
                shape = (diag.size, diag.size)
                ax = LinearOperator(shape, lambda x: matvec_dynamic(freq, x))
                mx = LinearOperator(shape, lambda x: matdiv_dynamic(freq, x))
                x0 = matdiv_dynamic(freq, b)
                x, info = gcrotmk(ax, b, x0=x0, M=mx, atol=0, tol=conv_tol, m=30)

                for q, e in enumerate(es):
                    if info == 0:
                        gf[w, q, p] += np.dot(e, x)
                    else:
                        gf[w, q, p] = np.nan

        mpi_helper.barrier()
        gf = mpi_helper.allreduce(gf)

        calls[matvec] = mpi_helper.allreduce(matvec_dynamic.count)

        return gf

    imds = gfccsd.make_imds(eris=eris)

    eom = ccsd.eomip_method()
    diag = -eom.get_diag(imds)
    matvec = lambda v: -eom.matvec(v, imds, diag)
    gf_occ = _part(diag, matvec, get_b_hole, get_e_hole)

    eom = ccsd.eomea_method()
    diag = eom.get_diag(imds)
    matvec = lambda v: eom.matvec(v, imds, diag)
    gf_vir = _part(diag, matvec, get_b_part, get_e_part)

    gf = gf_occ + gf_vir

    for key, val in calls.items():
        logger.debug(gfccsd, "%d calls to %s", val, key)

    return gf


def get_ip_rdm_moments(gfccsd, rdm1=None, rdm2=None):
    """Get the zeroth- and first-order occupied moments via the first-
    and second-order reduced density matrices.
    """

    if rdm1 is None:
        rdm1 = gfccsd._cc.make_rdm1(l1=gfccsd.l1, l2=gfccsd.l2, ao_repr=False)
    if rdm2 is None:
        rdm2 = gfccsd._cc.make_rdm2(l1=gfccsd.l1, l2=gfccsd.l2, ao_repr=False)

    mf = gfccsd._cc._scf
    h1e = np.linalg.multi_dot((mf.mo_coeff.T.conj(), mf.get_hcore(), mf.mo_coeff))
    h2e = ao2mo.incore.full(mf._eri, mf.mo_coeff, compact=False)
    h2e = h2e.reshape((mf.mo_occ.size,) * 4)

    t0 = rdm1.copy()

    t1 =  lib.einsum("jk,ik->ij", h1e, t0)
    t1 += lib.einsum("jklm,iklm->ij", h2e, rdm2)

    return np.array([t0, t1]) / 2.0


def get_ea_rdm_moments(gfccsd, rdm1=None, rdm2=None):
    """Get the zeroth- and first-order occupied moments via the first-
    and second-order reduced density matrices.
    """

    if rdm1 is None:
        rdm1 = gfccsd._cc.make_rdm1(l1=gfccsd.l1, l2=gfccsd.l2, ao_repr=False)
    if rdm2 is None:
        rdm2 = gfccsd._cc.make_rdm2(l1=gfccsd.l1, l2=gfccsd.l2, ao_repr=False)

    mf = gfccsd._cc._scf
    h1e = np.linalg.multi_dot((mf.mo_coeff.conj().T, mf.get_hcore(), mf.mo_coeff))
    h2e = ao2mo.incore.full(mf._eri, mf.mo_coeff, compact=False)
    h2e = h2e.reshape((mf.mo_occ.size,) * 4)

    rdm2_full =  lib.einsum("kl,jb->bjkl", rdm1, np.eye(gfccsd.nmo)) * 2.0
    rdm2_full -= lib.einsum("kj,bl->bjkl", rdm1, np.eye(gfccsd.nmo))
    #rdm2_full -= lib.einsum("bjkl->blkj", rdm2)
    rdm2_full -= rdm2

    t0 = np.eye(gfccsd.nmo) * 2.0 - rdm1

    t1 =  lib.einsum("aj,jb->ab", h1e, t0)
    t1 += lib.einsum("ajkl,bjkl->ab", h2e, rdm2_full)

    return np.array([t0, t1]) / 2.0


def mat_sqrt(m):
    """Square root a matrix.
    """

    w, v = np.linalg.eig(m)

    return np.dot(v * w[None]**(0.5+0j), np.linalg.inv(v))


def mat_isqrt(m, check_null_space=False, tol=1e-16):
    """Inverse square root of the non-null-space of a matrix.
    """

    w, v = np.linalg.eig(m)
    vinv = np.linalg.inv(v)

    mask = np.abs(w) > tol
    w, v = w[mask], v[:, mask]
    vinv = vinv[mask]

    return np.dot(v * w[None]**(-0.5+0j), vinv)


def block_lanczos(gfccsd, t, nmom, verbose=None):
    """Compute the on- and off-diagonal blocks from the non-Hermitian moments.
    """

    log = logger.new_logger(gfccsd, verbose)

    nmo = t[0].shape[-1]
    dtype = np.complex128

    a = np.zeros((nmom+1, nmo, nmo), dtype=dtype)
    b = np.zeros((nmom, nmo, nmo), dtype=dtype)
    c = np.zeros((nmom, nmo, nmo), dtype=dtype)
    torth = np.zeros((len(t), nmo, nmo), dtype=dtype)
    s = {}
    r = {}
    v = defaultdict(lambda: np.zeros((nmo, nmo), dtype=dtype))
    w = defaultdict(lambda: np.zeros((nmo, nmo), dtype=dtype))
    v[0, 0] = np.eye(nmo).astype(dtype)
    w[0, 0] = np.eye(nmo).astype(dtype)

    eigvals = np.linalg.eigvals(t[0])
    log.info("Smallest eigenvalue of T[0]:  %.12g", np.min(np.abs(eigvals)))
    log.info("Largest imaginary part of T[0] eigenvalues:  %.12g", np.max(np.abs(eigvals.imag)))

    bi = mat_isqrt(t[0])

    def matrix_info(x):
        xnorm = np.abs(np.einsum("pq,qp->", x, x))
        eigvals = np.linalg.eigvals(x)
        mineig = np.min(np.abs(eigvals))
        maxeig = np.max(np.abs(eigvals))
        return xnorm, mineig, maxeig

    log.info("Raw moments:")
    for i in range(len(t)):
        log.info("  T[%d]:    norm=%.12g  min(|eig|)=%.12g  max(|eig|)=%.12g", i, *matrix_info(t[i]))

    log.info("Orthogonalised moments:")
    for i in range(len(t)):
        torth[i] = np.linalg.multi_dot((bi, t[i], bi))
        log.info("  S[%d]:    norm=%.12g  min(|eig|)=%.12g  max(|eig|)=%.12g", i, *matrix_info(torth[i]))

    a[0] = torth[1]

    for i in range(nmom):
        log.info("Iteration %d", i)

        # Compute B_{i} and C_{i}
        b2 = np.zeros((nmo, nmo), dtype=dtype)
        c2 = np.zeros((nmo, nmo), dtype=dtype)

        for j in range(i+2):
            for l in range(i+1):
                b2 += np.linalg.multi_dot((w[i, l], torth[j+l+1], v[i, j-1]))
                c2 += np.linalg.multi_dot((w[i, j-1], torth[j+l+1], v[i, l]))

        b2 -= np.dot(a[i], a[i])
        c2 -= np.dot(a[i], a[i])
        if i:
            b2 -= np.dot(c[i-1], c[i-1])
            c2 -= np.dot(b[i-1], b[i-1])

        b[i] = mat_sqrt(b2)
        c[i] = mat_sqrt(c2)

        binv = mat_isqrt(b2)
        cinv = mat_isqrt(c2)

        log.info("  B[%d]^2:  norm=%.12g  min(|eig|)=%.12g  max(|eig|)=%.12g", i, *matrix_info(b2))
        log.info("  C[%d]^2:  norm=%.12g  min(|eig|)=%.12g  max(|eig|)=%.12g", i, *matrix_info(c2))
        log.info("  B[%d]:    norm=%.12g  min(|eig|)=%.12g  max(|eig|)=%.12g", i, *matrix_info(b[i]))
        log.info("  C[%d]:    norm=%.12g  min(|eig|)=%.12g  max(|eig|)=%.12g", i, *matrix_info(c[i]))
        log.info("  B[%d]^-1: norm=%.12g", i, matrix_info(binv)[0])
        log.info("  C[%d]^-1: norm=%.12g", i, matrix_info(cinv)[0])

        for j in range(i+2):
            # Compute R
            r[j] = (
                + v[i, j-1]
                - np.dot(v[i, j], a[i])
                - np.dot(v[i-1, j], b[i-1])
            )

            # Compute S
            s[j] = (
                + w[i, j-1]
                - np.dot(a[i], w[i, j])
                - np.dot(c[i-1], w[i-1, j])
            )

        if gfccsd.biorthogonalise:
            #for j in range(i):
            #    # Biorthogonalise R = R - V^{j} (W^{j} R)
            #    wr = 0
            #    for k in range(i+2):
            #        for l in range(i+2):
            #            wr += np.linalg.multi_dot((w[j, k], torth[k+l], r[l]))
            #    for k in range(i+2):
            #        r[k] -= np.dot(v[j, k], wr)

            #    # Biorthogonalise S = S - (W^{j} V^{j}) S
            #    wv = 0
            #    for k in range(i+2):
            #        for l in range(i+2):
            #            wv += np.linalg.multi_dot((w[j, k], torth[k+l], v[j, l]))
            #    for k in range(i+2):
            #        s[k] -= np.dot(wv, s[k])

            pass

        for j in range(i+2):
            # Compute V^{i+1}
            v[i+1, j] = np.dot(r[j], cinv)

            # Compute W^{i+1}
            w[i+1, j] = np.dot(binv, s[j])

        if gfccsd.biorthogonalise:
            ## Biorthogonalise W and V
            #for k in range(i+1):
            #    # V^{j} = V^{j} - V^{k} (W^{k} V^{j})
            #    wv = 0
            #    for m in range(i+2):
            #        for l in range(i+2):
            #            wv += np.linalg.multi_dot((w[k, m], torth[m+l], v[i+1, l]))
            #    for m in range(i+2):
            #        v[i+1, m] -= np.dot(v[k, m], wv)

            #    # W^{j} = W^{j} - (W^{k} V^{k}) W^{j}
            #    wv = 0
            #    for m in range(i+2):
            #        for l in range(i+2):
            #            wv += np.linalg.multi_dot((w[k, m], torth[m+l], v[k, l]))
            #    for m in range(i+2):
            #        w[i+1, m] -= np.dot(wv, w[i+1, m])


            inner_prod = np.zeros((nmo*(i+2), nmo*(i+2)), dtype=dtype)
            for n in range(i+2):
                ns = slice(n*nmo, (n+1)*nmo)
                for m in range(i+2):
                    ms = slice(m*nmo, (m+1)*nmo)
                    for k in range(i+2):
                        for l in range(i+2):
                            if (n, k) in w and (m, l) in v:
                                inner_prod[ns, ms] += np.linalg.multi_dot((w[n, k], torth[k+l], v[m, l]))

            lu = np.block(inner_prod)
            l, u = scipy.linalg.lu(lu, permute_l=True)

            linv = np.linalg.inv(l)
            uinv = np.linalg.inv(u)

            vnew = defaultdict(lambda: np.zeros((nmo, nmo), dtype=dtype))
            wnew = defaultdict(lambda: np.zeros((nmo, nmo), dtype=dtype))

            for n in range(i+2):
                ns = slice(n*nmo, (n+1)*nmo)
                for j in range(i+2):
                    js = slice(j*nmo, (j+1)*nmo)
                    if (n, j) in w:
                        wnew[n, j] = np.dot(linv[ns, js], w[n, j])
                    if (n, j) in v:
                        vnew[n, j] = np.dot(v[n, j], uinv[ns, js])

            w, v = wnew, vnew


        # Compute A[i+1]
        for j in range(i+2):
            for l in range(i+2):
                a[i+1] += np.linalg.multi_dot((w[i+1, l], torth[j+l+1], v[i+1, j]))

        log.info("  A[%d]:    norm=%.12g  min(|eig|)=%.12g  max(|eig|)=%.12g", i+1, *matrix_info(a[i+1]))

        log.info("  Error in biorthogonal vector pairs:")
        for j in range(i+2):
            orth = np.zeros((nmo, nmo), dtype=dtype)
            for k in range(i+2):
                for l in range(i+2):
                    orth += np.linalg.multi_dot((w[i+1, l], torth[k+l], v[j, k]))
            log.info("    W[%d] * V[%d]:  %.12g", i+1, j, np.max(np.abs(orth - np.eye(nmo) * ((i+1) == j))))
        for j in range(i+2):
            orth = np.zeros((nmo, nmo), dtype=dtype)
            for k in range(i+2):
                for l in range(i+2):
                    orth += np.linalg.multi_dot((w[j, l], torth[k+l], v[i+1, k]))
            log.info("    W[%d] * V[%d]:  %.12g", j, i+1, np.max(np.abs(orth - np.eye(nmo) * ((i+1) == j))))

    return a, b, c


def build_block_tridiagonal(m, b, c=None):
    """Construct a block tridiagonal matrix from a list of on-diagonal
    and off-diagonal blocks.
    """

    z = np.zeros_like(m[0], dtype=m[0].dtype)

    if c is None:
        c = [x.T.conj() for x in b]

    h = np.block([[
        m[i] if i == j   else
        b[j] if j == i-1 else
        c[i] if i == j-1 else z
        for j in range(len(m))]
        for i in range(len(m))]
    )

    return h


def get_b_hole(ccsd, orb):
    """Get the first- and second-order contributions to the right-hand
    transformed vector for orbital p for the hole part of the Green's
    function.
    """

    nocc = ccsd.nocc
    nvir = ccsd.nmo - nocc

    if orb < nocc:
        b1 = np.eye(nocc)[orb]
        b2 = np.zeros((nocc, nocc, nvir))

    else:
        orb -= nocc
        b1 = ccsd.t1[:, orb]
        b2 = ccsd.t2[:, :, orb]

    return b1, b2


def get_e_hole(ccsd, orb):
    """Get the first- and second-orer contributions to the left-hand
    transformed vector for orbital p for the hole part of the Green's
    function.
    """

    nocc = ccsd.nocc

    if orb < nocc:
        e1 = (
                + np.eye(nocc)[orb]
                - lib.einsum("ie,e->i", ccsd.l1, ccsd.t1[orb])
                - lib.einsum("imef,mef->i", ccsd.l2, ccsd.t2[orb]) * 2.0
                + lib.einsum("imef,mfe->i", ccsd.l2, ccsd.t2[orb])
        )
        e2 = (
                - lib.einsum("ijea,e->ija", ccsd.l2, ccsd.t1[orb]) * 2.0
                + lib.einsum("jiea,e->ija", ccsd.l2, ccsd.t1[orb])
                + lib.einsum("ja,i->ija", ccsd.l1, np.eye(nocc)[orb]) * 2.0
                - lib.einsum("ia,j->ija", ccsd.l1, np.eye(nocc)[orb])
        )

    else:
        orb -= nocc
        e1 = ccsd.l1[:, orb]
        e2 = (
                + ccsd.l2[:, :, orb] * 2.0
                - ccsd.l2[:, :, :, orb]
        )

    return e1, e2


def get_b_part(ccsd, orb):
    """Get the first- and second-order contributions to the right-hand
    transformed vector for orbital p for the particle part of the Green's
    function.
    """

    nocc = ccsd.nocc
    nvir = ccsd.nmo - nocc

    if orb < nocc:
        b1 = -ccsd.t1[orb]
        b2 = -ccsd.t2[orb]

    else:
        orb -= nocc
        b1 = np.eye(nvir)[orb]
        b2 = np.zeros((nocc, nvir, nvir))

    return b1, b2


def get_e_part(ccsd, orb):
    """Get the first- and second-order contributions to the left-hand
    transformed vector for orbital p for the particle part of the Green's
    function.
    """

    nocc = ccsd.nocc
    nvir = ccsd.nmo - nocc

    if orb < nocc:
        e1 = -ccsd.l1[orb]
        e2 = (
                - ccsd.l2[orb] * 2.0
                + ccsd.l2[:, orb]
        )

    else:
        orb -= nocc
        e1 = (
                + np.eye(nvir)[orb]
                - lib.einsum("mb,m->b", ccsd.l1, ccsd.t1[:, orb])
                - lib.einsum("kmeb,kme->b", ccsd.l2, ccsd.t2[:, :, :, orb]) * 2.0
                + lib.einsum("kmeb,mke->b", ccsd.l2, ccsd.t2[:, :, :, orb])
        )
        e2 = (
                - lib.einsum("ikba,k->iab", ccsd.l2, ccsd.t1[:, orb]) * 2.0
                + lib.einsum("ikab,k->iab", ccsd.l2, ccsd.t1[:, orb])
                + lib.einsum("ib,a->iab", ccsd.l1, np.eye(nvir)[orb]) * 2.0
                - lib.einsum("ia,b->iab", ccsd.l1, np.eye(nvir)[orb])
        )

    return e1, e2


class GFCCSD(lib.StreamObject):
    """Green's function coupled cluster singles and doubles.
    """

    def __init__(self, mycc, nmom=(2, 2)):
        self._cc = mycc
        self.verbose = mycc.verbose
        self.stdout = mycc.stdout

        if isinstance(nmom, int):
            self.nmom = (nmom, nmom)
        else:
            self.nmom = nmom
        self.weight_tol = 1e-1
        self.biorthogonalise = False
        self.rdm_moments = False
        self.e_ip = None
        self.v_ip = None
        self.e_ea = None
        self.v_ea = None
        self.t1 = None
        self.t2 = None
        self.l1 = None
        self.l2 = None
        self.chkfile = self._cc.chkfile
        self._keys = set(self.__dict__.keys())

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info("")
        log.info('******** %s ********', self.__class__)
        log.info("nmom = %s", self.nmom)
        log.info("nmo = %s", self.nmo)
        log.info("nocc = %s", self.nocc)
        log.info("weight_tol = %s", self.weight_tol)
        log.info("rdm_moments = %s", self.rdm_moments)
        log.info("chkfile = %s", self.chkfile)

    def _finalize(self):
        v_ip_l, v_ip_r = self.v_ip
        mask = np.abs(np.sum(v_ip_l * v_ip_r.conj(), axis=0)) > self.weight_tol
        e_ip = self.e_ip[mask].real
        v_ip_l, v_ip_r = v_ip_l[:, mask], v_ip_r[:, mask]

        v_ea_l, v_ea_r = self.v_ea
        mask = np.abs(np.sum(v_ea_l * v_ea_r.conj(), axis=0)) > self.weight_tol
        e_ea = self.e_ea[mask].real
        v_ea_l, v_ea_r = v_ea_l[:, mask], v_ea_r[:, mask]

        logger.note(self, "Ionisation potentials:")
        logger.note(self, "  %4s %12s %12s" % ("root", "energy", "qpwt"))
        for nroot in range(min(5, len(e_ip))):
            qpwt = np.abs(np.sum(v_ip_l[:, nroot] * v_ip_r[:, nroot].conj())).real
            logger.note(self, "  %4d %12.6f %12.6g" % (nroot, e_ip[nroot], qpwt))

        logger.note(self, "Electron affinity:")
        logger.note(self, "  %4s %12s %12s" % ("root", "energy", "qpwt"))
        for nroot in range(min(5, len(e_ea))):
            qpwt = np.abs(np.sum(v_ea_l[:, nroot] * v_ea_r[:, nroot].conj())).real
            logger.note(self, "  %4d %12.6f %12.6g" % (nroot, e_ea[nroot], qpwt))

        return self

    def reset(self, mol=None):
        if cc is not None:
            self._cc.mol = mol
        self._cc.reset(mol)
        return self

    @property
    def eomip_method(self):
        return self._cc.eomip_method()

    @property
    def eomea_method(self):
        return self._cc.eomea_method()

    get_e_hole = get_e_hole
    get_e_part = get_e_part
    get_b_hole = get_b_hole
    get_b_part = get_b_part

    #TODO: freeze p, q if frozen in self._cc
    def get_ip_moments(self, imds=None, nmom=None):
        """Get the moments of the IP-EOM-CCSD Green's function.
        """

        if nmom is None:
            nmom = 2 * self.nmom[0] + 2
        moms = np.zeros((nmom, self.nmo, self.nmo))

        if self.rdm_moments:
            assert nmom == 2
            return get_ip_rdm_moments(self)

        cput0 = (logger.process_clock(), logger.perf_counter())

        eom = self.eomip_method()
        if imds is None:
            imds = self.make_imds(ea=False)
        diag = eom.get_diag(imds)

        def matvec(v):
            matvec.count += 1
            return -eom.l_matvec(v, imds, diag)
        matvec.count = 0

        bras = [None] * self.nmo
        for p in range(self.nmo):
            r1, r2 = self.get_b_hole(p)
            bras[p] = eom.amplitudes_to_vector(r1, r2)

        for p in mpi_helper.nrange(self.nmo):
            r1, r2 = self.get_e_hole(p)
            ket = eom.amplitudes_to_vector(r1, r2)

            for n in range(nmom):
                for q in range(self.nmo):
                    bra = bras[q]
                    moms[n, q, p] += np.dot(bra, ket)

                if (n+1) != nmom:
                    ket = matvec(ket)

        mpi_helper.barrier()
        moms = mpi_helper.allreduce(moms)

        logger.timer(self, "IP-EOM-CCSD moments", *cput0)
        logger.debug(self, "%d calls to %s", mpi_helper.allreduce(matvec.count), matvec)

        return moms

    def get_ea_moments(self, imds=None, nmom=None):
        """Get the moments of the EA-EOM-CCSD Green's function.
        """

        if nmom is None:
            nmom = 2 * self.nmom[1] + 2
        moms = np.zeros((nmom, self.nmo, self.nmo))

        if self.rdm_moments:
            assert nmom == 2
            return get_ea_rdm_moments(self)

        cput0 = (logger.process_clock(), logger.perf_counter())

        eom = self.eomea_method()
        if imds is None:
            imds = self.make_imds(ip=False)
        diag = eom.get_diag(imds)

        def matvec(v):
            matvec.count += 1
            return eom.l_matvec(v, imds, diag)
        matvec.count = 0

        bras = [None] * self.nmo
        for p in range(self.nmo):
            r1, r2 = self.get_b_part(p)
            bras[p] = eom.amplitudes_to_vector(r1, r2)

        for p in mpi_helper.nrange(self.nmo):
            r1, r2 = self.get_e_part(p)
            ket = eom.amplitudes_to_vector(r1, r2)

            for n in range(nmom):
                for q in range(self.nmo):
                    bra = bras[q]
                    moms[n, q, p] += np.dot(bra, ket)

                if (n+1) != nmom:
                    ket = matvec(ket)

        mpi_helper.barrier()
        moms = mpi_helper.allreduce(moms)

        logger.timer(self, "EA-EOM-CCSD moments", *cput0)
        logger.debug(self, "%d calls to %s", mpi_helper.allreduce(matvec.count), matvec)

        return moms

    def eigh_moments(self, t, nmom):
        """Block tridiagonalise the matrix under the constraint of
        conservation of a given number of moments, and diagonalise
        the block tridiagonal matrix.
        """

        a, b, c = block_lanczos(self, t, nmom)
        h_tri = build_block_tridiagonal(a, b, c)
        e, u = np.linalg.eig(h_tri)
        bi = mat_sqrt(t[0])

        return e, u, bi

    def make_imds(self, eris=None, ip=True, ea=True):
        """Build EOM intermediates.
        """

        imds = cc.eom_rccsd._IMDS(self._cc, eris=eris)

        if ip:
            imds.make_ip()
        if ea:
            imds.make_ea()

        return imds

    def make_rdm1(self, ao_repr=False, eris=None, imds=None):
        """Build the first-order reduced density matrix at the CCSD
        level using the zeroth moment of the hole part of the CCSD
        Green's function.
        """

        if imds is None:
            imds = self.make_imds(eris=eris, ea=False)

        dm1 = self.get_ip_moments(imds=imds, nmom=0)[0]
        dm1 = dm1 + dm1.T.conj()

        if ao_repr:
            mo = self._cc.mo_coeff
            dm1 = np.linalg.multi_dot((mo, dm1, mo.T.conj()))

        return dm1

    def kernel(self, th=None, tp=None, t1=None, t2=None, l1=None, l2=None, eris=None):
        if self.verbose >= logger.WARN:
            self.check_sanity()
        self.dump_flags()

        if t1 is not None:
            self.t1 = t1
        if t2 is not None:
            self.t2 = t2
        if l1 is not None:
            self.l1 = l1
        if l2 is not None:
            self.l2 = l2

        if self.t1 is None:
            self.t1 = self._cc.t1
        if self.t2 is None:
            self.t2 = self._cc.t2
        if self.l1 is None:
            self.l1 = self._cc.l1
        if self.l2 is None:
            self.l2 = self._cc.l2

        if self.l1 is None or self.l2 is None:
            raise RuntimeError(
                    "solve_lambda() must be called on the %s "
                    "object before passing to %s."
                    % (self._cc.__class__.__name__, self.__class__.__name__)
            )

        self.e_ip, self.v_ip, self.e_ea, self.v_ea = kernel(self, th=th, tp=tp, eris=eris)

        if self.chkfile is not None:
            self.dump_chk()

        self._finalize()

        return self.e_ip, self.v_ip, self.e_ea, self.v_ea

    def dump_chk(self, chkfile=None, key="gfccsd"):
        if chkfile is None:
            chkfile = self.chkfile
        lib.chkfile.dump(chkfile, "%s/e_ip" % key, self.e_ip)
        lib.chkfile.dump(chkfile, "%s/v_ip_left" % key, self.v_ip[0])
        lib.chkfile.dump(chkfile, "%s/v_ip_right" % key, self.v_ip[1])
        lib.chkfile.dump(chkfile, "%s/e_ea" % key, self.e_ea)
        lib.chkfile.dump(chkfile, "%s/v_ea_left" % key, self.v_ea[0])
        lib.chkfile.dump(chkfile, "%s/v_ea_right" % key, self.v_ea[1])
        lib.chkfile.dump(chkfile, "%s/nmom" % key, np.array(self.nmom))
        return self

    def update_from_chk_(self, chkfile=None, key="gfccsd"):
        if chkfile is None:
            chkfile = self.chkfile
        self.e_ip = lib.chkfile.load(chkfile, "%s/e_ip" % key)
        self.v_ip = (
                lib.chkfile.load(chkfile, "%s/v_ip_left" % key),
                lib.chkfile.load(chkfile, "%s/v_ip_right" % key),
        )
        self.e_ea = lib.chkfile.load(chkfile, "%s/e_ea" % key)
        self.v_ea = (
                lib.chkfile.load(chkfile, "%s/v_ea_left" % key),
                lib.chkfile.load(chkfile, "%s/v_ea_right" % key),
        )
        self.nmom = tuple(lib.chkfile.load(chkfile, "%s/nmom" % key))
        return self

    update = update_from_chk = update_from_chk_

    def ipccsd(self, nroots=5):
        """Print and return IPs.
        """

        e_ip = self.e_ip
        v_ip_l, v_ip_r = self.v_ip
        mask = np.abs(np.sum(v_ip_l * v_ip_r.conj(), axis=0)) > self.weight_tol

        e_ip = self.e_ip[mask]
        v_ip_l, v_ip_r = v_ip_l[:, mask], v_ip_r[:, mask]

        nroots = min(nroots, len(e_ip))

        e_ip = e_ip[:nroots]
        v_ip_l = list(v_ip_l[:, :nroots].T)
        v_ip_r = list(v_ip_r[:, :nroots].T)

        logger.note(self, "  %2s %2s %16s %10s" % ("", "", "Energy", "Weight"))
        for n, en, vnl, vnr in zip(range(nroots), e_ip, v_ip_l, v_ip_r):
            qpwt = np.abs(np.sum(vnl * vnr.conj())).real
            warn = "(Warning: complex excitation)" if np.abs(en.imag) > 1e-8 else ""
            logger.note(self, "  %2s %2d %16.10g %10.6g %s" % ("IP", n, en.real, qpwt, warn))

        if nroots == 1:
            return e_ip[0].real, v_ip_l[0], v_ip_r[0]
        else:
            return e_ip.real, v_ip_l, v_ip_r

    def eaccsd(self, nroots=5):
        """Print and return EAs.
        """

        e_ea = self.e_ea
        v_ea_l, v_ea_r = self.v_ea
        mask = np.abs(np.sum(v_ea_l * v_ea_r.conj(), axis=0)) > self.weight_tol

        e_ea = e_ea[mask]
        v_ea_l, v_ea_r = v_ea_l[:, mask], v_ea_r[:, mask]

        nroots = min(nroots, len(e_ea))

        e_ea = e_ea[:nroots]
        v_ea_l = list(v_ea_l[:, :nroots].T)
        v_ea_r = list(v_ea_r[:, :nroots].T)

        logger.note(self, "  %2s %2s %16s %10s" % ("", "", "Energy", "Weight"))
        for n, en, vnl, vnr in zip(range(nroots), e_ea, v_ea_l, v_ea_r):
            qpwt = np.abs(np.sum(vnl * vnr.conj())).real
            warn = "(Warning: complex excitation)" if np.abs(en.imag) > 1e-8 else ""
            logger.note(self, "  %2s %2d %16.10g %10.6g %s" % ("EA", n, en.real, qpwt, warn))

        if nroots == 1:
            return e_ea[0].real, v_ea_l[0], v_ea_r[0]
        else:
            return e_ea.real, v_ea_l, v_ea_r

    @property
    def nmo(self):
        return self._cc.nmo

    @property
    def nocc(self):
        return self._cc.nocc

    @property
    def nvir(self):
        return self.nmo - self.nocc


if __name__ == "__main__":
    from pyscf import gto, scf, cc

    mol = gto.M(atom="O 0 0 0.11779; H 0 0.755453 -0.471161; H 0 -0.755453 -0.471161", basis="6-31g", verbose=5)
    mf = scf.RHF(mol).run()
    ccsd = cc.CCSD(mf).run()
    ccsd.solve_lambda()

    gfcc = GFCCSD(ccsd, nmom=(5, 5))
    gfcc.kernel()

    ip1, vip1 = ccsd.ipccsd(nroots=8)
    ip2, vip2, uip2 = gfcc.ipccsd(nroots=8)

    ea1, vea1 = ccsd.eaccsd(nroots=8)
    ea2, vea2, uea2 = gfcc.eaccsd(nroots=8)

    print(np.abs(ip1[0]-ip2[0]))
    print(np.abs(ip1[1]-ip2[1]))
    print(np.abs(ea1[0]-ea2[0]))
    print(np.abs(ea1[1]-ea2[1]))
