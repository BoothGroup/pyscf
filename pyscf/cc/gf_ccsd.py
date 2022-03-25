"""Perform Green's function coupled cluster truncated to singles and
doubles using a modified block Lanczos solver to conserve spectral
moments of the similarity-transformed Hamiltonian.
"""

import numpy as np
from collections import defaultdict

from pyscf import lib, cc
from pyscf.lib import logger
from pyscf.agf2 import mpi_helper, chempot, GreensFunction


def kernel(gfccsd, th=None, tp=None, eris=None):
    """Run GF-CCSD for the IP and EA excitations up to a given
    number of moments of the Green's function.
    """

    if th is None or tp is None:
        imds = gfccsd.make_imds(eris=eris)

    if th is None:
        th = gfccsd.get_ip_moments(imds=imds)
    eh, uh = gfccsd.eigh_moments(th, gfccsd.nmom[0])
    eh *= -1

    if tp is None:
        tp = gfccsd.get_ea_moments(imds=imds)
    ep, up = gfccsd.eigh_moments(tp, gfccsd.nmom[1])

    e = np.concatenate([eh, ep])
    u = np.concatenate([uh, up], axis=1).T.conj()

    norm = np.linalg.norm(u, axis=0, keepdims=True)
    norm[np.abs(norm) < 1e-14] = 1e-14
    u /= norm

    p = np.eye(e.size) - np.dot(u, u.T.conj())
    w, v = np.linalg.eigh(p)
    del p

    mask = np.abs(w) > 1e-14
    w, v = w[mask], v[:, mask]
    u = np.block([u, v * w[None]])

    h = np.dot(u.T.conj() * e[None], u)
    e, c = np.linalg.eigh(h)
    del h

    # Remove low poles with unphysical energies FIXME
    mask = np.linalg.norm(c[:gfccsd.nmo], axis=0)**2 > gfccsd.weight_tol
    e, c = e[mask], c[:, mask]

    return e, c


def _kernel_dynamic(gfccsd, grid, eta=1e-2, eris=None, conv_tol=1e-6, max_cycle=100):
    """Run a more traditional GF-CCSD calculation for a series of
    frequencies using vector correction and DIIS.
    """

    ccsd = gfccsd._cc
    grid = np.array(grid)

    def _part(diag, matvec, get_b, get_e):
        gf = np.zeros((grid.size, ccsd.nmo, ccsd.nmo), dtype=np.complex128)

        def matvec_dynamic(freq, vec, out=None):
            # Compute (freq - H - i\eta) vec
            if out is None:
                out = np.zeros((diag.size,), dtype=np.complex128)
            out = (freq - 1.0j * eta) * vec
            out -= matvec(vec)
            return out

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

            for w, freq in enumerate(grid):
                diis = lib.diis.DIIS()
                omega = freq - 1.0j * eta
                converged = False
                r = Hx = None

                # Guess X using the inverse of H, approximated via the diagonal
                x = b / (omega - diag)

                for i in range(1, max_cycle+1):
                    # Compute product of current X with H
                    Hx = omega * x - matvec(x)

                    # Find error and compute residual
                    e = b - Hx
                    residual = np.linalg.norm(e)

                    # Compute correction vector using the inverse diagonal
                    r = matdiv_dynamic(freq, e, out=r)

                    # Check if convergence criteria are met
                    if residual < conv_tol:
                        converged = True
                        break

                    # Update X using correction vector and solve linear system
                    x += r
                    x = diis.update(x, xerr=r)

                for q, e in enumerate(es):
                    if converged:
                        gf[w, q, p] += np.dot(e, x)
                    else:
                        gf[w, q, p] = np.nan

        gf = 0.5 * (gf + gf.swapaxes(1, 2)).conj()

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

    return gf


def mat_sqrt(m):
    """Square root a matrix.
    """

    w, v = np.linalg.eigh(m)

    mask = w >= 0.0
    w, v = w[mask], v[:, mask]

    return np.dot(v * w[None]**0.5, v.T)


def mat_isqrt(m, check_null_space=False, tol=1e-14):
    """Inverse square root of the non-null-space of a matrix.
    """

    w, v = np.linalg.eigh(m)

    null_space = np.any(w < tol)
    mask = w >= tol
    w, v = w[mask], v[:, mask]

    if check_null_space:
        return np.dot(v * w[None]**-0.5, v.T), null_space
    else:
        return np.dot(v * w[None]**-0.5, v.T)


def block_lanczos(t, nmom):
    """Compute the on- and off-diagonal blocks from the moments.
    """

    nmo = t[0].shape[-1]

    m = np.zeros((nmom+1, nmo, nmo))
    b = np.zeros((nmom, nmo, nmo))
    s = np.zeros_like(t)
    c = defaultdict(lambda: np.zeros((nmo, nmo)))
    c[0, 0] = np.eye(nmo)

    bi = mat_isqrt(t[0])

    for i in range(len(t)):
        s[i] = np.linalg.multi_dot((bi, t[i], bi))

    m[0] = s[1]

    for i in range(nmom):
        # Compute B_{i}
        b2 = np.zeros((nmo, nmo))

        for j in range(i+2):
            for l in range(i+1):
                b2 += np.linalg.multi_dot((c[i, l].T, s[j+l+1], c[i, j-1]))

        b2 -= np.dot(m[i], m[i].T)
        if i:
            b2 -= np.dot(b[i-1], b[i-1].T).T

        b[i] = mat_sqrt(b2)
        binv, null = mat_isqrt(b2, check_null_space=True)

        # Compute C_{i,n}
        for j in range(i+2):
            tmp = (
                + c[i, j-1]
                - np.dot(c[i, j], m[i])
                - np.dot(c[i-1, j], b[i-1].T)
            )
            c[i+1, j] = np.dot(tmp, binv)

        for j in range(i+2):
            for l in range(i+2):
                m[i+1] += np.linalg.multi_dot((c[i+1, l].T, s[j+l+1], c[i+1, j]))

    return m, b


def build_block_tridiagonal(m, b):
    """Construct a block tridiagonal matrix from a list of on-diagonal
    and off-diagonal blocks.
    """

    z = np.zeros_like(m[0])

    h = np.block([[
        m[i]   if i == j   else
        b[j]   if j == i-1 else
        b[i].T if i == j-1 else z
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
        self.weight_tol = 1e-14
        self.e = None
        self.c = None
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
        log.info("chkfile = %s", self.chkfile)

    def _finalize(self):
        cpt = chempot.binsearch_chempot((self.e, self.c), self.nmo, self.nocc*2)[0]

        mask = np.linalg.norm(self.c[:self.nmo], axis=0)**2 > 0.01
        e, c = self.e[mask], self.c[:, mask]

        ips = e < cpt
        eas = e >= cpt

        logger.note(self, "Ionisation potentials:")
        logger.note(self, "  %4s %12s %12s" % ("root", "energy", "qpwt"))
        for nroot in range(min(5, np.sum(ips))):
            e_ip = -e[ips][-(nroot+1)]
            qpwt = np.linalg.norm(c[:self.nmo][:, ips][:, -(nroot+1)])**2
            logger.note(self, "  %4d %12.6f %12.6g" % (nroot, e_ip, qpwt))

        logger.note(self, "Electron affinity:")
        logger.note(self, "  %4s %12s %12s" % ("root", "energy", "qpwt"))
        for nroot in range(min(5, np.sum(eas))):
            e_ea = e[eas][nroot]
            qpwt = np.linalg.norm(c[:self.nmo][:, eas][:, nroot])**2
            logger.note(self, "%4d %12.6f %12.6g" % (nroot, e_ea, qpwt))

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

        cput0 = (logger.process_clock(), logger.perf_counter())

        eom = self.eomip_method()
        if imds is None:
            imds = self.make_imds(ea=False)
        diag = eom.get_diag(imds)
        matvec = lambda v: eom.l_matvec(v, imds, diag)

        if nmom is None:
            nmom = 2 * self.nmom[0] + 2
        moms = np.zeros((nmom, self.nmo, self.nmo))

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
                    moms[n, p, q] += np.dot(bra, ket)

                if (n+1) != nmom:
                    ket = matvec(ket)

        mpi_helper.barrier()
        moms = mpi_helper.allreduce(moms)

        moms = 0.5 * (moms + moms.swapaxes(1, 2))

        logger.timer(self, "IP-EOM-CCSD moments", *cput0)

        return moms

    def get_ea_moments(self, imds=None, nmom=None):
        """Get the moments of the EA-EOM-CCSD Green's function.
        """

        cput0 = (logger.process_clock(), logger.perf_counter())

        eom = self.eomea_method()
        if imds is None:
            imds = self.make_imds(ip=False)
        diag = eom.get_diag(imds)
        matvec = lambda v: eom.l_matvec(v, imds, diag)

        if nmom is None:
            nmom = 2 * self.nmom[1] + 2
        moms = np.zeros((nmom, self.nmo, self.nmo))

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
                    moms[n, p, q] += np.dot(bra, ket)

                if (n+1) != nmom:
                    ket = matvec(ket)

        mpi_helper.barrier()
        moms = mpi_helper.allreduce(moms)

        moms = 0.5 * (moms + moms.swapaxes(1, 2))

        logger.timer(self, "EA-EOM-CCSD moments", *cput0)

        return moms

    ##TODO: freeze p, q if frozen in self._cc
    #def get_ip_moments(self, imds=None, nmom=None):
    #    """Get the moments of the IP-EOM-CCSD Green's function.
    #    """

    #    cput0 = (logger.process_clock(), logger.perf_counter())

    #    eom = self.eomip_method()
    #    if imds is None:
    #        imds = self.make_imds(ea=False)
    #    diag = eom.get_diag(imds)
    #    matvec = lambda v: eom.matvec(v, imds, diag)

    #    if nmom is None:
    #        nmom = 2 * self.nmom[0] + 2
    #    moms = np.zeros((nmom, self.nmo, self.nmo))

    #    bras = [None] * self.nmo
    #    for p in range(self.nmo):
    #        r1, r2 = self.get_e_hole(p)
    #        bras[p] = eom.amplitudes_to_vector(r1, r2)

    #    for p in mpi_helper.nrange(self.nmo):
    #        r1, r2 = self.get_b_hole(p)
    #        ket = eom.amplitudes_to_vector(r1, r2)

    #        for n in range(nmom):
    #            for q in range(self.nmo):
    #                bra = bras[q]
    #                moms[n, p, q] += np.dot(bra, ket)

    #            if (n+1) != nmom:
    #                ket = matvec(ket)

    #    mpi_helper.barrier()
    #    moms = mpi_helper.allreduce(moms)

    #    moms = 0.5 * (moms + moms.swapaxes(1, 2))

    #    logger.timer(self, "IP-EOM-CCSD moments", *cput0)

    #    return moms

    #def get_ea_moments(self, imds=None, nmom=None):
    #    """Get the moments of the EA-EOM-CCSD Green's function.
    #    """

    #    cput0 = (logger.process_clock(), logger.perf_counter())

    #    eom = self.eomea_method()
    #    if imds is None:
    #        imds = self.make_imds(ip=False)
    #    diag = eom.get_diag(imds)
    #    matvec = lambda v: eom.matvec(v, imds, diag)

    #    if nmom is None:
    #        nmom = 2 * self.nmom[1] + 2
    #    moms = np.zeros((nmom, self.nmo, self.nmo))

    #    bras = [None] * self.nmo
    #    for p in range(self.nmo):
    #        r1, r2 = self.get_e_part(p)
    #        bras[p] = eom.amplitudes_to_vector(r1, r2)

    #    for p in mpi_helper.nrange(self.nmo):
    #        r1, r2 = self.get_b_part(p)
    #        ket = eom.amplitudes_to_vector(r1, r2)

    #        for n in range(nmom):
    #            for q in range(self.nmo):
    #                bra = bras[q]
    #                moms[n, p, q] += np.dot(bra, ket)

    #            if (n+1) != nmom:
    #                ket = matvec(ket)

    #    mpi_helper.barrier()
    #    moms = mpi_helper.allreduce(moms)

    #    moms = 0.5 * (moms + moms.swapaxes(1, 2))

    #    logger.timer(self, "EA-EOM-CCSD moments", *cput0)

    #    return moms

    def eigh_moments(self, t, nmom):
        """Block tridiagonalise the matrix under the constraint of
        conservation of a given number of moments, and diagonalise
        the block tridiagonal matrix.
        """

        m, b = block_lanczos(t, nmom)
        h_tri = build_block_tridiagonal(m, b)
        bi = mat_sqrt(t[0])

        e, u = np.linalg.eigh(h_tri)
        u = np.dot(bi.T.conj(), u[:self.nmo])

        return e, u

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

        dm1 = self.get_ip_moments(imds=imds, nmom=0)[0] * 2.0

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

        e, c = self.e, self.c = kernel(self, th=th, tp=tp, eris=eris)

        if self.chkfile is not None:
            self.dump_chk()

        self._finalize()

        return self.e, self.c

    def dump_chk(self, chkfile=None, key="gfccsd"):
        if chkfile is None:
            chkfile = self.chkfile
        lib.chkfile.dump(chkfile, "%s/e" % key, self.e)
        lib.chkfile.dump(chkfile, "%s/c" % key, self.c)
        lib.chkfile.dump(chkfile, "%s/nmom" % key, np.array(self.nmom))
        return self

    def update_from_chk_(self, chkfile=None, key="gfccsd"):
        if chkfile is None:
            chkfile = self.chkfile
        self.e = lib.chkfile.load(chkfile, "%s/e" % key)
        self.c = lib.chkfile.load(chkfile, "%s/c" % key)
        self.nmom = tuple(lib.chkfile.load(chkfile, "%s/nmom" % key))
        return self

    update = update_from_chk = update_from_chk_

    def ipccsd(self, nroots=5):
        """Print and return IPs.
        """

        cpt = chempot.binsearch_chempot((self.e, self.c), self.nmo, self.nocc*2)[0]
        ips = self.e < cpt

        e = list(-self.e[ips][::-1][:nroots])
        c = list(self.c[:, ips][:, ::-1][:, :nroots].T)

        nroots = max(nroots, len(e))

        for n, en, cn in zip(range(nroots), e, c):
            qpwt = np.linalg.norm(cn[:self.nmo])**2
            logger.note(self, "  %2s %2d %16.10g %0.6g" % ("IP", n, en, qpwt))

        if nroots == 1:
            return e[0], c[0]
        else:
            return e, c

    def eaccsd(self, nroots=5):
        """Print and return EAs.
        """

        cpt = chempot.binsearch_chempot((self.e, self.c), self.nmo, self.nocc*2)[0]
        eas = self.e >= cpt

        e = list(self.e[eas][:nroots])
        c = list(self.c[:, eas][:, :nroots].T)

        nroots = max(nroots, len(e))

        for n, en, cn in zip(range(nroots), e, c):
            qpwt = np.linalg.norm(cn[:self.nmo])**2
            logger.note(self, "  %2s %2d %16.10g %0.6g" % ("EA", n, en, qpwt))

        if nroots == 1:
            return e[0], c[0]
        else:
            return e, c

    @property
    def nmo(self):
        return self._cc.nmo

    @property
    def nocc(self):
        return self._cc.nocc

    @property
    def nvir(self):
        return self.nmo - self.nocc

    @property
    def gf(self):
        if self.e is None:
            return None
        cpt = chempot.binsearch_chempot((self.e, self.c), self.nmo, self.nocc*2)[0]
        return GreensFunction(self.e, self.c[:self.nmo], chempot=cpt)


class GFADC(GFCCSD):
    """Green's function solver for algebraic diagrammatic construction.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.t1 = self._adc.t1[0]
        self.t2 = self._adc.t2[0]

    @property
    def _adc(self):
        return self._cc

    def eomip_method(self):
        from pyscf.adc.radc import RADCIP
        eom = RADCIP(self._adc)
        eris = eom.transform_integrals()
        M = eom.get_imds(eris=eris)
        fake = lambda: None
        fake.t1, fake.t2 = self._adc.t1[0], self._adc.t2[0]
        fake.get_diag = lambda imds: eom.get_diag(M)
        fake.matvec = lambda v, *args: (eom.matvec(M, eris))(v)
        fake.amplitudes_to_vector = lambda r1, r2: np.concatenate([r1.ravel(), r2.ravel()], axis=0)
        return fake

    def eomea_method(self):
        from pyscf.adc.radc import RADCEA
        eom = RADCEA(self._adc)
        eris = eom.transform_integrals()
        M = eom.get_imds(eris=eris)
        fake = lambda: None
        fake.t1, fake.t2 = self._adc.t1[0], self._adc.t2[0]
        fake.get_diag = lambda imds: eom.get_diag(M)
        fake.matvec = lambda v, *args: (eom.matvec(M, eris))(v)
        fake.amplitudes_to_vector = lambda r1, r2: np.concatenate([r1.ravel(), r2.ravel()], axis=0)
        return fake

    def make_imds(self, eris=None, ip=True, ea=True):
        return None

    def kernel(self, t1=None, t2=None, l1=None, l2=None, eris=None):
        if self.verbose >= logger.WARN:
            self.check_sanity()
        self.dump_flags()

        e, c = self.e, self.c = kernel(self, eris=eris)

        if self.chkfile is not None:
            self.dump_chk()

        self._finalize()

        return self.e, self.c

    ip_adc = ipadc = GFCCSD.ipccsd
    ea_adc = eaadc = GFCCSD.eaccsd

    get_b_hole = get_b_hole
    get_b_part = get_b_part
    get_e_hole = get_b_hole
    get_e_part = get_b_part

    @property
    def nmo(self):
        return self._adc._nmo

    @property
    def nocc(self):
        return self._adc._nocc



if __name__ == "__main__":
    from pyscf import gto, scf, cc

    mol = gto.M(atom="O 0 0 0.11779; H 0 0.755453 -0.471161; H 0 -0.755453 -0.471161", basis="6-31g", verbose=5)
    mf = scf.RHF(mol).run()
    ccsd = cc.CCSD(mf).run()
    ccsd.solve_lambda()

    gfcc = GFCCSD(ccsd, nmom=(5, 5))
    gfcc.kernel()

    ip1, vip1 = ccsd.ipccsd(nroots=8)
    ip2, vip2 = gfcc.ipccsd(nroots=8)

    ea1, vea1 = ccsd.eaccsd(nroots=8)
    ea2, vea2 = gfcc.eaccsd(nroots=8)

    print(np.abs(ip1[0]-ip2[0]))
    print(np.abs(ip1[1]-ip2[1]))
    print(np.abs(ip1[2]-ip2[2]))
    print(np.abs(ea1[0]-ea2[0]))
    print(np.abs(ea1[1]-ea2[1]))
    print(np.abs(ea1[2]-ea2[2]))

