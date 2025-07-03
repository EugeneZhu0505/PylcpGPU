
import copy
import numpy as np
import cupy as cp
from cupyx.scipy import sparse

from ivpCupy import solve_ivp
from scipy.interpolate import interp1d
from .fields import magField as magFieldObject
from .fields import laserBeams as laserBeamsObject
from .common import ProgressBar


def spherical2cart(A):

    return cp.array(
        [1 / cp.sqrt(2) * (-A[2] + A[0]),
         1j / cp.sqrt(2) * (A[2] + A[0]), A[1]]
    )

    return np.array([1/np.sqrt(2)*(-A[2]+A[0]), 1j/np.sqrt(2)*(A[2]+A[0]), A[1]])

def cart2spherical(A):

    return cp.array(
        [(A[0] - 1j * A[1]) / cp.sqrt(2), A[2], 
         -(A[0] + 1j * A[1]) / cp.sqrt(2)]
    )

    return np.array([(A[0]-1j*A[1])/np.sqrt(2), A[2], -(A[0]+1j*A[1])/np.sqrt(2)])


def cartesian_vector_tensor_dot(a, B):

    if B.ndim == 2 and a.ndim == 1:
        return cp.dot(B, a)
    elif B.ndim == 2:
        return cp.sum(a[cp.newaxis, ...] * B[..., cp.newaxis], axis=1)
    else:
        return cp.sum(a[cp.newaxis, ...] * B[...], axis=1)


class Obe(object):

    def __init__(
        self, laserBeams, magField, hamiltonian=None,
        a=cp.array([0., 0., 0.]), r0=cp.array([0., 0., 0.]),
        v0=cp.array([0., 0., 0.]), rho0=cp.random.randn(24, ),
        transformIntoReIm=True, useSparseMatrices=None, 
        includeMagForces=True, recordForce=False,
    ) -> None:

        self.r0 = r0
        self.v0 = v0
        self.rho0 = rho0
        self.numAtom = self.r0.shape[-1]

        self.laserBeams = {} 
        if isinstance(laserBeams, list):
            self.laserBeams['g->e'] = copy.copy(laserBeamsObject(laserBeams))
        elif isinstance(laserBeams, laserBeamsObject):
            self.laserBeams['g->e'] = copy.copy(laserBeams)
        elif isinstance(laserBeams, dict):
            for key in laserBeams.keys():
                if not isinstance(laserBeams[key], laserBeamsObject):
                    raise TypeError('Key %s in dictionary lasersBeams ' % key +
                                     'is in not of type laserBeams.')
            self.laserBeams = copy.copy(laserBeams)
        else:
            raise TypeError('laserBeams is not a valid type.')

        if callable(magField) or isinstance(magField, cp.ndarray):
            self.magField = magFieldObject(magField)
        elif isinstance(magField, magFieldObject):
            self.magField = copy.copy(magField)
        else:
            raise TypeError('The magnetic field must be either a lambda ' +
                            'function or a magField object.')
        
        if hamiltonian is not None:
            self.hamiltonian = copy.copy(hamiltonian)
            self.hamiltonian.make_full_matrices()
            self.__check_consistency_in_lasers_and_d_q()

        if not isinstance(a, cp.ndarray):
            raise TypeError('Constant acceleration must be an cupy array.')
        elif a.size != 3:
            raise ValueError('Constant acceleration must have length 3.')
        else:
            self.constant_accel = a

        self.transformIntoReIm = transformIntoReIm
        self.includeMagForces = includeMagForces
        self.recordForce = recordForce

        if useSparseMatrices is None:
            if self.hamiltonian.n > 10:
                self.useSparseMatrices = True
            else:
                self.useSparseMatrices = False
        else:
            self.useSparseMatrices = useSparseMatrices

        self.evMat = {}
        self.__build_decay_ev()
        self.__build_coherent_ev()

        if self.transformIntoReIm:
            self.__transform_ev_matrices()

        if self.useSparseMatrices:
            self.__convert_to_sparse()

        pass

    def __check_consistency_in_lasers_and_d_q(self):
        for laserKey in self.laserBeams.keys():
            if not laserKey in self.hamiltonian.laser_keys.keys():
                raise ValueError('laserBeams dictionary keys %s ' % laserKey +
                                 'does not have a corresponding key the '+
                                 'Hamiltonian d_q.')
        
    def __density_index(self, i, j):
        return i * self.hamiltonian.n + j
    
    def __build_coherent_ev_submatrix(self, H):
        
        evMat = cp.zeros(
            (self.hamiltonian.n**2, self.hamiltonian.n**2),
            dtype=cp.complex128
        )

        for i in range(self.hamiltonian.n):
            for j in range(self.hamiltonian.n):
                for k in range(self.hamiltonian.n):
                    evMat[self.__density_index(i, j), self.__density_index(i, k)] += 1j * H[k, j]
                    evMat[self.__density_index(i, j), self.__density_index(k, j)] -= 1j * H[i, k]

        return evMat
        
    def __build_coherent_ev(self):

        self.evMat['H0'] = self.__build_coherent_ev_submatrix(
            self.hamiltonian.H_0
        )

        self.evMat['B'] = [None] * 3

        for q in range(3):
            self.evMat['B'][q] = self.__build_coherent_ev_submatrix(
                self.hamiltonian.mu_q[q]
            )
        self.evMat['B'] = cp.array(self.evMat['B'])

        self.evMat['d_q'] = {}
        self.evMat['d_q*'] = {}
        for key in self.laserBeams.keys():
            gamma = self.hamiltonian.blocks[self.hamiltonian.laser_keys[key]].parameters['gamma']
            self.evMat['d_q'][key] = [None] * 3
            self.evMat['d_q*'][key] = [None] * 3
            for q in range(3):
                self.evMat['d_q'][key][q] = self.__build_coherent_ev_submatrix(
                    gamma * self.hamiltonian.d_q_bare[key][q] / 4.
                )
                self.evMat['d_q*'][key][q] = self.__build_coherent_ev_submatrix(
                    gamma * self.hamiltonian.d_q_star[key][q] / 4.
                )
            self.evMat['d_q'][key] = cp.array(self.evMat['d_q'][key])
            self.evMat['d_q*'][key] = cp.array(self.evMat['d_q*'][key])

    def __build_decay_ev(self):

        d_q_bare = {}
        d_q_star = {}
        for key in self.laserBeams.keys():
            d_q_bare[key] = cp.asarray(self.hamiltonian.d_q_bare[key])
            d_q_star[key] = cp.array(self.hamiltonian.d_q_star[key])

        self.decayRates = {}
        self.decayRatesTruncated = {}
        self.decayRhoIndices = {}
        self.recoilVelocity = {}

        self.evMat['decay'] = cp.zeros(
            (self.hamiltonian.n ** 2, self.hamiltonian.n ** 2),
            dtype=cp.complex128
        )

        for key in d_q_bare:
            evMat = cp.zeros(
                (self.hamiltonian.n ** 2, self.hamiltonian.n ** 2),
                dtype=cp.complex128
            )
            gamma = self.hamiltonian.blocks[self.hamiltonian.laser_keys[key]].parameters['gamma']

            for i in range(self.hamiltonian.n):
                for j in range(self.hamiltonian.n):
                    for k in range(self.hamiltonian.n):
                        for l in range(self.hamiltonian.n):
                            for m, q in enumerate(np.arange(-1., 2., 1)):
                                evMat[self.__density_index(i, j), self.__density_index(l, j)] -= \
                                    d_q_star[key][m, l, k] * d_q_bare[key][m, k, i]
                                evMat[self.__density_index(i, j), self.__density_index(k, l)] += \
                                    d_q_star[key][m, k, i] * d_q_bare[key][m, j, l]
                                evMat[self.__density_index(i, j), self.__density_index(l, k)] += \
                                    d_q_star[key][m, l, i] * d_q_bare[key][m, j, k]
                                evMat[self.__density_index(i, j), self.__density_index(i, l)] -= \
                                    d_q_star[key][m, j, k] * d_q_bare[key][m, k, l]
            
            evMat = 0.5 * gamma * evMat

            self.decayRates[key] = -cp.real(
                cp.array(
                    [evMat[self.__density_index(i, i), self.__density_index(i, i)] 
                     for i in range(self.hamiltonian.n)]
                )
            )

            self.decayRatesTruncated[key] = self.decayRates[key][self.decayRates[key] > 0]
            self.decayRhoIndices[key] = cp.array(
                [self.__density_index(i, i)
                 for i, rate in enumerate(self.decayRates[key]) if rate > 0]
            )
            self.recoilVelocity[key] = self.hamiltonian.blocks[self.hamiltonian.laser_keys[key]].parameters['k'] / self.hamiltonian.mass

            self.evMat['decay'] += evMat
        
        return self.evMat['decay']
                            

    def __build_transform_matrices(self):

        self.U = cp.zeros(
            (self.hamiltonian.n**2, self.hamiltonian.n**2),
            dtype=cp.complex128
        )
        self.Uinv = cp.zeros(
            (self.hamiltonian.n**2, self.hamiltonian.n**2),
            dtype=cp.complex128
        )

        for i in range(self.hamiltonian.n):
            self.U[self.__density_index(i, i), self.__density_index(i, i)] = 1.
            self.Uinv[self.__density_index(i, i), self.__density_index(i, i)] = 1.

        for i in range(self.hamiltonian.n):
            for j in range(i+1, self.hamiltonian.n):
                self.U[self.__density_index(i, j), self.__density_index(i, j)] = 1.
                self.U[self.__density_index(i, j), self.__density_index(j, i)] = 1j

                self.U[self.__density_index(j, i), self.__density_index(i, j)] = 1.
                self.U[self.__density_index(j, i), self.__density_index(j, i)] = -1j

        for i in range(self.hamiltonian.n):
            for j in range(i+1, self.hamiltonian.n):
                self.Uinv[self.__density_index(i, j), self.__density_index(i, j)] = 0.5
                self.Uinv[self.__density_index(i, j), self.__density_index(j, i)] = 0.5

                self.Uinv[self.__density_index(j, i), self.__density_index(i, j)] = -0.5*1j
                self.Uinv[self.__density_index(j, i), self.__density_index(j, i)] = +0.5*1j

    def __transform_ev_matrix(self, evMat):

        if not hasattr(self, 'U'):
            self.__build_transform_matrices()
        
        evMatNew = self.Uinv @ evMat @ self.U

        if cp.allclose(cp.imag(evMatNew), 0.):
            return cp.real(evMatNew)
        else:
            raise ValueError('Something went dreadfully wrong.')

    def __transform_ev_matrices(self):

        self.evMat['decay'] = self.__transform_ev_matrix(self.evMat['decay'])
        self.evMat['H0'] = self.__transform_ev_matrix(self.evMat['H0'])

        self.evMat['reE'] = {}
        self.evMat['imE'] = {}

        for key in self.evMat['d_q'].keys():
            self.evMat['reE'][key] = cp.array(
                [self.__transform_ev_matrix(self.evMat['d_q'][key][j] + self.evMat['d_q*'][key][j])
                 for j in range(3)]
            )
            self.evMat['imE'][key] = cp.array(
                [self.__transform_ev_matrix(1j * (self.evMat['d_q'][key][j] - self.evMat['d_q*'][key][j]))
                 for j in range(3)]
            )
        
        self.evMat['B'] = spherical2cart(self.evMat['B'])

        for j in range(3):
            self.evMat['B'][j] = self.__transform_ev_matrix(
                self.evMat['B'][j]
            )
        self.evMat['B'] = cp.real(self.evMat['B'])

        del self.evMat['d_q']
        del self.evMat['d_q*']

    def __convert_based_on_shape(self, matrix):
        if matrix.shape == (3, self.hamiltonian.n**2, self.hamiltonian.n**2):
            newList = [None] * 3
            for j in range(3):
                newList[j] = sparse.csr_matrix(matrix[j])
            return newList
        else:
            return sparse.csr_matrix(matrix)
        
    def __convert_to_sparse(self):

        for key in self.evMat:
            if isinstance(self.evMat[key], dict):
                for subKey in self.evMat[key]:
                    self.evMat[key][subKey] = self.__convert_based_on_shape(
                        self.evMat[key][subKey]
                    )
            else:
                self.evMat[key] = self.__convert_based_on_shape(
                    self.evMat[key]
                )

    def __reshape_rho(self, rho):
        # rho shape (576, N)
        if self.transformIntoReIm:
            rho = rho.astype(cp.complex128)
            rho = self.U @ rho
        
        rho = rho.reshape(self.hamiltonian.n, self.hamiltonian.n, -1) # shape (24, 24, N)

        # return rho

        return rho
    
    def __reshape_sol(self, sol):
        sol.y = sol.y.reshape((-1, self.numAtom, sol.t.shape[0])).transpose(2, 0, 1) # (t, 582, N)
        if self.transformIntoReIm:
            sol.rho = sol.y[:, :-6].astype(cp.complex128) # (t, 576, N)
            sol.rho = cp.einsum("ij,tjk->tik", self.U, sol.rho)
        sol.rho = sol.rho.reshape(sol.t.shape[0], self.hamiltonian.n, self.hamiltonian.n, -1) # shape (24, 24, N)
        sol.r = cp.real(sol.y[:, -3:])
        sol.v = cp.real(sol.y[:, -6:-3])
        del sol.y
        return sol

    
    def __drhodt(self, t, r, rho):
        
        drhodt = (self.evMat['decay'] @ rho) + (self.evMat['H0'] @ rho)

        for key in self.laserBeams.keys():
            if self.transformIntoReIm:
                Eq = self.laserBeams[key].total_electric_field(r, t)
                Eq[cp.abs(Eq) < 1e-10] = 0
                for i, q in enumerate(np.arange(-1., 2., 1)):
                    drhodt -= (
                        (-1)**q * cp.real(Eq[2-i]) * (self.evMat['reE'][key][i] @ rho)
                    )
                    drhodt -= (
                        (-1)**q * cp.imag(Eq[2-i]) * (self.evMat['imE'][key][i] @ rho)
                    )
            else:
                Eq = self.laserBeams[key].total_electric_field(cp.real(r), t)
                Eq[cp.abs(Eq) < 1e-10] = 0
                for i, q in enumerate(np.arange(-1., 2., 1)):
                    drhodt -= (
                        (-1)**q * cp.real(Eq[2-i]) * (self.evMat['d_q'][key][i] @ rho)
                    )
                    drhodt -= (
                        (-1)**q * cp.imag(Eq[2-i]) * (self.evMat['d_q*'][key][i] @ rho)
                    )

        B = self.magField.Field(r, t)
        B[cp.abs(B) < 1e-10] = 0

        for i in np.arange(B.shape[0]):
            if self.transformIntoReIm:
                drhodt -= self.evMat['B'][i] @ rho * B[i]
            else:
                Bq = cart2spherical(B)
                Bq[cp.abs(Bq) < 1e-10] = 0
                drhodt -= self.evMat['B'][i] @ rho * cp.conjugate(Bq[i])
                
        return drhodt
    
    def observable(self, O, rho):

        avO = cp.einsum('ijk,jkl->il', O, rho)
        if cp.allclose(cp.imag(avO), 0):
            return cp.real(avO)
        else:
            return avO

    
    def force(self, r, t, rho):

        if rho.shape[0] != self.hamiltonian.n:
            rho = self.__reshape_rho(rho)

        f = cp.zeros((3,) + rho.shape[2:])

        if self.recordForce:
            f_laser_q = {}
            f_laser = {}
        for key in self.laserBeams:
            d_q_bare = self.hamiltonian.d_q_bare[key]
            mu_q_av = self.observable(
                d_q_bare, rho
            )

            gamma = self.hamiltonian.blocks[self.hamiltonian.laser_keys[key]].parameters['gamma']
            
            if not self.recordForce:
                delE = self.laserBeams[key].total_electric_field_gradient(r, t)
                for j, q in enumerate(np.arange(-1., 2., 1.)):
                    f += cp.real(
                        (-1) ** q * gamma * mu_q_av[j] * delE[:, 2 - j]
                    ) / 2
            else:
                f_laser_q[key] = cp.zeros(
                    (3, 3, self.laserBeams[key].num_of_beams) + rho.shape[2:]
                )

                f_laser[key] = cp.zeros(
                    (3, self.laserBeams[key].num_of_beams) + rho.shape[2:]
                )
                for ii, beam in enumerate(self.laserBeams[key].beam_vector):

                    delE = beam.electric_field_gradient(r, t)
                    for jj, q in enumerate(np.arange(-1., 2., 1.)):

                        f_laser_q[key][:, jj, ii] += cp.real(
                            (-1) ** q * gamma * mu_q_av[jj] * delE[:, 2 - jj]
                        ) / 2

                    f_laser[key][:, ii] = cp.sum(f_laser_q[key][:, :, ii], axis=1)

                f = f + cp.sum(f_laser[key], axis=1)

        if self.includeMagForces:
            delB = self.magField.gradField(r)
            mu = cp.asarray(self.hamiltonian.mu)
            av_mu = self.observable(mu, rho)
            f_mag = cartesian_vector_tensor_dot(av_mu, delB)

            f = f + f_mag  

        return f
    
    def evolve_motion(
            self, tSpan, tEval, freezeAxis=cp.array([False, False, False]), 
            randomRecoilFlag=False, maxScatterProbability=0.1,
            progressBarFlag=False, 
    ):
        
        freeAxes = cp.bitwise_not(freezeAxis)

        if progressBarFlag:
            progressBar = ProgressBar()

        # 记录求解全过程中的力
        if self.recordForce:
            ts = []
            Fs = []

        # 定义导数
        def dydt(t, y):

            y = y.reshape(-1, self.numAtom)
            t = t.item() if isinstance(t, cp.ndarray) else t

            # 更新进度条
            if progressBarFlag:
                progressBar.update(t / tSpan[1].item())
            
            if self.recordForce:
                # 将 force 求解的力记录下来
                F = self.force(y[-3:], t, y[:-6])
                ts.append(t)
                Fs.append(F)
                F = F[0]
            else:
                F = self.force(y[-3:], t, y[:-6])

            # 态密度导数
            drhodt = self.__drhodt(t=t, r=y[-3:], rho=y[:-6])

            # 速度导数
            dvdt = freeAxes[:, np.newaxis] * F / self.hamiltonian.mass + self.constant_accel[:, cp.newaxis]

            # 位置导数
            drdt = y[-6:-3]

            return cp.concatenate((drhodt, dvdt, drdt)).flatten()


        sol = solve_ivp(
            fun=dydt,
            t_span=tSpan,
            t_eval=tEval,
            y0=cp.concatenate((self.rho0, self.v0, self.r0)).flatten(),
        )

        if progressBarFlag:
            progressBar.update(1.)

        sol = self.__reshape_sol(sol=sol)

        if self.recordForce:
            f = interp1d(ts[:-1], cp.array([f[0] for f in Fs[:-1]]).T)
            sol.F = f(sol.t)

            f = interp1d(ts[:-1], cp.array([f[3] for f in Fs[:-1]]).T)
            sol.fMag = f(sol.t)

            sol.f = {}
            for key in Fs[0][1]:
                f = interp1d(ts[:-1], cp.array([f[1][key] for f in Fs[:-1]]).T)
                sol.f[key] = f(sol.t)
                sol.f[key] = cp.swapaxes(sol.f[key], 0, 1)

        return sol

    