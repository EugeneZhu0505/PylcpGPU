import numpy as np
import numba
import scipy.sparse as sparse
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from .integration_tools import solve_ivp_random
from .common import (ProgressBar, random_vector,
                     cart2spherical, spherical2cart)
from .governingeq import governingeq


@numba.vectorize([numba.float64(numba.complex128), numba.float32(numba.complex64)])
def abs2(x):
    return x.real**2 + x.imag**2


@numba.jit(nopython=True)
def dot(A, x):
    return A @ x


@numba.jit(nopython=True)
def dot_and_add(A, x, b):
    b += A @ x


def cartesian_vector_tensor_dot(a, B):
    if B.ndim == 2 and a.ndim == 1:
        return np.dot(B, a)
    elif B.ndim == 2:
        return np.sum(a[np.newaxis, ...]*B[..., np.newaxis], axis=1)
    else:
        return np.sum(a[np.newaxis, ...]*B[...], axis=1)


class obe(governingeq):

    def __init__(self, laserBeams, magField, hamitlonian,
                 a=np.array([0., 0., 0.]), transform_into_re_im=True,
                 use_sparse_matrices=None, include_mag_forces=True,
                 r0=np.array([0., 0., 0.]), v0=np.array([0., 0., 0.])):

        super().__init__(laserBeams, magField, hamitlonian, a=a,
                         r0=r0, v0=v0)

        self.transform_into_re_im = transform_into_re_im
        self.include_mag_forces = include_mag_forces

        if use_sparse_matrices is None:
            if self.hamiltonian.n > 10:
                self.use_sparse_matrices = True
            else:
                self.use_sparse_matrices = False
        else:
            self.use_sparse_matrices = use_sparse_matrices

        self.profile = {}

        self.sol = None

        self.ev_mat = {}
        self.__build_decay_ev()
        self.__build_coherent_ev()

        if self.transform_into_re_im:
            self.__transform_ev_matrices()

        if self.use_sparse_matrices:
            self.__convert_to_sparse()

    def __density_index(self, ii, jj):
        return ii + jj*self.hamiltonian.n

    def __build_coherent_ev_submatrix(self, H):
        ev_mat = np.zeros((self.hamiltonian.n**2, self.hamiltonian.n**2),
                          dtype='complex128')

        for ii in range(self.hamiltonian.n):
            for jj in range(self.hamiltonian.n):
                for kk in range(self.hamiltonian.n):
                    ev_mat[self.__density_index(ii, jj),
                           self.__density_index(ii, kk)] += 1j*H[kk, jj]
                    ev_mat[self.__density_index(ii, jj),
                           self.__density_index(kk, jj)] -= 1j*H[ii, kk]

        return ev_mat

    def __build_coherent_ev(self):
        self.ev_mat['H0'] = self.__build_coherent_ev_submatrix(
            self.hamiltonian.H_0
        )

        self.ev_mat['B'] = [None]*3
        for q in range(3):
            self.ev_mat['B'][q] = self.__build_coherent_ev_submatrix(
                self.hamiltonian.mu_q[q]
            )
        self.ev_mat['B'] = np.array(self.ev_mat['B'])

        self.ev_mat['d_q'] = {}
        self.ev_mat['d_q*'] = {}
        for key in self.laserBeams.keys():
            gamma = self.hamiltonian.blocks[self.hamiltonian.laser_keys[key]
                                            ].parameters['gamma']
            self.ev_mat['d_q'][key] = [None]*3
            self.ev_mat['d_q*'][key] = [None]*3
            for q in range(3):
                self.ev_mat['d_q'][key][q] = self.__build_coherent_ev_submatrix(
                    gamma*self.hamiltonian.d_q_bare[key][q]/4.
                )
                self.ev_mat['d_q*'][key][q] = self.__build_coherent_ev_submatrix(
                    gamma*self.hamiltonian.d_q_star[key][q]/4.
                )
            self.ev_mat['d_q'][key] = np.array(self.ev_mat['d_q'][key])
            self.ev_mat['d_q*'][key] = np.array(self.ev_mat['d_q*'][key])

    def __build_decay_ev(self):
        d_q_bare = self.hamiltonian.d_q_bare
        d_q_star = self.hamiltonian.d_q_star

        self.decay_rates = {}
        self.decay_rates_truncated = {}
        self.decay_rho_indices = {}
        self.recoil_velocity = {}

        self.ev_mat['decay'] = np.zeros((self.hamiltonian.n**2,
                                         self.hamiltonian.n**2),
                                        dtype='complex128')

        for key in d_q_bare:
            ev_mat = np.zeros((self.hamiltonian.n**2, self.hamiltonian.n**2),
                              dtype='complex128')
            gamma = self.hamiltonian.blocks[self.hamiltonian.laser_keys[key]
                                            ].parameters['gamma']

            for ii in range(self.hamiltonian.n):
                for jj in range(self.hamiltonian.n):
                    for kk in range(self.hamiltonian.n):
                        for ll in range(self.hamiltonian.n):
                            for mm, q in enumerate(np.arange(-1., 2., 1)):
                                ev_mat[self.__density_index(ii, jj),
                                       self.__density_index(ll, jj)] -= \
                                    d_q_star[key][mm, ll, kk] * \
                                    d_q_bare[key][mm, kk, ii]
                                ev_mat[self.__density_index(ii, jj),
                                       self.__density_index(kk, ll)] += \
                                    d_q_star[key][mm, kk, ii] * \
                                    d_q_bare[key][mm, jj, ll]

                                ev_mat[self.__density_index(ii, jj),
                                       self.__density_index(ll, kk)] += \
                                    d_q_star[key][mm, ll, ii] * \
                                    d_q_bare[key][mm, jj, kk]
                                ev_mat[self.__density_index(ii, jj),
                                       self.__density_index(ii, ll)] -= \
                                    d_q_star[key][mm, jj, kk] * \
                                    d_q_bare[key][mm, kk, ll]

            ev_mat = 0.5*gamma*ev_mat

            self.decay_rates[key] = -np.real(np.array(
                [ev_mat[self.__density_index(ii, ii), self.__density_index(ii, ii)]
                 for ii in range(self.hamiltonian.n)]
            ))

            self.decay_rates_truncated[key] = self.decay_rates[key][self.decay_rates[key] > 0]
            self.decay_rho_indices[key] = np.array([self.__density_index(ii, ii)
                                                    for ii, rate in enumerate(self.decay_rates[key]) if rate > 0]
                                                   )
            self.recoil_velocity[key] = \
                self.hamiltonian.blocks[self.hamiltonian.laser_keys[key]].parameters['k']\
                / self.hamiltonian.mass

            self.ev_mat['decay'] += ev_mat

        return self.ev_mat['decay']

    def __build_transform_matrices(self):
        self.U = np.zeros((self.hamiltonian.n**2, self.hamiltonian.n**2),
                          dtype='complex128')
        self.Uinv = np.zeros((self.hamiltonian.n**2, self.hamiltonian.n**2),
                             dtype='complex128')

        for ii in range(self.hamiltonian.n):
            self.U[self.__density_index(ii, ii),
                   self.__density_index(ii, ii)] = 1.
            self.Uinv[self.__density_index(ii, ii),
                      self.__density_index(ii, ii)] = 1.

        for ii in range(self.hamiltonian.n):
            for jj in range(ii+1, self.hamiltonian.n):
                self.U[self.__density_index(ii, jj),
                       self.__density_index(ii, jj)] = 1.
                self.U[self.__density_index(ii, jj),
                       self.__density_index(jj, ii)] = 1j

                self.U[self.__density_index(jj, ii),
                       self.__density_index(ii, jj)] = 1.
                self.U[self.__density_index(jj, ii),
                       self.__density_index(jj, ii)] = -1j

        for ii in range(self.hamiltonian.n):
            for jj in range(ii+1, self.hamiltonian.n):
                self.Uinv[self.__density_index(ii, jj),
                          self.__density_index(ii, jj)] = 0.5
                self.Uinv[self.__density_index(ii, jj),
                          self.__density_index(jj, ii)] = 0.5

                self.Uinv[self.__density_index(jj, ii),
                          self.__density_index(ii, jj)] = -0.5*1j
                self.Uinv[self.__density_index(jj, ii),
                          self.__density_index(jj, ii)] = +0.5*1j

    def __transform_ev_matrix(self, ev_mat):
        if not hasattr(self, 'U'):
            self.__build_transform_matrices()

        ev_mat_new = self.Uinv @ ev_mat @ self.U

        if np.allclose(np.imag(ev_mat_new), 0):
            return np.real(ev_mat_new)
        else:
            raise ValueError('Something went dreadfully wrong.')

    def __transform_ev_matrices(self):
        self.ev_mat['decay'] = self.__transform_ev_matrix(self.ev_mat['decay'])
        self.ev_mat['H0'] = self.__transform_ev_matrix(self.ev_mat['H0'])

        self.ev_mat['reE'] = {}
        self.ev_mat['imE'] = {}
        for key in self.ev_mat['d_q'].keys():
            self.ev_mat['reE'][key] = np.array([self.__transform_ev_matrix(
                self.ev_mat['d_q'][key][jj] + self.ev_mat['d_q*'][key][jj]
            ) for jj in range(3)])
            self.ev_mat['imE'][key] = np.array([self.__transform_ev_matrix(
                1j*(self.ev_mat['d_q'][key][jj] - self.ev_mat['d_q*'][key][jj])
            ) for jj in range(3)])

        self.ev_mat['B'] = spherical2cart(self.ev_mat['B'])

        for jj in range(3):
            self.ev_mat['B'][jj] = self.__transform_ev_matrix(
                self.ev_mat['B'][jj])
        self.ev_mat['B'] = np.real(self.ev_mat['B'])

        del self.ev_mat['d_q']
        del self.ev_mat['d_q*']

    def __convert_to_sparse(self):
        def convert_based_on_shape(matrix):
            if matrix.shape == (3, self.hamiltonian.n**2, self.hamiltonian.n**2):
                new_list = [None]*3
                for jj in range(3):
                    new_list[jj] = sparse.csr_matrix(matrix[jj])

                return new_list
            else:
                return sparse.csr_matrix(matrix)

        for key in self.ev_mat:
            if isinstance(self.ev_mat[key], dict):
                for subkey in self.ev_mat[key]:
                    self.ev_mat[key][subkey] = convert_based_on_shape(
                        self.ev_mat[key][subkey]
                    )
            else:
                self.ev_mat[key] = convert_based_on_shape(self.ev_mat[key])

    def __reshape_rho(self, rho):
        if self.transform_into_re_im:
            rho = rho.astype('complex128')

            if len(rho.shape) == 1:
                rho = self.U @ rho
            else:
                for jj in range(rho.shape[1]):
                    rho[:, jj] = self.U @ rho[:, jj]

        rho = rho.reshape((self.hamiltonian.n, self.hamiltonian.n) +
                          rho.shape[1:])

        return rho

    def __reshape_sol(self):
        self.sol.rho = self.__reshape_rho(self.sol.y[:-6])
        self.sol.r = np.real(self.sol.y[-3:])
        self.sol.v = np.real(self.sol.y[-6:-3])

        del self.sol.y

    def set_initial_rho(self, rho0):
        if np.any(np.isnan(rho0)) or np.any(np.isinf(rho0)):
            raise ValueError('rho0 has NaNs or Infs!')

        if rho0.size != self.hamiltonian.n**2:
            raise ValueError('rho0 should have n^2 elements.')

        if rho0.shape == (self.hamiltonian.n, self.hamiltonian.n):
            rho0 = rho0.flatten()

        if self.transform_into_re_im and rho0.dtype is np.dtype('complex128'):
            self.rho0 = self.Uinv @ rho0
        elif (not self.transform_into_re_im and
              not rho0.dtype is np.dtype('complex128')):
            self.rho0 = rho0.astype('complex128')
        else:
            self.rho0 = rho0

    def __drhodt(self, r, t, rho):
        drhodt = (self.ev_mat['decay'] @ rho) + (self.ev_mat['H0'] @ rho)

        for key in self.laserBeams.keys():
            if self.transform_into_re_im:
                Eq = self.laserBeams[key].total_electric_field(r, t)
                for ii, q in enumerate(np.arange(-1., 2., 1)):
                    if np.abs(Eq[2-ii]) > 1e-10:
                        drhodt -= ((-1.)**q*np.real(Eq[2-ii]) *
                                   (self.ev_mat['reE'][key][ii] @ rho))
                        drhodt -= ((-1.)**q*np.imag(Eq[2-ii]) *
                                   (self.ev_mat['imE'][key][ii] @ rho))
            else:
                Eq = self.laserBeams[key].total_electric_field(np.real(r), t)
                for ii, q in enumerate(np.arange(-1., 2., 1)):
                    if np.abs(Eq[2-ii]) > 1e-10:
                        drhodt -= ((-1.)**q*Eq[2-ii] *
                                   (self.ev_mat['d_q'][key][ii] @ rho))
                        drhodt -= ((-1.)**q*np.conjugate(Eq[2-ii]) *
                                   (self.ev_mat['d_q*'][key][ii] @ rho))

        B = self.magField.Field(r, t)
        for ii, q in enumerate(range(-1, 2)):
            if self.transform_into_re_im:
                if np.abs(B[ii]) > 1e-10:
                    drhodt -= self.ev_mat['B'][ii]*B[ii] @ rho
            else:
                Bq = cart2spherical(B)
                if np.abs(Bq[ii]) > 1e-10:
                    drhodt -= self.ev_mat['B'][ii]*np.conjugate(Bq[ii]) @ rho

        return drhodt

    def evolve_motion(self, t_span, freeze_axis=[False, False, False],
                      random_recoil=False, max_scatter_probability=0.1,
                      progress_bar=False, record_force=False,
                      rng=np.random.default_rng(), **kwargs):
        free_axes = np.bitwise_not(freeze_axis)
        random_recoil_flag = random_recoil

        if progress_bar:
            progress = progressBar()

        if record_force:
            ts = []
            Fs = []

        def dydt(t, y):
            if progress_bar:
                progress.update(t/t_span[1])

            if record_force:
                F = self.force(y[-3:], t, y[:-6], return_details=True)

                ts.append(t)
                Fs.append(F)

                F = F[0]
            else:
                F = self.force(y[-3:], t, y[:-6], return_details=False)

            return np.concatenate((
                self.__drhodt(y[-3:], t, y[:-6]),
                F*free_axes/self.hamiltonian.mass + self.constant_accel,
                y[-6:-3]
            ))

        def random_recoil(t, y, dt):
            num_of_scatters = 0
            total_P = 0.

            for key in self.decay_rates:
                P = dt*self.decay_rates_truncated[key] * \
                    np.real(y[self.decay_rho_indices[key]])

                dice = rng.random(len(P))

                for ii in range(np.sum(dice < P)):
                    num_of_scatters += 1
                    y[-6:-3] += self.recoil_velocity[key]*(random_vector(rng, free_axes) +
                                                           random_vector(rng, free_axes))

                total_P += np.sum(P)

            new_dt_max = (max_scatter_probability/total_P)*dt

            return (num_of_scatters, new_dt_max)

        if not random_recoil_flag:
            self.sol = solve_ivp(
                dydt, t_span, np.concatenate((self.rho0, self.v0, self.r0)),
                **kwargs)
        else:
            self.sol = solve_ivp_random(
                dydt, random_recoil, t_span,
                np.concatenate((self.rho0, self.v0, self.r0)),
                **kwargs
            )

        if progress_bar:
            progress.update(1.)

        self.__reshape_sol()

        if record_force:
            f = interp1d(ts[:-1], np.array([f[0] for f in Fs[:-1]]).T)
            self.sol.F = f(self.sol.t)

            f = interp1d(ts[:-1], np.array([f[3] for f in Fs[:-1]]).T)
            self.sol.fmag = f(self.sol.t)

            self.sol.f = {}
            for key in Fs[0][1]:
                f = interp1d(ts[:-1], np.array([f[1][key] for f in Fs[:-1]]).T)
                self.sol.f[key] = f(self.sol.t)
                self.sol.f[key] = np.swapaxes(self.sol.f[key], 0, 1)

        return self.sol

    def observable(self, O, rho=None):
        if rho is None:
            rho = self.sol.rho

        if rho.shape[:2] != (self.hamiltonian.n, self.hamiltonian.n):
            raise ValueError('rho must have dimensions (n, n,...), where n ' +
                             'corresponds to the number of states in the ' +
                             'generating Hamiltonian. ' +
                             'Instead, shape of rho is %s.' % str(rho.shape))
        elif O.shape[-2:] != (self.hamiltonian.n, self.hamiltonian.n):
            raise ValueError('O must have dimensions (..., n, n), where n ' +
                             'corresponds to the number of states in the ' +
                             'generating Hamiltonian. ' +
                             'Instead, shape of O is %s.' % str(O.shape))
        else:
            avO = np.tensordot(O, rho, axes=[(-2, -1), (0, 1)])
            if np.allclose(np.imag(avO), 0):
                return np.real(avO)
            else:
                return avO

    def force(self, r, t, rho, return_details=False):
        if rho.shape[0] != self.hamiltonian.n:
            rho = self.__reshape_rho(rho)

        f = np.zeros((3,) + rho.shape[2:])
        if return_details:
            f_laser_q = {}
            f_laser = {}

        for key in self.laserBeams:
            # First, determine the average mu_q:
            # This returns a (3,) + rho.shape[2:] array
            mu_q_av = self.observable(self.hamiltonian.d_q_bare[key], rho)
            gamma = self.hamiltonian.blocks[self.hamiltonian.laser_keys[key]
                                            ].parameters['gamma']

            if not return_details:
                delE = self.laserBeams[key].total_electric_field_gradient(
                    np.real(r), t)
                for jj, q in enumerate(np.arange(-1., 2., 1.)):
                    f += np.real((-1)**q*gamma*mu_q_av[jj]*delE[:, 2-jj])/2
            else:
                f_laser_q[key] = np.zeros((3, 3, self.laserBeams[key].num_of_beams)
                                          + rho.shape[2:])
                f_laser[key] = np.zeros((3, self.laserBeams[key].num_of_beams)
                                        + rho.shape[2:])

                for ii, beam in enumerate(self.laserBeams[key].beam_vector):
                    if not self.transform_into_re_im:
                        delE = beam.electric_field_gradient(np.real(r), t)
                    else:
                        delE = beam.electric_field_gradient(r, t)

                    for jj, q in enumerate(np.arange(-1., 2., 1.)):
                        f_laser_q[key][:, jj, ii] += \
                            np.real((-1)**q*gamma*mu_q_av[jj]*delE[:, 2-jj])/2

                    f_laser[key][:, ii] = np.sum(
                        f_laser_q[key][:, :, ii], axis=1)

                f += np.sum(f_laser[key], axis=1)

        if self.include_mag_forces:
            delB = self.magField.gradField(np.real(r))

            av_mu = self.observable(self.hamiltonian.mu, rho)

            f_mag = cartesian_vector_tensor_dot(av_mu, delB)

            f += f_mag
        elif return_details:
            f_mag = np.zeros(f.shape)

        if return_details:
            return f, f_laser, f_laser_q, f_mag
        else:
            return f
