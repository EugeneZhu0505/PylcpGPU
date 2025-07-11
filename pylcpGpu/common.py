import time
import copy
import numpy as np
from scipy.optimize import fsolve

class ProgressBar(object):
    def __init__(self, decimals=1, fill='█', prefix='Progress:',
                 suffix='', time_remaining_prefix=' time left', length=30,
                 update_rate=0.5):
        self.tic = time.time()
        self.decimals = decimals
        self.fill = fill
        self.length = length
        self.prefix = prefix
        self.suffix = suffix
        self.time_remaining_prefix = time_remaining_prefix
        self.finished = False
        self.max_written_length = 0
        self.last_update = 0.
        self.update_rate = update_rate

    def format_time(self, tic_toc):
        # Print a final time of completion
        if tic_toc>3600:
            time_str = "%d:%02d:%02d" % ((tic_toc)/3600.0,
                                        ((tic_toc)/60.0)%60.0,
                                        (tic_toc)%60.0)
        elif tic_toc>60:
            time_str = "%d:%02d" % ((tic_toc)/60.0,
                                    (tic_toc)%60.0)
        else:
            time_str = "%.2f s" % (tic_toc)

        return time_str

    def print_string(self, string1):
        # Update the maximum length of string written:
        self.max_written_length = max(self.max_written_length, len(string1))
        pad = ''.join([' ']*(self.max_written_length - len(string1)))
        print(string1 + pad, end='\r')

    def update(self, percentage):
        toc = time.time()
        if percentage>0 and percentage<1 and (toc-self.last_update)>self.update_rate:
            percent = ("{0:." + str(self.decimals) + "f}").format(100*percentage)
            filledLength = int(self.length*percentage)
            bar = self.fill*filledLength + '-'*(self.length - filledLength)

            remaining_time = (1-percentage)*((toc-self.tic)/percentage)
            if remaining_time>0:
                time_str = self.format_time(remaining_time)
            else:
                time_str = "0.00 s"
            self.print_string('%s |%s| %s%%%s;%s: %s' %
                              (self.prefix, bar, percent, self.suffix,
                               self.time_remaining_prefix, time_str))
            self.last_update = toc
        elif percentage>=1:
            if not self.finished:
                self.finished = True
                time_str = self.format_time(toc-self.tic)
                self.print_string('Completed in %s.' % time_str)
                print()


def cart2spherical(A):
    return np.array([(A[0]-1j*A[1])/np.sqrt(2), A[2], -(A[0]+1j*A[1])/np.sqrt(2)])

def spherical2cart(A):
    return np.array([1/np.sqrt(2)*(-A[2]+A[0]), 1j/np.sqrt(2)*(A[2]+A[0]), A[1]])

def spherical_dot(A, B):
    return np.tensordot(A, np.array([-1., 1., -1.])*B[::-1], axes=(0, 0))
    #return np.tensordot(A, np.conjugate(B), axes=(0,0))

class base_force_profile():
    """
    Base force profile

    The force profile object stores all of the calculated quantities created by
    the governingeq.generate_force_profile() method.  It has the following
    attributes:

    Attributes
    ----------
    R : array_like, shape (3, ...)
        Positions at which the force profile was calculated.
    V : array_like, shape (3, ...)
        Velocities at which the force profile was calculated.
    F : array_like, shape (3, ...)
        Total equilibrium force at position R and velocity V.
    f_mag : array_like, shape (3, ...)
        Magnetic force at position R and velocity V.
    f : dictionary of array_like
        The forces due to each laser, indexed by the
        manifold the laser addresses.  The dictionary is keyed by the transition
        driven, and individual lasers are in the same order as in the
        pylcp.laserBeams object used to create the governing equation.
    Neq : array_like
        Equilibrium population found.
    """
    def __init__(self, R, V, laserBeams, hamiltonian):
        if not isinstance(R, np.ndarray):
            R = np.array(R)
        if not isinstance(V, np.ndarray):
            V = np.array(V)

        if R.shape[0] != 3 or V.shape[0] != 3:
            raise TypeError('R and V must have first dimension of 3.')

        self.R = copy.copy(R)
        self.V = copy.copy(V)

        if hamiltonian is None:
            self.Neq = None
        else:
            self.Neq = np.zeros(R[0].shape + (hamiltonian.n,))

        self.f = {}
        for key in laserBeams:
            self.f[key] = np.zeros(R.shape + (len(laserBeams[key].beam_vector),))

        self.f_mag = np.zeros(R.shape)

        self.F = np.zeros(R.shape)

    def store_data(self, ind, Neq, F, F_laser, F_mag):
        if not Neq is None:
            self.Neq[ind] = Neq

        for jj in range(3):
            #self.f[(jj,) + ind] = f[jj]
            self.F[(jj,) + ind] = F[jj]
            for key in F_laser:
                self.f[key][(jj,) + ind] = F_laser[key][jj]

            self.f_mag[(jj,) + ind] = F_mag[jj]


def random_vector(rng, free_axes=[True, True, True]):
    """
    This function returns a random vector in either 1D, 2D or 3D

    Parameters:
    -----------
        rng: numpy.random.Generator()
            Properly seeded random number generator.
        free_axis: list of 3 boolean (option)
            Which axes (x, y, z) are considered free axes.  Default:
            [True, True, True], i.e., a 3D vector.

    Returns
    -------
        vector: array_like of shape (3,)
            Random vector with unit length.
    """
    if np.sum(free_axes)==1:
        return (np.sign(rng.random(1)-0.5)*free_axes).astype('float64')
    elif np.sum(free_axes)==2:
        phi = 2*np.pi*rng.random(1)[0]
        x = np.array([np.cos(phi), np.sin(phi)])
        y =np.zeros((3,))
        y[free_axes] = x
        return y
    elif np.sum(free_axes)==3:
        a, b = rng.random(2)
        th = np.arccos(2*b-1)
        phi = 2*np.pi*a

        return np.array([np.sin(th)*np.cos(phi), np.sin(th)*np.sin(phi),
                         np.cos(th)])
    else:
        raise StandardError('free_axes must be a boolean array of length 3.')


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    vectors = []
    for n in range(500):
        vectors.append(random_vector([True, True, True]))

    vectors = np.array(vectors)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(vectors[:,0], vectors[:,1], vectors[:,2])
    ax.view_init(elev=-90., azim=0.)
