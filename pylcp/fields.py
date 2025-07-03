# import numpy as np
import cupy as cp
from inspect import signature
from pylcp.common import cart2spherical, spherical2cart
from .integration_tools import parallelIntegrator
from scipy.spatial.transform import Rotation

# import numba

def dot2D(a, b):
    c = cp.zeros((a.shape[1],), dtype=a.dtype)
    for ii in range(a.shape[1]):
        c[ii] = cp.sum(a[:, ii]*b[:, ii])
    return c

def electric_field(r, t, amp, pol, k, phase):
    return pol*amp*cp.exp(-1j*(k[0]*r[0]+k[1]*r[1]+k[2]*r[2]) + 1j*phase)


def return_constant_val(R, t, val):
    if R.shape==(3,):
        return val
    elif R.shape[0] == 3:
        return val*cp.ones(R[0].shape)
    else:
        raise ValueError('The first dimension of R should have length 3, ' +
                         'not %d.'%R.shape[0])

def return_constant_vector(R, t, vector):
    if R.shape==(3,):
        return vector
    elif R.shape[0] == 3:
        return cp.outer(vector, cp.ones(R[0].shape))
    else:
        raise ValueError('The first dimension of R should have length 3, ' +
                         'not %d.'% R.shape[0])

def return_constant_val_t(t, val):
    if isinstance(t, cp.ndarray):
        return val*cp.ones(t.shape)
    else:
        return cp.array(val)

def promote_to_lambda(val, var_name='', type='Rt'):
    
    if type == 'Rt':
        if not callable(val):
            if isinstance(val, list) or isinstance(val, cp.ndarray):
                func = lambda R=cp.array([0., 0., 0.]), t=0.: return_constant_vector(R, t, val)
            else:
                func = lambda R=cp.array([0., 0., 0.]), t=0.: return_constant_val(R, t, val)
            sig = '()'
        else:
            sig = str(signature(val))
            if ('(R)' in sig or '(r)' in sig or '(x)' in sig):
                func = lambda R=cp.array([0., 0., 0.]), t=0.: val(R)
                sig = '(R)'
            elif ('(R, t)' in sig or '(r, t)' in sig or '(x, t)' in sig):
                func = lambda R=cp.array([0., 0., 0.]), t=0.: val(R, t)
                sig = '(R, t)'
            elif '(t)' in sig:
                func = lambda R=cp.array([0., 0., 0.]), t=0.: val(t)
                sig = '(R, t)'
            else:
                raise TypeError('Signature [%s] of function %s not'+
                                'understood.'% (sig, var_name))

        return func, sig
    elif type == 't':
        if not callable(val):
            func = lambda t=0.: return_constant_val_t(t, val)
            sig = '()'
        else:
            sig = str(signature(val))
            if '(t)' in sig:
                func = lambda t=0.: val(t)
            else:
                raise TypeError('Signature [%s] of function %s not '+
                                'understood.'% (sig, var_name))

        return func, sig

def return_dx_dy_dz(R, eps):
    if R.shape == (3,):
        dx = cp.array([eps, 0., 0.])
        dy = cp.array([0., eps, 0.])
        dz = cp.array([0., 0., eps])
    else:
        dx = cp.zeros(R.shape)
        dy = cp.zeros(R.shape)
        dz = cp.zeros(R.shape)

        dx[0] = eps
        dy[1] = eps
        dz[2] = eps

    return dx, dy, dz


class magField(object):
    
    def __init__(self, field, eps=1e-5):
        self.eps = eps

        R = cp.random.rand(3) # Pick a random point for testing

        # Promote it to a lambda func:
        self.Field, self.FieldSig = promote_to_lambda(field, var_name='for field')

        # Try it out:
        response = self.Field(R, 0.)
        if (isinstance(response, float) or isinstance(response, int) or
            len(response) != 3):
            raise ValueError('Magnetic field function must return a vector.')

    def FieldMag(self, R=cp.array([0., 0., 0.]), t=0):
        
        return cp.linalg.norm(self.Field(R, t))

    def gradFieldMag(self, R=cp.array([0., 0., 0.]), t=0):
        
        dx, dy, dz = return_dx_dy_dz(R, self.eps)

        return cp.array([
            (self.FieldMag(R+dx, t)-self.FieldMag(R-dx, t))/2/self.eps,
            (self.FieldMag(R+dy, t)-self.FieldMag(R-dy, t))/2/self.eps,
            (self.FieldMag(R+dz, t)-self.FieldMag(R-dz, t))/2/self.eps
            ])

    def gradField(self, R=cp.array([0., 0., 0.]), t=0):
        
        dx, dy, dz = return_dx_dy_dz(R, self.eps)

        return cp.array([
            (self.Field(R+dx, t) - self.Field(R-dx, t))/2/self.eps,
            (self.Field(R+dy, t) - self.Field(R-dy, t))/2/self.eps,
            (self.Field(R+dz, t) - self.Field(R-dz, t))/2/self.eps
            ])

class iPMagneticField(magField):
    
    def __init__(self, B0, B1, B2, eps = 1e-5):
        super().__init__(lambda R, t: cp.array([B1*R[0]-B2*R[0]*R[2]/2, -R[1]*B1-B2*R[1]*R[2]/2, B0+B2/2*(R[2]**2 - (R[0]**2+R[1]**2)/2)]))
        self.B0 = B0
        self.B1 = B1
        self.B2 = B2

    #Analytical form, not numerical for this and gradField
    def gradFieldMag(self, R=cp.array([0., 0., 0.]), t=0):
        a = self.B0
        b = self.B1
        c = self.B2
        x = R[0]
        y = R[1]
        z = R[2]
        mag = self.FieldMag(R, t)
        xcom = 0.5*(2*b**2*x-a*c*x+(c**2)*(x**3)/4+(c**2)*(y**2)*x/4-2*b*c*z*x)/mag
        ycom = 0.5*(2*b**2*(y)-a*c*y+(c**2)*(x**2)*y/4 + (c**2)*(y**3)/4+2*b*c*z*y)/mag
        zcom = 0.5*(0-b*c*(x**2)+b*c*(y**2)+2*a*c*z+(c**2)*(z**3))/mag
        return cp.array([xcom, ycom, zcom])

    def gradField(self, R=cp.array([0., 0., 0.]), t=0):
        B0 = self.B0
        B1 = self.B1
        B2 = self.B2
        x = R[0]
        y = R[1]
        z = R[2]
        xcom = cp.array([B1-B2*z/2, 0, -B2*x/2])
        ycom = cp.array([0, -B1-B2*z/2, B2*y/2])
        zcom = cp.array([-B2*x/2, -B2*y/2, B2*z])

        return cp.array([
            cp.array([B1-B2*z/2, 0, -B2*x/2]),
            cp.array([0, -B1-B2*z/2, B2*y/2]),
            cp.array([-B2*x/2, -B2*y/2, B2*z])
            ])


class constantMagneticField(magField):
    
    def __init__(self, B0):
        super().__init__(lambda R, t: B0)

        self.constant_grad_field_mag = cp.zeros((3,))
        self.constant_grad_field = cp.zeros((3,3))

    def gradFieldMag(self, R=cp.array([0., 0., 0.]), t=0):
       
        return self.constant_grad_field_mag

    def gradField(self, R=cp.array([0., 0., 0.]), t=0):
        
        return self.constant_grad_field


class quadrupoleMagneticField(magField):
   
    def __init__(self, alpha, eps=1e-5):
        super().__init__(lambda R, t: alpha*cp.array([-0.5*R[0], -0.5*R[1], R[2]]))
        self.alpha = alpha

        self.constant_grad_field = alpha*\
            cp.array([[-0.5, 0., 0.], [0., -0.5, 0.], [0., 0., 1.]])

    def gradField(self, R=cp.array([0., 0., 0.]), t=0):
        
        return self.constant_grad_field


class MOT2DMagneticField(magField):
    
    def __init__(self, alpha, eps=1e-5):
        super().__init__(lambda R, t: alpha*cp.array([-R[0], R[1], -0.2*R[2]]))
        self.alpha = alpha

        self.constant_grad_field = alpha*\
            cp.array([[-1., 0., 0.], [0., 1, 0.], [0., 0., -0.2]])

    def gradField(self, R=cp.array([0., 0., 0.]), t=0):
        
        return self.constant_grad_field

# First, define the laser beam class:
class laserBeam(object):
   
    def __init__(self, kvec=None, s=None, pol=None, delta=None,
                 phase=0., pol_coord='spherical', eps=1e-5):

        if not kvec is None:
            self.kvec, self.kvec_sig = promote_to_lambda(kvec, var_name='kvector')

        if not s is None:
            self.intensity, self.intensity_sig = promote_to_lambda(s, var_name='s')

        if not pol is None:
            if not callable(pol):
                pol = self.__parse_constant_polarization(pol, pol_coord)

            self.pol, self.pol_sig = promote_to_lambda(pol, var_name='polarization')

        if not delta is None:
            self.delta, self.delta_sig = promote_to_lambda(delta, var_name='delta', type='t')

        if self.delta_sig == '(t)':
            self.delta_phase = parallelIntegrator(self.delta)
        elif self.delta_sig == '()':
            self.delta_phase = lambda t: delta*t

        if not phase is None:
            self.phase, self.phase_sig = promote_to_lambda(phase, var_name='phase', type='t')

        self.eps = eps

    def __parse_constant_polarization(self, pol, pol_coord):
        if isinstance(pol, float) or isinstance(pol, int):
            
            if cp.sign(pol)<0:
                self.pol = cp.array([1., 0., 0.], dtype='complex')
            else:
                self.pol = cp.array([0., 0., 1.], dtype='complex')

            self.pol, self.pol_sig = promote_to_lambda(self.pol, var_name='polarization')

            self.pol = self.project_pol(self.kvec()/cp.linalg.norm(self.kvec()),
                                        invert=True).astype('complex128')

        elif isinstance(pol, cp.ndarray):
            if pol.shape != (3,):
                raise ValueError("pol, when a vector, must be a (3,) array")

            if pol_coord=='cartesian':
                if cp.abs(cp.dot(self.kvec(), pol)) > 1e-9:
                    raise ValueError("I'm sorry; light is a transverse wave")

                self.pol = cart2spherical(pol).astype('complex128')

            elif pol_coord=='spherical':
                pol_cart = spherical2cart(pol)

                if cp.abs(cp.dot(self.kvec(), pol_cart)) > 1e-9:
                    raise ValueError("I'm sorry; light is a transverse wave")

                self.pol = pol.astype('complex128')

            self.pol = self.pol/cp.linalg.norm(self.pol)
        else:
            raise ValueError("pol must be +1, -1, or a numpy array")

        return self.pol


    def kvec(self, R=cp.array([0., 0., 0.]), t=0.):
        
        pass

    def intensity(self, R=cp.array([0., 0., 0.]), t=0.):
        
        pass

    def pol(self, R=cp.array([0., 0., 0.]), t=0.):
        
        pass

    def delta(self, t=0.):
        
        pass

    # TODO: add testing of kvec/pol orthogonality.
    def project_pol(self, quant_axis, R=cp.array([0., 0., 0.]), t=0,
                    treat_nans=False, calculate_norm=False, invert=False):
    

        pol = self.pol(R, t)

        if calculate_norm:
            quant_axis2 = cp.zeros(quant_axis.shape)
            quant_axis[2] = 1.0  # Make the third entry all ones.
            quant_axis_norm = cp.linalg.norm(quant_axis, axis=0)
            for ii in range(3):
                quant_axis2[ii][quant_axis_norm!=0] = \
                    quant_axis2[ii][quant_axis_norm!=0]/\
                    quant_axis_norm[quant_axis_norm!=0]
            quant_axis=quant_axis2
        elif treat_nans:
            for ii in range(quant_axis.shape[0]):
                if ii<quant_axis.shape[0]-1:
                    quant_axis[ii][cp.isnan(quant_axis[-1])] = 0.0
                else:
                    quant_axis[ii][cp.isnan(quant_axis[-1])] = 1.0

        cosbeta = quant_axis[2]
        sinbeta = cp.sqrt(1-cosbeta**2)
        if isinstance(cosbeta, (float, int)):
            if cp.abs(cosbeta)<1:
                gamma = cp.arctan2(quant_axis[1], quant_axis[0])
            else:
                gamma = 0
            alpha = 0
        else:
            gamma = cp.zeros(cosbeta.shape)
            inds = cp.abs(quant_axis[2])<1
            gamma[inds] = cp.arctan2(quant_axis[1][inds],
                                         quant_axis[0][inds])
            alpha = cp.zeros(cosbeta.shape)

        quant_axis = quant_axis.astype('float64')
        pol = pol.astype('complex128')

        D = cp.array([
            [(1+cosbeta)/2*cp.exp(-1j*alpha + 1j*gamma),
             -sinbeta/cp.sqrt(2)*cp.exp(-1j*alpha),
             (1-cosbeta)/2*cp.exp(-1j*alpha - 1j*gamma)],
            [sinbeta/cp.sqrt(2)*cp.exp(1j*gamma),
             cosbeta,
             -sinbeta/cp.sqrt(2)*cp.exp(-1j*gamma)],
            [(1-cosbeta)/2*cp.exp(1j*alpha+1j*gamma),
             sinbeta/cp.sqrt(2),
             (1+cosbeta)/2*cp.exp(1j*alpha-1j*gamma)]
             ])

        if invert:
            D = cp.linalg.inv(D)

        if pol.shape == (3,) and quant_axis.shape == (3,):
            return D @ pol
        else:
            return cp.tensordot(D, pol, ([1],[0]))


    def cartesian_pol(self, R=cp.array([0., 0., 0.]), t=0):

        pol = self.pol(R, t)
        return spherical2cart(pol)

    def jones_vector(self, xp, yp, R=cp.array([0., 0., 0.]), t=0):
        
        if cp.abs(cp.dot(xp, yp)) > 1e-10:
            raise ValueError('xp and yp must be orthogonal.')
        if cp.abs(cp.dot(xp, self.kvec(R, t))) > 1e-10:
            raise ValueError('xp and k must be orthogonal.')
        if cp.abs(cp.dot(yp, self.kvec(R, t))) > 1e-10:
            raise ValueError('yp and k must be orthogonal.')
        if cp.sum(cp.abs(cp.cross(xp, yp) - self.kvec(R, t))) > 1e-10:
            raise ValueError('xp, yp, and k must form a right-handed' +
                             'coordinate system.')

        pol_cart = self.cartesian_pol(R, t)

        if cp.abs(cp.dot(pol_cart, self.kvec(R, t))) > 1e-9:
            raise ValueError('Something is terribly, terribly wrong.')

        return cp.array([cp.dot(pol_cart, xp), cp.dot(pol_cart, yp)])


    def stokes_parameters(self, xp, yp, R=cp.array([0., 0., 0.]), t=0):
        
        jones_vector = self.jones_vector(xp, yp, R, t)

        Q = cp.abs(jones_vector[0])**2 - cp.abs(jones_vector[1])**2
        U = 2*cp.real(jones_vector[0]*cp.conj(jones_vector[1]))
        V = -2*cp.imag(jones_vector[0]*cp.conj(jones_vector[1]))

        return (Q, U, V)


    def polarization_ellipse(self, xp, yp, R=cp.array([0., 0., 0.]), t=0):

        Q, U, V = self.stokes_parameters(xp, yp, R, t)

        psi = cp.arctan2(U, Q)
        while psi<0:
            psi+=2*cp.pi
        psi = psi%(2*cp.pi)/2
        if cp.sqrt(Q**2+U**2)>1e-10:
            chi = 0.5*cp.arctan(V/cp.sqrt(Q**2+U**2))
        else:
            chi = cp.pi/4*cp.sign(V)

        return (psi, chi)


    def electric_field(self, R, t):

        kvec = self.kvec(R, t)
        s = self.intensity(R, t)
        pol = self.pol(R, t)
        delta_phase = self.delta_phase(t)
        phase = self.phase(t)

        amp = cp.sqrt(2*s)

        if isinstance(t, float):
            Eq = electric_field(R, t, amp, pol, kvec, delta_phase - phase)
        else:
            Eq = pol.reshape(3, t.size)*\
            (amp*cp.exp(-1j*dot2D(kvec, R) + 1j*delta_phase - 1j*phase)).reshape(1, t.size)

        return Eq


    def electric_field_gradient(self, R, t):
        """
        The full derivative of electric field at position R and t

        Parameters
        ----------
        R : array_like, size (3,)
            vector of the position at which to return the kvector.  By default,
            the origin.
        t : float
            time at which to return the k-vector.  By default, t=0.

        Returns
        -------
        dEq : array_like, shape (3, 3)
            The full gradient of the electric field, in spherical coordinates.

            .. math::
              \\begin{pmatrix}
                \\frac{dE_{-1}}{dx} & \\frac{dE_0}{dx} & \\frac{dE_{+1}}{dx} \\\\
                \\frac{dE_{-1}}{dy} & \\frac{dE_0}{dy} & \\frac{dE_{+1}}{dy} \\\\
                \\frac{dE_{-1}}{dz} & \\frac{dE_0}{dz} & \\frac{dE_{+1}}{dz} \\\\
              \\end{pmatrix}
        """
        (dx, dy, dz) = return_dx_dy_dz(R, self.eps)
        delEq = cp.array([
            (self.electric_field(R+dx, t) -
             self.electric_field(R-dx, t))/2/self.eps,
            (self.electric_field(R+dy, t) -
             self.electric_field(R-dy, t))/2/self.eps,
            (self.electric_field(R+dz, t) -
             self.electric_field(R-dz, t))/2/self.eps
            ])

        return delEq


class infinitePlaneWaveBeam(laserBeam):
    """
    Infinte plane wave beam

    A beam which has spatially constant intensity, k-vector, and polarization.

    .. math::
        \\frac{1}{2}\\hat{\\boldsymbol{\\epsilon}} E_0e^{i\\mathbf{k}\\cdot\\mathbf{r}-i \\int dt\\Delta(t) + i\\phi(r, t)}

    where :math:`\\hat{\\boldsymbol{\\epsilon}}` is the polarization, :math:`E_0`
    is the electric field magnitude, :math:`\\mathbf{k}(r,t)` is the k-vector,
    :math:`\\mathbf{r}` is the position, :math:`\\Delta(t)` is the deutning,
    :math:`t` is the time, and :math:`\\phi` is the phase.

    Parameters
    ----------
    kvec : array_like with shape (3,) or callable
        The k-vector of the laser beam, specified as either a three-element
        list or numpy array.
    pol : int, float, array_like with shape (3,), or callable
        The polarization of the laser beam, specified as either an integer, float
        array_like with shape(3,).  If an integer or float,
        if `pol<0` the polarization will be left circular polarized relative to
        the k-vector of the light.  If `pol>0`, the polarization will be right
        circular polarized.  If array_like, polarization will be specified by the
        vector, whose basis is specified by `pol_coord`.
    s : float or callable
        The intensity of the laser beam, specified as either a float or as
        callable function.
    delta: float or callable
        Detuning of the laser beam.  If a callable, it must have a
        signature like (t) where t is a float and it must return a float.
    **kwargs :
        Additional keyword arguments to pass to laserBeam superclass.

    Notes
    -----
    This implementation is much faster, when it can be used, compared to the
    base laserBeam class.
    """
    def __init__(self, kvec, pol, s, delta, **kwargs):
        if callable(kvec):
            raise TypeError('kvec cannot be a function for an infinite plane wave.')

        if callable(s):
            raise TypeError('s cannot be a function for an infinite plane wave.')

        if callable(pol):
            raise TypeError('Polarization cannot be a function for an infinite plane wave.')

        # Use the super class to define the functions kvec, s, pol, and delta.
        super().__init__(kvec=kvec, s=s, pol=pol, delta=delta,
                         **kwargs)

        # Save the constant values (might be useful):
        self.con_kvec = kvec
        self.con_s = s
        self.con_pol = self.pol(cp.array([0., 0., 0.]), 0.)
        # Define attributes to speed up gradient calculation:
        self.amp = cp.sqrt(2*self.con_s)
        self.dEq_prefactor = (-1j*self.amp*self.con_kvec.reshape(3, 1)*
                              self.con_pol.reshape(1, 3))

    def electric_field_gradient(self, R, t):
        # With a plane wave, this is simple:
        delta_phase = self.delta_phase(t)
        phase = self.phase(t)

        if isinstance(t, float) or (isinstance(t, cp.ndarray) and t.size==1):
            delEq = self.dEq_prefactor*\
            cp.exp(-1j*cp.dot(self.con_kvec, R) + 1j*delta_phase - 1j*phase)
        else:
            delEq = self.dEq_prefactor.reshape(3, 3, 1)*\
            cp.exp(-1j*cp.dot(self.con_kvec, R) + 1j*delta_phase -1j*phase).reshape(1, 1, t.size)

        return delEq


class gaussianBeam(laserBeam):
    """
    Collimated Gaussian beam

    A beam which has spatially constant k-vector and polarization, with a
    Gaussian intensity modulation.  Specifically,

    .. math::
      \\frac{1}{2}\\hat{\\boldsymbol{\\epsilon}} E_0 e^{-\\mathbf{r}^2/w_b^2} e^{i\\mathbf{k}\\cdot\\mathbf{r}-i \\int dt\\Delta(t) + i\\phi(r, t)}

    where :math:`\\hat{\\boldsymbol{\\epsilon}}` is the polarization, :math:`E_0`
    is the electric field magnitude, :math:`\\mathbf{k}(r,t)` is the k-vector,
    :math:`\\mathbf{r}` is the position, :math:`\\Delta(t)` is the deutning,
    :math:`t` is the time, and :math:`\\phi` is the phase.  Note that because
    :math:`I\\propto E^2`, :math:`w_b` is the :math:`1/e^2` radius.

    Parameters
    ----------
    kvec : array_like with shape (3,) or callable
        The k-vector of the laser beam, specified as either a three-element
        list or numpy array.
    pol : int, float, array_like with shape (3,), or callable
        The polarization of the laser beam, specified as either an integer, float
        array_like with shape(3,).  If an integer or float,
        if `pol<0` the polarization will be left circular polarized relative to
        the k-vector of the light.  If `pol>0`, the polarization will be right
        circular polarized.  If array_like, polarization will be specified by the
        vector, whose basis is specified by `pol_coord`.
    s : float or callable
        The maximum intensity of the laser beam at the center, specified as
        either a float or as callable function.
    delta : float or callable
        Detuning of the laser beam.  If a callable, it must have a
        signature like (t) where t is a float and it must return a float.
    wb : float
        The :math:`1/e^2` radius of the beam.
    **kwargs:
        Additional keyword arguments to pass to the laserBeam superclass.
    """
    def __init__(self, kvec, pol, s, delta, wb, **kwargs):
        if callable(kvec):
            raise TypeError('kvec cannot be a function for a Gaussian beam.')

        if callable(pol):
            raise TypeError('Polarization cannot be a function for a Gaussian beam.')

        # Use super class to define kvec(R, t), pol(R, t), and delta(t)
        super().__init__(kvec=kvec, pol=pol, delta=delta, **kwargs)

        # Save the constant values (might be useful):
        self.con_kvec = kvec
        self.con_khat = kvec/cp.linalg.norm(kvec)
        self.con_pol = self.pol(cp.array([0., 0., 0.]), 0.)

        # Save the parameters specific to the Gaussian beam:
        self.s_max = s # central saturation parameter
        self.wb = wb # 1/e^2 radius
        self.define_rotation_matrix()

    def define_rotation_matrix(self):
        # Angles of rotation:
        th = cp.arccos(self.con_khat[2])
        phi = cp.arctan2(self.con_khat[1], self.con_khat[0])

        # Use scipy to define the rotation matrix
        self.rmat = cp.asarray(Rotation.from_euler('ZY', [cp.asnumpy(phi), cp.asnumpy(th)]).inv().as_matrix())

    def intensity(self, R=cp.array([0., 0., 0.]), t=0.):
        # Rotate up to the z-axis where we can apply formulas:
        Rp = cp.einsum('ij,j...->i...', self.rmat, R)
        rho_sq=cp.sum(Rp[:2]**2, axis=0)
        # Return the intensity:
        return self.s_max*cp.exp(-2*rho_sq/self.wb**2)


class clippedGaussianBeam(gaussianBeam):
    """
    Clipped, collimated Gaussian beam

    A beam which has spatially constant k-vector and polarization, with a
    Gaussian intensity modulation.  Specifically,

    .. math::
      \\frac{1}{2}\\hat{\\boldsymbol{\\epsilon}} E_0 e^{-\\mathbf{r}^2/w_b^2} (|\\mathbf{r}|<r_s) e^{i\\mathbf{k}\\cdot\\mathbf{r}-i \\int dt\\Delta(t) + i\\phi(r, t)}

    where :math:`\\hat{\\boldsymbol{\\epsilon}}` is the polarization, :math:`E_0`
    is the electric field magnitude, :math:`r_s` is the radius of the stop,
    :math:`\\mathbf{k}(r,t)` is the k-vector,
    :math:`\\mathbf{r}` is the position, :math:`\\Delta(t)` is the deutning,
    :math:`t` is the time, and :math:`\\phi` is the phase. Note that because
    :math:`I\\propto E^2`, :math:`w_b` is the :math:`1/e^2` radius.

    Parameters
    ----------
    kvec : array_like with shape (3,) or callable
        The k-vector of the laser beam, specified as either a three-element
        list or numpy array.
    pol : int, float, array_like with shape (3,), or callable
        The polarization of the laser beam, specified as either an integer, float
        array_like with shape(3,).  If an integer or float,
        if `pol<0` the polarization will be left circular polarized relative to
        the k-vector of the light.  If `pol>0`, the polarization will be right
        circular polarized.  If array_like, polarization will be specified by the
        vector, whose basis is specified by `pol_coord`.
    s : float or callable
        The maximum intensity of the laser beam at the center, specified as
        either a float or as callable function.
    delta : float or callable
        Detuning of the laser beam.  If a callable, it must have a
        signature like (t) where t is a float and it must return a float.
    wb : float
        The :math:`1/e^2` radius of the beam.
    rs : float
        The radius of the stop.
    **kwargs:
        Additional keyword arguments to pass to the laserBeam superclass.
    """
    def __init__(self, kvec, pol, s, delta, wb, rs, **kwargs):
        super().__init__(kvec=kvec, pol=pol, s=s, delta=delta, wb=wb, **kwargs)

        self.rs = rs # Save the radius of the stop.

    def intensity(self, R=cp.array([0., 0., 0.]), t=0.):
        Rp = cp.einsum('ij,j...->i...', self.rmat, R)
        rho_sq = cp.sum(Rp[:2]**2, axis=0)
        return self.s_max*cp.exp(-2*rho_sq/self.wb**2)*(cp.sqrt(rho_sq)<self.rs)


class laserBeams(object):
    """
    The base class for a collection of laser beams

    Parameters
    ----------
    laserbeamparams : array_like of laserBeam or array_like of dictionaries
        If array_like contains laserBeams, the laserBeams in the array will be joined
        together to form a collection.  If array_like is a list of dictionaries, the
        dictionaries will be passed as keyword arguments to beam_type
    beam_type : laserBeam or laserBeam subclass, optional
        Type of beam to use in the collection of laserBeams.  By default
        `beam_type=laserBeam`.
    """
    def __init__(self, laserbeamparams=None, beam_type=laserBeam):
        if laserbeamparams is not None:
            if not isinstance(laserbeamparams, list):
                raise ValueError('laserbeamparams must be a list.')
            self.beam_vector = []
            for laserbeamparam in laserbeamparams:
                if isinstance(laserbeamparam, dict):
                    self.beam_vector.append(beam_type(**laserbeamparam))
                elif isinstance(laserbeamparam, laserBeam):
                    self.beam_vector.append(laserbeamparam)
                else:
                    raise TypeError('Each element of laserbeamparams must either ' +
                                    'be a list of dictionaries or list of ' +
                                    'laserBeams')

            self.num_of_beams = len(self.beam_vector)
        else:
            self.beam_vector = []
            self.num_of_beams = 0

    def __iadd__(self, other):
        self.beam_vector += other.beam_vector
        self.num_of_beams = len(self.beam_vector)

        return self

    def __add__(self, other):
        return laserBeams(self.beam_vector + other.beam_vector)

    def add_laser(self, new_laser):
        """
        Add a laser to the collection

        Parameters
        ----------
        new_laser : laserBeam or laserBeam subclass
        """
        if isinstance(new_laser, laserBeam):
            self.beam_vector.append(new_laser)
            self.num_of_beams = len(self.beam_vector)
        elif isinstance(new_laser, dict):
            self.beam_vector.append(laserBeam(**new_laser))
        else:
            raise TypeError('new_laser should by type laserBeam or a dictionary' +
                            'of arguments to initialize the laserBeam class.')

    def pol(self, R=cp.array([0., 0., 0.]), t=0.):
        """
        Returns the polarization of the laser beam at position R and t

        The polarization is returned in the spherical basis.

        Parameters
        ----------
        R : array_like, size (3,), optional
            vector of the position at which to return the kvector.  By default,
            the origin.
        t : float, optional
            time at which to return the k-vector.  By default, t=0.

        Returns
        -------
        pol : list of array_like, size (3,)
            polarization of each laser beam at R and t in spherical basis.
        """
        return cp.array([beam.pol(R, t) for beam in self.beam_vector])

    def intensity(self, R=cp.array([0., 0., 0.]), t=0.):
        """
        Returns the intensity of the laser beam at position R and t

        Parameters
        ----------
        R : array_like, size (3,), optional
            vector of the position at which to return the kvector.  By default,
            the origin.
        t : float, optional
            time at which to return the k-vector.  By default, t=0.

        Returns
        -------
        s : list of float or array_like
            Saturation parameters of all laser beams at R and t.
        """
        return cp.array([beam.intensity(R, t) for beam in self.beam_vector])

    def kvec(self, R=cp.array([0., 0., 0.]), t=0.):
        """
        Returns the k-vector of the laser beam

        Parameters
        ----------
        R : array_like, size (3,), optional
            vector of the position at which to return the kvector.  By default,
            the origin.
        t : float, optional
            time at which to return the k-vector.  By default, t=0.

        Returns
        -------
        kvec : list of array_like, size(3,)
            the k vector at position R and time t for each laser beam.
        """
        return cp.array([beam.kvec(R, t) for beam in self.beam_vector])

    def delta(self, t=0):
        """
        Returns the detuning of the laser beam at time t

        Parameters
        ----------
        t : float, optional
            time at which to return the k-vector.  By default, t=0.

        Returns
        -------
        delta : float or array like
            detuning of the laser beam at time t for all laser beams
        """
        return cp.array([beam.delta(t) for beam in self.beam_vector])

    def electric_field(self, R=cp.array([0., 0., 0.]), t=0.):
        """
        Returns the electric field of the laser beams

        Parameters
        ----------
        R : array_like, size (3,), optional
            vector of the position at which to return the kvector.  By default,
            the origin.
        t : float, optional
            time at which to return the k-vector.  By default, t=0.

        Returns
        -------
        E : list of array_like, size(3,)
            the electric field vectors at position R and time t for each laser beam.
        """
        return cp.array([beam.electric_field(R, t) for beam in self.beam_vector])

    def electric_field_gradient(self, R=cp.array([0., 0., 0.]), t=0.):
        """
        Returns the gradient of the electric field of the laser beams

        Parameters
        ----------
        R : array_like, size (3,), optional
            vector of the position at which to return the kvector.  By default,
            the origin.
        t : float, optional
            time at which to return the k-vector.  By default, t=0.

        Returns
        -------
        dE : list of array_like, size(3,)
            the electric field gradient matrices at position R and time t for each laser beam.
        """
        return cp.array([beam.electric_field_gradient(R, t)
                         for beam in self.beam_vector])

    def total_electric_field(self, R=cp.array([0., 0., 0.]), t=0.):
        """
        Returns the total electric field of the laser beams

        Parameters
        ----------
        R : array_like, size (3,), optional
            vector of the position at which to return the kvector.  By default,
            the origin.
        t : float, optional
            time at which to return the k-vector.  By default, t=0.

        Returns
        -------
        E : array_like, size(3,)
            the total electric field vector at position R and time t of all
            the laser beams
        """
        return cp.sum(self.electric_field(R, t), axis=0)

    def total_electric_field_gradient(self, R=cp.array([0., 0., 0.]), t=0.):
        """
        Returns the total gradient of the electric field of the laser beams

        Parameters
        ----------
        R : array_like, size (3,), optional
            vector of the position at which to return the kvector.  By default,
            the origin.
        t : float, optional
            time at which to return the k-vector.  By default, t=0.

        Returns
        -------
        dE : array_like, size(3,)
            the total electric field gradient matrices at position R and time t
            of all laser beams.
        """
        return cp.sum(self.electric_field_gradient(R, t), axis=0)


    def project_pol(self, quant_axis, R=cp.array([0., 0., 0.]), t=0, **kwargs):
        """
        Project the polarization onto a quantization axis.

        Parameters
        ----------
        quant_axis : array_like, shape (3,)
            A normalized 3-vector of the quantization axis direction.
        R : array_like, shape (3,), optional
            If polarization is a function of R is the
            3-vectors at which the polarization shall be calculated.
        calculate_norm : bool, optional
            If true, renormalizes the quant_axis.  By default, False.
        treat_nans : bool, optional
            If true, every place that nan is encoutnered, replace with the
            $hat{z}$ axis as the quantization axis.  By default, False.
        invert : bool, optional
            If true, invert the process to project the quantization axis
            onto the specified polarization.

        Returns
        -------
        projected_pol : list of array_like, shape (3,)
            The polarization projected onto the quantization axis for all
            laser beams
        """
        cosbeta = quant_axis[2]
        sinbeta = cp.sqrt(1-cosbeta**2)
        if isinstance(cosbeta, float):
            if cp.abs(cosbeta)<1:
                gamma = cp.arctan2(quant_axis[1], quant_axis[0])
            else:
                gamma = 0
            alpha = 0
        else:
            gamma = cp.zeros(cosbeta.shape)
            inds = cp.abs(quant_axis[2])<1
            gamma[inds] = cp.arctan2(quant_axis[1][inds],
                                         quant_axis[0][inds])
            alpha = cp.zeros(cosbeta.shape)

        quant_axis = quant_axis.astype('float64')

        D = cp.array([
            [(1+cosbeta)/2*cp.exp(-1j*alpha + 1j*gamma),
             -sinbeta/cp.sqrt(2)*cp.exp(-1j*alpha),
             (1-cosbeta)/2*cp.exp(-1j*alpha - 1j*gamma)],
            [sinbeta/cp.sqrt(2)*cp.exp(1j*gamma),
             cosbeta,
             -sinbeta/cp.sqrt(2)*cp.exp(-1j*gamma)],
            [(1-cosbeta)/2*cp.exp(1j*alpha+1j*gamma),
             sinbeta/cp.sqrt(2),
             (1+cosbeta)/2*cp.exp(1j*alpha-1j*gamma)]
             ])

        if quant_axis.shape == (3,) and R.shape == (3,):
            return [D @ beam.pol(R, t) for beam in self.beam_vector]
        else:
            return [cp.tensordot(D, beam.pol(R, t), ([1],[0]))
                    for beam in self.beam_vector]

    def cartesian_pol(self, R=cp.array([0., 0., 0.]), t=0):
        """
        Returns the polarization of all laser beams in Cartesian coordinates.

        Parameters
        ----------
        R : array_like, size (3,), optional
            vector of the position at which to return the polarization.  By default,
            the origin.
        t : float, optional
            time at which to return the polarization.  By default, t=0.

        Returns
        -------
        pol : array_like, shape (num_of_beams, 3)
            polarization of the laser beam at R and t in Cartesian basis.
        """
        return [beam.cartesian_pol(R, t) for beam in self.beam_vector]

    def jones_vector(self, xp, yp, R=cp.array([0., 0., 0.]), t=0):
        """
        Jones vector at position R and time t

        Parameters
        ----------
        xp : array_like, shape (3,)
            The x vector of the basis in which to calculate the Jones vector.
            Must be orthogonal to k.
        yp : array_like, shape (3,)
            The y vector of the basis in which to calculate the Jones vector.
            Must be orthogonal to k and `xp`.
        R : array_like, size (3,), optional
            vector of the position at which to evaluate the Jones vector.  By default,
            the origin.
        t : float, optional
            time at which to evaluate the Jones vector.  By default, t=0.

        Returns
        -------
        pol : array_like, size (num_of_beams, 2)
            Jones vector of the laser beams at R and t in Cartesian basis.
        """

        return [beam.jones_vector(xp, yp, R, t) for beam in self.beam_vector]

    def stokes_parameters(self, xp, yp, R=cp.array([0., 0., 0.]), t=0):
        """
        The Stokes Parameters of the laser beam at R and t

        Parameters
        ----------
        xp : array_like, shape (3,)
            The x vector of the basis in which to calculate the Stokes parameters.
            Must be orthogonal to k.
        yp : array_like, shape (3,)
            The y vector of the basis in which to calculate the Stokes parameters.
            Must be orthogonal to k and `xp`.
        R : array_like, size (3,), optional
            vector of the position at which to calculate the Stokes parameters.
            By default, the origin.
        t : float, optional
            time at which to calculate the Stokes parameters.  By default, t=0.

        Returns
        -------
        pol : array_like, shape (num_of_beams, 3)
            Stokes parameters for the laser beams, [Q, U, V]
        """
        return [beam.stokes_parameters(xp, yp, R, t) for beam in self.beam_vector]

    def polarization_ellipse(self, xp, yp, R=cp.array([0., 0., 0.]), t=0):
        """
        The polarization ellipse parameters of the laser beam at R and t

        Parameters
        ----------
        xp : array_like, shape (3,)
            The x vector of the basis in which to calculate the polarization ellipse.
            Must be orthogonal to k.
        yp : array_like, shape (3,)
            The y vector of the basis in which to calculate the polarization ellipse.
            Must be orthogonal to k and `xp`.
        R : array_like, size (3,), optional
            vector of the position at which to return the kvector.  By default,
            the origin.
        t : float, optional
            time at which to return the k-vector.  By default, t=0.

        Returns
        -------
        list of (psi, chi) : list of tuples
            list of (:math:`\\psi`, :math:`\\chi`) parameters of the
            polarization ellipses for each laser beam
        """
        return [beam.polarization_ellipse(xp, yp, R, t) for beam in self.beam_vector]


class conventional3DMOTBeams(laserBeams):
    """
    A collection of laser beams for 6-beam MOT

    The standard geometry is to generate counter-progagating beams along all
    orthogonal axes :math:`(\\hat{x}, \\hat{y}, \\hat{z})`.

    Parameters
    ----------
    k : float, optional
        Magnitude of the k-vector for the six laser beams.  Default: 1
    pol : int or float, optional
        Sign of the circular polarization for the beams moving along
        :math:`\\hat{z}`.  Default: +1.  Orthogonal beams have opposite
        polarization by default.
    rotation_angles : array_like
        List of angles to define a rotated MOT.  Default: [0., 0., 0.]
    rotation_spec : str
        String to define the convention of the Euler rotations.  Default: 'ZYZ'
    beam_type : pylcp.laserBeam or subclass
        Type of beam to generate.
    **kwargs :
        other keyword arguments to pass to beam_type
    """
    def __init__(self, k=1, pol=+1, rotation_angles=[0., 0., 0.],
                 rotation_spec='ZYZ', beam_type=laserBeam, **kwargs):
        super().__init__()

        rot_mat = Rotation.from_euler(rotation_spec, rotation_angles).as_matrix()

        kvecs = [cp.array([ 1.,  0.,  0.]), cp.array([-1.,  0.,  0.]),
                 cp.array([ 0.,  1.,  0.]), cp.array([ 0., -1.,  0.]),
                 cp.array([ 0.,  0.,  1.]), cp.array([ 0.,  0., -1.])]
        pols = [-pol, -pol, -pol, -pol, +pol, +pol]

        for kvec, pol in zip(kvecs, pols):
            self.add_laser(beam_type(kvec=rot_mat @ (k*kvec), pol=pol, **kwargs))

