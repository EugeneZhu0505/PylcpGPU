from __future__ import division, print_function, absolute_import
import inspect
import numpy as np
from inspect import signature
from scipy.integrate._ivp.bdf import BDF
from scipy.integrate._ivp.radau import Radau
from scipy.integrate._ivp.rk import RK23, RK45
from scipy.integrate._ivp.lsoda import LSODA
from scipy.optimize import OptimizeResult
from scipy.integrate._ivp.common import EPS, OdeSolution
from scipy.integrate._ivp.base import OdeSolver
from scipy.integrate._ivp.ivp import (prepare_events, solve_event_equation,
                                      handle_events, find_active_events)
import time
from .common import progressBar

METHODS = {'RK23': RK23,
           'RK45': RK45,
           'Radau': Radau,
           'BDF': BDF,
           'LSODA': LSODA}


MESSAGES = {0: "The solver successfully reached the end of the integration interval.",
            1: "A termination event occurred."}


class RandomOdeResult(OptimizeResult):
    """
    Optimize result is a dictionary where each key becomes an attribute.  Neat.
    """
    pass


class parallelIntegrator(object):
    """
    parallelIntegrator: a class to integrate a function as it is being called

    Parameters:
    ----------
    func : callable
        The function that is to be integrated.  It can have the form func(t) or
        func(t,y).

    y0 : float or array, optional
        The initial value of y.  Default value is 0.

    method : string, optional
        Integration method to use:
            * 'RK45' (default): Explicit Runge-Kutta method of order 5(4) [1]_.
              The error is controlled assuming accuracy of the fourth-order
              method, but steps are taken using the fifth-order accurate
              formula (local extrapolation is done). A quartic interpolation
              polynomial is used for the dense output [2]_. Can be applied in
              the complex domain.
            * 'RK23': Explicit Runge-Kutta method of order 3(2) [3]_. The error
              is controlled assuming accuracy of the second-order method, but
              steps are taken using the third-order accurate formula (local
              extrapolation is done). A cubic Hermite polynomial is used for the
              dense output. Can be applied in the complex domain.
            * 'DOP853': Explicit Runge-Kutta method of order 8 [13]_.
              Python implementation of the "DOP853" algorithm originally
              written in Fortran [14]_. A 7-th order interpolation polynomial
              accurate to 7-th order is used for the dense output.
              Can be applied in the complex domain.
            * 'Radau': Implicit Runge-Kutta method of the Radau IIA family of
              order 5 [4]_. The error is controlled with a third-order accurate
              embedded formula. A cubic polynomial which satisfies the
              collocation conditions is used for the dense output.
            * 'BDF': Implicit multi-step variable-order (1 to 5) method based
              on a backward differentiation formula for the derivative
              approximation [5]_. The implementation follows the one described
              in [6]_. A quasi-constant step scheme is used and accuracy is
              enhanced using the NDF modification. Can be applied in the
              complex domain.
            * 'LSODA': Adams/BDF method with automatic stiffness detection and
              switching [7]_, [8]_. This is a wrapper of the Fortran solver
              from ODEPACK.
    tmax : float, optional
        Maximum magnitude of the time.  By default, 1e9.
    kwargs :
        Options passed to a chosen OdeSolver.

    Attributes
    ----------
    t0 : initial time of the integrator
    tlast : last time evaluated
    direction : direction of the integrator
    tmax : maximium value of integrator
    """
    def __init__(self, func, y0=[0.], method='RK45', tmax=1e9, **kwargs):
        if '(t, y' in str(signature(func)):
            self.func = func
        elif '(t' in str(signature(func)):
            self.func = lambda t, y: func(t)
        else:
            raise ValueError('signature %s for func not recognized'%str(signature(func)))

        self.t0 = None
        self.tlast = None
        self.direction = +1
        self.tmax = tmax
        # Now we can actually create the integrator:
        if method == 'RK45':
            self.intobj = RK45
        elif method == 'RK23':
            self.intobj = RK23
        elif method == 'DOP853':
            self.intobj = DOP853
        elif method == 'Radau':
            self.intobj = Radau
        elif method == 'BDF':
            self.intobj = BDF
        elif method == 'LSODA':
            self.intobj = LSODA
        else:
            raise ValueError('Method %s not recognized'%self.method)

        self.y0 = np.array(y0)
        self.extra_kwargs = kwargs

    def __call__(self, t):
        """
        __call: return value at time t:

        Parameters:
        -----------
        t : float or array_like
            time at which to evaluate function

        Returns:
        --------
        y : float or array_like
            value of the function at time t.
        """
        if isinstance(t, np.ndarray):
            self.__step(np.amin(t)) # start the intergrator
            self.__step(np.amax(t)) # step the integrator through to max value

            # Rebuild the solution:
            sol = OdeSolution(self.ts, self.interpolants)

            return sol(t) # return the full array
        else:
            self.__step(t) # step integrator

            if t==self.t0:
                return self.y0
            else:
                # Rebuild the solution:
                sol = OdeSolution(self.ts, self.interpolants)

                return sol(t)

    def __step(self, t):
        # Is this the first call, or did we return to the initial time?
        if self.t0 is None or t==self.t0:
            self.t0 = t
            self.tlast = None
            self.interpolants = []
            self.ts = [t]

            # Return the initial value:
            return self.y0

        # Second call, we will now establish a direction and create the solver:
        elif self.tlast == None:
            if t>self.t0:
                self.direction = +1
            elif t<self.t0:
                self.direction = -1

            self.integrator=self.intobj(self.func, self.t0, self.y0,
                                        self.direction*self.tmax, **self.extra_kwargs)

        # Did we go to a value smaller than our initial value, given the
        # direction?
        elif (t<self.t0 and self.direction==+1) or (t>self.t0 and self.direction==-1):
            # Reset the integrator.
            self.t0=None
            self.tlast=None
            # Cute way to reset the integrator to a new starting t:
            return self(t)

        # If we made it here, we did not reset yet:
        self.tlast = t

        # Now integrator up to t:
        while self.integrator.t<t:
            self.integrator.step()
            sol = self.integrator.dense_output()
            self.interpolants.append(sol)
            self.ts.append(self.integrator.t)

def solve_ivp_random(fun, random_func, t_span, y0,  method='RK45', t_eval=None,
                     dense_output=False, events=None, vectorized=False,
                     args=None, **options):
   
    if method not in METHODS and not (
            inspect.isclass(method) and issubclass(method, OdeSolver)):
        raise ValueError("`method` must be one of {} or OdeSolver class."
                         .format(METHODS))

    t0, tf = float(t_span[0]), float(t_span[1])

    if args is not None:
        fun = lambda t, x, fun=fun: fun(t, x, *args)
        jac = options.get('jac')
        if callable(jac):
            options['jac'] = lambda t, x: jac(t, x, *args)

    if t_eval is not None:
        t_eval = np.asarray(t_eval)
        if t_eval.ndim != 1:
            raise ValueError("`t_eval` must be 1-dimensional.")

        if np.any(t_eval < min(t0, tf)) or np.any(t_eval > max(t0, tf)):
            raise ValueError("Values in `t_eval` are not within `t_span`.")

        d = np.diff(t_eval)
        if tf > t0 and np.any(d <= 0) or tf < t0 and np.any(d >= 0):
            raise ValueError("Values in `t_eval` are not properly sorted.")

        if tf > t0:
            t_eval_i = 0
        else:
            t_eval = t_eval[::-1]
            t_eval_i = t_eval.shape[0]

    if method in METHODS:
        method = METHODS[method]

    max_step_initial = options.pop('initial_max_step', np.inf)
    max_step_global = options.pop('max_step', np.inf)

    solver = method(fun, t0, y0, tf, vectorized=vectorized,
                    max_step=max_step_initial, **options)

    if t_eval is None:
        ts = [t0]
        ys = [y0]
    elif t_eval is not None and dense_output:
        ts = []
        ti = [t0]
        ys = []
    else:
        ts = []
        ys = []

    interpolants = []

    events, is_terminal, event_dir = prepare_events(events)

    if events is not None:
        if args is not None:

            events = [lambda t, x, event=event: event(t, x, *args)
                      for event in events]
        g = [event(t0, y0) for event in events]
        t_events = [[] for _ in range(len(events))]
        y_events = [[] for _ in range(len(events))]
    else:
        t_events = None
        y_events = None

    t_random = []
    n_random = []

    status = None
    while status is None:
        message = solver.step()

        if solver.status == 'finished':
            status = 0
        elif solver.status == 'failed':
            status = -1
            break

        (random_event_number, max_step) = random_func(solver.t, solver.y,
                                                      solver.step_size)
        if not max_step is None:
            solver.max_step = np.min([max_step, max_step_global])

        t_old = solver.t_old
        t = solver.t
        y = solver.y

        if random_event_number>0:
            t_random.append(t)
            n_random.append(int(random_event_number))

        if dense_output:
            sol = solver.dense_output()
            interpolants.append(sol)
        else:
            sol = None

        if events is not None:
            g_new = [event(t, y) for event in events]
            active_events = find_active_events(g, g_new, event_dir)
            if active_events.size > 0:
                if sol is None:
                    sol = solver.dense_output()

                root_indices, roots, terminate = handle_events(
                    sol, events, active_events, is_terminal, t_old, t)

                for e, te in zip(root_indices, roots):
                    t_events[e].append(te)
                    y_events[e].append(sol(te))

                if terminate:
                    status = 1
                    t = roots[-1]
                    y = sol(t)

            g = g_new

        if t_eval is None:
            ts.append(t)
            ys.append(y)
        else:
            if solver.direction > 0:
                t_eval_i_new = np.searchsorted(t_eval, t, side='right')
                t_eval_step = t_eval[t_eval_i:t_eval_i_new]
            else:
                t_eval_i_new = np.searchsorted(t_eval, t, side='left')
                t_eval_step = t_eval[t_eval_i_new:t_eval_i][::-1]

            if t_eval_step.size > 0:
                if sol is None:
                    sol = solver.dense_output()
                ts.append(t_eval_step)
                ys.append(sol(t_eval_step))
                t_eval_i = t_eval_i_new

        if t_eval is not None and dense_output:
            ti.append(t)

    message = MESSAGES.get(status, message)

    if t_events is not None:
        t_events = [np.asarray(te) for te in t_events]
        y_events = [np.asarray(ye) for ye in y_events]

    if t_eval is None:
        ts = np.array(ts)
        ys = np.vstack(ys).T
    else:
        ts = np.hstack(ts)
        ys = np.hstack(ys)

    if len(t_random)>0:
        t_random = np.array(t_random)
        n_random = np.vstack(n_random)

    if dense_output:
        if t_eval is None:
            sol = OdeSolution(ts, interpolants)
        else:
            sol = OdeSolution(ti, interpolants)
    else:
        sol = None

    inds_random = np.zeros(ts.shape, dtype='bool')
    for t_i in t_random:
        inds_random = np.bitwise_or(inds_random, ts==t_i)

    return RandomOdeResult(t=ts, y=ys, sol=sol, t_events=t_events,
                           y_events=y_events, t_random=t_random,
                           n_random=n_random, inds_random=inds_random,
                           nfev=solver.nfev, njev=solver.njev,
                           nlu=solver.nlu, status=status, message=message,
                           success=status >= 0)

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    def dydt(t, y):
        return np.array([-y[1], y[0]])

    def func2(t, y, dt):
        if np.random.rand()<2*dt:
            y[1]+=5*np.random.randn()
            return (True, max(0.1, y[1]))
        else:
            return (False, max(0.1, y[1]))

    sol = solve_ivp_random(dydt, func2, [0, 2*np.pi], [0, 1],
                           max_step=0.1, method='RK45')

    plt.figure()
    plt.plot(sol.t, sol.y.T)
    plt.plot(sol.t_random, sol.y[:, sol.inds_random].T, '.')
