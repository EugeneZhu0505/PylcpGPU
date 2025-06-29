import inspect
# import numpy as np
import cupy as cp
from .bdf import BDF
from .radau import Radau
from .rk import RK23, RK45, DOP853
from .lsoda import LSODA
from scipy.optimize import OptimizeResult
from .common import EPS, OdeSolution
from .base import OdeSolver


METHODS = {'RK23': RK23,
           'RK45': RK45,
           'DOP853': DOP853,
           'Radau': Radau,
           'BDF': BDF,
           'LSODA': LSODA}


MESSAGES = {0: "The solver successfully reached the end of the integration interval.",
            1: "A termination event occurred."}


class OdeResult(OptimizeResult):
    pass


def prepare_events(events):
    if callable(events):
        events = (events,)

    if events is not None:
        is_terminal = cp.empty(len(events), dtype=bool)
        direction = cp.empty(len(events))
        for i, event in enumerate(events):
            try:
                is_terminal[i] = event.terminal
            except AttributeError:
                is_terminal[i] = False

            try:
                direction[i] = event.direction
            except AttributeError:
                direction[i] = 0
    else:
        is_terminal = None
        direction = None

    return events, is_terminal, direction


def solve_event_equation(event, sol, t_old, t):

    from scipy.optimize import brentq
    return brentq(lambda t: event(t, sol(t)), t_old, t,
                  xtol=4 * EPS, rtol=4 * EPS)


def handle_events(sol, events, active_events, is_terminal, t_old, t):

    roots = [solve_event_equation(events[event_index], sol, t_old, t)
             for event_index in active_events]

    roots = cp.asarray(roots)

    if cp.any(is_terminal[active_events]):
        if t > t_old:
            order = cp.argsort(roots)
        else:
            order = cp.argsort(-roots)
        active_events = active_events[order]
        roots = roots[order]
        t = cp.nonzero(is_terminal[active_events])[0][0]
        active_events = active_events[:t + 1]
        roots = roots[:t + 1]
        terminate = True
    else:
        terminate = False

    return active_events, roots, terminate


def find_active_events(g, g_new, direction):
    
    g, g_new = cp.asarray(g), cp.asarray(g_new)
    up = (g <= 0) & (g_new >= 0)
    down = (g >= 0) & (g_new <= 0)
    either = up | down
    mask = (up & (direction > 0) |
            down & (direction < 0) |
            either & (direction == 0))

    return cp.nonzero(mask)[0]


def solve_ivp(fun, t_span, y0, method='RK45', t_eval=None, dense_output=False,
              events=None, vectorized=False, args=None, **options):

    if method not in METHODS and not (
            inspect.isclass(method) and issubclass(method, OdeSolver)):
        raise ValueError("`method` must be one of {} or OdeSolver class."
                         .format(METHODS))

    t0, tf = map(float, t_span)

    if args is not None:
        # Wrap the user's fun (and jac, if given) in lambdas to hide the
        # additional parameters.  Pass in the original fun as a keyword
        # argument to keep it in the scope of the lambda.
        try:
            _ = [*(args)]
        except TypeError as exp:
            suggestion_tuple = (
                "Supplied 'args' cannot be unpacked. Please supply `args`"
                f" as a tuple (e.g. `args=({args},)`)"
            )
            raise TypeError(suggestion_tuple) from exp

        fun = lambda t, x, fun=fun: fun(t, x, *args)
        jac = options.get('jac')
        if callable(jac):
            options['jac'] = lambda t, x: jac(t, x, *args)

    if t_eval is not None:
        t_eval = cp.asarray(t_eval)
        if t_eval.ndim != 1:
            raise ValueError("`t_eval` must be 1-dimensional.")

        if cp.any(t_eval < min(t0, tf)) or cp.any(t_eval > max(t0, tf)):
            raise ValueError("Values in `t_eval` are not within `t_span`.")

        d = cp.diff(t_eval)
        if tf > t0 and cp.any(d <= 0) or tf < t0 and cp.any(d >= 0):
            raise ValueError("Values in `t_eval` are not properly sorted.")

        if tf > t0:
            t_eval_i = 0
        else:
            # Make order of t_eval decreasing to use np.searchsorted.
            t_eval = t_eval[::-1]
            # This will be an upper bound for slices.
            t_eval_i = t_eval.shape[0]

    if method in METHODS:
        method = METHODS[method]

    solver = method(fun, t0, y0, tf, vectorized=vectorized, **options)

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
            # Wrap user functions in lambdas to hide the additional parameters.
            # The original event function is passed as a keyword argument to the
            # lambda to keep the original function in scope (i.e., avoid the
            # late binding closure "gotcha").
            events = [lambda t, x, event=event: event(t, x, *args)
                      for event in events]
        g = [event(t0, y0) for event in events]
        t_events = [[] for _ in range(len(events))]
        y_events = [[] for _ in range(len(events))]
    else:
        t_events = None
        y_events = None

    status = None
    while status is None:
        message = solver.step()

        if solver.status == 'finished':
            status = 0
        elif solver.status == 'failed':
            status = -1
            break

        t_old = solver.t_old
        t = solver.t
        y = solver.y

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
            # The value in t_eval equal to t will be included.
            if solver.direction > 0:
                t_eval_i_new = cp.searchsorted(t_eval, t, side='right')
                t_eval_step = t_eval[t_eval_i:t_eval_i_new]
            else:
                t_eval_i_new = cp.searchsorted(t_eval, t, side='left')
                # It has to be done with two slice operations, because
                # you can't slice to 0th element inclusive using backward
                # slicing.
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
        t_events = [cp.asarray(te) for te in t_events]
        y_events = [cp.asarray(ye) for ye in y_events]

    if t_eval is None:
        ts = cp.array(ts)
        ys = cp.vstack(ys).T
    elif ts:
        ts = cp.hstack(ts)
        ys = cp.hstack(ys)

    if dense_output:
        if t_eval is None:
            sol = OdeSolution(ts, interpolants)
        else:
            sol = OdeSolution(ti, interpolants)
    else:
        sol = None

    return OdeResult(t=ts, y=ys, sol=sol, t_events=t_events, y_events=y_events,
                     nfev=solver.nfev, njev=solver.njev, nlu=solver.nlu,
                     status=status, message=message, success=status >= 0)
