# import numpy as np
import cupy as cp
from .base import OdeSolver, DenseOutput
from .common import validate_max_step, validate_tol, warn_extraneous, validate_first_step
from scipy.optimize import OptimizeResult

SAFETY = 0.9
MIN_FACTOR = 0.2
MAX_FACTOR = 10

def rk_step_matrix(fun, t, y, f, h, A, B, C, K):
    K[0] = f
    for s, (a, c) in enumerate(zip(A[1:], C[1:]), start=1):
        dy = cp.zeros_like(y)
        for i in range(s):
            dy += a[i] * K[i]
        dy *= h
        K[s] = fun(t + c * h, y + dy)
    y_new = y.copy()
    for i in range(len(B)):
        y_new += h * B[i] * K[i]
    f_new = fun(t + h, y_new)
    K[-1] = f_new
    return y_new, f_new

class RungeKuttaMatrix(OdeSolver):
    C: cp.ndarray = NotImplemented
    A: cp.ndarray = NotImplemented
    B: cp.ndarray = NotImplemented
    E: cp.ndarray = NotImplemented
    P: cp.ndarray = NotImplemented
    order: int = NotImplemented
    error_estimator_order: int = NotImplemented
    n_stages: int = NotImplemented

    def __init__(self, fun, t0, y0, t_bound, max_step=cp.inf,
                 rtol=1e-3, atol=1e-6, vectorized=False,
                 first_step=None, **extraneous):
        warn_extraneous(extraneous)
        y0 = cp.asarray(y0)
        if cp.issubdtype(y0.dtype, cp.complexfloating):
            dtype = complex
        else:
            dtype = float
        y0 = y0.astype(dtype, copy=False)
        if y0.ndim == 1:
            y0 = y0.reshape(-1, 1)
        elif y0.ndim != 2:
            raise ValueError("`y0` must be 1-dimensional or 2-dimensional for matrix solver.")
        def fun_wrapped(t, y):
            return cp.asarray(fun(t, y), dtype=dtype)
        self.t = self.t_old = float(t0)
        self.t_bound = float(t_bound)
        if self.t == self.t_bound:
            raise ValueError("Integration time is zero.")
        if self.t_bound > self.t:
            self.direction = 1
        else:
            self.direction = -1
        self.y = y0
        self.n = y0.shape[0] * y0.shape[1]
        self.status = 'running'
        self.nfev = 0
        self.njev = 0
        self.nlu = 0
        self.fun = fun_wrapped
        self.vectorized = vectorized
        self.support_complex = True
        self.y_old = None
        self.max_step = validate_max_step(max_step)
        self.TOO_SMALL_STEP = "Required step size is less than spacing between numbers."
        self._fun_single = self.fun
        self._fun_vectorized = self.fun
        if y0.ndim == 2:
            n_total = y0.shape[0] * y0.shape[1]
        else:
            n_total = y0.shape[0]
        self.rtol, self.atol = validate_tol(rtol, atol, n_total)
        if y0.ndim == 2:
            self.rtol = cp.full(y0.shape, self.rtol)
            self.atol = cp.full(y0.shape, self.atol)
        self.f = self._fun_single(self.t, self.y)
        self.nfev += 1
        if first_step is None:
            self.h_abs = self._select_initial_step_matrix()
        else:
            self.h_abs = validate_first_step(first_step, t0, t_bound)
        self.K = cp.empty((self.n_stages + 1,) + self.y.shape, dtype=self.y.dtype)
        self.error_exponent = -1 / (self.error_estimator_order + 1)
        self.h_previous = None
    
    def _select_initial_step_matrix(self):
        t0 = self.t
        y0 = self.y
        f0 = self.f
        direction = self.direction
        order = self.order
        rtol = self.rtol
        atol = self.atol
        if y0.size == 0:
            return cp.inf
        scale = atol + cp.abs(y0) * rtol
        d0 = cp.sqrt(cp.mean((y0 / scale) ** 2))
        d1 = cp.sqrt(cp.mean((f0 / scale) ** 2))
        if d0 < 1e-5 or d1 < 1e-5:
            h0 = 1e-6
        else:
            h0 = 0.01 * d0 / d1
        y1 = y0 + h0 * direction * f0
        f1 = self._fun_single(t0 + h0 * direction, y1)
        self.nfev += 1
        d2 = cp.sqrt(cp.mean(((f1 - f0) / scale) ** 2)) / h0
        if d1 <= 1e-15 and d2 <= 1e-15:
            h1 = max(1e-6, h0 * 1e-3)
        else:
            h1 = (0.01 / max(d1, d2)) ** (1.0 / (order + 1))
        return min(100 * h0, h1)
    
    def step(self):
        if self.status != 'running':
            raise RuntimeError("Solver is not running.")
        success, message = self._step_impl()
        if not success:
            self.status = 'failed'
            return message
        self.t_old = self.t
        if self.direction * (self.t - self.t_bound) >= 0:
            self.status = 'finished'
        return None
    
    def dense_output(self):
        return self._dense_output_impl()
    
    def _estimate_error(self, K, h):
        error = cp.zeros_like(self.y)
        for i, e in enumerate(self.E):
            if i < K.shape[0]:
                error += e * K[i]
        return error * h

    def _estimate_error_norm(self, K, h, scale):
        error = self._estimate_error(K, h)
        return cp.sqrt(cp.mean((error / scale) ** 2))

    def _step_impl(self):
        t = self.t
        y = self.y
        max_step = self.max_step
        rtol = self.rtol
        atol = self.atol
        min_step = 10 * cp.abs(cp.nextafter(t, self.direction * cp.inf) - t)
        if self.h_abs > max_step:
            h_abs = max_step
        elif self.h_abs < min_step:
            h_abs = min_step
        else:
            h_abs = self.h_abs
        step_accepted = False
        step_rejected = False
        while not step_accepted:
            if h_abs < min_step:
                return False, self.TOO_SMALL_STEP
            h = h_abs * self.direction
            t_new = t + h
            if self.direction * (t_new - self.t_bound) > 0:
                t_new = self.t_bound
                h = t_new - t
                h_abs = cp.abs(h)
            y_new, f_new = rk_step_matrix(self._fun_single, t, y, self.f, h, self.A,
                                        self.B, self.C, self.K)
            self.nfev += self.n_stages + 1
            scale = atol + cp.maximum(cp.abs(y), cp.abs(y_new)) * rtol
            error_norm = self._estimate_error_norm(self.K, h, scale)
            if error_norm < 1:
                if error_norm == 0:
                    factor = MAX_FACTOR
                else:
                    factor = min(MAX_FACTOR,
                                SAFETY * error_norm ** self.error_exponent)
                if step_rejected:
                    factor = min(1, factor)
                h_abs *= factor
                step_accepted = True
            else:
                h_abs *= max(MIN_FACTOR,
                             SAFETY * error_norm ** self.error_exponent)
                step_rejected = True
        self.h_previous = h
        self.y_old = y
        self.t = t_new
        self.y = y_new
        self.h_abs = h_abs
        self.f = f_new
        return True, None

    def _dense_output_impl(self):
        Q = cp.zeros((self.K.shape[0] - 1,) + self.y.shape)
        for i in range(self.K.shape[0] - 1):
            for j in range(self.P.shape[1]):
                Q[i] += self.P[i, j] * self.K[j]
        return RkDenseOutputMatrix(self.t_old, self.t, self.y_old, Q)

class RK45Matrix(RungeKuttaMatrix):
    order = 5
    error_estimator_order = 4
    n_stages = 6
    C = cp.array([0, 1/5, 3/10, 4/5, 8/9, 1])
    A = cp.array([
        [0, 0, 0, 0, 0],
        [1/5, 0, 0, 0, 0],
        [3/40, 9/40, 0, 0, 0],
        [44/45, -56/15, 32/9, 0, 0],
        [19372/6561, -25360/2187, 64448/6561, -212/729, 0],
        [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656]
    ])
    B = cp.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84])
    E = cp.array([-71/57600, 0, 71/16695, -71/1920, 17253/339200, -22/525,
                  1/40])
    P = cp.array([
        [1, -8048581381/2820520608, 8663915743/2820520608,
         -12715105075/11282082432],
        [0, 0, 0, 0],
        [0, 131558114200/32700410799, -68118460800/10900136933,
         87487479700/32700410799],
        [0, -1754552775/470086768, 14199869525/1410260304,
         -10690763975/1880347072],
        [0, 127303824393/49829197408, -318862633887/49829197408,
         701980252875 / 199316789632],
        [0, -282668133/205662961, 2019193451/616988883, -1453857185/822651844],
        [0, 40617522/29380423, -110615467/29380423, 69997945/29380423]])


class RkDenseOutputMatrix(DenseOutput):
    def __init__(self, t_old, t, y_old, Q):
        super().__init__(t_old, t)
        self.y_old = y_old
        self.Q = Q

    def _call_impl(self, t):
        dt = self.t - self.t_old
        if abs(dt) < 1e-15:
            return self.y_old
        x = (t - self.t_old) / dt
        if cp.isscalar(x):
            x = cp.asarray(x)
            p = cp.array([1, x, x**2, x**3])
            result = self.y_old.copy()
            for i in range(self.Q.shape[0]):
                result += p[i] * self.Q[i]
        else:
            x = cp.asarray(x)
            result = cp.zeros((len(x),) + self.y_old.shape)
            for j in range(len(x)):
                x_val = cp.asarray(x[j])
                result[j] = self.y_old.copy()
                p_val = cp.array([1, x_val, x_val**2, x_val**3])
                for i in range(min(len(p_val), self.Q.shape[0])):
                    result[j] += p_val[i] * self.Q[i]
        return result


def solve_ivp_matrix(fun, t_span, y0, method='RK45', t_eval=None, 
                    dense_output=False, events=None, vectorized=False, 
                    args=None, **options):
    if method == 'RK45':
        method = RK45Matrix
    else:
        raise ValueError(f"Matrix version not implemented for method {method}")
    
    t0, tf = map(float, t_span)
    y0 = cp.asarray(y0)
    
    if args is not None:
        try:
            _ = [*(args)]
        except TypeError as exp:
            suggestion_tuple = (
                "Supplied 'args' cannot be unpacked. Please supply `args`"
                f" as a tuple (e.g. `args=({args},)`)"
            )
            raise TypeError(suggestion_tuple) from exp
        fun = lambda t, x, fun=fun: fun(t, x, *args)
    
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
            t_eval = t_eval[::-1]
            t_eval_i = t_eval.shape[0]
    
    solver = method(fun, t0, y0, tf, vectorized=vectorized, **options)
    
    if t_eval is None:
        ts = [t0]
        ys = [y0]
    else:
        ts = []
        ys = []
    
    status = None
    while status is None:
        message = solver.step()
        
        if solver.status == 'finished':
            status = 0
        elif solver.status == 'failed':
            status = -1
            break
        
        t = solver.t
        y = solver.y
        
        if t_eval is None:
            ts.append(t)
            ys.append(y.copy())
        else:
            if solver.direction > 0:
                t_eval_i_new = cp.searchsorted(t_eval, t, side='right')
                t_eval_step = t_eval[t_eval_i:t_eval_i_new]
            else:
                t_eval_i_new = cp.searchsorted(t_eval, t, side='left')
                t_eval_step = t_eval[t_eval_i_new:t_eval_i][::-1]
            
            if t_eval_step.size > 0:
                sol = solver.dense_output()
                ts.append(t_eval_step)
                ys.append(sol(t_eval_step))
                t_eval_i = t_eval_i_new
    
    if t_eval is None:
        ts = cp.array(ts)
        ys = cp.array(ys).transpose(1, 2, 0)
    elif ts:
        ts = cp.hstack(ts)
        ys_concat = []
        for y_batch in ys:
            if y_batch.ndim == 3:
                ys_concat.append(y_batch.transpose(1, 2, 0))
            else:
                ys_concat.append(y_batch[:, :, cp.newaxis])
        ys = cp.concatenate(ys_concat, axis=2)
    
    return OptimizeResult(t=ts, y=ys)