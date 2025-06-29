from itertools import groupby
from warnings import warn
import numpy as np
import cupy as cp
# from scipy.sparse import find, coo_matrix
from cupyx.scipy.sparse import find, coo_matrix


EPS = cp.finfo(float).eps


def validate_first_step(first_step, t0, t_bound):
    if first_step <= 0:
        raise ValueError("`first_step` must be positive.")
    if first_step > cp.abs(t_bound - t0):
        raise ValueError("`first_step` exceeds bounds.")
    return first_step


def validate_max_step(max_step):
    if max_step <= 0:
        raise ValueError("`max_step` must be positive.")
    return max_step


def warn_extraneous(extraneous):
    
    if extraneous:
        warn("The following arguments have no effect for a chosen solver: {}."
             .format(", ".join("`{}`".format(x) for x in extraneous)))


def validate_tol(rtol, atol, n):

    if np.any(rtol < 100 * EPS):
        warn("At least one element of `rtol` is too small. "
             f"Setting `rtol = cp.maximum(rtol, {100 * EPS})`.")
        rtol = cp.maximum(rtol, 100 * EPS)

    atol = cp.asarray(atol)
    if atol.ndim > 0 and atol.shape != (n,):
        raise ValueError("`atol` has wrong shape.")

    if np.any(atol < 0):
        raise ValueError("`atol` must be positive.")

    return rtol, atol


def norm(x):
    if hasattr(x, '__array_ufunc__') and x.__class__.__module__ == 'cupy':
        return cp.linalg.norm(x) / x.size ** 0.5
    else:
        return cp.linalg.norm(x) / x.size ** 0.5


def select_initial_step(fun, t0, y0, f0, direction, order, rtol, atol):
   
    if y0.size == 0:
        return cp.inf

    scale = atol + cp.abs(y0) * rtol
    d0 = norm(y0 / scale)
    d1 = norm(f0 / scale)
    if d0 < 1e-5 or d1 < 1e-5:
        h0 = 1e-6
    else:
        h0 = 0.01 * d0 / d1

    y1 = y0 + h0 * direction * f0
    f1 = fun(t0 + h0 * direction, y1)
    d2 = norm((f1 - f0) / scale) / h0

    if d1 <= 1e-15 and d2 <= 1e-15:
        h1 = max(1e-6, h0 * 1e-3)
    else:
        h1 = (0.01 / max(d1, d2)) ** (1 / (order + 1))

    return min(100 * h0, h1)


class OdeSolution:
    
    def __init__(self, ts, interpolants):
        ts = cp.asarray(ts)
        d = cp.diff(ts)
        # The first case covers integration on zero segment.
        if not ((ts.size == 2 and ts[0] == ts[-1])
                or cp.all(d > 0) or cp.all(d < 0)):
            raise ValueError("`ts` must be strictly increasing or decreasing.")

        self.n_segments = len(interpolants)
        if ts.shape != (self.n_segments + 1,):
            raise ValueError("Numbers of time stamps and interpolants "
                             "don't match.")

        self.ts = ts
        self.interpolants = interpolants
        if ts[-1] >= ts[0]:
            self.t_min = ts[0]
            self.t_max = ts[-1]
            self.ascending = True
            self.ts_sorted = ts
        else:
            self.t_min = ts[-1]
            self.t_max = ts[0]
            self.ascending = False
            self.ts_sorted = ts[::-1]

    def _call_single(self, t):
        # Here we preserve a certain symmetry that when t is in self.ts,
        # then we prioritize a segment with a lower index.
        if self.ascending:
            ind = cp.searchsorted(self.ts_sorted, t, side='left')
        else:
            ind = cp.searchsorted(self.ts_sorted, t, side='right')

        segment = min(max(ind - 1, 0), self.n_segments - 1)
        if not self.ascending:
            segment = self.n_segments - 1 - segment

        return self.interpolants[segment](t)

    def __call__(self, t):
        
        t = cp.asarray(t)

        if t.ndim == 0:
            return self._call_single(t)

        order = cp.argsort(t)
        reverse = cp.empty_like(order)
        reverse[order] = cp.arange(order.shape[0])
        t_sorted = t[order]

        # See comment in self._call_single.
        if self.ascending:
            segments = cp.searchsorted(self.ts_sorted, t_sorted, side='left')
        else:
            segments = cp.searchsorted(self.ts_sorted, t_sorted, side='right')
        segments -= 1
        segments[segments < 0] = 0
        segments[segments > self.n_segments - 1] = self.n_segments - 1
        if not self.ascending:
            segments = self.n_segments - 1 - segments

        ys = []
        group_start = 0
        for segment, group in groupby(segments):
            group_end = group_start + len(list(group))
            y = self.interpolants[segment](t_sorted[group_start:group_end])
            ys.append(y)
            group_start = group_end

        ys = cp.hstack(ys)
        ys = ys[:, reverse]

        return ys


NUM_JAC_DIFF_REJECT = EPS ** 0.875
NUM_JAC_DIFF_SMALL = EPS ** 0.75
NUM_JAC_DIFF_BIG = EPS ** 0.25
NUM_JAC_MIN_FACTOR = 1e3 * EPS
NUM_JAC_FACTOR_INCREASE = 10
NUM_JAC_FACTOR_DECREASE = 0.1


def num_jac(fun, t, y, f, threshold, factor, sparsity=None):
    
    y = cp.asarray(y)
    n = y.shape[0]
    if n == 0:
        return cp.empty((0, 0)), factor

    if factor is None:
        factor = cp.full(n, EPS ** 0.5)
    else:
        factor = factor.copy()

    
    f_sign = 2 * (cp.real(f) >= 0).astype(float) - 1
    y_scale = f_sign * cp.maximum(threshold, cp.abs(y))
    h = (y + factor * y_scale) - y


    for i in cp.nonzero(h == 0)[0]:
        while h[i] == 0:
            factor[i] *= 10
            h[i] = (y[i] + factor[i] * y_scale[i]) - y[i]

    if sparsity is None:
        return _dense_num_jac(fun, t, y, f, h, factor, y_scale)
    else:
        structure, groups = sparsity
        return _sparse_num_jac(fun, t, y, f, h, factor, y_scale,
                               structure, groups)


def _dense_num_jac(fun, t, y, f, h, factor, y_scale):
    n = y.shape[0]
    h_vecs = cp.diag(h)
    f_new = fun(t, y[:, None] + h_vecs)
    diff = f_new - f[:, None]
    max_ind = cp.argmax(cp.abs(diff), axis=0)
    r = cp.arange(n)
    max_diff = cp.abs(diff[max_ind, r])
    scale = cp.maximum(cp.abs(f[max_ind]), cp.abs(f_new[max_ind, r]))

    diff_too_small = max_diff < NUM_JAC_DIFF_REJECT * scale
    if cp.any(diff_too_small):
        ind, = cp.nonzero(diff_too_small)
        new_factor = NUM_JAC_FACTOR_INCREASE * factor[ind]
        h_new = (y[ind] + new_factor * y_scale[ind]) - y[ind]
        h_vecs[ind, ind] = h_new
        f_new = fun(t, y[:, None] + h_vecs[:, ind])
        diff_new = f_new - f[:, None]
        max_ind = cp.argmax(cp.abs(diff_new), axis=0)
        r = cp.arange(ind.shape[0])
        max_diff_new = cp.abs(diff_new[max_ind, r])
        scale_new = cp.maximum(cp.abs(f[max_ind]), cp.abs(f_new[max_ind, r]))

        update = max_diff[ind] * scale_new < max_diff_new * scale[ind]
        if cp.any(update):
            update, = cp.nonzero(update)
            update_ind = ind[update]
            factor[update_ind] = new_factor[update]
            h[update_ind] = h_new[update]
            diff[:, update_ind] = diff_new[:, update]
            scale[update_ind] = scale_new[update]
            max_diff[update_ind] = max_diff_new[update]

    diff /= h

    factor[max_diff < NUM_JAC_DIFF_SMALL * scale] *= NUM_JAC_FACTOR_INCREASE
    factor[max_diff > NUM_JAC_DIFF_BIG * scale] *= NUM_JAC_FACTOR_DECREASE
    factor = cp.maximum(factor, NUM_JAC_MIN_FACTOR)

    return diff, factor


def _sparse_num_jac(fun, t, y, f, h, factor, y_scale, structure, groups):
    n = y.shape[0]
    n_groups = cp.max(groups) + 1
    h_vecs = cp.empty((n_groups, n))
    for group in range(n_groups):
        e = cp.equal(group, groups)
        h_vecs[group] = h * e
    h_vecs = h_vecs.T

    f_new = fun(t, y[:, None] + h_vecs)
    df = f_new - f[:, None]

    i, j, _ = find(structure)
    diff = coo_matrix((df[i, groups[j]], (i, j)), shape=(n, n)).tocsc()
    max_ind = cp.array(abs(diff).argmax(axis=0)).ravel()
    r = cp.arange(n)
    max_diff = cp.asarray(cp.abs(diff[max_ind, r])).ravel()
    scale = cp.maximum(cp.abs(f[max_ind]),
                       cp.abs(f_new[max_ind, groups[r]]))

    diff_too_small = max_diff < NUM_JAC_DIFF_REJECT * scale
    if cp.any(diff_too_small):
        ind, = cp.nonzero(diff_too_small)
        new_factor = NUM_JAC_FACTOR_INCREASE * factor[ind]
        h_new = (y[ind] + new_factor * y_scale[ind]) - y[ind]
        h_new_all = cp.zeros(n)
        h_new_all[ind] = h_new

        groups_unique = cp.unique(groups[ind])
        groups_map = cp.empty(n_groups, dtype=int)
        h_vecs = cp.empty((groups_unique.shape[0], n))
        for k, group in enumerate(groups_unique):
            e = cp.equal(group, groups)
            h_vecs[k] = h_new_all * e
            groups_map[group] = k
        h_vecs = h_vecs.T

        f_new = fun(t, y[:, None] + h_vecs)
        df = f_new - f[:, None]
        i, j, _ = find(structure[:, ind])
        diff_new = coo_matrix((df[i, groups_map[groups[ind[j]]]],
                               (i, j)), shape=(n, ind.shape[0])).tocsc()

        max_ind_new = cp.array(abs(diff_new).argmax(axis=0)).ravel()
        r = cp.arange(ind.shape[0])
        max_diff_new = cp.asarray(cp.abs(diff_new[max_ind_new, r])).ravel()
        scale_new = cp.maximum(
            cp.abs(f[max_ind_new]),
            cp.abs(f_new[max_ind_new, groups_map[groups[ind]]]))

        update = max_diff[ind] * scale_new < max_diff_new * scale[ind]
        if cp.any(update):
            update, = cp.nonzero(update)
            update_ind = ind[update]
            factor[update_ind] = new_factor[update]
            h[update_ind] = h_new[update]
            diff[:, update_ind] = diff_new[:, update]
            scale[update_ind] = scale_new[update]
            max_diff[update_ind] = max_diff_new[update]

    diff.data /= cp.repeat(h, cp.diff(diff.indptr))

    factor[max_diff < NUM_JAC_DIFF_SMALL * scale] *= NUM_JAC_FACTOR_INCREASE
    factor[max_diff > NUM_JAC_DIFF_BIG * scale] *= NUM_JAC_FACTOR_DECREASE
    factor = cp.maximum(factor, NUM_JAC_MIN_FACTOR)

    return diff, factor
