import copy
import numpy as np
from abc import ABC, abstractmethod
from enum import Enum
from typing import List


class InterpMethod(Enum):

    PIECEWISE_CONSTANT_LEFT_CONTINUOUS = 'PIECEWISE_CONSTANT_LEFT_CONTINUOUS'
    LINEAR = 'LINEAR'
    MONOTONE_CONVEX = 'MONOTONE_CONVEX'

    @classmethod
    def from_string(cls, value: str) -> 'InterpMethod':
        if not isinstance(value, str):
            raise TypeError("value must be a string")
        try:
            return cls(value.upper())
        except ValueError:
            raise ValueError(f"Invalid token: {value}")

    def to_string(self) -> str:
        return self.value


class ExtrapMethod(Enum):

    FLAT = 'FLAT'
    LINEAR = 'LINEAR'

    @classmethod
    def from_string(cls, value: str) -> 'ExtrapMethod':
        if not isinstance(value, str):
            raise TypeError("value must be a string")
        try:
            return cls(value.upper())
        except ValueError:
            raise ValueError(f"Invalid token: {value}")

    def to_string(self) -> str:
        return self.value


class Interpolator1D(ABC):
    """Abstract interface for a 1-D interpolator."""

    def __init__(self,
                 axis1: np.ndarray,
                 values: np.ndarray,
                 interpolation_method: InterpMethod,
                 extrapolation_method: ExtrapMethod) -> None:

        self.axis1_ = axis1
        self.values_ = values
        self.interp_method_ = interpolation_method
        self.extrap_method_ = extrapolation_method
        self.length_ = len(self.axis1_)

    @abstractmethod
    def interpolate(self, x: float) -> float:
        pass

    @abstractmethod
    def integrate(self, start_x: float, end_x: float) -> float:
        pass

    @abstractmethod
    def gradient_wrt_ordinate(self, x: float) -> np.ndarray:
        pass

    @abstractmethod
    def gradient_of_integrated_value_wrt_ordinate(self, start_x: float, end_x: float) -> np.ndarray:
        pass

    @property
    def axis1(self) -> np.ndarray:
        return self.axis1_

    @property
    def values(self) -> np.ndarray:
        return self.values_

    @property
    def length(self) -> int:
        return self.length_

    @property
    def interp_method(self) -> str:
        return self.interp_method_.to_string()

    @property
    def extrap_method(self) -> str:
        return self.extrap_method_.to_string()


class Interpolator1DPCP(Interpolator1D):
    """Piecewise-constant left-continuous interpolator with FLAT extrapolation.

    With axis1 = [1, 3, 5, 7], values = [3, 4, 5, 6]:
        f(0.5) = 3, f(1) = 3, f(1.5) = 4, f(3) = 4, f(5.5) = 6, f(8) = 6
    """

    def __init__(self, axis1: np.ndarray, values: np.ndarray,
                 extrapolation_method: ExtrapMethod) -> None:
        super().__init__(axis1, values,
                         InterpMethod.PIECEWISE_CONSTANT_LEFT_CONTINUOUS,
                         extrapolation_method)
        assert self.extrap_method_ == ExtrapMethod.FLAT

    def interpolate(self, x: float) -> float:
        idx = min(int(np.searchsorted(self.axis1_, x, side='left')), self.length_ - 1)
        return float(self.values_[idx])

    def integrate(self, start_x: float, end_x: float) -> float:
        edges = np.concatenate(([-np.inf], np.asarray(self.axis1_, dtype=float)[:-1], [np.inf]))
        lo = np.maximum(start_x, edges[:-1])
        hi = np.minimum(end_x, edges[1:])
        overlaps = np.maximum(hi - lo, 0.0)
        return float(np.dot(np.asarray(self.values_, dtype=float), overlaps))

    def gradient_wrt_ordinate(self, x: float) -> np.ndarray:
        idx = min(int(np.searchsorted(self.axis1_, x, side='left')), self.length_ - 1)
        grad = np.zeros(self.length_)
        grad[idx] = 1.0
        return grad

    def gradient_of_integrated_value_wrt_ordinate(self, start_x: float, end_x: float) -> np.ndarray:
        edges = np.concatenate(([-np.inf], np.asarray(self.axis1_, dtype=float)[:-1], [np.inf]))
        lo = np.maximum(start_x, edges[:-1])
        hi = np.minimum(end_x, edges[1:])
        return np.maximum(hi - lo, 0.0)



class Interpolator1DMonotoneConvex(Interpolator1D):
    """Monotone convex piecewise-cubic Hermite interpolator with FLAT extrapolation.

    On each interval [t_i, t_{i+1}] (h = width, D = secant slope) the cubic is
        p(tau) = f_i + dL*tau + c2*tau^2 + c3*tau^3,
        c2 = (2a - b)/h,  c3 = (b - a)/h^2,
    where a = D - dL and b = dR - D are the end-slope deviations from the secant.

    Shape conditions on one interval (p'' is linear, so check the two ends):
        convex   <=>  a >= 0, b >= 0, a/2 <= b <= 2a
        monotone <=>  a <= D (if D > 0)  or  b <= |D| (if D < 0); a = b = 0 if D = 0
    Candidate node slopes are Bessel (3-point parabola) slopes, zero at local extrema,
    and natural (p'' = 0) at the end nodes. Each interval then projects its own
    (a, b) onto the feasible set, which always contains (0, 0) = linear. Hence the
    result always interpolates, is continuous, and is monotone and convex on every
    interval; C1 at a node is kept whenever the candidate slope is feasible on both
    sides, otherwise a small slope kink is accepted.
    """

    def __init__(self, axis1: np.ndarray, values: np.ndarray,
                 extrapolation_method: ExtrapMethod) -> None:
        super().__init__(axis1, values, InterpMethod.MONOTONE_CONVEX,
                         extrapolation_method)
        assert self.extrap_method_ == ExtrapMethod.FLAT
        assert self.length_ >= 1

        t = np.asarray(self.axis1_, dtype=float)
        f = np.asarray(self.values_, dtype=float)
        assert np.all(np.diff(t) > 0), "node times must be strictly increasing"
        self.t_ = t
        self.f_ = f
        n = self.length_ - 1  # number of intervals

        if n == 0:
            self.dL_ = self.c2_ = self.c3_ = np.zeros(0)
            self.cum_ = np.zeros(1)
            return

        h = np.diff(t)
        delta = np.diff(f) / h

        # Step 1: candidate (shared) node slopes.
        d = np.zeros(n + 1)
        if n >= 2:
            d[1:-1] = (h[1:] * delta[:-1] + h[:-1] * delta[1:]) / (h[:-1] + h[1:])
            d[1:-1][delta[:-1] * delta[1:] <= 0.0] = 0.0  # local extremum / flat
            d[0] = 0.5 * (3.0 * delta[0] - d[1])          # natural end: p''(t_0) = 0
            d[-1] = 0.5 * (3.0 * delta[-1] - d[-2])       # natural end: p''(t_n) = 0
        else:
            d[:] = delta[0]                               # two nodes: straight line

        # Step 2: per-interval projection of (a, b) onto the feasible set.
        a = delta - d[:-1]
        b = d[1:] - delta
        for i in range(n):
            a[i], b[i] = self._project(a[i], b[i], delta[i])

        self.h_ = h
        self.dL_ = delta - a
        self.dR_ = delta + b
        self.c2_ = (2.0 * a - b) / h
        self.c3_ = (b - a) / h**2

        # Cumulative integral from t_0 to each node.
        self.cum_ = np.concatenate(([0.0], np.cumsum(self._poly_integral(np.arange(n), h))))

    @staticmethod
    def _project(a: float, b: float, delta: float):
        if delta == 0.0:
            return 0.0, 0.0
        a, b = max(a, 0.0), max(b, 0.0)        # no concave part
        a, b = min(a, 2.0 * b), min(b, 2.0 * a)  # factor-2 band: p'' >= 0 at both ends
        if delta > 0.0 and a > delta:            # monotone increasing: dL >= 0
            a = delta
        elif delta < 0.0 and b > -delta:         # monotone decreasing: dR <= 0
            b = -delta
        a, b = min(a, 2.0 * b), min(b, 2.0 * a)  # re-apply band after the cap
        return a, b

    def _poly_integral(self, idx, u):
        """Integral of p_idx over [t_idx, t_idx + u]."""
        return (self.f_[idx] * u + 0.5 * self.dL_[idx] * u**2
                + self.c2_[idx] * u**3 / 3.0 + 0.25 * self.c3_[idx] * u**4)

    def _interval(self, x: float) -> int:
        return min(max(int(np.searchsorted(self.t_, x, side='right')) - 1, 0), self.length_ - 2)

    def _antiderivative(self, x: float) -> float:
        """F(x) = integral of f from t_0 to x (negative for x < t_0)."""
        t0, tn = self.t_[0], self.t_[-1]
        if x <= t0:
            return self.f_[0] * (x - t0)
        if x >= tn:
            return self.cum_[-1] + self.f_[-1] * (x - tn)
        i = self._interval(x)
        return self.cum_[i] + self._poly_integral(i, x - self.t_[i])

    def interpolate(self, x: float) -> float:
        if x <= self.t_[0]:
            return float(self.f_[0])
        if x >= self.t_[-1]:
            return float(self.f_[-1])
        i = self._interval(x)
        tau = x - self.t_[i]
        return float(self.f_[i] + tau * (self.dL_[i] + tau * (self.c2_[i] + tau * self.c3_[i])))

    def integrate(self, start_x: float, end_x: float) -> float:
        return float(self._antiderivative(end_x) - self._antiderivative(start_x))

    # Optional risk interface
    # The test cells for risk are not prepared but you can do it by your own.
    def gradient_wrt_ordinate(self, x: float) -> np.ndarray:
        raise NotImplementedError("Optional risk: ordinate gradient is not implemented")

    def gradient_of_integrated_value_wrt_ordinate(
        self, start_x: float, end_x: float,
    ) -> np.ndarray:
        raise NotImplementedError("Optional risk: integrated ordinate gradient is not implemented")


class InterpolatorFactory:

    @staticmethod
    def create_1d_interpolator(axis1: np.ndarray | List,
                               values: np.ndarray | List,
                               interpolation_method: InterpMethod,
                               extrapolation_method: ExtrapMethod):

        axis1_ = copy.deepcopy(axis1)
        values_ = copy.deepcopy(values)
        if isinstance(axis1_, list):
            axis1_ = np.array(axis1_)
        if isinstance(values_, list):
            values_ = np.array(values_)
        assert len(axis1_.shape) == 1 and len(values_.shape) == 1
        assert len(axis1_) == len(values_)
        assert np.all(np.diff(axis1_) >= 0)

        if interpolation_method == InterpMethod.PIECEWISE_CONSTANT_LEFT_CONTINUOUS:
            return Interpolator1DPCP(axis1_, values_, extrapolation_method)
        elif interpolation_method == InterpMethod.MONOTONE_CONVEX:
            return Interpolator1DMonotoneConvex(axis1_, values_, extrapolation_method)
        else:
            raise ValueError(f'Unsupported interpolation method: {interpolation_method}')
