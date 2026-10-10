"""
Grouped nonlinear mixed-effects model shared by the fitting algorithms.

For group i the parameters passed to the user's model function are

    phi_i = g^{-1}(eta_i),   eta_i = A_i @ beta + B_i @ b_i

where A_i (p x f) and B_i (p x r) are the fixed- and random-effects design
matrices (MATLAB's FE*/RE* name-value arguments) and g is the parameter
transform selected by ``ParamTransform`` (0 identity, 1 log, 2 probit, 3 logit).
"""

import inspect
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.special import expit, ndtr

# Relative finite-difference step, as MATLAB's default statset DerivStep.
_DERIV_STEP = np.finfo(float).eps ** (1 / 3)


def inverse_transform(eta: np.ndarray, codes: np.ndarray | None) -> np.ndarray:
    """Map transformed parameters eta to model parameters phi = g^{-1}(eta)."""
    if codes is None:
        return np.array(eta, dtype=float, copy=True)
    phi = np.array(eta, dtype=float, copy=True)
    phi[..., codes == 1] = np.exp(phi[..., codes == 1])
    phi[..., codes == 2] = ndtr(phi[..., codes == 2])
    phi[..., codes == 3] = expit(phi[..., codes == 3])
    return phi


def with_jacobian(modelfun: Callable, jacobian: Callable) -> Callable:
    """Attach analytic derivatives to a model function.

    ``jacobian`` takes the same arguments as ``modelfun`` and returns
    d f / d phi with one row per observation and one column per parameter
    (phi on the model's scale; pynlme applies the chain rule through
    ParamTransform). The fitting functions then use it instead of finite
    differences.
    """

    def model(*args):
        return modelfun(*args)

    model.__signature__ = inspect.signature(modelfun)
    model.jacobian = jacobian
    return model


def _accepts_third_argument(fun: Callable) -> bool:
    """Whether fun(phi, x) would fail because a third argument is required."""
    try:
        params = list(inspect.signature(fun).parameters.values())
    except (TypeError, ValueError):
        return False
    if any(p.kind is p.VAR_POSITIONAL for p in params):
        return False
    positional = [
        p for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    return len(positional) >= 3 and positional[2].default is inspect.Parameter.empty


@dataclass
class GroupedModel:
    """Data split by group together with the parameter mapping and model."""

    X: list[np.ndarray]  # X[i]: (n_i, h) predictors for group i
    y: list[np.ndarray]  # y[i]: (n_i,) responses for group i
    V: np.ndarray | None  # (m, g) group-level predictors
    A: np.ndarray  # (m, p, f) fixed-effects design per group
    B: np.ndarray  # (m, p, r) random-effects design per group
    transform: np.ndarray | None  # (p,) ParamTransform codes
    modelfun: Callable
    rows: list[np.ndarray]  # rows[i]: indices of group i in the stacked data
    # "singlephi": modelfun(phi, x_i[, v_i]) per group, phi a vector.
    # "full": one call for many groups at once, phi with one row per
    # observation (MATLAB's Vectorization="Full").
    vectorization: str = "singlephi"

    def __post_init__(self) -> None:
        if self.vectorization not in ("singlephi", "full"):
            raise ValueError(f"Unknown vectorization '{self.vectorization}'")
        self._pass_none = self.V is None and _accepts_third_argument(self.modelfun)
        self.n_calls = 0
        self._lengths = np.array([len(y) for y in self.y])
        # Padded layout (groups x max observations) for batched linear algebra.
        self.n_max = int(self._lengths.max()) if len(self._lengths) else 0
        self.mask = np.arange(self.n_max)[None, :] < self._lengths[:, None]
        self.y_pad = np.zeros(self.mask.shape)
        self.y_pad[self.mask] = np.concatenate(self.y) if self.y else []
        # Index arrays per transform code, computed once: predict() is called
        # thousands of times per fit.
        codes = self.transform if self.transform is not None else np.zeros(0, dtype=int)
        self._exp_idx = np.flatnonzero(codes == 1)
        self._probit_idx = np.flatnonzero(codes == 2)
        self._logit_idx = np.flatnonzero(codes == 3)

    def _phi(self, eta: np.ndarray) -> np.ndarray:
        """g^{-1}(eta) for a vector or for an array of vectors (last axis)."""
        phi = eta.copy()
        if self._exp_idx.size:
            phi[..., self._exp_idx] = np.exp(phi[..., self._exp_idx])
        if self._probit_idx.size:
            phi[..., self._probit_idx] = ndtr(phi[..., self._probit_idx])
        if self._logit_idx.size:
            phi[..., self._logit_idx] = expit(phi[..., self._logit_idx])
        return phi

    def predict_flat(self, groups, etas):
        """Predictions for several (group, eta) pairs as one flat array.

        Returns the concatenated predictions and a boolean per pair telling
        whether its predictions are all finite. With vectorization "full"
        this is a single model call.
        """
        groups = np.asarray(groups, dtype=int)
        etas = np.asarray(etas, dtype=float)
        lengths = self._lengths[groups]
        if self.vectorization != "full":
            parts, valid = [], np.ones(len(groups), dtype=bool)
            for k, (g, eta) in enumerate(zip(groups, etas, strict=True)):
                try:
                    parts.append(self.predict(g, eta))
                except FloatingPointError:
                    parts.append(np.full(lengths[k], np.nan))
                    valid[k] = False
            return (np.concatenate(parts) if parts else np.zeros(0)), valid
        phi = np.repeat(self._phi(etas), lengths, axis=0)
        x = np.concatenate([self.X[g] for g in groups])
        if self.V is not None:
            values = self.modelfun(phi, x, np.repeat(self.V[groups], lengths, axis=0))
        elif self._pass_none:
            values = self.modelfun(phi, x, None)
        else:
            values = self.modelfun(phi, x)
        self.n_calls += 1
        values = np.asarray(values, dtype=float).reshape(-1)
        if values.shape[0] != lengths.sum():
            raise ValueError(
                f"modelfun returned {values.shape[0]} values for {lengths.sum()} "
                "observations "
                "(with Vectorization 'Full', phi has one row per observation)"
            )
        finite = np.isfinite(values)
        if finite.all():
            return values, np.ones(len(groups), dtype=bool)
        starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
        valid = (
            np.logical_and.reduceat(finite, starts) if len(starts) else np.ones(0, bool)
        )
        return values, valid

    def predict_many(self, groups, etas, allow_invalid: bool = False) -> list:
        """Predictions for several (group, eta) pairs, one array per pair.

        Entries with non-finite predictions are None if allow_invalid, else
        an error.
        """
        groups = np.asarray(groups, dtype=int)
        values, valid = self.predict_flat(groups, etas)
        if not valid.all() and not allow_invalid:
            raise FloatingPointError(
                f"modelfun returned non-finite values for group {groups[~valid][0]}"
            )
        out = np.split(values, np.cumsum(self._lengths[groups])[:-1])
        return [f if ok else None for f, ok in zip(out, valid, strict=True)]

    def etas_all(self, beta, b=None) -> np.ndarray:
        """Transformed parameters for every group, shape (m, p); b is (r, m)."""
        etas = self.A @ beta
        if b is not None and self.n_random:
            etas = etas + np.einsum("mpr,rm->mp", self.B, b)
        return etas

    def predict_padded(self, etas) -> np.ndarray:
        """Predictions for all groups as an (m, n_max) array, zero-padded."""
        values, valid = self.predict_flat(np.arange(self.n_groups), etas)
        if not valid.all():
            raise FloatingPointError("modelfun returned non-finite values")
        out = np.zeros(self.mask.shape)
        out[self.mask] = values
        return out

    @property
    def has_jacobian(self) -> bool:
        return callable(getattr(self.modelfun, "jacobian", None))

    def _dphi_deta(self, etas: np.ndarray) -> np.ndarray:
        """Derivative of g^{-1} (the ParamTransform) at eta, elementwise."""
        d = np.ones_like(etas)
        if self._exp_idx.size:
            d[..., self._exp_idx] = np.exp(etas[..., self._exp_idx])
        if self._probit_idx.size:
            z = etas[..., self._probit_idx]
            d[..., self._probit_idx] = np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi)
        if self._logit_idx.size:
            q = expit(etas[..., self._logit_idx])
            d[..., self._logit_idx] = q * (1 - q)
        return d

    def _call_jacobian(self, phi, x, v):
        jac = self.modelfun.jacobian
        if self.V is not None:
            return jac(phi, x, v)
        if self._pass_none:
            return jac(phi, x, None)
        return jac(phi, x)

    def jacobian_flat(self, groups, etas) -> np.ndarray:
        """Analytic d f / d eta for (group, eta) pairs, stacked rows (N, p).

        Uses ``modelfun.jacobian`` (d f / d phi) and the chain rule through the
        parameter transform.
        """
        groups = np.asarray(groups, dtype=int)
        etas = np.asarray(etas, dtype=float)
        lengths = self._lengths[groups]
        scale = self._dphi_deta(etas)
        phis = self._phi(etas)
        if self.vectorization == "full":
            x = np.concatenate([self.X[g] for g in groups])
            v = None if self.V is None else np.repeat(self.V[groups], lengths, axis=0)
            J = np.asarray(
                self._call_jacobian(np.repeat(phis, lengths, axis=0), x, v), dtype=float
            )
            J = J.reshape(lengths.sum(), -1) * np.repeat(scale, lengths, axis=0)
        else:
            parts = []
            for k, (g, phi, sc) in enumerate(zip(groups, phis, scale, strict=True)):
                v = None if self.V is None else self.V[g]
                Jg = np.asarray(
                    self._call_jacobian(phi, self.X[g], v), dtype=float
                ).reshape(lengths[k], -1)
                parts.append(Jg * sc)
            J = np.vstack(parts) if parts else np.zeros((0, etas.shape[1]))
        self.n_calls += 1
        if not np.all(np.isfinite(J)):
            raise FloatingPointError("modelfun.jacobian returned non-finite values")
        return J

    def jacobian_padded(self, etas) -> np.ndarray:
        """d f / d eta for all groups, shape (m, n_max, p)."""
        m, p = etas.shape
        if self.has_jacobian:
            out = np.zeros((m, self.n_max, p))
            out[self.mask] = self.jacobian_flat(np.arange(m), etas)
            return out
        h = _DERIV_STEP * np.maximum(np.abs(etas), 1.0)
        steps = h[:, :, None] * np.eye(p)[None]
        up = (etas[:, None, :] + steps).reshape(-1, p)
        down = (etas[:, None, :] - steps).reshape(-1, p)
        groups = np.repeat(np.arange(m), p)
        values, valid = self.predict_flat(
            np.concatenate([groups, groups]), np.vstack([up, down])
        )
        if not valid.all():
            raise FloatingPointError("modelfun returned non-finite values")
        # Scatter (group, param) segments into (m, p, n_max), then transpose.
        seg_mask = np.repeat(self.mask, p, axis=0)  # (m*p, n_max)
        half = values.size // 2
        f_up = np.zeros(seg_mask.shape)
        f_down = np.zeros(seg_mask.shape)
        f_up[seg_mask] = values[:half]
        f_down[seg_mask] = values[half:]
        J = (f_up - f_down).reshape(m, p, self.n_max) / (2 * h[:, :, None])
        return np.swapaxes(J, 1, 2)

    def jacobian_padded_matlab(self, etas, f0=None) -> np.ndarray:
        """d f / d eta as MATLAB's statjacobian, shape (m, n_max, p).

        Forward differences with step DerivStep * eta (relative and signed);
        for a zero entry DerivStep * ||eta row|| (DerivStep if the row is
        zero). Analytic derivatives are used if the model has them.
        """
        m, p = etas.shape
        if self.has_jacobian:
            return self.jacobian_padded(etas)
        if f0 is None:
            f0 = self.predict_padded(etas)
        h = _DERIV_STEP * etas
        norm = np.linalg.norm(etas, axis=1, keepdims=True)
        h = np.where(h == 0, _DERIV_STEP * np.where(norm == 0, 1.0, norm), h)
        up = (etas[:, None, :] + h[:, :, None] * np.eye(p)[None]).reshape(-1, p)
        values, valid = self.predict_flat(np.repeat(np.arange(m), p), up)
        if not valid.all():
            raise FloatingPointError("modelfun returned non-finite values")
        seg_mask = np.repeat(self.mask, p, axis=0)
        f_up = np.zeros(seg_mask.shape)
        f_up[seg_mask] = values
        J = (f_up.reshape(m, p, self.n_max) - f0[:, None, :]) / h[:, :, None]
        return np.swapaxes(J, 1, 2) * self.mask[..., None]

    def jacobian_many(self, groups, etas) -> list[np.ndarray]:
        """Jacobians d f / d eta for several (group, eta) pairs."""
        groups = np.asarray(groups, dtype=int)
        J = self.jacobian_rows(groups, etas)
        return np.split(J, np.cumsum(self._lengths[groups])[:-1])

    def jacobian_rows(self, groups, etas) -> np.ndarray:
        """d f / d eta for several (group, eta) pairs, rows stacked by group (N, p):
        analytic if the model provides one, else central differences (one model
        call with Vectorization="Full")."""
        groups = np.asarray(groups, dtype=int)
        etas = np.asarray(etas, dtype=float)
        if self.has_jacobian:
            return self.jacobian_flat(groups, etas)
        k, p = etas.shape
        h = _DERIV_STEP * np.maximum(np.abs(etas), 1.0)  # (k, p)
        steps = h[:, :, None] * np.eye(p)[None]  # (k, p, p)
        up = (etas[:, None, :] + steps).reshape(-1, p)
        down = (etas[:, None, :] - steps).reshape(-1, p)
        rep = np.repeat(groups, p)
        values, valid = self.predict_flat(
            np.concatenate([rep, rep]), np.vstack([up, down])
        )
        if not valid.all():
            raise FloatingPointError("modelfun returned non-finite values")
        half = values.size // 2
        lengths = self._lengths[groups]
        # Rows of pair (group j, parameter c) are consecutive; divide each by 2 h.
        diff = (values[:half] - values[half:]) / np.repeat(
            2 * h.reshape(-1), np.repeat(lengths, p)
        )
        blocks = np.split(diff, np.cumsum(lengths * p)[:-1])
        return np.concatenate(
            [b.reshape(p, n).T for b, n in zip(blocks, lengths, strict=True)]
        )

    @property
    def n_groups(self) -> int:
        return len(self.y)

    @property
    def n_obs(self) -> int:
        return sum(len(y) for y in self.y)

    @property
    def n_fixed(self) -> int:
        return self.A.shape[2]

    @property
    def n_random(self) -> int:
        return self.B.shape[2]

    def eta(self, i: int, beta: np.ndarray, b: np.ndarray | None) -> np.ndarray:
        eta = self.A[i] @ beta
        if b is not None and self.n_random:
            eta = eta + self.B[i] @ b
        return eta

    def predict(self, i: int, eta: np.ndarray) -> np.ndarray:
        """Model prediction for group i at transformed parameters eta."""
        phi = self._phi(eta)
        if self.V is not None:
            out = self.modelfun(phi, self.X[i], self.V[i])
        elif self._pass_none:
            out = self.modelfun(phi, self.X[i], None)
        else:
            out = self.modelfun(phi, self.X[i])
        self.n_calls += 1
        out = np.asarray(out, dtype=float).reshape(-1)
        if out.shape[0] != len(self.y[i]):
            raise ValueError(
                f"modelfun returned {out.shape[0]} values for a group with "
                f"{len(self.y[i])} observations"
            )
        if not np.all(np.isfinite(out)):
            raise FloatingPointError(
                f"modelfun returned non-finite values for group {i} at phi={phi}"
            )
        return out

    def jacobian(self, i: int, eta: np.ndarray) -> np.ndarray:
        """Jacobian d f_i / d eta, shape (n_i, p): analytic if the model
        provides one, else central differences."""
        if self.has_jacobian:
            return self.jacobian_flat([i], eta[None, :])
        p = eta.shape[0]
        J = np.empty((len(self.y[i]), p))
        for k in range(p):
            h = _DERIV_STEP * max(abs(eta[k]), 1.0)
            up = eta.copy()
            up[k] += h
            down = eta.copy()
            down[k] -= h
            J[:, k] = (self.predict(i, up) - self.predict(i, down)) / (2 * h)
        return J

    def predictions(self, beta: np.ndarray, b: np.ndarray | None) -> np.ndarray:
        """Stacked predictions in the original row order; b is (r, m) or None."""
        out = np.empty(self.n_obs)
        for i in range(self.n_groups):
            bi = None if b is None else b[:, i]
            out[self.rows[i]] = self.predict(i, self.eta(i, beta, bi))
        return out


def _design_from_select(
    select: Sequence | np.ndarray, p: int, one_based: bool
) -> np.ndarray:
    """p x k selection matrix from a boolean mask or a list of indices."""
    select = np.asarray(select).reshape(-1)
    if select.dtype == bool:
        if select.shape[0] != p:
            raise ValueError(f"Logical parameter selection must have length {p}")
        idx = np.flatnonzero(select)
    else:
        idx = select.astype(int) - (1 if one_based else 0)
        if np.any(idx < 0) or np.any(idx >= p):
            base = "1-based" if one_based else "0-based"
            raise ValueError(
                f"Parameter selection {select.tolist()} out of range ({base}, p={p})"
            )
    D = np.zeros((p, len(idx)))
    D[idx, np.arange(len(idx))] = 1.0
    return D


def build_design(
    p: int | None,
    m: int,
    params_select,
    const_design,
    group_design,
    one_based: bool,
    name: str,
) -> np.ndarray:
    """Design matrices of shape (m, p, k) from MATLAB-style options.

    ``p`` may be None when it is to be inferred from an explicit design.
    """
    given = [o is not None for o in (params_select, const_design, group_design)]
    if sum(given) > 1:
        raise ValueError(
            f"Specify at most one of {name}ParamsSelect, {name}ConstDesign "
            f"and {name}GroupDesign"
        )
    if group_design is not None:
        D = np.asarray(group_design, dtype=float)
        if D.ndim != 3 or D.shape[2] != m:
            raise ValueError(f"{name}GroupDesign must be p-by-k-by-m with m={m} groups")
        return np.ascontiguousarray(np.moveaxis(D, 2, 0))
    if const_design is not None:
        D = np.asarray(const_design, dtype=float)
        if D.ndim != 2:
            raise ValueError(f"{name}ConstDesign must be a p-by-k matrix")
    elif params_select is not None:
        D = _design_from_select(params_select, p, one_based)
    else:
        D = np.eye(p)
    return np.broadcast_to(D, (m, *D.shape)).copy()
