//! Compiled kernels for pynlme's Lindstrom-Bates iterations.
//!
//! The Python code makes thousands of calls on tiny matrices (one QR per
//! subject of a (k + r) x (r + f + 1) block), where NumPy's per-call overhead
//! dominates. These kernels do the same arithmetic in one call.

use numpy::ndarray::{Array1, Array2, Array3, ArrayView2, ArrayView3, Axis};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3,
};
use pyo3::prelude::*;

/// In-place Householder QR of a row-major `rows x cols` matrix; afterwards the
/// upper triangle of the first `min(rows, cols)` rows holds R (below-diagonal
/// entries are left undefined).
fn householder_r(a: &mut [f64], rows: usize, cols: usize) {
    let steps = rows.min(cols);
    let mut v = vec![0.0; rows];
    for j in 0..steps {
        // Householder vector for column j, rows j..
        let mut norm2 = 0.0;
        for i in j..rows {
            let x = a[i * cols + j];
            norm2 += x * x;
        }
        let norm = norm2.sqrt();
        if norm == 0.0 {
            continue;
        }
        let x0 = a[j * cols + j];
        let alpha = if x0 >= 0.0 { -norm } else { norm };
        // v = x - alpha e1, normalised implicitly through beta = 2 / (v'v)
        v[j] = x0 - alpha;
        for i in (j + 1)..rows {
            v[i] = a[i * cols + j];
        }
        let vtv = norm2 - x0 * x0 + v[j] * v[j];
        if vtv == 0.0 {
            continue;
        }
        let beta = 2.0 / vtv;
        a[j * cols + j] = alpha;
        for i in (j + 1)..rows {
            a[i * cols + j] = 0.0;
        }
        for c in (j + 1)..cols {
            let mut s = 0.0;
            for i in j..rows {
                s += v[i] * a[i * cols + c];
            }
            s *= beta;
            for i in j..rows {
                a[i * cols + c] -= s * v[i];
            }
        }
    }
}

/// Solve the upper-triangular system R x = b in place (R row-major, n x n,
/// stored with leading dimension `ld`).
fn solve_upper(r: &[f64], ld: usize, n: usize, b: &mut [f64]) {
    for i in (0..n).rev() {
        let mut s = b[i];
        for j in (i + 1)..n {
            s -= r[i * ld + j] * b[j];
        }
        b[i] = s / r[i * ld + i];
    }
}

struct LmeResult {
    logdet: f64,
    beta: Array1<f64>,
    rss: f64,
    r_xx: Array2<f64>,
    u: Array2<f64>,
    r_zz: Array3<f64>,
}

fn lme_core(z: ArrayView3<f64>, xw: ArrayView3<f64>, l: ArrayView2<f64>) -> LmeResult {
    let (m, k, r) = z.dim();
    let q1 = xw.len_of(Axis(2)); // f + 1
    let f = q1 - 1;
    let rows = k + r;
    let cols = r + q1;
    let mut a = vec![0.0; rows * cols];
    let mut blocks = vec![0.0; m * q1 * q1];
    let mut r_zz = Array3::<f64>::zeros((m, r, r));
    let mut r_zx = vec![0.0; m * r * q1]; // [R_zx | r_zw] per group
    let mut logdet = 0.0;
    for g in 0..m {
        a.iter_mut().for_each(|x| *x = 0.0);
        // [Z_g L, X_g, w_g] on top, [I, 0, 0] below
        for i in 0..k {
            for c in 0..r {
                let mut s = 0.0;
                for t in 0..r {
                    s += z[[g, i, t]] * l[[t, c]];
                }
                a[i * cols + c] = s;
            }
            for c in 0..q1 {
                a[i * cols + r + c] = xw[[g, i, c]];
            }
        }
        for i in 0..r {
            a[(k + i) * cols + i] = 1.0;
        }
        householder_r(&mut a, rows, cols);
        for i in 0..r {
            logdet += 2.0 * a[i * cols + i].abs().ln();
            for c in i..r {
                r_zz[[g, i, c]] = a[i * cols + c];
            }
            for c in 0..q1 {
                r_zx[(g * r + i) * q1 + c] = a[i * cols + r + c];
            }
        }
        for i in 0..q1 {
            for c in 0..q1 {
                blocks[(g * q1 + i) * q1 + c] = if c >= i {
                    a[(r + i) * cols + r + c]
                } else {
                    0.0
                };
            }
        }
    }
    householder_r(&mut blocks, m * q1, q1);
    let mut r_xx = Array2::<f64>::zeros((f, f));
    let mut beta = vec![0.0; f];
    for i in 0..f {
        for c in i..f {
            r_xx[[i, c]] = blocks[i * q1 + c];
        }
        beta[i] = blocks[i * q1 + f];
    }
    solve_upper(&blocks, q1, f, &mut beta);
    let s = blocks[f * q1 + f];
    let rss = (s * s).max(1e-300);
    let mut u = Array2::<f64>::zeros((m, r));
    let mut rzz_g = vec![0.0; r * r];
    for g in 0..m {
        let mut rhs = vec![0.0; r];
        for i in 0..r {
            let base = (g * r + i) * q1;
            let mut v = r_zx[base + f];
            for c in 0..f {
                v -= r_zx[base + c] * beta[c];
            }
            rhs[i] = v;
            for c in 0..r {
                rzz_g[i * r + c] = r_zz[[g, i, c]];
            }
        }
        solve_upper(&rzz_g, r, r, &mut rhs);
        for i in 0..r {
            u[[g, i]] = rhs[i];
        }
    }
    LmeResult {
        logdet,
        beta: Array1::from(beta),
        rss,
        r_xx,
        u,
        r_zz,
    }
}

/// Lower Cholesky factor of a symmetric positive-definite n x n matrix.
fn cholesky_lower(a: &[f64], n: usize) -> Vec<f64> {
    let mut l = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[i * n + j];
            for k in 0..j {
                s -= l[i * n + k] * l[j * n + k];
            }
            l[i * n + j] = if i == j {
                s.max(0.0).sqrt()
            } else {
                s / l[j * n + j]
            };
        }
    }
    l
}

/// nlme's EM update of the relative covariance factor (internal_EM, ML), with
/// the QR-based LME solve in each step (as nlme does): Psi / sigma^2 =
/// mean_i(Var(b_i | y) / sigma^2 + b_i b_i' / sigma^2), Var(b_i | y) / sigma^2 =
/// L R_i^-1 R_i^-T L'. Returns the updated factor L (r, r).
fn em_core(
    z: ArrayView3<f64>,
    xw: ArrayView3<f64>,
    l0: ArrayView2<f64>,
    n_obs: f64,
    full: bool,
    n_iter: usize,
) -> Array2<f64> {
    let (m, _, r) = z.dim();
    let mut l = l0.to_owned();
    for _ in 0..n_iter {
        let res = lme_core(z, xw, l.view());
        let sigma2 = res.rss / n_obs;
        let mut psi = vec![0.0; r * r];
        let mut rg = vec![0.0; r * r];
        for g in 0..m {
            for i in 0..r {
                for c in 0..r {
                    rg[i * r + c] = res.r_zz[[g, i, c]];
                }
            }
            // R^-1 by back substitution, column by column
            let mut rinv = vec![0.0; r * r];
            for c in 0..r {
                let mut e = vec![0.0; r];
                e[c] = 1.0;
                solve_upper(&rg, r, r, &mut e);
                for i in 0..r {
                    rinv[i * r + c] = e[i];
                }
            }
            // W = L R^-1; cond cov = W W'; b = L u
            let mut w = vec![0.0; r * r];
            let mut b = vec![0.0; r];
            for i in 0..r {
                for c in 0..r {
                    let mut s = 0.0;
                    for k in 0..r {
                        s += l[[i, k]] * rinv[k * r + c];
                    }
                    w[i * r + c] = s;
                }
                let mut s = 0.0;
                for k in 0..r {
                    s += l[[i, k]] * res.u[[g, k]];
                }
                b[i] = s;
            }
            for i in 0..r {
                for j in 0..r {
                    let mut s = 0.0;
                    for k in 0..r {
                        s += w[i * r + k] * w[j * r + k];
                    }
                    psi[i * r + j] += s + b[i] * b[j] / sigma2;
                }
            }
        }
        for x in psi.iter_mut() {
            *x /= m as f64;
        }
        let new: Vec<f64> = if full {
            cholesky_lower(&psi, r)
        } else {
            let mut d = vec![0.0; r * r];
            for i in 0..r {
                d[i * r + i] = psi[i * r + i].sqrt();
            }
            d
        };
        l = Array2::from_shape_vec((r, r), new).unwrap();
    }
    l
}

#[pyfunction]
#[pyo3(signature = (z, xw, l, n_obs, full, n_iter=20))]
fn em_factor<'py>(
    py: Python<'py>,
    z: PyReadonlyArray3<'py, f64>,
    xw: PyReadonlyArray3<'py, f64>,
    l: PyReadonlyArray2<'py, f64>,
    n_obs: f64,
    full: bool,
    n_iter: usize,
) -> Bound<'py, PyArray2<f64>> {
    em_core(
        z.as_array(),
        xw.as_array(),
        l.as_array(),
        n_obs,
        full,
        n_iter,
    )
    .into_pyarray(py)
}

// ----- R's optif9 (nlm) as nlme_increment calls it -------------------------

const DBL_MAX: f64 = f64::MAX;

/// Lower-triangular solve L x = b (n x n, row-major).
fn solve_lower(l: &[f64], n: usize, b: &mut [f64]) {
    for i in 0..n {
        let mut s = b[i];
        for j in 0..i {
            s -= l[i * n + j] * b[j];
        }
        b[i] = s / l[i * n + i];
    }
}

/// Inverse of a symmetric positive-definite n x n matrix via its Cholesky factor.
fn spd_inverse(a: &[f64], n: usize) -> Vec<f64> {
    let l = cholesky_lower(a, n);
    let mut inv = vec![0.0; n * n];
    for c in 0..n {
        let mut e = vec![0.0; n];
        e[c] = 1.0;
        solve_lower(&l, n, &mut e);
        // L' x = e
        for i in (0..n).rev() {
            let mut s = e[i];
            for j in (i + 1)..n {
                s -= l[j * n + i] * e[j];
            }
            e[i] = s / l[i * n + i];
        }
        for i in 0..n {
            inv[i * n + c] = e[i];
        }
    }
    inv
}

fn norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// R's optif9 with nlme's settings (method 1: line search; secant BFGS on the
/// Cholesky factor; forward differences, central after a failed line search;
/// typsiz 1, fscale 1, ndigit 15, gradtl eps^(1/3), steptl sqrt(eps), default
/// stepmx, at most `itnlim` iterations). Returns the vector optif9 leaves in
/// its input argument: the last accepted iterate before the terminating one.
// Names and their counterparts in R's uncmin.c (Dennis & Schnabel's notation):
//   max_step (stepmx), noise (rnf), grad_tol (gradtl), step_tol (steptl);
//   line search: slope (slp), newton_len (sln), rel_len (rln), min_step (rmnlmb),
//   step / prev_step (lambda / plmbda), prev_value (pfpls), trial (tlmbda),
//   took_max (mxtake), status (iretcd); stopping: rel_change (rsx);
//   BFGS update: sy (den1), s_hs (den2), scale (alp).
fn optif9<F: FnMut(&[f64]) -> f64>(mut fcn: F, x0: &[f64], itnlim: usize) -> Vec<f64> {
    let n = x0.len();
    let eps = f64::EPSILON;
    let noise = 1e-15_f64.max(eps);
    let grad_tol = eps.powf(1.0 / 3.0);
    let step_tol = eps.sqrt();
    let mut x = x0.to_vec();
    let max_step = 1000.0 * norm(&x).max(1.0);
    let mut evaluate = |v: &[f64]| {
        let value = fcn(v);
        if value.is_finite() {
            value
        } else {
            DBL_MAX
        }
    };
    let forward_gradient = |evaluate: &mut dyn FnMut(&[f64]) -> f64, v: &[f64], fv: f64| {
        let mut g = vec![0.0; n];
        for j in 0..n {
            let h = noise.sqrt() * v[j].abs().max(1.0);
            let mut w = v.to_vec();
            w[j] += h;
            g[j] = (evaluate(&w) - fv) / h;
        }
        g
    };
    let central_gradient = |evaluate: &mut dyn FnMut(&[f64]) -> f64, v: &[f64]| {
        let mut g = vec![0.0; n];
        for j in 0..n {
            let h = noise.powf(1.0 / 3.0) * v[j].abs().max(1.0);
            let mut up = v.to_vec();
            let mut down = v.to_vec();
            up[j] += h;
            down[j] -= h;
            g[j] = (evaluate(&up) - evaluate(&down)) / (2.0 * h);
        }
        g
    };
    let converged_gradient = |v: &[f64], fv: f64, g: &[f64]| {
        let mut worst = 0.0_f64;
        for j in 0..n {
            worst = worst.max(g[j].abs() * v[j].abs().max(1.0));
        }
        worst / fv.abs().max(1.0) <= grad_tol
    };

    let mut f = evaluate(&x);
    let mut g = forward_gradient(&mut evaluate, &x, f);
    if converged_gradient(&x, f, &g) {
        return x;
    }
    let mut l = vec![0.0; n * n]; // Cholesky factor of the secant Hessian
    for i in 0..n {
        l[i * n + i] = 1.0;
    }
    let mut central = false;
    let mut n_max_steps = 0;
    for iteration in 1..=itnlim {
        let (x_new, f_new, status, took_max) = loop {
            // p = -(L L')^-1 g
            let mut p: Vec<f64> = g.iter().map(|v| -v).collect();
            solve_lower(&l, n, &mut p);
            for i in (0..n).rev() {
                let mut s = p[i];
                for j in (i + 1)..n {
                    s -= l[j * n + i] * p[j];
                }
                p[i] = s / l[i * n + i];
            }
            // Line search (A6.3.1)
            let mut newton_len = norm(&p);
            if newton_len > max_step {
                let scale = max_step / newton_len;
                p.iter_mut().for_each(|v| *v *= scale);
                newton_len = max_step;
            }
            let slope: f64 = g.iter().zip(&p).map(|(a, b)| a * b).sum();
            let mut rel_len = 0.0_f64;
            for j in 0..n {
                rel_len = rel_len.max(p[j].abs() / x[j].abs().max(1.0));
            }
            let min_step = step_tol / rel_len;
            let (mut step, mut first_backtrack, mut prev_step, mut prev_value) =
                (1.0_f64, true, 0.0_f64, 0.0_f64);
            let result = loop {
                let x_new: Vec<f64> = x.iter().zip(&p).map(|(a, b)| a + step * b).collect();
                let f_new = evaluate(&x_new);
                if f_new <= f + slope * 1e-4 * step {
                    break (x_new, f_new, 0, step == 1.0 && newton_len > max_step * 0.99);
                }
                if step < min_step {
                    break (x_new, f_new, 1, false);
                }
                if f_new >= DBL_MAX {
                    step *= 0.1;
                    first_backtrack = true;
                    continue;
                }
                let mut trial;
                if first_backtrack {
                    trial = -step * slope / ((f_new - f - slope) * 2.0);
                    first_backtrack = false;
                } else {
                    let c1 = f_new - f - step * slope;
                    let c2 = prev_value - f - prev_step * slope;
                    let c3 = 1.0 / (step - prev_step);
                    let cubic_a = 3.0 * c3 * (c1 / (step * step) - c2 / (prev_step * prev_step));
                    let b =
                        c3 * (c2 * step / (prev_step * prev_step) - c1 * prev_step / (step * step));
                    let discriminant = b * b - cubic_a * slope;
                    trial = if discriminant > b * b {
                        (-b + if cubic_a < 0.0 {
                            -discriminant.sqrt()
                        } else {
                            discriminant.sqrt()
                        }) / cubic_a
                    } else {
                        (-b + if cubic_a < 0.0 {
                            discriminant.sqrt()
                        } else {
                            -discriminant.sqrt()
                        }) / cubic_a
                    };
                    trial = trial.min(step * 0.5);
                }
                prev_step = step;
                prev_value = f_new;
                step = if trial < step * 0.1 {
                    step * 0.1
                } else {
                    trial
                };
            };
            if result.2 == 1 && !central {
                central = true; // retry with central differences
                g = central_gradient(&mut evaluate, &x);
                continue;
            }
            break result;
        };
        if status == 1 {
            return x;
        }
        let g_new = if central {
            central_gradient(&mut evaluate, &x_new)
        } else {
            forward_gradient(&mut evaluate, &x_new, f_new)
        };
        if converged_gradient(&x_new, f_new, &g_new) {
            return x;
        }
        let mut rel_change = 0.0_f64;
        for j in 0..n {
            rel_change = rel_change.max((x_new[j] - x[j]).abs() / x_new[j].abs().max(1.0));
        }
        if rel_change <= step_tol || iteration >= itnlim {
            return x;
        }
        if took_max {
            n_max_steps += 1;
            if n_max_steps >= 5 {
                return x;
            }
        } else {
            n_max_steps = 0;
        }
        // BFGS update of the factor (secfac), with the first_backtrack-update scaling.
        let s: Vec<f64> = x_new.iter().zip(&x).map(|(a, b)| a - b).collect();
        let y: Vec<f64> = g_new.iter().zip(&g).map(|(a, b)| a - b).collect();
        let sy: f64 = s.iter().zip(&y).map(|(a, b)| a * b).sum();
        if sy >= eps.sqrt() * norm(&s) * norm(&y) {
            let mut u = vec![0.0; n]; // L' s
            for i in 0..n {
                for k in 0..n {
                    u[i] += l[k * n + i] * s[k];
                }
            }
            let mut s_hs: f64 = u.iter().map(|v| v * v).sum();
            let mut scale = (sy / s_hs).sqrt();
            if iteration == 1 {
                l.iter_mut().for_each(|v| *v *= scale);
                u.iter_mut().for_each(|v| *v *= scale);
                s_hs = sy;
                scale = 1.0;
            }
            let _ = s_hs;
            let mut w = vec![0.0; n]; // L u
            for i in 0..n {
                for k in 0..n {
                    w[i] += l[i * n + k] * u[k];
                }
            }
            let reltol = if central { noise } else { noise.sqrt() };
            let skip_update =
                (0..n).all(|i| (y[i] - w[i]).abs() < reltol * g[i].abs().max(g_new[i].abs()));
            if !skip_update {
                for i in 0..n {
                    w[i] = y[i] - scale * w[i];
                    u[i] *= scale / sy;
                }
                // R of L' + u w', then L = R' with a positive diagonal
                let mut a = vec![0.0; n * n];
                for i in 0..n {
                    for j in 0..n {
                        a[i * n + j] = l[j * n + i] + u[i] * w[j];
                    }
                }
                householder_r(&mut a, n, n);
                l = vec![0.0; n * n];
                for i in 0..n {
                    let sign = if a[i * n + i] < 0.0 { -1.0 } else { 1.0 };
                    for j in i..n {
                        l[j * n + i] = sign * a[i * n + j];
                    }
                }
            }
        }
        x = x_new;
        f = f_new;
        g = g_new;
    }
    x
}

/// nlme_increment's refinement of the factor after EM: optif9 on
/// 0.5 (N log RSS + log det) (nlme's internal_loglik, negated) over log Delta
/// (pdDiag) or the log-Cholesky parameters of the precision factor (pdLogChol).
fn nlm_core(
    z: ArrayView3<f64>,
    xw: ArrayView3<f64>,
    l0: ArrayView2<f64>,
    n_obs: f64,
    full: bool,
) -> Array2<f64> {
    let r = l0.len_of(Axis(0));
    let cols: Vec<(usize, usize)> = (0..r).flat_map(|j| (0..j).map(move |i| (i, j))).collect();
    let to_l = |theta: &[f64]| -> Option<Array2<f64>> {
        let mut l = Array2::<f64>::zeros((r, r));
        if !full {
            for i in 0..r {
                l[[i, i]] = (-theta[i]).exp();
            }
            return Some(l);
        }
        let mut u = vec![0.0; r * r];
        for i in 0..r {
            u[i * r + i] = theta[i].exp();
        }
        for (k, &(i, j)) in cols.iter().enumerate() {
            u[i * r + j] = theta[r + k];
        }
        let mut utu = vec![0.0; r * r];
        for i in 0..r {
            for j in 0..r {
                for k in 0..r {
                    utu[i * r + j] += u[k * r + i] * u[k * r + j];
                }
            }
        }
        let c = cholesky_lower(&spd_inverse(&utu, r), r);
        if c.iter().any(|v| !v.is_finite()) {
            return None;
        }
        for i in 0..r {
            for j in 0..r {
                l[[i, j]] = c[i * r + j];
            }
        }
        Some(l)
    };
    let theta0: Vec<f64> = if !full {
        (0..r).map(|i| -l0[[i, i]].ln()).collect()
    } else {
        let mut llt = vec![0.0; r * r];
        for i in 0..r {
            for j in 0..r {
                for k in 0..r {
                    llt[i * r + j] += l0[[i, k]] * l0[[j, k]];
                }
            }
        }
        let c = cholesky_lower(&spd_inverse(&llt, r), r); // U = c'
        let mut t: Vec<f64> = (0..r).map(|i| c[i * r + i].ln()).collect();
        for &(i, j) in &cols {
            t.push(c[j * r + i]);
        }
        t
    };
    let fcn = |theta: &[f64]| -> f64 {
        match to_l(theta) {
            Some(l) => {
                let res = lme_core(z, xw, l.view());
                0.5 * (n_obs * res.rss.ln() + res.logdet)
            }
            None => f64::INFINITY,
        }
    };
    to_l(&optif9(fcn, &theta0, 20)).unwrap_or_else(|| l0.to_owned())
}

#[pyfunction]
fn nlm_factor<'py>(
    py: Python<'py>,
    z: PyReadonlyArray3<'py, f64>,
    xw: PyReadonlyArray3<'py, f64>,
    l: PyReadonlyArray2<'py, f64>,
    n_obs: f64,
    full: bool,
) -> Bound<'py, PyArray2<f64>> {
    nlm_core(z.as_array(), xw.as_array(), l.as_array(), n_obs, full).into_pyarray(py)
}

/// optif9 on a Python function (for testing the Rust implementation against
/// pynlme's Python version on smooth functions).
#[pyfunction]
fn optif9_py(py: Python<'_>, fcn: Py<PyAny>, x0: Vec<f64>, itnlim: usize) -> PyResult<Vec<f64>> {
    let mut err = None;
    let x = optif9(
        |v: &[f64]| match fcn
            .call1(py, (v.to_vec(),))
            .and_then(|r| r.extract::<f64>(py))
        {
            Ok(value) => value,
            Err(e) => {
                err.get_or_insert(e);
                f64::INFINITY
            }
        },
        &x0,
        itnlim,
    );
    match err {
        Some(e) => Err(e),
        None => Ok(x),
    }
}

/// d logl / d L at L (ML), as LinearizedLME.gradient: by the envelope theorem
/// d RSS/dL = -2 sum_i Z_i' r_i u_i', and d log|I + L'Z'ZL|/dL = 2 Z'Z L M^-1.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn lme_gradient<'py>(
    py: Python<'py>,
    z: PyReadonlyArray3<'py, f64>,
    xw: PyReadonlyArray3<'py, f64>,
    l: PyReadonlyArray2<'py, f64>,
    beta: numpy::PyReadonlyArray1<'py, f64>,
    u: PyReadonlyArray2<'py, f64>,
    rss: f64,
    r_zz: PyReadonlyArray3<'py, f64>,
    n_obs: f64,
) -> Bound<'py, PyArray2<f64>> {
    let (z, xw, l, beta, u, r_zz) = (
        z.as_array(),
        xw.as_array(),
        l.as_array(),
        beta.as_array(),
        u.as_array(),
        r_zz.as_array(),
    );
    let (m, k, r) = z.dim();
    let f = beta.len();
    let mut d_rss = vec![0.0; r * r];
    let mut d_logdet = vec![0.0; r * r];
    let mut rg = vec![0.0; r * r];
    for g in 0..m {
        // residual r_i = w - X beta - Z L u
        let mut zl_u = vec![0.0; k];
        for i in 0..k {
            let mut s = 0.0;
            for a in 0..r {
                let mut lu = 0.0;
                for b in 0..r {
                    lu += l[[a, b]] * u[[g, b]];
                }
                s += z[[g, i, a]] * lu;
            }
            zl_u[i] = s;
        }
        for i in 0..k {
            let mut res = xw[[g, i, f]] - zl_u[i];
            for c in 0..f {
                res -= xw[[g, i, c]] * beta[c];
            }
            for a in 0..r {
                for b in 0..r {
                    d_rss[a * r + b] += -2.0 * z[[g, i, a]] * res * u[[g, b]];
                }
            }
        }
        // M^-1 = R^-1 R^-T
        for i in 0..r {
            for c in 0..r {
                rg[i * r + c] = r_zz[[g, i, c]];
            }
        }
        let mut rinv = vec![0.0; r * r];
        for c in 0..r {
            let mut e = vec![0.0; r];
            e[c] = 1.0;
            solve_upper(&rg, r, r, &mut e);
            for i in 0..r {
                rinv[i * r + c] = e[i];
            }
        }
        let mut minv = vec![0.0; r * r];
        for i in 0..r {
            for j in 0..r {
                for t in 0..r {
                    minv[i * r + j] += rinv[i * r + t] * rinv[j * r + t];
                }
            }
        }
        // Z'Z L M^-1
        let mut ztz = vec![0.0; r * r];
        for i in 0..k {
            for a in 0..r {
                for b in 0..r {
                    ztz[a * r + b] += z[[g, i, a]] * z[[g, i, b]];
                }
            }
        }
        for a in 0..r {
            for p in 0..r {
                let mut s = 0.0;
                for b in 0..r {
                    let mut lm = 0.0;
                    for t in 0..r {
                        lm += l[[b, t]] * minv[t * r + p];
                    }
                    s += ztz[a * r + b] * lm;
                }
                d_logdet[a * r + p] += 2.0 * s;
            }
        }
    }
    let mut grad = Array2::<f64>::zeros((r, r));
    for a in 0..r {
        for b in 0..r {
            grad[[a, b]] = -0.5 * n_obs / rss * d_rss[a * r + b] - 0.5 * d_logdet[a * r + b];
        }
    }
    grad.into_pyarray(py)
}

// ----- MATLAB nlmefit's LME objective --------------------------------------

/// QR blocks of [Z_i X_i w_i; Delta 0] (nlmefit.m: R11, [R10 c1], R00).
struct DeltaQr {
    r_b: Array3<f64>,
    r_bx: Array3<f64>,
    r_x: Array2<f64>,
}

/// Per-group QR of [Z_i X_i w_i; Delta 0] (as nlmefit's LMEnegloglikelihood and
/// GNfitPNLS), and the combined fixed-effects factor R00 (f+1, f+1).
fn delta_qr_core(z: ArrayView3<f64>, xw: ArrayView3<f64>, delta: ArrayView2<f64>) -> DeltaQr {
    let (m, k, r) = z.dim();
    let q1 = xw.len_of(Axis(2));
    let rows = (k + r).max(r + q1);
    let cols = r + q1;
    let mut a = vec![0.0; rows * cols];
    let mut r_b = Array3::<f64>::zeros((m, r, r));
    let mut r_bx = Array3::<f64>::zeros((m, r, q1));
    let mut blocks = vec![0.0; m * q1 * q1];
    for g in 0..m {
        a.iter_mut().for_each(|x| *x = 0.0);
        for i in 0..k {
            for c in 0..r {
                a[i * cols + c] = z[[g, i, c]];
            }
            for c in 0..q1 {
                a[i * cols + r + c] = xw[[g, i, c]];
            }
        }
        for i in 0..r {
            for c in 0..r {
                a[(k + i) * cols + c] = delta[[i, c]];
            }
        }
        householder_r(&mut a, rows, cols);
        for i in 0..r {
            for c in i..r {
                r_b[[g, i, c]] = a[i * cols + c];
            }
            for c in 0..q1 {
                r_bx[[g, i, c]] = a[i * cols + r + c];
            }
        }
        for i in 0..q1 {
            for c in i..q1 {
                blocks[(g * q1 + i) * q1 + c] = a[(r + i) * cols + r + c];
            }
        }
    }
    householder_r(&mut blocks, m * q1, q1);
    let mut r_x = Array2::<f64>::zeros((q1, q1));
    for i in 0..q1 {
        for c in i..q1 {
            r_x[[i, c]] = blocks[i * q1 + c];
        }
    }
    DeltaQr { r_b, r_bx, r_x }
}

/// (R_b, R_bx, R_x) as NumPy arrays.
type DeltaQrArrays<'py> = (
    Bound<'py, PyArray3<f64>>,
    Bound<'py, PyArray3<f64>>,
    Bound<'py, PyArray2<f64>>,
);

#[pyfunction]
fn delta_qr<'py>(
    py: Python<'py>,
    z: PyReadonlyArray3<'py, f64>,
    xw: PyReadonlyArray3<'py, f64>,
    delta: PyReadonlyArray2<'py, f64>,
) -> DeltaQrArrays<'py> {
    let d = delta_qr_core(z.as_array(), xw.as_array(), delta.as_array());
    (
        d.r_b.into_pyarray(py),
        d.r_bx.into_pyarray(py),
        d.r_x.into_pyarray(py),
    )
}

/// nlmefit's LMEnegloglikelihood (varying part of Pinheiro & Bates eq. 2.21 or
/// 2.23): returns (value, c_1, R00).
#[pyfunction]
fn matlab_negloglik<'py>(
    py: Python<'py>,
    z: PyReadonlyArray3<'py, f64>,
    xw: PyReadonlyArray3<'py, f64>,
    delta: PyReadonlyArray2<'py, f64>,
    n_obs: f64,
    reml: bool,
) -> (f64, f64, Bound<'py, PyArray2<f64>>) {
    let delta = delta.as_array();
    let d = delta_qr_core(z.as_array(), xw.as_array(), delta);
    let (m, r, _) = d.r_b.dim();
    let q = d.r_x.len_of(Axis(0)) - 1;
    let resid_norm = d.r_x[[q, q]].abs();
    let mut log_det_delta = 0.0;
    for i in 0..r {
        log_det_delta += delta[[i, i]].abs().ln(); // Delta is upper triangular
    }
    let mut log_ratio = m as f64 * log_det_delta;
    for g in 0..m {
        for i in 0..r {
            log_ratio -= d.r_b[[g, i, i]].abs().ln();
        }
    }
    let value = if reml {
        let mut log_rx = 0.0;
        for i in 0..q {
            log_rx += d.r_x[[i, i]].abs().ln();
        }
        (n_obs - q as f64) * resid_norm.ln() + log_rx - log_ratio
    } else {
        n_obs * resid_norm.ln() - log_ratio
    };
    (value, resid_norm, d.r_x.into_pyarray(py))
}

/// MATLAB's eps(x).
fn spacing(x: f64) -> f64 {
    let a = x.abs();
    if a == 0.0 {
        f64::from_bits(1)
    } else if !a.is_finite() {
        f64::NAN
    } else {
        f64::from_bits(a.to_bits() + 1) - a
    }
}

// Nelder-Mead coefficients
const REFLECT: f64 = 1.0;
const EXPAND: f64 = 2.0;
const CONTRACT: f64 = 0.5;
const SHRINK: f64 = 0.5;

/// Nelder-Mead as MATLAB's fminsearch (Lagarias et al. 1998), as pynlme's
/// matlab_lb.fminsearch. Returns (x, f(x), converged).
fn fminsearch<F: FnMut(&[f64]) -> f64>(
    mut fun: F,
    x0: &[f64],
    tol_x: f64,
    tol_fun: f64,
    max_iter: usize,
    max_fun_evals: f64,
) -> (Vec<f64>, f64, bool) {
    let n = x0.len();
    let mut simplex: Vec<Vec<f64>> = Vec::with_capacity(n + 1);
    let mut values: Vec<f64> = Vec::with_capacity(n + 1);
    simplex.push(x0.to_vec());
    values.push(fun(x0));
    for j in 0..n {
        let mut vertex = x0.to_vec();
        vertex[j] = if vertex[j] != 0.0 {
            (1.0 + 0.05) * vertex[j]
        } else {
            0.00025
        };
        values.push(fun(&vertex));
        simplex.push(vertex);
    }
    let sort = |simplex: &mut Vec<Vec<f64>>, values: &mut Vec<f64>| {
        let mut idx: Vec<usize> = (0..values.len()).collect();
        idx.sort_by(|&a, &b| {
            values[a]
                .partial_cmp(&values[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        }); // stable
        *simplex = idx.iter().map(|&i| simplex[i].clone()).collect();
        *values = idx.iter().map(|&i| values[i]).collect();
    };
    sort(&mut simplex, &mut values);
    let (mut n_evals, mut iterations) = ((n + 1) as f64, 1usize);
    while n_evals < max_fun_evals && iterations < max_iter {
        let fspread = (1..=n)
            .map(|j| (values[0] - values[j]).abs())
            .fold(0.0, f64::max);
        let xspread = (1..=n)
            .flat_map(|j| {
                (0..n).map({
                    let simplex = &simplex;
                    move |i| (simplex[j][i] - simplex[0][i]).abs()
                })
            })
            .fold(0.0, f64::max);
        let vmax = simplex[0].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        if fspread <= tol_fun.max(10.0 * spacing(values[0]))
            && xspread <= tol_x.max(10.0 * spacing(vmax))
        {
            break;
        }
        let mut centroid = vec![0.0; n];
        for row in simplex.iter().take(n) {
            for i in 0..n {
                centroid[i] += row[i];
            }
        }
        centroid.iter_mut().for_each(|x| *x /= n as f64);
        let affine = |a: f64, b: f64, vertex: &Vec<f64>| -> Vec<f64> {
            (0..n).map(|i| a * centroid[i] + b * vertex[i]).collect()
        };
        let reflected = affine(1.0 + REFLECT, -REFLECT, &simplex[n]);
        let f_reflected = fun(&reflected);
        n_evals += 1.0;
        let mut do_shrink = false;
        if f_reflected < values[0] {
            let expanded = affine(1.0 + REFLECT * EXPAND, -REFLECT * EXPAND, &simplex[n]);
            let f_expanded = fun(&expanded);
            n_evals += 1.0;
            if f_expanded < f_reflected {
                simplex[n] = expanded;
                values[n] = f_expanded;
            } else {
                simplex[n] = reflected;
                values[n] = f_reflected;
            }
        } else if f_reflected < values[n - 1] {
            simplex[n] = reflected;
            values[n] = f_reflected;
        } else if f_reflected < values[n] {
            let outside = affine(1.0 + CONTRACT * REFLECT, -CONTRACT * REFLECT, &simplex[n]);
            let f_outside = fun(&outside);
            n_evals += 1.0;
            if f_outside <= f_reflected {
                simplex[n] = outside;
                values[n] = f_outside;
            } else {
                do_shrink = true;
            }
        } else {
            let inside = affine(1.0 - CONTRACT, CONTRACT, &simplex[n]);
            let f_inside = fun(&inside);
            n_evals += 1.0;
            if f_inside < values[n] {
                simplex[n] = inside;
                values[n] = f_inside;
            } else {
                do_shrink = true;
            }
        }
        if do_shrink {
            for j in 1..=n {
                let vertex: Vec<f64> = (0..n)
                    .map(|i| simplex[0][i] + SHRINK * (simplex[j][i] - simplex[0][i]))
                    .collect();
                values[j] = fun(&vertex);
                simplex[j] = vertex;
            }
            n_evals += n as f64;
        }
        sort(&mut simplex, &mut values);
        iterations += 1;
    }
    let converged = n_evals < max_fun_evals && iterations < max_iter;
    (simplex[0].clone(), values[0], converged)
}

/// fminsearch on a Python function (for testing the Rust implementation against
/// pynlme's Python version).
#[pyfunction]
fn fminsearch_py(
    py: Python<'_>,
    fcn: Py<PyAny>,
    x0: Vec<f64>,
    tol_x: f64,
    tol_fun: f64,
    max_iter: usize,
) -> PyResult<(Vec<f64>, f64, bool)> {
    let mut err = None;
    let out = fminsearch(
        |v: &[f64]| match fcn
            .call1(py, (v.to_vec(),))
            .and_then(|r| r.extract::<f64>(py))
        {
            Ok(value) => value,
            Err(e) => {
                err.get_or_insert(e);
                f64::INFINITY
            }
        },
        &x0,
        tol_x,
        tol_fun,
        max_iter,
        f64::INFINITY,
    );
    match err {
        Some(e) => Err(e),
        None => Ok(out),
    }
}

/// nlmefit's LME step for the parametrizations computed here: "chol" (Delta
/// set from theta at the pattern positions) or "logm" with a diagonal pattern
/// (Delta = sqrt(exp(theta))). fminsearch on LMEnegloglikelihood.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn matlab_lme_step(
    z: PyReadonlyArray3<'_, f64>,
    xw: PyReadonlyArray3<'_, f64>,
    theta0: Vec<f64>,
    rows: Vec<usize>,
    cols: Vec<usize>,
    chol: bool,
    n_obs: f64,
    reml: bool,
    tol_x: f64,
    tol_fun: f64,
    max_iter: usize,
) -> (Vec<f64>, f64, bool) {
    let (z, xw) = (z.as_array(), xw.as_array());
    let (m, _, r) = z.dim();
    let q = xw.len_of(Axis(2)) - 1;
    let objective = |theta: &[f64]| -> f64 {
        let mut delta = Array2::<f64>::zeros((r, r));
        for (k, (&i, &j)) in rows.iter().zip(&cols).enumerate() {
            delta[[i, j]] = if chol {
                theta[k]
            } else {
                let e = theta[k].exp();
                if !(e > 0.0 && e.is_finite()) {
                    return 1.0 / f64::EPSILON; // expm/chol failure in nlmefit
                }
                e.sqrt()
            };
        }
        let d = delta_qr_core(z, xw, delta.view());
        let resid_norm = d.r_x[[q, q]].abs();
        let mut log_ratio = 0.0;
        let mut log_det_delta = 0.0;
        for i in 0..r {
            log_det_delta += delta[[i, i]].abs().ln();
        }
        log_ratio += m as f64 * log_det_delta;
        for g in 0..m {
            for i in 0..r {
                log_ratio -= d.r_b[[g, i, i]].abs().ln();
            }
        }
        if reml {
            let log_rx: f64 = (0..q).map(|i| d.r_x[[i, i]].abs().ln()).sum();
            (n_obs - q as f64) * resid_norm.ln() + log_rx - log_ratio
        } else {
            n_obs * resid_norm.ln() - log_ratio
        }
    };
    fminsearch(objective, &theta0, tol_x, tol_fun, max_iter, f64::INFINITY)
}

/// (logdet, beta, rss, R_xx, u, R_zz) as Python values.
type LmeArrays<'py> = (
    f64,
    Bound<'py, PyArray1<f64>>,
    f64,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray3<f64>>,
);

/// LME core for all groups: returns (logdet, beta, rss, R_xx, u, R_zz), as
/// LinearizedLME.solve computes them (R up to row signs).
#[pyfunction]
fn lme_solve<'py>(
    py: Python<'py>,
    z: PyReadonlyArray3<'py, f64>,
    xw: PyReadonlyArray3<'py, f64>,
    l: PyReadonlyArray2<'py, f64>,
) -> LmeArrays<'py> {
    let res = lme_core(z.as_array(), xw.as_array(), l.as_array());
    (
        res.logdet,
        res.beta.into_pyarray(py),
        res.rss,
        res.r_xx.into_pyarray(py),
        res.u.into_pyarray(py),
        res.r_zz.into_pyarray(py),
    )
}

/// Solve the n x n system a x = b in place (b becomes x) by LU decomposition with
/// partial pivoting, as LAPACK's gesv (numpy.linalg.solve).
fn lu_solve(a: &mut [f64], n: usize, b: &mut [f64]) {
    for k in 0..n {
        let pivot = (k..n)
            .max_by(|&i, &j| a[i * n + k].abs().total_cmp(&a[j * n + k].abs()))
            .unwrap();
        if pivot != k {
            for c in 0..n {
                a.swap(k * n + c, pivot * n + c);
            }
            b.swap(k, pivot);
        }
        for i in k + 1..n {
            let factor = a[i * n + k] / a[k * n + k];
            for c in k + 1..n {
                a[i * n + c] -= factor * a[k * n + c];
            }
            b[i] -= factor * b[k];
        }
    }
    for k in (0..n).rev() {
        let mut v = b[k];
        for c in k + 1..n {
            v -= a[k * n + c] * b[c];
        }
        b[k] = v / a[k * n + k];
    }
}

/// The model's group-level data for the conditional modes: designs A_i (p x f) and
/// B_i (p x r), weighted responses, and where each group's rows start.
struct ModesProblem<'a> {
    a: ArrayView3<'a, f64>,
    b: ArrayView3<'a, f64>,
    y: &'a [f64],
    w: &'a [f64],
    starts: Vec<usize>,
    lengths: Vec<usize>,
    beta: &'a [f64],
    l: ArrayView2<'a, f64>,
}

impl ModesProblem<'_> {
    /// eta_i = A_i beta + B_i L u.
    fn eta(&self, i: usize, u: &[f64]) -> Vec<f64> {
        let (p, f) = (self.a.shape()[1], self.a.shape()[2]);
        let r = self.b.shape()[2];
        let lu: Vec<f64> = (0..r)
            .map(|j| (0..r).map(|k| self.l[[j, k]] * u[k]).sum())
            .collect();
        (0..p)
            .map(|q| {
                let fixed: f64 = (0..f).map(|k| self.a[[i, q, k]] * self.beta[k]).sum();
                let random: f64 = (0..r).map(|k| self.b[[i, q, k]] * lu[k]).sum();
                fixed + random
            })
            .collect()
    }

    /// Weighted residuals w (y - f) of group i from its predictions.
    fn residuals(&self, i: usize, f: &[f64]) -> Vec<f64> {
        let o = self.starts[i];
        (0..self.lengths[i])
            .map(|k| self.w[o + k] * (self.y[o + k] - f[k]))
            .collect()
    }
}

fn sum_sq(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum()
}

/// Evaluate the model for several groups through a Python callback
/// `predict(groups, etas) -> (values, valid)`; values are stacked by group.
fn call_predict(
    py: Python<'_>,
    predict: &Py<PyAny>,
    groups: &[usize],
    etas: Vec<Vec<f64>>,
) -> PyResult<(Vec<f64>, Vec<bool>)> {
    let p = etas.first().map_or(0, |e| e.len());
    let etas = Array2::from_shape_vec((groups.len(), p), etas.concat()).unwrap();
    let out = predict.call1(py, (groups.to_vec(), etas.into_pyarray(py)))?;
    let (values, valid): (PyReadonlyArray1<f64>, PyReadonlyArray1<bool>) = out.extract(py)?;
    Ok((values.as_array().to_vec(), valid.as_array().to_vec()))
}

/// Conditional modes of the random effects for FOCE, as
/// first_order.conditional_modes: u_i minimising ||w_i (y_i - f_i(beta, L u_i))||^2 +
/// ||u_i||^2 by Levenberg-Marquardt, all groups in lock-step. The model is
/// evaluated by the Python callbacks `predict(groups, etas) -> (values, valid)` and
/// `jacobian(groups, etas) -> rows` (d f / d eta, stacked by group), once per round
/// for all groups that need it.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn conditional_modes<'py>(
    py: Python<'py>,
    predict: Py<PyAny>,
    jacobian: Py<PyAny>,
    a: PyReadonlyArray3<'py, f64>,
    b: PyReadonlyArray3<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    w: PyReadonlyArray1<'py, f64>,
    lengths: Vec<usize>,
    beta: PyReadonlyArray1<'py, f64>,
    l: PyReadonlyArray2<'py, f64>,
    u0: PyReadonlyArray2<'py, f64>,
    max_iter: usize,
    tol: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let (y, w, beta) = (y.as_slice()?, w.as_slice()?, beta.as_slice()?);
    let mut starts = vec![0; lengths.len()];
    for i in 1..lengths.len() {
        starts[i] = starts[i - 1] + lengths[i - 1];
    }
    let problem = ModesProblem {
        a: a.as_array(),
        b: b.as_array(),
        y,
        w,
        starts,
        lengths,
        beta,
        l: l.as_array(),
    };
    let u0 = u0.as_array();
    let (m, r) = (u0.shape()[0], u0.shape()[1]);
    let p = problem.b.shape()[1];
    let mut u: Vec<Vec<f64>> = (0..m).map(|i| u0.row(i).to_vec()).collect();
    let mut etas: Vec<Vec<f64>> = (0..m).map(|i| problem.eta(i, &u[i])).collect();
    let all: Vec<usize> = (0..m).collect();
    let (values, valid) = call_predict(py, &predict, &all, etas.clone())?;
    if let Some(i) = valid.iter().position(|ok| !ok) {
        return Err(pyo3::exceptions::PyFloatingPointError::new_err(format!(
            "modelfun returned non-finite values for group {i}"
        )));
    }
    let mut res: Vec<Vec<f64>> = (0..m)
        .map(|i| {
            let o = problem.starts[i];
            problem.residuals(i, &values[o..o + problem.lengths[i]])
        })
        .collect();
    let mut objective: Vec<f64> = (0..m).map(|i| sum_sq(&res[i]) + sum_sq(&u[i])).collect();
    let mut lam = vec![1e-3; m];
    let mut iterations = vec![0usize; m];
    let mut need_jacobian = vec![true; m];
    let mut active = vec![true; m];
    let mut system = vec![vec![0.0; r * r]; m]; // M_i = H'H + I
    let mut gradient = vec![vec![0.0; r]; m]; // g_i = H' res - u
    let l_mat = problem.l;
    while active.iter().any(|&x| x) {
        // Groups starting an iteration: Jacobian, gradient and Gauss-Newton matrix.
        let starting: Vec<usize> = (0..m).filter(|&i| active[i] && need_jacobian[i]).collect();
        if !starting.is_empty() {
            let etas_start = Array2::from_shape_vec(
                (starting.len(), p),
                starting.iter().flat_map(|&i| etas[i].clone()).collect(),
            )
            .unwrap();
            let rows: PyReadonlyArray2<f64> = jacobian
                .call1(py, (starting.clone(), etas_start.into_pyarray(py)))?
                .extract(py)?;
            let rows = rows.as_array();
            let mut row = 0;
            for &i in &starting {
                let (o, n) = (problem.starts[i], problem.lengths[i]);
                // H = (w J) B_i L, as (n x p)(p x r)(r x r).
                let mut h = vec![0.0; n * r];
                for k in 0..n {
                    let jb: Vec<f64> = (0..r)
                        .map(|c| {
                            (0..p)
                                .map(|q| {
                                    problem.w[o + k] * rows[[row + k, q]] * problem.b[[i, q, c]]
                                })
                                .sum()
                        })
                        .collect();
                    for c in 0..r {
                        h[k * r + c] = (0..r).map(|j| jb[j] * l_mat[[j, c]]).sum();
                    }
                }
                row += n;
                for c in 0..r {
                    gradient[i][c] =
                        (0..n).map(|k| h[k * r + c] * res[i][k]).sum::<f64>() - u[i][c];
                    for d in 0..r {
                        system[i][c * r + d] =
                            (0..n).map(|k| h[k * r + c] * h[k * r + d]).sum::<f64>()
                                + if c == d { 1.0 } else { 0.0 };
                    }
                }
                iterations[i] += 1;
                need_jacobian[i] = false;
                if lam[i] >= 1e10 {
                    active[i] = false; // no damping left to try
                }
            }
        }
        let trying: Vec<usize> = (0..m).filter(|&i| active[i]).collect();
        if trying.is_empty() {
            break;
        }
        let mut steps = Vec::with_capacity(trying.len());
        let mut u_new = Vec::with_capacity(trying.len());
        let mut etas_new = Vec::with_capacity(trying.len());
        for &i in &trying {
            let mut damped = system[i].clone();
            for c in 0..r {
                damped[c * r + c] += lam[i] * system[i][c * r + c];
            }
            let mut step = gradient[i].clone();
            lu_solve(&mut damped, r, &mut step);
            let candidate: Vec<f64> = (0..r).map(|c| u[i][c] + step[c]).collect();
            etas_new.push(problem.eta(i, &candidate));
            u_new.push(candidate);
            steps.push(step);
        }
        let (values, valid) = call_predict(py, &predict, &trying, etas_new.clone())?;
        let mut offset = 0;
        for (k, &i) in trying.iter().enumerate() {
            let n = problem.lengths[i];
            let f = &values[offset..offset + n];
            offset += n;
            let (res_new, objective_new) = if valid[k] {
                let res_new = problem.residuals(i, f);
                let value = sum_sq(&res_new) + sum_sq(&u_new[k]);
                (res_new, value)
            } else {
                (Vec::new(), f64::INFINITY)
            };
            if objective_new > objective[i] {
                lam[i] *= 10.0;
                if lam[i] >= 1e10 {
                    active[i] = false; // no improving step: done
                }
                continue;
            }
            u[i] = u_new[k].clone();
            etas[i] = etas_new[k].clone();
            res[i] = res_new;
            objective[i] = objective_new;
            lam[i] = (lam[i] / 10.0).max(1e-12);
            let largest_step = steps[k].iter().fold(0.0f64, |acc, v| acc.max(v.abs()));
            let largest_u = u[i].iter().fold(0.0f64, |acc, v| acc.max(v.abs()));
            if largest_step <= tol * (1.0 + largest_u) || iterations[i] >= max_iter {
                active[i] = false;
            } else {
                need_jacobian[i] = true;
            }
        }
    }
    let flat: Vec<f64> = u.concat();
    Ok(Array2::from_shape_vec((m, r), flat)
        .unwrap()
        .into_pyarray(py))
}

/// log|I + H'H| and the quadratic form of the model linearised at b_i = L u_i, as
/// first_order.linearized_marginal: with H_i = (w_i J_i) B_i L and
/// v_i = w_i (y_i - f_i) + H_i u_i, sums 2 log det C_i (C_i C_i' = I + H_i'H_i) and
/// v_i'v_i - |C_i^-1 H_i'v_i|^2 over the groups.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn linearized_marginal<'py>(
    jacobian_rows: PyReadonlyArray2<'py, f64>,
    f: PyReadonlyArray1<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    w: PyReadonlyArray1<'py, f64>,
    lengths: Vec<usize>,
    b: PyReadonlyArray3<'py, f64>,
    l: PyReadonlyArray2<'py, f64>,
    u: PyReadonlyArray2<'py, f64>,
) -> PyResult<(f64, f64)> {
    let (jac, b, l, u) = (
        jacobian_rows.as_array(),
        b.as_array(),
        l.as_array(),
        u.as_array(),
    );
    let (f, y, w) = (f.as_slice()?, y.as_slice()?, w.as_slice()?);
    let (p, r) = (b.shape()[1], b.shape()[2]);
    let (mut logdet, mut quad) = (0.0, 0.0);
    let mut o = 0;
    for (i, &n) in lengths.iter().enumerate() {
        let mut h = vec![0.0; n * r];
        for k in 0..n {
            let jb: Vec<f64> = (0..r)
                .map(|c| {
                    (0..p)
                        .map(|q| w[o + k] * jac[[o + k, q]] * b[[i, q, c]])
                        .sum()
                })
                .collect();
            for c in 0..r {
                h[k * r + c] = (0..r).map(|j| jb[j] * l[[j, c]]).sum();
            }
        }
        let v: Vec<f64> = (0..n)
            .map(|k| {
                w[o + k] * (y[o + k] - f[o + k])
                    + (0..r).map(|c| h[k * r + c] * u[[i, c]]).sum::<f64>()
            })
            .collect();
        let mut system = vec![0.0; r * r];
        for c in 0..r {
            for d in 0..r {
                system[c * r + d] = (0..n).map(|k| h[k * r + c] * h[k * r + d]).sum::<f64>()
                    + if c == d { 1.0 } else { 0.0 };
            }
        }
        let chol = cholesky_lower(&system, r);
        logdet += 2.0 * (0..r).map(|c| chol[c * r + c].ln()).sum::<f64>();
        let mut hv: Vec<f64> = (0..r)
            .map(|c| (0..n).map(|k| h[k * r + c] * v[k]).sum())
            .collect();
        solve_lower(&chol, r, &mut hv);
        quad += sum_sq(&v) - sum_sq(&hv);
        o += n;
    }
    Ok((logdet, quad))
}

/// The sampler's state after an E-step: latent parameters, predictions, and the
/// single-coordinate and full-vector random-walk scales.
type McmcState<'py> = (
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<f64>>,
    Vec<f64>,
    Vec<f64>,
);

/// SAEM's target acceptance rate for the random-walk kernels.
const TARGET_ACCEPT: f64 = 0.4;

/// Error model of the SAEM likelihood: standard deviation g(f) = a (constant),
/// max(b |f|, eps) (proportional) or max(a + b |f|, eps) (combined).
struct SaemError {
    kind: u8, // 0 constant, 1 proportional, 2 combined
    a: f64,
    b: f64,
}

impl SaemError {
    fn sd(&self, f: f64) -> f64 {
        match self.kind {
            1 => (self.b * f.abs()).max(f64::EPSILON),
            2 => (self.a + self.b * f.abs()).max(f64::EPSILON),
            _ => self.a,
        }
    }

    /// 0.5 sum(((y - f) / g)^2) + sum(log g) over one row; infinite if not finite.
    fn nll(&self, y: &[f64], f: &[f64]) -> f64 {
        let value: f64 = y
            .iter()
            .zip(f)
            .map(|(&yk, &fk)| {
                let g = self.sd(fk);
                0.5 * ((yk - fk) / g).powi(2) + g.ln()
            })
            .sum();
        if value.is_finite() {
            value
        } else {
            f64::INFINITY
        }
    }
}

/// -0.5 (phi - mu)' Omega^-1 (phi - mu) for one row.
fn log_prior(phi: &[f64], mu: &[f64], omega_inv: &ArrayView2<f64>) -> f64 {
    let d = phi.len();
    let diff: Vec<f64> = (0..d).map(|c| phi[c] - mu[c]).collect();
    let mut value = 0.0;
    for i in 0..d {
        for j in 0..d {
            value += diff[i] * omega_inv[[i, j]] * diff[j];
        }
    }
    -0.5 * value
}

/// The state of SAEM's Metropolis-Hastings sampler over all rows (subjects x
/// chains): latent parameters, flat predictions, and per-row likelihood terms.
struct Sampler<'a> {
    phi: Vec<Vec<f64>>,
    f: Vec<f64>,
    nll: Vec<f64>,
    prior: Vec<f64>,
    mu: ArrayView2<'a, f64>,
    omega_inv: ArrayView2<'a, f64>,
    y: &'a [f64],
    starts: Vec<usize>,
    lengths: Vec<usize>,
    error: SaemError,
}

impl Sampler<'_> {
    /// Evaluate proposals for all rows and accept each with the Metropolis-Hastings
    /// rule (with the prior ratio unless proposed from the prior); returns which
    /// rows accepted.
    fn propose(
        &mut self,
        py: Python<'_>,
        predict: &Py<PyAny>,
        proposal: Vec<Vec<f64>>,
        with_prior: bool,
        uniforms: &[f64],
    ) -> PyResult<Vec<bool>> {
        let (rows, d) = (proposal.len(), self.phi[0].len());
        let array = Array2::from_shape_vec((rows, d), proposal.concat()).unwrap();
        let out = predict.call1(py, (array.into_pyarray(py),))?;
        let (values, valid): (PyReadonlyArray1<f64>, PyReadonlyArray1<bool>) = out.extract(py)?;
        let (values, valid) = (values.as_slice()?, valid.as_slice()?);
        let mut accepted = vec![false; rows];
        for k in 0..rows {
            let (o, n) = (self.starts[k], self.lengths[k]);
            let nll_new = if valid[k] {
                self.error.nll(&self.y[o..o + n], &values[o..o + n])
            } else {
                f64::INFINITY
            };
            let mu_k: Vec<f64> = self.mu.row(k).to_vec();
            let prior_new = log_prior(&proposal[k], &mu_k, &self.omega_inv);
            let mut log_ratio = self.nll[k] - nll_new;
            if with_prior {
                log_ratio += prior_new - self.prior[k];
            }
            if uniforms[k].ln() < log_ratio {
                accepted[k] = true;
                self.phi[k] = proposal[k].clone();
                self.nll[k] = nll_new;
                self.prior[k] = prior_new;
                self.f[o..o + n].copy_from_slice(&values[o..o + n]);
            }
        }
        Ok(accepted)
    }
}

fn acceptance(accepted: &[bool]) -> f64 {
    accepted.iter().filter(|&&a| a).count() as f64 / accepted.len() as f64
}

/// One E-step of SAEM, as saem._SAEM.mcmc: Metropolis-Hastings with three kernels
/// (independent draws from the prior, single-coordinate random walks, full-vector
/// random walks) over all rows. The random numbers come from Python, in the order
/// the NumPy code draws them (for each proposal its normals, then one uniform per
/// row), so both give the same chain. `predict(phi) -> (values, valid)` evaluates
/// the model. Returns the new latent parameters, predictions and random-walk
/// scales.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn saem_mcmc<'py>(
    py: Python<'py>,
    predict: Py<PyAny>,
    phi: PyReadonlyArray2<'py, f64>,
    f: PyReadonlyArray1<'py, f64>,
    mu: PyReadonlyArray2<'py, f64>,
    omega_inv: PyReadonlyArray2<'py, f64>,
    chol: PyReadonlyArray2<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    lengths: Vec<usize>,
    error: (u8, f64, f64),
    n_kernel: (usize, usize, usize),
    scales: (Vec<f64>, Vec<f64>),
    normals: PyReadonlyArray1<'py, f64>,
    uniforms: PyReadonlyArray1<'py, f64>,
) -> PyResult<McmcState<'py>> {
    let phi_in = phi.as_array();
    let (rows, d) = (phi_in.shape()[0], phi_in.shape()[1]);
    let chol = chol.as_array();
    let (normals, uniforms) = (normals.as_slice()?, uniforms.as_slice()?);
    let mut starts = vec![0; rows];
    for k in 1..rows {
        starts[k] = starts[k - 1] + lengths[k - 1];
    }
    let mut sampler = Sampler {
        phi: (0..rows).map(|k| phi_in.row(k).to_vec()).collect(),
        f: f.as_slice()?.to_vec(),
        nll: Vec::new(),
        prior: Vec::new(),
        mu: mu.as_array(),
        omega_inv: omega_inv.as_array(),
        y: y.as_slice()?,
        starts,
        lengths,
        error: SaemError {
            kind: error.0,
            a: error.1,
            b: error.2,
        },
    };
    sampler.nll = (0..rows)
        .map(|k| {
            let (o, n) = (sampler.starts[k], sampler.lengths[k]);
            sampler
                .error
                .nll(&sampler.y[o..o + n], &sampler.f[o..o + n])
        })
        .collect();
    sampler.prior = (0..rows)
        .map(|k| {
            log_prior(
                &sampler.phi[k],
                &sampler.mu.row(k).to_vec(),
                &sampler.omega_inv,
            )
        })
        .collect();
    let (mut single_scales, mut full_scales) = scales;
    let (mut z, mut u) = (0, 0); // positions in the random numbers

    // Kernel 1: independent proposals from the prior, mu + chol z.
    for _ in 0..n_kernel.0 {
        let proposal: Vec<Vec<f64>> = (0..rows)
            .map(|k| {
                (0..d)
                    .map(|c| {
                        sampler.mu[[k, c]]
                            + (0..d)
                                .map(|j| normals[z + k * d + j] * chol[[c, j]])
                                .sum::<f64>()
                    })
                    .collect()
            })
            .collect();
        z += rows * d;
        sampler.propose(py, &predict, proposal, false, &uniforms[u..u + rows])?;
        u += rows;
    }
    // Kernel 2: single-coordinate random walks.
    for _ in 0..n_kernel.1 {
        for j in 0..d {
            let mut proposal = sampler.phi.clone();
            for (k, row) in proposal.iter_mut().enumerate() {
                row[j] += single_scales[j] * normals[z + k];
            }
            z += rows;
            let accepted = sampler.propose(py, &predict, proposal, true, &uniforms[u..u + rows])?;
            u += rows;
            single_scales[j] *= 1.0 + 0.4 * (acceptance(&accepted) - TARGET_ACCEPT);
        }
    }
    // Kernel 3: full-vector random walks.
    for _ in 0..n_kernel.2 {
        let proposal: Vec<Vec<f64>> = (0..rows)
            .map(|k| {
                (0..d)
                    .map(|c| sampler.phi[k][c] + full_scales[c] * normals[z + k * d + c])
                    .collect()
            })
            .collect();
        z += rows * d;
        let accepted = sampler.propose(py, &predict, proposal, true, &uniforms[u..u + rows])?;
        u += rows;
        let factor = 1.0 + 0.4 * (acceptance(&accepted) - TARGET_ACCEPT);
        for scale in full_scales.iter_mut() {
            *scale *= factor;
        }
    }
    let phi_out = Array2::from_shape_vec((rows, d), sampler.phi.concat()).unwrap();
    Ok((
        phi_out.into_pyarray(py),
        Array1::from_vec(sampler.f).into_pyarray(py),
        single_scales,
        full_scales,
    ))
}

#[pymodule]
fn _kernels(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(lme_solve, m)?)?;
    m.add_function(wrap_pyfunction!(em_factor, m)?)?;
    m.add_function(wrap_pyfunction!(nlm_factor, m)?)?;
    m.add_function(wrap_pyfunction!(lme_gradient, m)?)?;
    m.add_function(wrap_pyfunction!(optif9_py, m)?)?;
    m.add_function(wrap_pyfunction!(delta_qr, m)?)?;
    m.add_function(wrap_pyfunction!(matlab_negloglik, m)?)?;
    m.add_function(wrap_pyfunction!(matlab_lme_step, m)?)?;
    m.add_function(wrap_pyfunction!(fminsearch_py, m)?)?;
    m.add_function(wrap_pyfunction!(conditional_modes, m)?)?;
    m.add_function(wrap_pyfunction!(linearized_marginal, m)?)?;
    m.add_function(wrap_pyfunction!(saem_mcmc, m)?)?;
    Ok(())
}
