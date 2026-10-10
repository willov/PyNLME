//! Compiled kernels for pynlme's Lindstrom-Bates iterations.
//!
//! The Python code makes thousands of calls on tiny matrices (one QR per
//! subject of a (k + r) x (r + f + 1) block), where NumPy's per-call overhead
//! dominates. These kernels do the same arithmetic in one call.

use numpy::ndarray::{Array1, Array2, Array3, ArrayView2, ArrayView3, Axis};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray2, PyReadonlyArray3};
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
                blocks[(g * q1 + i) * q1 + c] = if c >= i { a[(r + i) * cols + r + c] } else { 0.0 };
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
    LmeResult { logdet, beta: Array1::from(beta), rss, r_xx, u, r_zz }
}


/// Inverse of a small dense n x n matrix (row-major) by Gauss-Jordan
/// elimination with partial pivoting.
fn inverse(a: &[f64], n: usize) -> Vec<f64> {
    let mut m = a.to_vec();
    let mut inv = vec![0.0; n * n];
    for i in 0..n {
        inv[i * n + i] = 1.0;
    }
    for c in 0..n {
        let mut p = c;
        for i in (c + 1)..n {
            if m[i * n + c].abs() > m[p * n + c].abs() {
                p = i;
            }
        }
        if p != c {
            for j in 0..n {
                m.swap(c * n + j, p * n + j);
                inv.swap(c * n + j, p * n + j);
            }
        }
        let d = m[c * n + c];
        for j in 0..n {
            m[c * n + j] /= d;
            inv[c * n + j] /= d;
        }
        for i in 0..n {
            if i != c {
                let fct = m[i * n + c];
                if fct != 0.0 {
                    for j in 0..n {
                        m[i * n + j] -= fct * m[c * n + j];
                        inv[i * n + j] -= fct * inv[c * n + j];
                    }
                }
            }
        }
    }
    inv
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
            l[i * n + j] = if i == j { s.max(0.0).sqrt() } else { s / l[j * n + j] };
        }
    }
    l
}

/// nlme's EM update of the relative covariance factor (internal_EM, ML), from
/// the groups' cross-products: zz (m, r, r) = Z'Z, zxw (m, r, f+1) = Z'[X w],
/// xx (f+1, f+1) = sum of [X w]'[X w]. Returns the updated factor L (r, r).
fn em_core(zz: ArrayView3<f64>, zxw: ArrayView3<f64>, xx: ArrayView2<f64>, l0: ArrayView2<f64>,
           n_obs: f64, full: bool, n_iter: usize) -> Array2<f64> {
    let (m, r, _) = zz.dim();
    let q1 = xx.len_of(Axis(0));
    let f = q1 - 1;
    let mut l: Vec<f64> = l0.iter().cloned().collect();
    let mut minv = vec![0.0; m * r * r];
    let mut t = vec![0.0; m * r * q1];
    for _ in 0..n_iter {
        // P = (L L')^-1
        let mut llt = vec![0.0; r * r];
        for i in 0..r {
            for j in 0..r {
                let mut s = 0.0;
                for k in 0..r {
                    s += l[i * r + k] * l[j * r + k];
                }
                llt[i * r + j] = s;
            }
        }
        let p = inverse(&llt, r);
        let mut sm: Vec<f64> = xx.iter().cloned().collect();
        for g in 0..m {
            let mut mg = vec![0.0; r * r];
            for i in 0..r {
                for j in 0..r {
                    mg[i * r + j] = zz[[g, i, j]] + p[i * r + j];
                }
            }
            let mi = inverse(&mg, r);
            minv[g * r * r..(g + 1) * r * r].copy_from_slice(&mi);
            for i in 0..r {
                for c in 0..q1 {
                    let mut s = 0.0;
                    for k in 0..r {
                        s += mi[i * r + k] * zxw[[g, k, c]];
                    }
                    t[(g * r + i) * q1 + c] = s;
                }
            }
            // S -= ZXw' T
            for a in 0..q1 {
                for c in 0..q1 {
                    let mut s = 0.0;
                    for k in 0..r {
                        s += zxw[[g, k, a]] * t[(g * r + k) * q1 + c];
                    }
                    sm[a * q1 + c] -= s;
                }
            }
        }
        // beta = S[:f, :f]^-1 S[:f, f]
        let mut sff = vec![0.0; f * f];
        for i in 0..f {
            for j in 0..f {
                sff[i * f + j] = sm[i * q1 + j];
            }
        }
        let sinv = inverse(&sff, f);
        let mut beta = vec![0.0; f];
        for i in 0..f {
            let mut s = 0.0;
            for j in 0..f {
                s += sinv[i * f + j] * sm[j * q1 + f];
            }
            beta[i] = s;
        }
        let mut rss = sm[f * q1 + f];
        for j in 0..f {
            rss -= sm[f * q1 + j] * beta[j];
        }
        let sigma2 = rss / n_obs;
        // Psi / sigma^2 = mean(M^-1 + b b' / sigma^2)
        let mut psi = vec![0.0; r * r];
        for g in 0..m {
            let mut b = vec![0.0; r];
            for i in 0..r {
                let mut v = t[(g * r + i) * q1 + f];
                for c in 0..f {
                    v -= t[(g * r + i) * q1 + c] * beta[c];
                }
                b[i] = v;
            }
            for i in 0..r {
                for j in 0..r {
                    psi[i * r + j] += minv[g * r * r + i * r + j] + b[i] * b[j] / sigma2;
                }
            }
        }
        for x in psi.iter_mut() {
            *x /= m as f64;
        }
        if full {
            l = cholesky_lower(&psi, r);
        } else {
            l = vec![0.0; r * r];
            for i in 0..r {
                l[i * r + i] = psi[i * r + i].sqrt();
            }
        }
    }
    Array2::from_shape_vec((r, r), l).unwrap()
}

#[pyfunction]
#[pyo3(signature = (zz, zxw, xx, l, n_obs, full, n_iter=20))]
#[allow(clippy::too_many_arguments)]
fn em_factor<'py>(
    py: Python<'py>,
    zz: PyReadonlyArray3<'py, f64>,
    zxw: PyReadonlyArray3<'py, f64>,
    xx: PyReadonlyArray2<'py, f64>,
    l: PyReadonlyArray2<'py, f64>,
    n_obs: f64,
    full: bool,
    n_iter: usize,
) -> Bound<'py, PyArray2<f64>> {
    em_core(zz.as_array(), zxw.as_array(), xx.as_array(), l.as_array(), n_obs, full, n_iter).into_pyarray(py)
}

/// LME core for all groups: returns (logdet, beta, rss, R_xx, u, R_zz), as
/// LinearizedLME.solve computes them (R up to row signs).
#[pyfunction]
fn lme_solve<'py>(
    py: Python<'py>,
    z: PyReadonlyArray3<'py, f64>,
    xw: PyReadonlyArray3<'py, f64>,
    l: PyReadonlyArray2<'py, f64>,
) -> (f64, Bound<'py, PyArray1<f64>>, f64, Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>, Bound<'py, PyArray3<f64>>) {
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

#[pymodule]
fn pynlme_kernels(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(lme_solve, m)?)?;
    m.add_function(wrap_pyfunction!(em_factor, m)?)?;
    Ok(())
}
