#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn import preprocessing
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from math import sqrt
import numba as nb
import scipy.sparse as sp
from scipy.optimize import fmin_l_bfgs_b, fmin_ncg
from numpy.lib.stride_tricks import as_strided
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils import check_consistent_length, check_array
from sklearn.utils import check_random_state
from sklearn.base import BaseEstimator

from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from functools import reduce


# In[ ]:


def trcg(Ax, r, x, n_iterations=1000, tr_delta=0, rtol=1e-5, atol=1e-8, args=(), verbose=False):
    if n_iterations > 0:
        n_iterations = min(n_iterations, len(x))

    p, iteration = r.copy(), 0
    tr_delta_sq = tr_delta ** 2

    rtr, rtr_old = np.dot(r, r), 1.0
    cg_tol = sqrt(rtr) * rtol + atol
    region_breached = False
    while (iteration < n_iterations) and (sqrt(rtr) > cg_tol):
        Ap = Ax(p, *args)
        iteration += 1
        if verbose:
            print("""iter %2d |Ap| %5.3e |p| %5.3e """
                  """|r| %5.3e |x| %5.3e beta %5.3e""" %
                  (iteration, np.linalg.norm(Ap), np.linalg.norm(p),
                   np.linalg.norm(r), np.linalg.norm(x), rtr / rtr_old))
        # end if

        # ddot(&n, p, &inc, Ap, &inc);
        alpha = rtr / np.dot(p, Ap)
        # daxpy(&n, &alpha, p, &inc, x, &inc);
        x += alpha * p
        # daxpy(&n, &( -alpha ), Ap, &inc, r, &inc);
        r -= alpha * Ap

        # check trust region (diverges from tron.cpp in liblinear and leml-imf)
        if tr_delta_sq > 0:
            xTx = np.dot(x, x)
            if xTx > tr_delta_sq:
                xTp = np.dot(x, p)
                if xTp > 0:
                    # backtrack into the trust region
                    p_nrm = np.linalg.norm(p)

                    q = xTp / p_nrm
                    eta = (q - sqrt(max(q * q + tr_delta_sq - xTx, 0))) / p_nrm

                    # reproject onto the boundary of the region
                    r += eta * Ap
                    x -= eta * p
                else:
                    # this never happens maybe due to CG iteration properties
                    pass
                # end if

                region_breached = True
                break
            # end if
        # end if

        # ddot(&n, r, &inc, r, &inc);
        rtr, rtr_old = np.dot(r, r), rtr
        # dscal(&n, &(rtr / rtr_old), p, &inc);
        p *= rtr / rtr_old
        # daxpy(&n, &one, r, &1, p, &1);
        p += r
    # end while

    return iteration, region_breached


def tron(func, x, n_iterations=1000, rtol=1e-3, atol=1e-5, args=(),
         verbose=False):
    
    eta0, eta1, eta2 = 1e-4, 0.25, 0.75
    sigma1, sigma2, sigma3 = 0.25, 0.5, 4.0

    f_valp_, f_grad_, f_hess_ = func

    iteration, cg_iter = 0, 0

    fval = f_valp_(x, *args)
    grad = f_grad_(x, *args)
    grad_norm = np.linalg.norm(grad)

    # make a copy of `-grad` and zeros like `x`
    # r, z = -grad, np.zeros_like(x)
    delta, grad_norm_tol = grad_norm, grad_norm * rtol + atol
    while iteration < n_iterations and grad_norm > grad_norm_tol:
        r, z = -grad, np.zeros_like(x)
        # tolerances and n_iterations as in leml-imf
        cg_iter, region_breached = trcg(
            f_hess_, r, z, tr_delta=delta, args=args,
            n_iterations=20, rtol=1e-1, atol=0.0)

        z_norm = np.linalg.norm(z)
        if iteration == 0:
            delta = min(delta, z_norm)

        # trcg finds z and r s.t. r + A z = -g and \|r\|\to \min
        # f(x) - f(x+z) ~ -0.5 * (2 g'z + z'Az) = -0.5 * (g'z + z'(-r))
        linear = np.dot(z, grad)
        approxred = -0.5 * (linear - np.dot(z, r))

        # The value and the actual reduction: compute the forward pass.
        fnew = f_valp_(x + z, *args)
        actualred = fval - fnew

        if linear + actualred < 0:
            alpha = max(sigma1, 0.5 * linear / (linear + actualred))

        else:
            alpha = sigma3

        # end if

        if actualred < eta0 * approxred:
            delta = min(max(alpha, sigma1) * z_norm, sigma2 * delta)

        elif actualred < eta1 * approxred:
            delta = max(sigma1 * delta, min(alpha * z_norm, sigma2 * delta))

        elif actualred < eta2 * approxred:
            delta = max(sigma1 * delta, min(alpha * z_norm, sigma3 * delta))

        else:
            # patch 2018-08-30: new addition from tron.cpp at
            #  https://github.com/cjlin1/liblinear/blob/master/tron.cpp
            if region_breached:
                delta = sigma3 * delta

            else:
                delta = max(delta, min(alpha * z_norm, sigma3 * delta))

        # end if

        if verbose:
            print("""iter %2d act %5.3e pre %5.3e delta %5.3e """
                  """f %5.3e |z| %5.3e |g| %5.3e CG %3d""" %
                  (iteration, actualred, approxred,
                   delta, fval, z_norm, grad_norm, cg_iter))
        # end if

        if actualred > eta0 * approxred:
            x += z
            fval, grad = fnew, f_grad_(x, *args)
            grad_norm = np.linalg.norm(grad)
            iteration += 1

            # r, z = -grad, np.zeros_like(x)
        # end if

        if fval < -1e32:
            if verbose:
                print("WARNING: f < -1.0e+32")
            break
        # end if

        if abs(actualred) <= 0 and approxred <= 0:
            if verbose:
                print("WARNING: actred and prered <= 0")
            break
        # end if

        if abs(actualred) <= 1e-12 * abs(fval) and            abs(approxred) <= 1e-12 * abs(fval):
            if verbose:
                print("WARNING: actred and prered too small")
            break
        # end if

        if delta <= rtol * (z_norm + atol):
            if verbose:
                print("WARNING: degenerate trust region")
            break
        # end if
    # end while

    return cg_iter


# In[ ]:


@nb.njit("float64[:,::1](float64[:,::1], float64[:,::1])",fastmath=True, cache=False, error_model="numpy")
def ar_resid(Z, phi):
    """Compute the AR(p) residuals of the multivariate data in Z."""
    n_components, n_order = phi.shape

    resid = Z[n_order:].copy()
    for k in range(n_order):
        # r_t -= y_{t-(p-k)} * \beta_{p - k} (\phi is reversed \beta)
        resid -= Z[k:k - n_order] * phi[:, k]

    return resid

#@nb.njit("float64[:,::1](float64[:,::1], float64[:,::1], float64[:,::1])",fastmath=True, cache=False, error_model="numpy")
def ar_hess_vect(V, Z, phi):
    """Compute the Hessian-vector product of the AR(p) square loss for `V`."""
    n_components, n_order = phi.shape

    # compute the AR(p) residuals over V
    resid = ar_resid(V, phi)

    # get the derivative w.r.t. the series
    hess_v = np.zeros_like(V)
    hess_v[n_order:] = resid
    for k in range(n_order):
        hess_v[k:k - n_order] -= resid * phi[:, k]

    return hess_v

#@nb.njit("float64[:,::1](float64[:,::1], float64[:,::1])",         fastmath=True, cache=False, error_model="numpy")
def ar_grad(Z, phi):
    """Compute the gradient of the AR(p) l2 loss w.r.t. the time-series `Z`."""
    return ar_hess_vect(Z, Z, phi)


def precompute_graph_reg(adj):
    """Precompute the neighbor average discrepancy operator."""

    # make a copy of the adjacency matrix and the outbound degree
    resid, deg = adj.astype(float), adj.getnnz(axis=1)

    # scale the rows : D^{-1} A
    resid.data /= deg[adj.nonzero()[0]]

    # subtract the matrix from the diagonalized mask
    return sp.diags((deg > 0).astype(float)) - resid


def graph_resid(F, adj):
    """Get the residual of the outgoing neighbor average of `F`."""
    return safe_sparse_dot(adj, F.T).T


def graph_grad(F, adj):
    """Compute the gradient of the outgoing neighbors average w.r.t. `F`."""
    return safe_sparse_dot(adj.T, graph_resid(F, adj).T).T


def graph_hess_vect(V, F, adj):
    """Get the Hessian-vector product of the outgoing neighbors average."""
    return graph_grad(V, adj)

def l2_loss_valj(Y, Z, F):
    if sp.issparse(Y):
        R = csr_gemm(1, Z, F, -1, Y.copy())
        return sp.linalg.norm(R, ord="fro") ** 2

    return np.linalg.norm(Y - np.dot(Z, F), ord="fro") ** 2

def f_step_tron_valj(f, Y, Z, C_F, eta_F, adj):
    """Compute current value the f-step loss."""
    (n_samples, n_targets), n_components = Y.shape, Z.shape[1]

    F = f.reshape(n_components, n_targets)
    objective = l2_loss_valj(Y, Z, F)

    if sp.issparse(Y):
        coef = C_F * Y.nnz / (n_components * n_targets)
    else:
        coef = C_F * n_samples / n_components

    if C_F > 0:
        if eta_F < 1:
            reg_f_l2 = np.linalg.norm(F, ord="fro") ** 2
        else:
            reg_f_l2, eta_F = 0., 1.
        # end if

        if sp.issparse(adj) and (eta_F > 0):
            reg_f_graph = np.linalg.norm(graph_resid(F, adj), ord="fro") ** 2
        else:
            reg_f_graph, eta_F = 0., 0.
        # end if

        reg_f = reg_f_l2 * (1 - eta_F) + reg_f_graph * eta_F
        objective += reg_f * coef
    # end if

    return 0.5 * objective

def f_step_tron_grad(f, Y, Z, C_F, eta_F, adj):
    """Compute the gradient of the f-step objective."""
    (n_samples, n_targets), n_components = Y.shape, Z.shape[1]

    F = f.reshape(n_components, n_targets)
    if sp.issparse(Y):
        coef = C_F * Y.nnz / (n_components * n_targets)

        grad = safe_sparse_dot(Z.T, csr_gemm(1, Z, F, -1, Y.copy()))
        grad += (1 - eta_F) * coef * F

    else:
        coef = C_F * n_samples / n_components

        ZTY, ZTZ = np.dot(Z.T, Y), np.dot(Z.T, Z)
        if C_F > 0 and eta_F < 1:
            ZTZ.flat[::n_components + 1] += (1 - eta_F) * coef

        grad = np.dot(ZTZ, F) - ZTY
    # end if

    if C_F > 0 and sp.issparse(adj) and eta_F > 0:
        grad += graph_grad(F, adj) * (eta_F * coef)

    return grad.reshape(-1)

def f_step_tron_hess(v, Y, Z, C_F, eta_F, adj):
    """Get the Hessian-vector product for the f-step objective."""
    (n_samples, n_targets), n_components = Y.shape, Z.shape[1]

    V = v.reshape(n_components, n_targets)
    if sp.issparse(Y):
        coef = C_F * Y.nnz / (n_components * n_targets)

        hess_v = safe_sparse_dot(Z.T, csr_gemm(1, Z, V, 0, Y.copy()))
        hess_v += (1 - eta_F) * coef * V

    else:
        coef = C_F * n_samples / n_components

        ZTZ = np.dot(Z.T, Z)
        if C_F > 0 and eta_F < 1:
            ZTZ.flat[::n_components + 1] += (1 - eta_F) * coef

        hess_v = np.dot(ZTZ, V)
    # end if

    if C_F > 0 and sp.issparse(adj) and eta_F > 0:
        hess_v += graph_grad(V, adj) * (eta_F * coef)

    return hess_v.reshape(-1)


def f_step_tron(F, Y, Z, C_F, eta_F, adj, rtol=5e-2, atol=1e-4, verbose=False,
                **kwargs):
    """TRON solver for the f-step minimization problem."""
    f_call = f_step_tron_valj, f_step_tron_grad, f_step_tron_hess

    tron(f_call, F.ravel(), n_iterations=5, rtol=rtol, atol=atol,
         args=(Y, Z, C_F, eta_F, adj), verbose=verbose)

    return F


def f_step_ncg_hess_(F, v, Y, Z, C_F, eta_F, adj):
    """A wrapper of the hess-vector product for ncg calls."""
    return f_step_tron_hess(v, Y, Z, C_F, eta_F, adj)


def f_step_ncg(F, Y, Z, C_F, eta_F, adj, **kwargs):
    """Solve the F-step using scipy's Newton-CG."""
    FF = fmin_ncg(f=f_step_tron_valj, x0=F.ravel(), disp=False,
                  fprime=f_step_tron_grad, fhess_p=f_step_ncg_hess_,
                  args=(Y, Z, C_F, eta_F, adj))

    return FF.reshape(F.shape)


def f_step_lbfgs(F, Y, Z, C_F, eta_F, adj, **kwargs):
    """Solve the F-step using scipy's L-BFGS method."""
    FF, f, d = fmin_l_bfgs_b(func=f_step_tron_valj, x0=F.ravel(), iprint=0,
                             fprime=f_step_tron_grad,
                             args=(Y, Z, C_F, eta_F, adj))

    return FF.reshape(F.shape)


# In[33]:
def f_step_prox_func(F, Y, Z, C_F, eta_F, adj):
    """An interface to the f-step objective for unraveled matrices."""
    return f_step_tron_valj(F.ravel(), Y, Z, C_F, eta_F, adj)


# In[34]:
def f_step_prox_grad(F, Y, Z, C_F, eta_F, adj):
    """An interface to the f-step objective gradient for unraveled matrices."""
    return f_step_tron_grad(F.ravel(), Y, Z, C_F, eta_F, adj).reshape(F.shape)


# In[35]:
def f_step_prox(F, Y, Z, C_F, eta_F, adj, lip=1e-2, n_iter=25, alpha=1.0,
                **kwargs):
    gamma_u, gamma_d = 2, 1.1

    # get the gradient
    grad = f_step_prox_grad(F, Y, Z, C_F, eta_F, adj)
    grad_F = np.dot(grad.flat, F.flat)

    f0, lip0 = f_step_prox_func(F, Y, Z, C_F, eta_F, adj), lip
    for _ in range(n_iter):
        # F_new = (1 -  alpha) * F + alpha * np.maximum(F - lr * grad, 0.)
        # prox-sgd operation
        F_new = np.maximum(F - grad / lip, 0.)

        # FGM Lipschitz search
        delta = f_step_prox_func(F_new, Y, Z, C_F, eta_F, adj) - f0
        linear = np.dot(grad.flat, F_new.flat) - grad_F
        quad = np.linalg.norm(F_new - F, ord="fro") ** 2
        if delta <= linear + lip * quad / 2:
            break

        lip *= gamma_u
    # end for

    # lip = max(lip0, lip / gamma_d)
    lip = lip / gamma_d

    return F_new, lip

def f_step(F, Y, Z, C_F, eta_F, adj, kind="fgm", **kwargs):
    """A common subroutine solving the f-step minimization problem."""
    lip = np.inf
    if kind == "fgm":
        F, lip = f_step_prox(F, Y, Z, C_F, eta_F, adj, **kwargs)
    elif kind == "tron":
        F = f_step_tron(F, Y, Z, C_F, eta_F, adj, **kwargs)
    elif kind == "ncg":
        F = f_step_ncg(F, Y, Z, C_F, eta_F, adj, **kwargs)
    elif kind == "lbfgs":
        F = f_step_lbfgs(F, Y, Z, C_F, eta_F, adj, **kwargs)
    else:
        raise ValueError(f"""Unrecognized method `{kind}`""")

    return F, lip


# In[41]:
def phi_step(phi, Z, C_Z, C_phi, eta_Z, nugget=1e-8):
    # return a set of independent AR(p) ridge estimates.
    (n_components, n_order), n_samples = phi.shape, Z.shape[0]
    if n_order < 1 or n_components < 1:
        return np.empty((n_components, n_order))

    if not (C_Z > 0 and eta_Z > 0):
        return np.zeros_like(phi)

    # embed into the last dimensions
    shape = Z.shape[1:] + (Z.shape[0] - n_order, n_order + 1)
    strides = Z.strides[1:] + Z.strides[:1] + Z.strides[:1]
    Z_view = as_strided(Z, shape=shape, strides=strides)

    # split into y (d x T-p) and Z (d x T-p x p) (all are views!)
    y, Z_lagged = Z_view[..., -1], Z_view[..., :-1]

    # compute the SVD: thin, but V is d x p x p
    U, s, Vh = np.linalg.svd(Z_lagged, full_matrices=False)
    if C_phi > 0:
        # the {V^{H}}^{H} (\Sigma^2 + C I)^{-1} \Sigma part is reduced
        #  to columnwise operations
        gain = (C_Z * eta_Z * n_order) * s
        gain /= gain * s + C_phi * (n_samples - n_order)
    else:
        # do the same cutoff as in np.linalg.pinv(...)
        large = s > nugget * np.max(s, axis=-1, keepdims=True)
        gain = np.divide(1, s, where=large, out=s)
        gain[~large] = 0
    # end if

    # get the U' y part and the final estimate
    # $\phi_j$ corresponds to $p-j$-th lag $j = 0,\,\ldots,\,p-1$
    return np.einsum("ijk,ij,isj,is->ik", Vh, gain, U, y)

def z_step_tron_valh(z, Y, F, phi, C_Z, eta_Z):
    """Get the value of the z-step objective."""
    n_samples, n_targets = Y.shape
    n_components, n_order = phi.shape

    Z = z.reshape(n_samples, n_components)
    objective = l2_loss_valj(Y, Z, F)

    if sp.issparse(Y):
        coef = C_Z * Y.nnz / (n_samples * n_components)
    else:
        coef = C_Z * n_targets / n_components

    if C_Z > 0:
        if eta_Z < 1:
            reg_z_l2 = np.linalg.norm(Z, ord="fro") ** 2
        else:
            reg_z_l2, eta_Z = 0., 1.
        # end if

        if eta_Z > 0 and n_samples > n_order:
            reg_z_ar_j = np.linalg.norm(ar_resid(Z, phi), ord=2, axis=0) ** 2
            reg_z_ar = np.sum(reg_z_ar_j) * n_samples / (n_samples - n_order)
        else:
            reg_z_ar, eta_Z = 0., 0.
        # end if

        # reg_z was implicitly scaled by T d or nnz(Y)
        reg_z = reg_z_l2 * (1 - eta_Z) + reg_z_ar * eta_Z
        objective += reg_z * coef
    # end if

    return 0.5 * objective

def z_step_tron_grad(z, Y, F, phi, C_Z, eta_Z):
    """Compute the gradient of the z-step objective."""
    n_samples, n_targets = Y.shape
    n_components, n_order = phi.shape

    Z = z.reshape(n_samples, n_components)
    if sp.issparse(Y):
        coef = C_Z * Y.nnz / (n_samples * n_components)

        grad = safe_sparse_dot(csr_gemm(1, Z, F, -1, Y.copy()), F.T)
        grad += (1 - eta_Z) * coef * Z

    else:
        coef = C_Z * n_targets / n_components

        YFT, FFT = np.dot(Y, F.T), np.dot(F, F.T)
        if C_Z > 0 and eta_Z < 1:
            FFT.flat[::n_components + 1] += (1 - eta_Z) * coef

        grad = np.dot(Z, FFT) - YFT
    # end if

    if C_Z > 0 and eta_Z > 0:
        ratio = n_samples / (n_samples - n_order)
        grad += ar_grad(Z, phi) * (ratio * eta_Z * coef)

    return grad.reshape(-1)


# In[49]:
def z_step_tron_hess(v, Y, F, phi, C_Z, eta_Z):
    """Compute the Hessian-vector product of the z-step objective for v."""
    n_samples, n_targets = Y.shape
    n_components, n_order = phi.shape

    V = v.reshape(n_samples, n_components)
    if sp.issparse(Y):
        coef = C_Z * Y.nnz / (n_samples * n_components)

        hess_v = safe_sparse_dot(csr_gemm(1, V, F, 0, Y.copy()), F.T)
        hess_v += (1 - eta_Z) * coef * V

    else:
        coef = C_Z * n_targets / n_components

        FFT = np.dot(F, F.T)
        if C_Z > 0 and eta_Z < 1:
            FFT.flat[::n_components + 1] += (1 - eta_Z) * coef

        hess_v = np.dot(V, FFT)
    # end if

    if C_Z > 0 and eta_Z > 0:
        # should call ar_hess_vect(V, Z, adj) but no Z is available
        ratio = n_samples / (n_samples - n_order)
        hess_v += ar_grad(V, phi) * ratio * eta_Z * coef

    return hess_v.reshape(-1)


# In[50]:
def z_step_tron(Z, Y, F, phi, C_Z, eta_Z, rtol=5e-2, atol=1e-4, verbose=False):
    """TRON solver for the f-step minimization problem."""
    f_call = z_step_tron_valh, z_step_tron_grad, z_step_tron_hess

    tron(f_call, Z.ravel(), n_iterations=5, rtol=rtol, atol=atol,
         args=(Y, F, phi, C_Z, eta_Z), verbose=verbose)

    return Z


def z_step_ncg_hess_(Z, v, Y, F, phi, C_Z, eta_Z):
    """A wrapper of the hess-vector product for ncg calls."""
    return z_step_tron_hess(v, Y, F, phi, C_Z, eta_Z)


def z_step_ncg(Z, Y, F, phi, C_Z, eta_Z, **kwargs):
    """Solve the Z-step using scipy's Newton-CG."""
    ZZ = fmin_ncg(f=z_step_tron_valh, x0=Z.ravel(), disp=False,
                  fprime=z_step_tron_grad, fhess_p=z_step_ncg_hess_,
                  args=(Y, F, phi, C_Z, eta_Z))
    return ZZ.reshape(Z.shape)


def z_step_lbfgs(Z, Y, F, phi, C_Z, eta_Z, **kwargs):
    """Solve the Z-step using scipy's L-BFGS method."""
    ZZ, f, d = fmin_l_bfgs_b(func=z_step_tron_valh, x0=Z.ravel(), iprint=0,
                             fprime=z_step_tron_grad,
                             args=(Y, F, phi, C_Z, eta_Z))

    return ZZ.reshape(Z.shape)


def z_step(Z, Y, F, phi, C_Z, eta_Z, kind="tron", **kwargs):
    """A common subroutine solving the Z-step minimization problem."""
    if kind == "tron":
        Z = z_step_tron(Z, Y, F, phi, C_Z, eta_Z, **kwargs)
    elif kind == "ncg":
        Z = z_step_ncg(Z, Y, F, phi, C_Z, eta_Z, **kwargs)
    elif kind == "lbfgs":
        Z = z_step_lbfgs(Z, Y, F, phi, C_Z, eta_Z, **kwargs)
    else:
        raise ValueError(f"""Unrecognized method `{kind}`""")

    return Z


# In[ ]:



@nb.njit("(float64, float64[:,::1], float64[:,::1], ""float64, int32[::1], int32[::1], float64[::1])",fastmath=True, error_model="numpy", parallel=False, cache=False)
def _csr_gemm(alpha, X, D, beta, Sp, Sj, Sx):
    # computes\mathcal{P}_\Omega(X D) -- n1 x n2 sparse matrix
    if abs(beta) > 0:
        for i in nb.prange(len(X)):
            # compute e_i' XD e_{Sj[j]}
            for j in range(Sp[i], Sp[i+1]):
                dot = np.dot(X[i], D[:, Sj[j]])
                Sx[j] = beta * Sx[j] + alpha * dot
        # end for
    else:
        for i in nb.prange(len(X)):
            # compute e_i' XD e_{Sj[j]}
            for j in range(Sp[i], Sp[i+1]):
                Sx[j] = alpha * np.dot(X[i], D[:, Sj[j]])
        # end for
    # end if


def csr_gemm(alpha, X, D, beta, Y):
    _csr_gemm(alpha, X, D, beta, Y.indptr, Y.indices, Y.data)
    return Y


def csr_column_means(X):
    f_sums = np.bincount(X.indices, weights=X.data, minlength=X.shape[1])
    n_nnz = np.maximum(np.bincount(X.indices, minlength=X.shape[1]), 1.)

    return (f_sums / n_nnz)[np.newaxis]

def b_step_tron_valj(b, Y, X, C_B):
    """Compute current value the b-step loss."""
    (n_samples, n_targets), n_features = Y.shape, X.shape[1]

    B = b.reshape(n_features, n_targets)
    objective = l2_loss_valj(Y, X, B)

    if sp.issparse(Y):
        coef = C_B * Y.nnz / (n_features * n_targets)
    else:
        coef = C_B * n_samples / n_features

    if C_B > 0:
        reg_b = np.linalg.norm(B, ord="fro") ** 2

        objective += reg_b * coef
    # end if

    return 0.5 * objective


def b_step_tron_grad(b, Y, X, C_B):
    """Compute the gradient of the b-step objective."""
    (n_samples, n_targets), n_features = Y.shape, X.shape[1]

    B = b.reshape(n_features, n_targets)
    if sp.issparse(Y):
        coef = C_B * Y.nnz / (n_features * n_targets)

        grad = safe_sparse_dot(X.T, csr_gemm(1, X, B, -1, Y.copy()))
        grad += coef * B

    else:
        coef = C_B * n_samples / n_features

        XTY, XTX = np.dot(X.T, Y), np.dot(X.T, X)
        if C_B > 0:
            XTX.flat[::n_features + 1] += coef
        grad = np.dot(XTX, B) - XTY
    return grad.reshape(-1)

def b_step_tron_hess(v, Y, X, C_B):
    """Get the Hessian-vector product for the b-step objective."""
    (n_samples, n_targets), n_features = Y.shape, X.shape[1]
    V = v.reshape(n_features, n_targets)
    if sp.issparse(Y):
        coef = C_B * Y.nnz / (n_features * n_targets)
        hess_v = safe_sparse_dot(X.T, csr_gemm(1, X, V, 0, Y.copy()))
        hess_v += coef * V

    else:
        coef = C_B * n_samples / n_features
        XTX = np.dot(X.T, X)
        if C_B > 0:
            XTX.flat[::n_features + 1] += coef
        hess_v = np.dot(XTX, V)
    return hess_v.reshape(-1)

def b_step_tron(B, Y, X, C_B, rtol=5e-2, atol=1e-4, verbose=False, **kwargs):
    """TRON solver for the b-step minimization problem."""
    f_call = b_step_tron_valj, b_step_tron_grad, b_step_tron_hess

    tron(f_call, B.ravel(), n_iterations=5, rtol=rtol, atol=atol,
         args=(Y, X, C_B), verbose=verbose)

    return B

def soft_prox(x, c):
    return np.maximum(x - c, 0.) + np.minimum(x + c, 0.)

def b_step(B, Y, X, C_B, kind="tron", **kwargs):
    """A common subroutine solving the b-step minimization problem."""
    lip = np.inf
    if kind == "tron":
        B = b_step_tron(B, Y, X, C_B, **kwargs)
    # elif kind == "fgm":
    #     B, lip = b_step_prox(B, Y, X, C_B, **kwargs)
    else:
        raise ValueError(f"""Unrecognozed optimization `{kind}`""")

    return B, lip

def trmf_init(data, n_components, n_order, random_state=None):
    random_state = check_random_state(random_state)
    n_samples, n_targets = data.shape
    if sp.issparse(data):
        U, s, Vh = sp.linalg.svds(data, k=n_components)

        order = np.argsort(s)[::-1]
        U, s, Vh = U[:, order], s[order], Vh[order]
    else:
        U, s, Vh = np.linalg.svd(data, full_matrices=False)

    factors = U[:, :n_components].copy()
    loadings = Vh[:n_components].copy()
    loadings *= s[:n_components, np.newaxis]

    n_svd_factors = factors.shape[1]
    if n_svd_factors < n_components:
        random_factors = random_state.normal(
            scale=0.01, size=(n_samples, n_components - n_svd_factors))
        factors = np.concatenate([factors, random_factors], axis=1)

    n_svd_loadings = loadings.shape[0]
    if n_svd_loadings < n_components:
        random_loadings = random_state.normal(
            scale=0.01, size=(n_components - n_svd_loadings, n_targets))
        loadings = np.concatenate([loadings, random_loadings], axis=0)

    phi = np.zeros((n_components, n_order))
    ar_coef = phi_step(phi, factors, 1.0, 0., 1.0)
    return factors, loadings, ar_coef

def trmf(data, n_components, n_order, C_Z, C_F, C_phi, eta_Z,
         eta_F=0., adj=None, fit_intercept=False, regressors=None, C_B=0.0,
         tol=1e-6, n_max_iterations=2500, n_max_mf_iter=5,
         f_step_kind="fgm", z_step_kind="tron", random_state=None):
    if not all(C >= 0 for C in (C_Z, C_F, C_phi, C_B)):
        raise ValueError("""Negative ridge regularizer coefficient.""")

    if not all(0 <= eta <= 1 for eta in (eta_Z, eta_F)):
        raise ValueError("""Share `eta` is not within `[0, 1]`.""")

    if not (n_components > 0):
        raise ValueError("""Empty latent factors are not supported.""")

    if C_F > 0 and eta_F > 0:
        if not sp.issparse(adj):
            raise TypeError("""The adjacency matrix must be sparse.""")

        # precompute the outbound average dsicrepancy operator
        adj = precompute_graph_reg(adj)
    # end if

    # prepare the regressors
    n_samples, n_targets = data.shape
    if isinstance(regressors, str):
        if regressors != "auto" or True:
            raise ValueError(f"""Invalid regressor setting `{regressors}`""")

        if sp.issparse(data):
            raise ValueError("""`data` cannot be sparse in """
                             """autoregression mode""")

        # assumes order-1 explicit autoregression
        regressors, data = data[:-1], data[1:]
        n_samples, n_targets = data.shape

    elif regressors is None:
        regressors = np.empty((n_samples, 0))

    # end if

    check_consistent_length(regressors, data)

    # by default the intercept is zero
    intercept = np.zeros((1, n_targets))

    # initialize the regression coefficients
    n_regressors = regressors.shape[1]
    beta = np.zeros((n_regressors, n_targets))

    # prepare smart guesses
    factors, loadings, ar_coef = trmf_init(data, n_components, n_order,
                                           random_state=random_state)

    # prepare for estimating the coefs of the exogenous ridge regression
    if fit_intercept and n_regressors > 0:
        regressors_mean = regressors.mean(axis=0, keepdims=True)
        regressors_cntrd = regressors - regressors_mean
    else:
        regressors_cntrd = regressors
    # end if

    # initialize the outer loop
    ZF, lip_f, lip_b = np.dot(factors, loadings), 500.0, 500.0
    ZF_old_norm, delta = np.linalg.norm(ZF, ord="fro"), +np.inf

    if sp.issparse(data):
        # run the trmf algo
        for iteration in range(n_max_iterations):
            if (delta <= ZF_old_norm * tol) and (iteration > 0):
                break

            # Fit the exogenous ridge-regression with an optional intercept
            if fit_intercept or n_regressors > 0:
                resid = csr_gemm(-1, factors, loadings, 1, data.copy())

                if fit_intercept:
                    intercept = csr_column_means(resid)

                if n_regressors > 0:
                    if fit_intercept:
                        resid.data -= intercept[0, resid.indices]
                    # end if

                    # solve for beta
                    beta, lip_b = b_step(beta, resid, regressors_cntrd, C_B,
                                         kind="tron")

                    # mean(R) - mean(X) beta = mu
                    if fit_intercept:
                        intercept -= np.dot(regressors_mean, beta)
                    # end if
                # end if

                # prepare the residuals for the trmf loop
                resid = data.copy()
                if n_regressors > 0:
                    resid = csr_gemm(-1, regressors, beta, 1, resid)

                if fit_intercept:
                    resid.data -= intercept[0, resid.indices]
            else:
                resid = data
            # end if

            # update (F, Z), then phi
            for inner_iter in range(n_max_mf_iter):
                loadings, lip_f = f_step(loadings, resid, factors, C_F, eta_F,
                                         adj, kind=f_step_kind, lip=lip_f)

                factors = z_step(factors, resid, loadings, ar_coef,
                                 C_Z, eta_Z, kind=z_step_kind)
            # end for

            if n_order > 0:
                ar_coef = phi_step(ar_coef, factors, C_Z, C_phi, eta_Z)
            # end if

            # recompute the reconstruction and convergence criteria
            ZF, ZF_old = np.dot(factors, loadings), ZF
            delta = np.linalg.norm(ZF - ZF_old, ord="fro")
            ZF_old_norm = np.linalg.norm(ZF_old, ord="fro")
        # end for
    else:
        # run the trmf algo
        for iteration in range(n_max_iterations):
            #print("loop1")
            if (delta <= ZF_old_norm * tol) and (iteration > 0):
                break

            # Fit the exogenous ridge-regression with an optional intercept
            if fit_intercept or n_regressors > 0:
                resid = data - ZF
                if fit_intercept:
                    intercept = resid.mean(axis=0, keepdims=True)
                # end if

                if n_regressors > 0:
                    if fit_intercept:
                        resid -= intercept
                    # end if

                    # solve for beta
                    beta, lip_b = b_step(beta, resid, regressors_cntrd, C_B,
                                         kind="tron")

                    # mean(R) - mean(X) beta = mu
                    if fit_intercept:
                        intercept -= np.dot(regressors_mean, beta)
                    # end if
                # end if

                resid = data.copy()
                if n_regressors > 0:
                    resid -= np.dot(regressors, beta)

                if fit_intercept:
                    resid -= intercept
            else:
                resid = data
            # end if

            # update (F, Z), then phi
            for inner_iter in range(n_max_mf_iter):
                #print("loop2")
                loadings, lip_f = f_step(loadings, resid, factors, C_F, eta_F,
                                         adj, kind=f_step_kind, lip=lip_f)

                factors = z_step(factors, resid, loadings, ar_coef,
                                 C_Z, eta_Z, kind=z_step_kind)
            # end for

            if n_order > 0:
                ar_coef = phi_step(ar_coef, factors, C_Z, C_phi, eta_Z)
            # end if

            # recompute the reconstruction and convergence criteria
            ZF, ZF_old = np.dot(factors, loadings), ZF
            delta = np.linalg.norm(ZF - ZF_old, ord="fro")
            ZF_old_norm = np.linalg.norm(ZF_old, ord="fro")
        # end for
    # end if

    return factors, loadings, ar_coef, intercept, beta


# Modified `In[50]`
def trmf_forecast_factors(n_ahead, ar_coef, prehist):
    n_components, n_order = ar_coef.shape
    if n_ahead < 1:
        raise ValueError("""`n_ahead` must be a positive integer.""")

    if len(prehist) < n_order:
        raise TypeError("""Factor history is too short.""")

    forecast = np.concatenate([
        prehist[-n_order:] if n_order > 0 else prehist[:0],
        np.zeros((n_ahead, n_components))
    ], axis=0)

    # compute the dynamic forecast
    for h in range(n_order, n_order + n_ahead):
        # ar_coef are stored in little endian lag order: from lag p to lag 1
        #  from the least recent to the most recent!
        forecast[h] = np.einsum("il,li->i", ar_coef, forecast[h - n_order:h])

    return forecast[-n_ahead:]


# Extra functionality
def trmf_forecast_targets(n_ahead, loadings, ar_coef, intercept, beta,
                          factors, regressors=None, mode="exog"):
    n_regressors, n_targets = beta.shape
    if regressors is None:
        if n_regressors > 0:
            raise TypeError("""Regressors must be provided.""")
        regressors = np.empty((n_ahead, 0))

    #regressors = check_array(regressors, dtype="numeric",
     #                        accept_sparse=False, ensure_min_features=0)

    if regressors.shape[1] != n_regressors:
        raise TypeError("""Invalid number of regressor features.""")

    if mode == "exog":
        if regressors.shape[0] < n_ahead:
            raise TypeError("""Not enough future observations.""")

    elif mode == "auto":
        if n_regressors != n_targets:
            raise TypeError("""Invalid `beta` for mode `auto`.""")

        if regressors.shape[0] < 1:
            raise TypeError("""Insufficient history of targets.""")
    # end if

    # step 1: predict the latent factors
    forecast = trmf_forecast_factors(n_ahead, ar_coef, factors)
    factors_forecast = np.dot(forecast, loadings)

    # step 2: predict the targets
    if mode == "exog":
        # assume the regressors are exogenous
        targets = np.dot(regressors, beta) + factors_forecast + intercept

    elif mode == "auto":
        # Assume the regressors are order 1 autoregressors (can be
        #  order-q but needs embedding).
        targets = np.concatenate([
            regressors[-1:],
            np.zeros((n_ahead, n_regressors), dtype=regressors.dtype)
        ], axis=0)

        # compute the dynamic forecast
        for h in range(n_ahead):
            targets[h + 1] = intercept + np.dot(targets[h], beta)                              + factors_forecast[h]
        # end for
    # end if

    return targets[-n_ahead:]


# In[ ]:


class TRMFRegressor(BaseEstimator):
    def __init__(self,
                 n_components,
                 n_order,
                 C_Z=1e-1,
                 C_F=1e-1,
                 C_phi=1e-2,
                 eta_Z=0.5,
                 eta_F=0.,
                 adj=None,
                 C_B=0.0,
                 fit_regression=False,
                 fit_intercept=True,
                 nonnegative_factors=True,
                 tol=1e-5,
                 n_max_iterations=1000,
                 n_max_mf_iter=5,
                 z_step_kind="tron",
                 f_step_kind="tron",
                 random_state=None):
        super(TRMFRegressor, self).__init__()

        self.n_components = n_components
        self.n_order = n_order
        self.C_Z = C_Z
        self.C_F = C_F
        self.C_phi = C_phi
        self.eta_Z = eta_Z
        self.eta_F = eta_F
        self.adj = adj
        self.C_B = C_B
        self.fit_regression = fit_regression
        self.fit_intercept = fit_intercept
        self.tol = tol
        self.n_max_iterations = n_max_iterations
        self.n_max_mf_iter = n_max_mf_iter
        self.nonnegative_factors = nonnegative_factors
        self.random_state = random_state
        self.z_step_kind = z_step_kind
        self.f_step_kind = f_step_kind

    def fit(self, X, y=None, sample_weight=None):
        if not self.fit_regression:
            if y is not None:
                raise TypeError("""Exogenous regressors provided in `X`, """
                                """yet `fit_regression` is false.""")
            X, y = None, X

        else:
            if y is None:
                raise TypeError("""Endogenous data are is not provided """
                                """in `y`, yet `fit_regression` is True.""")
        # end if

        if X is None:
            X = np.empty((y.shape[0], 0))
        # end if

        check_consistent_length(X, y)

        f_step_kind = "fgm" if self.nonnegative_factors else self.f_step_kind
        estimates = trmf(y, self.n_components, self.n_order, self.C_Z,
                         self.C_F, self.C_phi, self.eta_Z, self.eta_F,
                         adj=self.adj, fit_intercept=self.fit_intercept,
                         regressors=X, C_B=self.C_B, tol=self.tol,
                         n_max_iterations=self.n_max_iterations,
                         n_max_mf_iter=self.n_max_mf_iter,
                         f_step_kind=f_step_kind,
                         z_step_kind=self.z_step_kind,
                         random_state=self.random_state)

        # Record the estimates in this instance's properties
        factors, loadings, ar_coef, intercept, beta = estimates

        self.factors_, self.loadings_ = factors, loadings
        self.ar_coef_ = ar_coef

        self.coef_, self.intercept_ = beta, intercept

        # self.fitted_ = np.dot(X, beta) + np.dot(factors, loadings) \
        #                + intercept

        return self
    
    def fit_predict(self, n_ahead):
        fitted_targets = np.dot(self.factors_,self.loadings_)
        forecast_targets = trmf_forecast_targets(n_ahead, self.loadings_, self.ar_coef_, self.intercept_, self.coef_,self.factors_)
        predicted = np.concatenate([fitted_targets,forecast_targets],axis=0)
        return predicted
        
    def forecast_factors(self, n_ahead):
        return trmf_forecast_factors(n_ahead, self.ar_coef_,
                                     prehist=self.factors_)

    def predict(self, X=None, n_ahead=10):
        if self.fit_regression:
            X = check_array(X, dtype="numeric", accept_sparse=False)
        else:
            X = np.empty((n_ahead, 0))
        # end if

        return trmf_forecast_targets(
            n_ahead, self.loadings_, self.ar_coef_, self.intercept_,
            self.coef_, self.factors_)


# In[ ]:


import os
os.listdir('/kaggle/input/m5-forecasting-accuracy/')


# In[ ]:


input_dir="/kaggle/input/m5-forecasting-accuracy/"


# In[ ]:


main_data=pd.read_csv(input_dir+'sales_train_evaluation.csv')
main_columns=main_data.columns[:6]


# In[ ]:


value_column=main_data.columns[6:]


# In[ ]:


value_column


# In[ ]:


def ReturnFirstSellDate(df,main_columns,value_column):
    main_data=df.copy()
    First_Sold_Date=pd.DataFrame(np.transpose(np.nonzero(main_data[value_column].values))).drop_duplicates(0).reset_index(drop=True)
    First_Sold_Date.columns=['Time_Series_Index','First_Date_sold']
    Feature_vector=pd.concat((main_data[main_columns],First_Sold_Date),axis=1)
    return Feature_vector


# In[ ]:


FeatureVector=ReturnFirstSellDate(main_data,main_columns,value_column)


# In[ ]:


dict_label_encoders={}
dict_one_hot_encoders={}
for col in main_columns[2:]:
    dict_label_encoders[col]= preprocessing.LabelEncoder()
    dict_one_hot_encoders[col]= preprocessing.OneHotEncoder()
    FeatureVector['lb_enc_'+col]=dict_label_encoders[col].fit_transform(main_data[col].values)
    dict_one_hot_encoders['onehot_enc_'+col]=preprocessing.OneHotEncoder(sparse=False)
    Base=pd.DataFrame(dict_one_hot_encoders['onehot_enc_'+col].fit_transform(main_data[col].values.reshape(main_data.shape[0],1)))
    Base.columns=['one_hot_col'+str(j) for j in Base.columns]
    FeatureVector[Base.columns]=Base
    


# In[ ]:


FeatureVector


# In[ ]:


X_data=main_data[value_column[-365*2-28:-28]].values
X_data.shape



# In[ ]:


Y_data=main_data[value_column[-28:]].values
Y_data.shape


# In[ ]:


#Use TRMF W matrix as Catagorical embedding 


# In[ ]:


from sklearn.decomposition import NMF


# In[ ]:


# X_data=main_data[value_column[-365*2-28:-28]].values
# for vec in range(100,200,50):
#     model = NMF(n_components=vec, init='random', random_state=0,l1_ratio=0.5)
#     W = model.fit_transform(X_data)
#     #H = model.components_
#     print(vec,model.reconstruction_err_)


# In[ ]:


# model = NMF(n_components=150, init='random', random_state=0,l1_ratio=0.5)
# W = model.fit_transform(X_data)
# #H = model.components_
# print(150,model.reconstruction_err_)


# In[ ]:





# In[ ]:


# %%time
# X_data=main_data[value_column].values
# X_data.shape
# # 60 12
# # Score 1.2208639911218397
# # 90 12
# # Score 1.2221116627832116
# # 120 12
# # Score 1.2243798351230097
# # CPU times: user 1h 9min 4s, sys: 11min 7s, total: 1h 20min 11s
# # Wall time: 33min 7s

# for k in [60,120,150]:
#     for j in [12]:
#         for p in [False]:
#             n_components = k
#             n_order = j

#             params = {
#                 "n_components" : n_components,
#                 "n_order" : n_order,
#                 "C_Z" : 5e1,
#                 "C_F" : 5e-4,
#                 "C_phi" : 1e-4,
#                 "eta_Z" : 0.25,
#                 "C_B" : 0.,
#                 "fit_regression" : False,
#                 "fit_intercept" : False,
#                 "nonnegative_factors" : p,
#                 "n_max_iterations" : 1000,
#                 "f_step_kind" : "tron",
#                 "tol" : 1e-5,'random_state':32
#             }
#             regressor= TRMFRegressor(**params)
#             regressor.fit(X_data.T)

#             import gc
#             from scipy.sparse import csr_matrix

#             calendar = pd.read_csv(input_dir+'calendar.csv')
#             sell_prices = pd.read_csv(input_dir+'sell_prices.csv')
#             sales = pd.read_csv(input_dir+'sales_train_validation.csv')

#             def transform(df):
#                 newdf = df.melt(id_vars=["id"], var_name="d", value_name="sale")
#                 newdf.sort_values(by=['id', "d"], inplace=True)
#                 newdf.reset_index(inplace=True)
#                 return newdf

#             from sklearn.metrics import mean_squared_error

#             def rmse(df, gt):
#                 df = transform(df)
#                 gt = transform(gt)
#                 return mean_squared_error(df["sale"], gt["sale"])
#             sample_submission = pd.read_csv(input_dir+'sample_submission.csv')
#             sample_submission = sample_submission[sample_submission.id.str.endswith('validation')]

#             NUM_ITEMS = sales.shape[0]    # 30490
#             DAYS_PRED = sample_submission.shape[1] - 1    # 28

#             # To make it simpler, I will run only the last 10 days
#             DAYS_PRED = 10
#             idcols = ["id", "item_id", "state_id", "store_id", "cat_id", "dept_id"]
#             product = sales[idcols]

#             # create weight matrix
#             pd.get_dummies(product.state_id, drop_first=False)
#             weight_mat = np.c_[
#                np.ones([NUM_ITEMS,1]).astype(np.int8), # level 1
#                pd.get_dummies(product.state_id, drop_first=False).values,
#                pd.get_dummies(product.store_id, drop_first=False).values,
#                pd.get_dummies(product.cat_id, drop_first=False).values,
#                pd.get_dummies(product.dept_id, drop_first=False).values,
#                pd.get_dummies(product.state_id + product.cat_id, drop_first=False).values,
#                pd.get_dummies(product.state_id + product.dept_id, drop_first=False).values,
#                pd.get_dummies(product.store_id + product.cat_id, drop_first=False).values,
#                pd.get_dummies(product.store_id + product.dept_id, drop_first=False).values,
#                pd.get_dummies(product.item_id, drop_first=False).values,
#                pd.get_dummies(product.state_id + product.item_id, drop_first=False).values,
#                np.identity(NUM_ITEMS).astype(np.int8) #item :level 12
#             ].T

#             weight_mat = weight_mat.astype("int8")
#             weight_mat, weight_mat.shape
#             weight_mat_csr = csr_matrix(weight_mat)
#             del weight_mat; gc.collect()
#             def cal_weight1(product):
#                 sales_train_val = sales
#                 d_name = ['d_' + str(i+1) for i in range(1913)]

#                 sales_train_val = weight_mat_csr * sales_train_val[d_name].values


#                 df_tmp = ((sales_train_val>0) * np.tile(np.arange(1,1914),(weight_mat_csr.shape[0],1)))

#                 start_no = np.min(np.where(df_tmp==0,9999,df_tmp),axis=1)-1

#                 flag = np.dot(np.diag(1/(start_no+1)) , np.tile(np.arange(1,1914),(weight_mat_csr.shape[0],1)))<1

#                 sales_train_val = np.where(flag, np.nan, sales_train_val)

#                 # denominator of RMSSE / RMSSE
#                 weight1 = np.nansum(np.diff(sales_train_val,axis=1)**2,axis=1)/(1913-start_no)

#                 return weight1

#             weight1 = cal_weight1(product)
#             # Get the last 28 days for weight2
#             cols = ["d_{}".format(i) for i in range(1886, 1886+28)]

#             data = sales[["id", 'store_id', 'item_id'] + cols]

#             data = data.melt(id_vars=["id", 'store_id', 'item_id'], var_name="d", value_name="sale")
#             data = pd.merge(data, calendar, how = 'left', left_on = ['d'], right_on = ['d'])
#             data = data[["id", 'store_id', 'item_id', "sale", "wm_yr_wk"]]
#             data = data.merge(sell_prices, on = ['store_id', 'item_id', 'wm_yr_wk'], how = 'left')
#             def cal_weight2(data):
#                 # calculate the sales amount for each item/level
#                 df_tmp = data
#                 df_tmp['amount'] = df_tmp['sale'] * df_tmp['sell_price']
#                 df_tmp =df_tmp.groupby(['id'])['amount'].apply(np.sum)
#                 df_tmp = df_tmp.values

#                 weight2 = weight_mat_csr * df_tmp 

#                 weight2 = weight2/np.sum(weight2)
#                 return weight2

#             weight2 = cal_weight2(data)
#             weight2.shape
#             def wrmsse(preds, y_true):
#                 # number of columns
#                 num_col = DAYS_PRED

#                 reshaped_preds = preds.reshape(num_col, NUM_ITEMS).T
#                 reshaped_true = y_true.reshape(num_col, NUM_ITEMS).T


#                 train = weight_mat_csr*np.c_[reshaped_preds, reshaped_true]

#                 score = np.sum(
#                             np.sqrt(
#                                 np.mean(
#                                     np.square(
#                                         train[:,:num_col] - train[:,num_col:])
#                                     ,axis=1) / weight1) * weight2)

#                 return score
#             # WRMSSE score
#             X_data.shape
#             all_pred=regressor.predict(n_ahead=28)
#             forecast_value=all_pred[-28:].T
#             (forecast_value-Y_data)
#             FeatureVector['emb_item']=[j for j in regressor.loadings_.T]
#             regressor.factors_.shape
#             forecast_value_df=pd.DataFrame(forecast_value)
#             Y_data_df=pd.DataFrame(Y_data)
#             forecast_value_df['id']=main_data['id']
#             Y_data_df['id']=main_data['id']
#             DAYS_PRED=28
#             dft = transform(forecast_value_df)
#             gtt = transform(Y_data_df)
#             print(n_components,n_order)
#             print('Score', wrmsse(dft["sale"].to_numpy(), gtt["sale"].to_numpy()))


# In[ ]:


get_ipython().run_cell_magic('time', '', 'X_data=main_data[value_column].values\nX_data.shape\n# 60 12\n# Score 1.2208639911218397\n# 90 12\n# Score 1.2221116627832116\n# 120 12\n# Score 1.2243798351230097\n# CPU times: user 1h 9min 4s, sys: 11min 7s, total: 1h 20min 11s\n# Wall time: 33min 7s\n\nfor k in [60]:\n    for j in [12]:\n        for p in [False]:\n            n_components = k\n            n_order = j\n\n            params = {\n                "n_components" : n_components,\n                "n_order" : n_order,\n                "C_Z" : 5e1,\n                "C_F" : 5e-4,\n                "C_phi" : 1e-4,\n                "eta_Z" : 0.25,\n                "C_B" : 0.,\n                "fit_regression" : False,\n                "fit_intercept" : False,\n                "nonnegative_factors" : p,\n                "n_max_iterations" : 1000,\n                "f_step_kind" : "tron",\n                "tol" : 1e-5,\'random_state\':32\n            }\n            regressor= TRMFRegressor(**params)\n            regressor.fit(X_data.T)\n\n            import gc\n            from scipy.sparse import csr_matrix\n\n            calendar = pd.read_csv(input_dir+\'calendar.csv\')\n            sell_prices = pd.read_csv(input_dir+\'sell_prices.csv\')\n            sales = pd.read_csv(input_dir+\'sales_train_validation.csv\')\n\n            def transform(df):\n                newdf = df.melt(id_vars=["id"], var_name="d", value_name="sale")\n                newdf.sort_values(by=[\'id\', "d"], inplace=True)\n                newdf.reset_index(inplace=True)\n                return newdf\n\n            from sklearn.metrics import mean_squared_error\n\n            def rmse(df, gt):\n                df = transform(df)\n                gt = transform(gt)\n                return mean_squared_error(df["sale"], gt["sale"])\n            sample_submission = pd.read_csv(input_dir+\'sample_submission.csv\')\n            sample_submission = sample_submission[sample_submission.id.str.endswith(\'validation\')]\n\n            NUM_ITEMS = sales.shape[0]    # 30490\n            DAYS_PRED = sample_submission.shape[1] - 1    # 28\n\n            # To make it simpler, I will run only the last 10 days\n            DAYS_PRED = 10\n            idcols = ["id", "item_id", "state_id", "store_id", "cat_id", "dept_id"]\n            product = sales[idcols]\n\n            # create weight matrix\n            pd.get_dummies(product.state_id, drop_first=False)\n            weight_mat = np.c_[\n               np.ones([NUM_ITEMS,1]).astype(np.int8), # level 1\n               pd.get_dummies(product.state_id, drop_first=False).values,\n               pd.get_dummies(product.store_id, drop_first=False).values,\n               pd.get_dummies(product.cat_id, drop_first=False).values,\n               pd.get_dummies(product.dept_id, drop_first=False).values,\n               pd.get_dummies(product.state_id + product.cat_id, drop_first=False).values,\n               pd.get_dummies(product.state_id + product.dept_id, drop_first=False).values,\n               pd.get_dummies(product.store_id + product.cat_id, drop_first=False).values,\n               pd.get_dummies(product.store_id + product.dept_id, drop_first=False).values,\n               pd.get_dummies(product.item_id, drop_first=False).values,\n               pd.get_dummies(product.state_id + product.item_id, drop_first=False).values,\n               np.identity(NUM_ITEMS).astype(np.int8) #item :level 12\n            ].T\n\n            weight_mat = weight_mat.astype("int8")\n            weight_mat, weight_mat.shape\n            weight_mat_csr = csr_matrix(weight_mat)\n            del weight_mat; gc.collect()\n            def cal_weight1(product):\n                sales_train_val = sales\n                d_name = [\'d_\' + str(i+1) for i in range(1913)]\n\n                sales_train_val = weight_mat_csr * sales_train_val[d_name].values\n\n\n                df_tmp = ((sales_train_val>0) * np.tile(np.arange(1,1914),(weight_mat_csr.shape[0],1)))\n\n                start_no = np.min(np.where(df_tmp==0,9999,df_tmp),axis=1)-1\n\n                flag = np.dot(np.diag(1/(start_no+1)) , np.tile(np.arange(1,1914),(weight_mat_csr.shape[0],1)))<1\n\n                sales_train_val = np.where(flag, np.nan, sales_train_val)\n\n                # denominator of RMSSE / RMSSE\n                weight1 = np.nansum(np.diff(sales_train_val,axis=1)**2,axis=1)/(1913-start_no)\n\n                return weight1\n\n            weight1 = cal_weight1(product)\n            # Get the last 28 days for weight2\n            cols = ["d_{}".format(i) for i in range(1886, 1886+28)]\n\n            data = sales[["id", \'store_id\', \'item_id\'] + cols]\n\n            data = data.melt(id_vars=["id", \'store_id\', \'item_id\'], var_name="d", value_name="sale")\n            data = pd.merge(data, calendar, how = \'left\', left_on = [\'d\'], right_on = [\'d\'])\n            data = data[["id", \'store_id\', \'item_id\', "sale", "wm_yr_wk"]]\n            data = data.merge(sell_prices, on = [\'store_id\', \'item_id\', \'wm_yr_wk\'], how = \'left\')\n            def cal_weight2(data):\n                # calculate the sales amount for each item/level\n                df_tmp = data\n                df_tmp[\'amount\'] = df_tmp[\'sale\'] * df_tmp[\'sell_price\']\n                df_tmp =df_tmp.groupby([\'id\'])[\'amount\'].apply(np.sum)\n                df_tmp = df_tmp.values\n\n                weight2 = weight_mat_csr * df_tmp \n\n                weight2 = weight2/np.sum(weight2)\n                return weight2\n\n            weight2 = cal_weight2(data)\n            weight2.shape\n            def wrmsse(preds, y_true):\n                # number of columns\n                num_col = DAYS_PRED\n\n                reshaped_preds = preds.reshape(num_col, NUM_ITEMS).T\n                reshaped_true = y_true.reshape(num_col, NUM_ITEMS).T\n\n\n                train = weight_mat_csr*np.c_[reshaped_preds, reshaped_true]\n\n                score = np.sum(\n                            np.sqrt(\n                                np.mean(\n                                    np.square(\n                                        train[:,:num_col] - train[:,num_col:])\n                                    ,axis=1) / weight1) * weight2)\n\n                return score\n            # WRMSSE score\n            X_data.shape\n            all_pred=regressor.predict(n_ahead=28)\n            forecast_value=all_pred[-28:].T\n            (forecast_value-Y_data)\n            FeatureVector[\'emb_item\']=[j for j in regressor.loadings_.T]\n            regressor.factors_.shape\n            forecast_value_df=pd.DataFrame(forecast_value)\n            Y_data_df=pd.DataFrame(Y_data)\n            forecast_value_df[\'id\']=main_data[\'id\']\n            Y_data_df[\'id\']=main_data[\'id\']\n            DAYS_PRED=28\n            dft = transform(forecast_value_df)\n            gtt = transform(Y_data_df)\n            print(n_components,n_order)\n            print(\'Score\', wrmsse(dft["sale"].to_numpy(), gtt["sale"].to_numpy()))')


# In[ ]:



X_data.shape
all_pred=regressor.fit_predict(n_ahead=28)
forecast_value=all_pred[-28:].T
(forecast_value-Y_data)
FeatureVector['emb_item']=[j for j in regressor.loadings_.T]
regressor.factors_.shape


# In[ ]:


regressor.loadings_.T.shape


# In[ ]:


regressor.factors_.shape


# In[ ]:


all_pred.shape


# In[ ]:




regressor.forecast_factors(n_ahead=28).shape


# In[ ]:


FeatureVector.to_pickle('FeatureVector_all_60_eval.pkl')
#regressor.factors_-eval.to_pickle('factors_.pkl')
np.save('factors_60_eval', regressor.factors_)
#regressor.forecast_factors(n_ahead=28).to_pickle('forecast_factors_all.pkl')
np.save('forecast_factors_all_60_eval', regressor.forecast_factors(n_ahead=28))


np.save('regressor_loadings__60_eval',regressor.loadings_.T)


# In[ ]:





# In[ ]:


regressor.loadings_.T.to_pickle('regressorloadings_60_eval.pkl')


# In[ ]:


#FeatureVector.to_pickle('FeatureVector_60.pkl')


# In[ ]:


#regressor.loadings_.T.to_pickle('regressorloadings.pkl')


# In[ ]:





# In[ ]:




