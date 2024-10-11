import numpy as np
from scipy.linalg import svd

def robust_approximation_m_uv_trace_norm_reg(M, W, r, lambda_val=1e-3, rho=1.05, max_iter_in=100, signM=0):
    """
    Robust low-rank matrix approximation with missing data and outliers.
    
    min |W.*(M-E)|_1 + lambda*|V|_*
    s.t., E = UV, U'*U = I

    Parameters:
    M: (m, n) data matrix
    W: (m, n) indicator matrix, with '1' means 'observed', and '0' 'missing'.
    r: rank of the matrix approximation
    lambda_val: weighting factor of the trace-norm regularization, default 1e-3.
    rho: increasing ratio of the penalty parameter mu, default 1.05.
    max_iter_in: maximum number of iterations for inner loop, default 100.
    signM: if M >= 0, then signM = 1, otherwise signM = 0.

    Returns:
    M_est: (m, n) full matrix approximation
    U_est: (m, r) matrix
    V_est: (r, n) matrix
    L1_error: L1-norm error of observed data only
    """
    # Normalization
    scale = np.max(np.abs(M))
    M = M / scale

    # Default parameters
    m, n = M.shape
    max_iter_out = 500
    max_mu = 1e20
    mu = 1e-6
    M_norm = np.linalg.norm(M, 'fro')
    tol = 1e-9 * M_norm

    cW = np.ones(W.shape) - W  # complement of W
    display = True

    # Initialize optimization variables as zeros
    E = np.zeros((m, n))
    U = np.zeros((m, r))
    V = np.zeros((r, n))
    Y = np.zeros((m, n))  # Lagrange multiplier

    # Start main outer loop
    iter_out = 0
    objs = []
    while iter_out < max_iter_out:
        iter_out += 1
        itr_in = 0
        obj_pre = 1e20

        # Inner loop
        while itr_in < max_iter_in:
            # Update U
            temp = (E + Y / mu) @ V.T
            Us, sigma, Ud = svd(temp, full_matrices=False)
            U = Us @ Ud

            # Update V
            temp = U.T @ (E + Y / mu)
            Vs, sigma, Vd = svd(temp, full_matrices=False)
            svp = np.sum(sigma > lambda_val / mu)
            if svp >= 1:
                sigma = sigma[:svp] - lambda_val / mu
            else:
                svp = 1
                sigma = np.zeros(1)
            V = np.dot(np.dot(Vs[:, :svp], np.diag(sigma)), Vd[:svp, :])

            # Update E
            UV = U @ V
            temp1 = UV - Y / mu
            temp = M - temp1
            E = np.maximum(0, temp - 1 / mu) + np.minimum(0, temp + 1 / mu)
            E = (M - E) * W + temp1 * cW

            if signM > 0:
                E[E < 0] = 0

            # Evaluate current objective
            obj_cur = np.sum(np.abs(W * (M - E))) + lambda_val * np.sum(sigma) + np.sum(Y * (E - UV)) + mu / 2 * np.linalg.norm(E - UV, 'fro')**2

            # Check convergence of inner loop
            if abs(obj_cur - obj_pre) <= 1e-8 * abs(obj_pre):
                break
            else:
                obj_pre = obj_cur
                itr_in += 1

        # Update Lagrange multiplier and penalty parameter
        leq = E - UV
        stopC = np.linalg.norm(leq, 'fro')
        if display:
            obj = np.sum(np.abs(W * (M - UV))) + lambda_val * np.sum(sigma)
            objs.append(obj)
            if iter_out == 1 or iter_out % 50 == 0 or stopC < tol:
                print(f'iter {iter_out}, mu={mu:.1e}, obj={obj}, stopALM={stopC:.3e}')

        if stopC < tol:
            break
        else:
            Y = Y + mu * leq
            mu = min(max_mu, mu * rho)

    # De-normalization
    U_est = np.sqrt(scale) * U
    V_est = np.sqrt(scale) * V
    M_est = np.dot(U_est, V_est)
    L1_error = np.sum(np.abs(W * (scale * M - M_est)))

    return M_est, U_est, V_est, L1_error

