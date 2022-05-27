import scipy as sp
import scipy.linalg.blas as FB
from scipy import stats
from scipy import sparse
from scipy.linalg import lstsq
from scipy.sparse.linalg import spsolve
from scipy.sparse import linalg as splinalg
from scipy.special import erfinv

import statistics as stat
import pandas as pd
import time

import numpy as np
from numpy import concatenate as concat
from numpy import array, sqrt, exp, log, random, repeat, sum, mean, ones, zeros, diag, eye, empty
from numpy.random import randn, uniform, seed, beta
from numpy.linalg import solve, cholesky

import matplotlib.pyplot as plt
from datetime import datetime as dt

import sys

idx = pd.IndexSlice

################################ Functions #################################

np.set_printoptions(precision=4)


def nsample(n):
#     seed(1)
    return np.sqrt(2) * erfinv(2 * np.random.rand(n) - 1)


def nsample2(n, m):
#     seed(1)
    return np.sqrt(2) * erfinv(2 * np.random.rand(m, n).T - 1)


# cholesky decomposition is 2 times faster than lu decomposition...
def sparse_cholesky(A): # The input matrix A must be a sparse symmetric positive-definite.
    n = A.shape[0]
    LU = splinalg.splu(A, permc_spec='NATURAL', diag_pivot_thresh=0) # sparse LU decomposition
    
    if ( LU.perm_r == np.arange(n) ).all() and ( LU.U.diagonal() > 0 ).all(): # check the matrix A is positive definite.
        return LU.L.dot( sparse.diags(LU.U.diagonal()**0.5) )
    else:
        sys.exit('The matrix is not positive definite')


def sample_ps(scale_eps, ps_prior, n_scl_eps):

    n1 = sum(scale_eps == 1)
    n2 = len(scale_eps) - n1
    ps = beta(ps_prior[0] + n1, ps_prior[1] + n2)
    prob_scl_eps_vec = ps*ones(n_scl_eps)
    prob_scl_eps_vec[1:] = (1-ps)/(n_scl_eps-1)

    return prob_scl_eps_vec


def sample_scale_eps(e, sigma, ind_e, scale_e_vec, prob_scale_e_vec):
    # 10-component mixture from Omori, Chib, Shephard, and Nakajima JOE (2007)
    r_p = np.array([0.00609, 0.04775, 0.13057, 0.20674, 0.22715, 0.18842, 0.12047, 0.05591, 0.01575, 0.00115])
    r_m = np.array([1.92677, 1.34744, 0.73504, 0.02266, -0.85173, -1.97278, -3.46788, -5.55246, -8.68384, -14.65000])
    r_v = np.array([0.11265, 0.17788, 0.26768, 0.40611, 0.62699, 0.98583, 1.57469, 2.54498, 4.16591, 7.33342])
    r_s = np.sqrt(r_v)
    ##### for debugging  #################
    #e = eps
    #sigma = sigma_eps
    #ind_e = ind_eps
    #scale_e_vec = scl_eps_vec
    #prob_scale_e_vec = prob_scl_eps_vec
    #seed(1)
    ######################################  

    n = len(scale_e_vec)
    T = len(e)

    # Compute mean of ln(e^2) associated with each scale value
    scl_mean = log(scale_e_vec**2)

    # Compute Mean and SD from chi-squared 1 mixture
    mean_cs = ind_e.dot(r_m)
    sd_cs = ind_e.dot(r_s)

    # ln(e.^2) residual
    c = 0.001
    lnres2 = log(e**2 + c)    # c = 0.001 factor from ksv, restud(1998), page 370) 
    res_e = lnres2 - mean_cs - log(sigma**2)

    # Compute scale-specific demeaned version
    try:
        res_e = res_e.to_numpy()
    except:
        pass
    
    res_e_mat = repeat(res_e[:, None], n, axis=1) - repeat(scl_mean[None, :], T, axis=0)
    tmp_scl = repeat((1/sqrt(sd_cs))[:, None], n, axis=1)
    tmp_exp = exp(-0.5*(res_e_mat/sd_cs[:, None])**2)
    den_prob = tmp_scl*tmp_exp*prob_scale_e_vec[None, :]
    den_marg = den_prob.sum(axis=1)
    p_post = den_prob/den_marg[:, None]  # Posterior probability of different scale factors 

    # Draw Scale Factors from Posterior
    U = uniform(0, 1, T)
    bb = n - (U[:, None] <= p_post.cumsum(axis=1)).sum(axis=1)

    scale_e = scale_e_vec[bb]
    
    return scale_e


def SVRW(x, z_tilde, z0, omega, a0, b0, Vomega):
    # 10-component mixture from Omori, Chib, Shephard, and Nakajima JOE (2007)
    r_p = np.array([0.00609, 0.04775, 0.13057, 0.20674, 0.22715, 0.18842, 0.12047, 0.05591, 0.01575, 0.00115])
    r_m = np.array([1.92677, 1.34744, 0.73504, 0.02266, -0.85173, -1.97278, -3.46788, -5.55246, -8.68384, -14.65000])
    r_v = np.array([0.11265, 0.17788, 0.26768, 0.40611, 0.62699, 0.98583, 1.57469, 2.54498, 4.16591, 7.33342])
    r_s = np.sqrt(r_v)
    
    T = len(x)
    n = len(r_p)

    z = z0 + omega*z_tilde
    
    # Compute likelihood for each mixture and each time period
    xrep = repeat(x[:, None], n, axis=1)
    zrep = repeat(z[:, None], n, axis=1)
    mrep = repeat(r_m[None, :], T, axis=0)
    srep = repeat(r_s[None, :], T, axis=0)
    prep = repeat(r_p[None, :], T, axis=0)

    # sample S from a n-point discrete distribution
    pxlike = prep*exp(-0.5*((xrep - zrep - mrep)/srep)**2)/srep
    p_post = pxlike / pxlike.sum(axis=1)[:, None]

    # If data are missing, posterior = prior (which is in prep); 
    p_post[np.isnan(p_post)] = prep[np.isnan(p_post)]
    
    # Draw Indicators from posterior
    U = uniform(0, 1, T)
    S = n - (U[:, None] <= p_post.cumsum(axis = 1)).sum(axis=1)
    ind_e = sparse.coo_matrix((ones(T), (range(T), S)), shape=(T,n)).toarray()    
    
    # sample z_tilde
    H = sparse.identity(T) - sparse.coo_matrix((ones(T-1), (range(1, T), range(T-1))), shape=(T,T));
    d_s = r_m[S]
    iOs = sparse.diags(1/r_v[S])
    Kh = H.T.dot(H) + omega**2*iOs;
    z_tilde_hat = spsolve(Kh, omega*iOs.dot(x - d_s - z0));
    z_tilde = z_tilde_hat + spsolve(sparse_cholesky(Kh).T, random.randn(T));

    # sample z0 and omegaz
    Xbeta = concat((ones((T,1)), z_tilde[:, None]), axis=1);
    iVbeta = diag([1/b0, 1/Vomega]);    
    Kbeta = iVbeta + Xbeta.T.dot(iOs.toarray()).dot(Xbeta);
    beta_hat = solve(Kbeta, (iVbeta.dot([a0, 0]) + Xbeta.T.dot(iOs.toarray()).dot(x - d_s)));
    beta = beta_hat + solve(cholesky(Kbeta).T, random.randn(2));
    z0 = beta[0]; omega = beta[1];

    # randomly permute the signs h_tilde and omegah
    U = -1 + 2*(random.uniform() > .5);
    z_tilde = U*z_tilde;
    omega = U*omega;
    
    # compute the mean and variance of the conditional density of omegah    
    Dbeta = solve(Kbeta, eye(2));
    omega_hat = beta_hat[1];
    Domega = Dbeta[1,1];
    
    sigma_e = np.exp(0.5*(z0 + omega*z_tilde))
    # omega = g_draw in draw_g function
    
    return z_tilde, z0, omega, omega_hat, Domega, ind_e, sigma_e


def sample_dalpha_sigma(dalpha, nu_prior_alpha, s2_prior_alpha):

    n = dalpha.shape[0]
    T = dalpha.shape[1]

    SSR_mat = sum(dalpha**2, axis=1)
    SSR_prior = nu_prior_alpha*s2_prior_alpha

    a = (T + nu_prior_alpha)/2
    
    sigma_draw = np.nan*zeros(n)
    for i in range(n):
        b = 2/(SSR_mat[i] + SSR_prior)
        var_dalpha = 1/random.gamma(a, b)
        sigma_draw[i] = sqrt(var_dalpha)

    return sigma_draw


def sample_alpha_tvp(y, prior_var_alpha, sigma_dalpha, tau, eps_common, sigma_eps_unique):
    ### UNTITLED Summary of this function goes here
    #   Detailed explanation goes here

    # sigma_eps_unique = sigma_eps_unique_scl
    tau_unique = tau[1:]
    tau_common = tau[0]
    
    n_y = y.shape[0]
    nobs = y.shape[1]
    y = y - tau_unique    # Eliminates tau_unique from y ;

    # Brute Force Calculation
    ns = 2*n_y

    # First n_y elements of state are alpha_eps; second n_y elements of state are alpha_tau
    Q = diag(sigma_dalpha**2)    # Q_t = Q
    F = eye(ns)                  # F_t = F
    H = concat((np.kron(eps_common[:, None], eye(3)), np.kron(tau_common[:, None], eye(3))), axis=1).reshape(-1, 3, 6)
    H = np.transpose(H, (1, 2, 0))
    R = np.transpose(eye(n_y) * sigma_eps_unique.T[:, None, :]**2, (1, 2, 0))
    
    # random draws from N(0, 1)
    rand1 = nsample2(ns, nobs)
    rand2 = nsample2(ns, nobs+1)
    
    # Set up KF to run
    # Initial conditions
    x0 = zeros(ns)
    P0 = prior_var_alpha
    x_u = zeros((n_state, nobs+1))
    P_u = zeros((n_state, n_state, nobs+1))
    x_p = zeros((n_state, nobs+1))
    P_p = zeros((n_state, n_state, nobs+1))
    x_u[:, 0] = x0
    P_u[:, :, 0] = P0
    
    # Draw From State
    x_draw = np.zeros((ns, nobs + 1))  # sample from N(x_t|T, P_t|T)

    for t in range(nobs):
        x1 = F.dot(x0)                                  # x_1|0, x_2|1, ...
        P1 = F.dot(P0).dot(F.T) + Q   # P_1|0, P_2|1, ...
        nu = y[:, t] - H[:, :, t].dot(x1)
        S = H[:, :, t].dot(P1).dot(H[:, :, t].T) + R[:, :, t]
        invS = np.linalg.pinv(S)      
        # invS = solve(S, eye(n_y))
        K = P1.dot(H[:, :, t].T).dot(invS)

        x0 = x1 + K.dot(nu)                                      # x_1|1, x_2|2, ...
        P0 = (np.eye(ns) - K.dot(H[:, :, t])).dot(P1)            # P_1|1, x_2|2, ...
        P0 = 0.5*(P0 + P0.T)
        x_p[:, t + 1] = x1
        P_p[:, :, t + 1] = P1
        x_u[:, t + 1] = x0
        P_u[:, :, t + 1] = P0

    ############# Kalman Smoothing starts ################
    # Initial Draw
    PT = P0
    xT = x0
    # x = x3 + cholesky(P3).dot(randn(ns))
    xd = xT + cholesky(PT).dot(rand2[:, -1])
    x_draw[:, -1] = xd
        
    for t in range(nobs)[::-1]:
        x0 = x_u[:, t]
        x1 = x_p[:, t + 1]
        P0 = P_u[:, :, t]
        P1 = P_p[:, :, t + 1]
        FP0 = F.dot(P0)
        AS = solve(P1, FP0)
        PT = P0 - AS.T.dot(FP0)
        PT = 0.5*(PT + PT.T)
        xT = x0 + AS.T.dot(xd - x1)
        xd = xT + cholesky(PT).dot(rand2[:, t])
        x_draw[:, t] = xd

    dalpha_eps = x_draw[:n_y, 1:] - x_draw[:n_y, :-1]
    dalpha_tau = x_draw[n_y:, 1:] - x_draw[n_y:, :-1]
    alpha_eps = x_draw[:n_y, 1:]
    alpha_tau = x_draw[n_y:, 1:]
    dalpha = concat((dalpha_eps, dalpha_tau), axis=0)

    return alpha_eps, alpha_tau, dalpha



def kfilt(y, X1, P1, H, F, R, Q): # x_{t-1|t-1}, P_{t-1|t-1}
    nstate = X1.shape[0]
    eye_ny = eye(len(yt))
    X2 = F.dot(X1)                          # x_t|t-1 ~ x2
    e = y - H.T.dot(X2)               
    P2 = F.dot(P1).dot(F.T) + Q             # P_t|t-1 ~ P2
    ht = H.T.dot(P2).dot(H) + R
    hti = solve(ht, eye_ny)
    K = P2.dot(H).dot(hti)
    X1 = X2 + K.dot(e)                      # x_t|t
    P1 = (eye(nstate) - K.dot(H.T)).dot(P2) # P_t|t
    P1 = 0.5*(P1 + P1.T)
    return X1, P1, X2, P2


def sample_eps_tau(y, alpha_eps, alpha_tau, sigma_eps_scl, sigma_dtau):

    sigma_eps_common = sigma_eps_scl[0]
    sigma_eps_unique = sigma_eps_scl[1:]
    sigma_dtau_common = sigma_dtau[0]
    sigma_dtau_unique = sigma_dtau[1:]

    samll = 1e-6; big = 1e6
    n_y = y.shape[0]
    nobs = y.shape[1]

    # Set up State Vector
    # --- State Vector
    #     (1) eps(t)
    #     (2) tau(t)
    #     (3) tau_u(t)

    ns = 2 + n_y        # size of state
    F = zeros((ns, ns));     F[1:, 1:] = eye(ns-1)
    H = concat((alpha_eps, alpha_tau, eye(n_y).reshape(-1, 1).dot(ones((1, nobs)))), axis=0).reshape(n_y, ns, -1, order='F')
    Q = eye(ns) * concat((sigma_eps_common[None, :]**2, sigma_dtau_common[None, :]**2, sigma_dtau_unique**2), axis=0).T[:, None, :]
    Q = np.transpose(Q, (1, 2, 0))
    R = eye(n_y) * sigma_eps_unique.T[:, None, :]**2
    R = np.transpose(R, (1, 2, 0))
    
    rand1 = nsample2(ns, nobs)
    rand2 = nsample2(ns, nobs+1)

    # Set up KF to run
    # Initial conditions
    x0 = zeros(ns)               # x_0|0
    P0 = zeros((ns, ns))         # P_0|0
    P0[2:, 2:] = big*eye(n_y)    # Vague prior for tau_unique initial values 

    x_u = zeros((ns, nobs + 1))      # store x_1|1, x_2|2, ..., x_nobs|nobs
    P_u = zeros((ns, ns, nobs + 1))
    x_u[:, 0] = x0
    P_u[:, :, 0] = P0
    
    x_p = zeros((ns, nobs + 1))      # store x_1|0, x_2|1, ..., x_nobs|nobs-1
    P_p = zeros((ns, ns, nobs + 1))

    ############# Kalman Filtering starts ################
    # Draw from filtered .. these are marginal draws, not joint (same as ucsv) ;
    x_draw_f = np.zeros((ns, nobs + 1))  # sample from N(x_t|t, P_t|t)
    # Draw From State
    x_draw = np.zeros((ns, nobs + 1))  # sample from N(x_t|T, P_t|T)

    for t in range(nobs):
        x1 = F.dot(x0)                                  # x_1|0, x_2|1, ...
        P1 = F.dot(P0).dot(F.T) + Q[:, :, t]            # P_1|0, P_2|1, ...
        nu = y[:, t] - H[:, :, t].dot(x1)
        S = H[:, :, t].dot(P1).dot(H[:, :, t].T) + R[:, :, t]
        invS = np.linalg.pinv(S)      
        # invS = solve(S, eye(n_y))
        K = P1.dot(H[:, :, t].T).dot(invS)

        x0 = x1 + K.dot(nu)                                      # x_1|1, x_2|2, ...
        P0 = (np.eye(ns) - K.dot(H[:, :, t])).dot(P1)            # P_1|1, x_2|2, ...
        P0 = 0.5*(P0 + P0.T)
        x_p[:, t + 1] = x1
        P_p[:, :, t + 1] = P1
        x_u[:, t + 1] = x0
        P_u[:, :, t + 1] = P0
        
        # X = X1 + cholesky(P0).dot(randn(ns))
        x = x0 + cholesky(P0).dot(rand1[:, t])
        x_draw_f[:, t + 1] = x

        
    ############# Kalman Smoothing starts ################
    # Initial Draw
    P3 = P0
    x3 = x0
    
    # x = x3 + cholesky(P3).dot(randn(ns))
    x = x3 + cholesky(P3).dot(rand2[:, -1])
    x_draw[:, -1] = x

    for t in range(nobs)[::-1]:
        x0 = x_u[:, t]
        x1 = x_p[:, t + 1]
        P0 = P_u[:, :, t]
        P1 = P_p[:, :, t + 1]
        FP0 = F.dot(P0)
        AS = solve(P1, FP0)
        P3 = P0 - AS.T.dot(FP0)
        P3 = 0.5*(P3 + P3.T)
        x3 = x0 + AS.T.dot(x - x1)
        x = x3
        if t > 0:
            # x = x + cholesky(P3).dot(randn(ns))
            x = x + cholesky(P3).dot(rand2[:, t])
        else:
            P3 = P3[2:, 2:]
            # x[2:] = x[2:] + cholesky(P3).dot(randn(ns - 2))
            x[2:] = x[2:] + cholesky(P3).dot(rand2[2:, t])
        x_draw[:, t] = x

    ########### return x_draw, x_draw_f ##############
    eps_common = x_draw[0, 1:]
    tau_f = x_draw_f[1:, 1:]    
    dtau = x_draw[1:, 1:] - x_draw[1:, :-1]
    tau = x_draw[1:, 1:]
    
    return eps_common, tau_f, dtau, tau


def sample_tau(y, sigma_dtau, sigma_eps_scl):
    ##### for debugging  #################
    # y = yn; sigma_eps = sigma_eps_scl;
    #seed(1);
    ######################################
    samll = 1e-6; big = 1e6
    var_dtau = sigma_dtau**2
    var_eps = sigma_eps_scl**2

    # P matrices
    nobs = len(y)
    p1t = np.zeros(nobs+1)
    p2t = np.zeros(nobs+1)
    x1t = np.zeros(nobs+1)
    x2t = np.zeros(nobs+1)
    tau_a = np.zeros(nobs+1)
    tau_f = np.zeros(nobs)

    # KF using special structure of problem
    x1 = 0
    p1 = big
    x1t[0] = x1
    p1t[0] = p1
    
    err = nsample(nobs);
    for t in range(nobs):
        x2 = x1
        p2 = p1 + var_dtau[t]
        ht = p2 + var_eps[t]
        k = p2 / ht
        x1 = x2 + k*(y[t]-x2)
        p1 = p2 - k*p2
        x1t[t+1] = x1
        p1t[t+1] = p1
        x2t[t+1] = x2
        p2t[t+1] = p2
        # Generate random draw from filtered mean and variance 
        # tau_f[t] = x1 + np.sqrt(p1)*randn()
        tau_f[t] = x1 + np.sqrt(p1)*err[t]

    # Generate Random Draws from Smoothed Distribution 
    x3mean = x1;
    p3 = p1;
    chol_p = sqrt(p3);
    x3 = x3mean + chol_p*err[0];
    tau_a[nobs] = x3;

    for t in range(1, nobs)[::-1]:
        x2 = x2t[t+1]
        p2 = p2t[t+1]
        x1 = x1t[t]
        p1 = p1t[t]
        p2i = 1/p2
        k = p1*p2i
        e = x3 - x2
        x3mean = x1 + k*e
        p3 = (1-k)*p1
        chol_p = sqrt(p3)
        x3 = x3mean + chol_p*err[t]
        tau_a[t] = x3

    return tau_a, tau_f


def sample_lcs_indicators(e, sigma_e):
    # 10-component mixture from Omori, Chib, Shephard, and Nakajima JOE (2007)
    r_p = np.array([0.00609, 0.04775, 0.13057, 0.20674, 0.22715, 0.18842, 0.12047, 0.05591, 0.01575, 0.00115])
    r_m = np.array([1.92677, 1.34744, 0.73504, 0.02266, -0.85173, -1.97278, -3.46788, -5.55246, -8.68384, -14.65000])
    r_v = np.array([0.11265, 0.17788, 0.26768, 0.40611, 0.62699, 0.98583, 1.57469, 2.54498, 4.16591, 7.33342])
    r_s = np.sqrt(r_v)
    ##### for debugging  #################
    # e=eps_scaled
    # sigma_e = sigma_eps
    #seed(1)
    ######################################
    
    n = len(r_p)
    T = len(e)
    c = 0.001
    try:
        e = e.to_numpy()
    except:
        pass
    x = log(e**2 + c) - log(sigma_e**2)    # c = 0.001 factor from ksv, restud(1998), page 370)

    # Compute likelihood for each mixture and each time period
    xrep = repeat(x[:, None], n, axis=1)
    mrep = repeat(r_m[None, :], T, axis=0)
    srep = repeat(r_s[None, :], T, axis=0)
    prep = repeat(r_p[None, :], T, axis=0)
    
    pxlike = prep*exp(-0.5*((xrep-mrep)/srep)**2)/srep
    p_post = pxlike / pxlike.sum(axis=1)[:, None]
    # If data are missing, posterior = prior (which is in prep); 
    p_post[np.isnan(p_post)] = prep[np.isnan(p_post)]

    # Draw Indicators from posterior
    U = uniform(0, 1, T)
    bb = n - (U[:, None] <= p_post.cumsum(axis = 1)).sum(axis=1)

    ind_e = sparse.coo_matrix((ones(T), (range(T), bb)), shape=(T,n)).toarray()
    
    return ind_e


def sample_g(e, g_prior, ind_e, i_init):
    # 10-component mixture from Omori, Chib, Shephard, and Nakajima JOE (2007)
    r_p = np.array([0.00609, 0.04775, 0.13057, 0.20674, 0.22715, 0.18842, 0.12047, 0.05591, 0.01575, 0.00115])
    r_m = np.array([1.92677, 1.34744, 0.73504, 0.02266, -0.85173, -1.97278, -3.46788, -5.55246, -8.68384, -14.65000])
    r_v = np.array([0.11265, 0.17788, 0.26768, 0.40611, 0.62699, 0.98583, 1.57469, 2.54498, 4.16591, 7.33342])
    r_s = np.sqrt(r_v)
    ##### for debugging  #################
    #e = eps_scaled
    #g_prior = g_eps_prior
    #ind_e = ind_eps
    #i_init = 1
    #seed(1)
    ######################################    
    big = 1e6;  # large number     
    T = len(e)
    c = 0.001
    lnres2 = log(e**2 + c)      # c = 0.001 factor from ksv, restud(1998), page 370) 
    mean_t = ind_e.dot(r_m)
    sigma_t = ind_e.dot(r_s)
    var_t = sigma_t**2;

    # Kalman Filtering 
    ye = lnres2 - mean_t
    p1t = zeros(T+1)
    p2t = zeros(T+1)
    x1t = zeros(T+1);
    x2t = zeros(T+1);
    x3_draw = zeros(T+1) 

    # Compute log-likelihood values for each g value
    n_g = g_prior.shape[1]
    g_values = g_prior[0]
    p_values = g_prior[1]
    llf_vec = np.empty(n_g)
    
    # yt = Ht * xt + et
    # xt = Ft * xt-1 + ut
    # Qt = Var(ut), Rt = Var(et)
    for ig in range(n_g):
        gam = g_values[ig]**2
        # compute covariance matrix
        x1 = 0
        if i_init == 1:
            p1 = big
        elif i_init == 0:
            p1 = 0
        llf = 0
        for t in range(T):
            x2 = x1             # x2 = Ft * x1
            p2 = p1 + gam       # p2 = Ft * p1 * Ft' + Qt, Qt = gam
            e = ye[t] - x2      # e = yt - Ht * x2, Ht = 1
            h = p2 + var_t[t]   # h = Ht * p2 * Ht' + Rt, Rt = var_t[t]
            k = p2 / h          # k = p2 * Ht * inv(h), Ht = 1
            p1 = p2 - k*p2      # p1 = (I - k * Ht) * p2 = p2 - k * p2
            x1 = x2 + k*e       # x1 = x2 + k*e
            llf = llf - 0.5*(log(h) + e*e/h)
        llf_vec[ig] = llf

    lf_vec = exp(llf_vec - max(llf_vec))
    lf_marg = lf_vec.dot(p_values)
    g_post = (lf_vec*p_values)/lf_marg
    g_post = g_post/g_post.sum()
    bb = 5 - (uniform() <= g_post.cumsum()).sum()
    g_draw = g_values[bb]
    
    return g_draw


def sample_sigma(e, g, ind_e, i_init):
    # 10-component mixture from Omori, Chib, Shephard, and Nakajima JOE (2007)
    r_p = np.array([0.00609, 0.04775, 0.13057, 0.20674, 0.22715, 0.18842, 0.12047, 0.05591, 0.01575, 0.00115])
    r_m = np.array([1.92677, 1.34744, 0.73504, 0.02266, -0.85173, -1.97278, -3.46788, -5.55246, -8.68384, -14.65000])
    r_v = np.array([0.11265, 0.17788, 0.26768, 0.40611, 0.62699, 0.98583, 1.57469, 2.54498, 4.16591, 7.33342])
    r_s = np.sqrt(r_v)
    ##### for debugging  #################
    #e = eps_scaled
    #g = g_eps
    #ind_e = ind_e
    #i_init = 1
    #seed(1)
    ######################################  
    # 10-component mixture from Omori, Chib, Shephard, and Nakajima JOE (2007)
    r_p = np.array([0.00609, 0.04775, 0.13057, 0.20674, 0.22715, 0.18842, 0.12047, 0.05591, 0.01575, 0.00115])
    r_m = np.array([1.92677, 1.34744, 0.73504, 0.02266, -0.85173, -1.97278, -3.46788, -5.55246, -8.68384, -14.65000])
    r_v = np.array([0.11265, 0.17788, 0.26768, 0.40611, 0.62699, 0.98583, 1.57469, 2.54498, 4.16591, 7.33342])
    r_s = np.sqrt(r_v)  

    big = 1e6  # large number 
    small = 1e-10
    T = len(e)
    c = 0.001
    gam = g*g
    lnres2 = log(e**2 + c)   # c = 0.001 factor from ksv, restud(1998), page 370) 
    mean_t = ind_e.dot(r_m)
    sigma_t = ind_e.dot(r_s)
    var_t = sigma_t**2

    # Kalman Filtering 
    ye = lnres2 - mean_t
    p1t = zeros(T+1)
    p2t = zeros(T+1)
    x1t = zeros(T+1)
    x2t = zeros(T+1)
    x3_draw = zeros(T+1)

    # -- Compute Covariance Matrix  -- 
    x1=0;
    if i_init == 1:
        p1=big
    elif i_init == 0:
        p1 = 0

    x1t[0] = x1 
    p1t[0] = p1

    for t in range(T):
        x2 = x1
        p2 = p1+gam
        h = p2 + var_t[t]
        k = p2/h
        p1 = p2 - k*p2
        x1 = x2 + k*(ye[t] - x2)
        p1t[t+1] = p1
        p2t[t+1] = p2
        x1t[t+1] = x1
        x2t[t+1] = x2

    utmp = nsample(T+1)

    x3mean = x1
    p3 = p1
    chol_p = sqrt(p3)
    x3 = x3mean + chol_p*utmp[T]
    x3_draw[T] = x3

    for t in range(1, T)[::-1]:
        x2 = x2t[t+1]
        p2 = p2t[t+1]
        x1 = x1t[t]
        p1 = p1t[t]
        if p2 > small:
            p2i = 1/p2
            k = p1*p2i
            x3mean = x1 + k*(x3-x2)
            p3 = p1 - k*p1
        else:
            x3mean = x1
            p3 = p1
        chol_p = sqrt(p3)
        x3 = x3mean + chol_p*utmp[t]
        x3_draw[t] = x3
    
    sigma_e = exp(x3_draw[1:]/2)
    
    return sigma_e


def SVRW7(ystar, h_tilde, h0, omegah, a0, b0, Vomegah):

    # 7-component mixture from Omori, Chib, Shephard, and Nakajima JOE (2007)
    r_p = array([0.0073, .10556, .00002, .04395, .34001, .24566, .2575])
    r_m = array([-10.12999, -3.97281, -8.56686, 2.77786, .61942, 1.79518, -1.08819]) - 1.2704  # means already adjusted!
    r_v = array([5.79596, 2.61369, 5.17950, .16735, .64009, .34023, 1.26261])
    r_s = sqrt(r_v)
    
    T = len(h_tilde)
    n = len(r_p)
    
    # Compute likelihood for each mixture and each time period
    xrep = repeat(ystar[:, None], n, axis=1)
    hrep = repeat((h0 + omegah*h_tilde)[:,None], n, axis=1)
    mrep = repeat(r_m[None, :], T, axis=0)
    srep = repeat(r_s[None, :], T, axis=0)
    prep = repeat(r_p[None, :], T, axis=0)
    
    # sample S from a n-point discrete distribution
    U = random.uniform(0, 1, T)
    pxlike = prep*exp(-0.5*((xrep - hrep - mrep)/srep)**2)/srep
    p_post = pxlike / pxlike.sum(axis=1)[:, None]
    # Draw Indicators from posterior
    S = n - (U[:, None] <= p_post.cumsum(axis = 1)).sum(axis=1)

    # sample h_tilde
    H = sparse.identity(T) - sparse.coo_matrix((ones(T-1), (range(1, T), range(T-1))), shape=(T,T));
    d_s = r_m[S]
    iOs = sparse.diags(1/r_v[S])
    Kh = H.T.dot(H) + omegah**2*iOs;
    h_tilde_hat = spsolve(Kh, omegah*iOs.dot(ystar-d_s-h0));
    h_tilde = h_tilde_hat + spsolve(sparse_cholesky(Kh).T, random.randn(T));

    # sample h0 and omegah
    Xbeta = concat((ones((T,1)), h_tilde[:, None]), axis=1);
    iVbeta = diag([1/b0, 1/Vomegah]);    
    Kbeta = iVbeta + Xbeta.T.dot(iOs.toarray()).dot(Xbeta);
    beta_hat = solve(Kbeta, (iVbeta.dot([a0, 0]) + Xbeta.T.dot(iOs.toarray()).dot(ystar-d_s)));
    beta = beta_hat + solve(cholesky(Kbeta).T, random.randn(2));
    h0 = beta[0]; omegah = beta[1];

    # randomly permute the signs h_tilde and omegah
    U = -1 + 2*(random.uniform()>.5);
    h_tilde = U*h_tilde;
    omegah = U*omegah;
    
    # compute the mean and variance of the conditional density of omegah    
    Dbeta = solve(Kbeta, eye(2));
    omegah_hat = beta_hat[1];
    Domegah = Dbeta[1,1];
    
    return h_tilde, h0, omegah, omegah_hat, Domegah


def SVRW10(ystar, h_tilde, h0, omegah, a0, b0, Vomegah):

    # 10-component mixture from Omori, Chib, Shephard, and Nakajima JOE (2007)
    r_p = np.array([0.00609, 0.04775, 0.13057, 0.20674, 0.22715, 0.18842, 0.12047, 0.05591, 0.01575, 0.00115])
    r_m = np.array([1.92677, 1.34744, 0.73504, 0.02266, -0.85173, -1.97278, -3.46788, -5.55246, -8.68384, -14.65000])
    r_v = np.array([0.11265, 0.17788, 0.26768, 0.40611, 0.62699, 0.98583, 1.57469, 2.54498, 4.16591, 7.33342])
    r_s = np.sqrt(r_v)
    
    T = len(h_tilde)
    n = len(r_p)
    
    # Compute likelihood for each mixture and each time period
    xrep = repeat(ystar[:, None], n, axis=1)
    hrep = repeat((h0 + omegah*h_tilde)[:,None], n, axis=1)
    mrep = repeat(r_m[None, :], T, axis=0)
    srep = repeat(r_s[None, :], T, axis=0)
    prep = repeat(r_p[None, :], T, axis=0)

    # sample S from a n-point discrete distribution    
    U = random.uniform(0, 1, T)
    pxlike = prep*exp(-0.5*((xrep - hrep - mrep)/srep)**2)/srep
    p_post = pxlike / pxlike.sum(axis=1)[:, None]
    # Draw Indicators from posterior
    S = n - (U[:, None] <= p_post.cumsum(axis = 1)).sum(axis=1)

    # sample h_tilde
    H = sparse.identity(T) - sparse.coo_matrix((ones(T-1), (range(1, T), range(T-1))), shape=(T,T));
    d_s = r_m[S]
    iOs = sparse.diags(1/r_v[S])
    Kh = H.T.dot(H) + omegah**2*iOs;
    h_tilde_hat = spsolve(Kh, omegah*iOs.dot(ystar-d_s-h0));
    h_tilde = h_tilde_hat + spsolve(sparse_cholesky(Kh).T, random.randn(T));

    # sample h0 and omegah
    Xbeta = concat((ones((T,1)), h_tilde[:, None]), axis=1);
    iVbeta = diag([1/b0, 1/Vomegah]);    
    Kbeta = iVbeta + Xbeta.T.dot(iOs.toarray()).dot(Xbeta);
    beta_hat = solve(Kbeta, (iVbeta.dot([a0, 0]) + Xbeta.T.dot(iOs.toarray()).dot(ystar-d_s)));
    beta = beta_hat + solve(cholesky(Kbeta).T, random.randn(2));
    h0 = beta[0]; omegah = beta[1];

    # randomly permute the signs h_tilde and omegah
    U = -1 + 2*(random.uniform()>.5);
    h_tilde = U*h_tilde;
    omegah = U*omegah;
    
    # compute the mean and variance of the conditional density of omegah    
    Dbeta = solve(Kbeta, eye(2));
    omegah_hat = beta_hat[1];
    Domegah = Dbeta[1,1];
    
    return h_tilde, h0, omegah, omegah_hat, Domegah

