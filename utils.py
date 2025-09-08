import numpy as np
import sympy as sp
import control as ct
from scipy.linalg import expm, eigh
from scipy.linalg import solve_continuous_lyapunov, eigvals
import math
from robest import System, AugmentedSystem
from norms import spectral_norm, log_norm_2


def make_f_dA(system: System, param_index: int):
    """Return a callable f_dA(*params) for 
    sensitivity to parameter of system"""
    theta = system.param_syms[param_index]
    dA_sym = sp.diff(system.A_sym, theta)
    return sp.lambdify(system.param_syms, dA_sym, "numpy")

def make_f_dAbar(aug_sys: AugmentedSystem, param_index: int):
    """Return a callable f_dAbar(*params) for 
    sensitivity to parameter of augmented system."""
    theta = aug_sys.param_syms[param_index]
    dAbar_sym = sp.diff(aug_sys.Abar_sym, theta)
    return sp.lambdify(aug_sys.param_syms, dAbar_sym, "numpy")

def make_f_xbar0(x0_expr: sp.Matrix, x0_tilde_expr: sp.Matrix, param_syms):
    """Return a callable f_xbar0(*params) for evaluating \bar x"""
    xbar_expr = sp.Matrix.vstack(x0_expr, x0_tilde_expr)  # 4x1
    f = sp.lambdify(param_syms, xbar_expr, "numpy")
    return lambda *thetas: np.array(f(*thetas), dtype=float).reshape(-1)

def make_f_dxbar0(x0_expr: sp.Matrix, x0_tilde_expr: sp.Matrix, param_syms, param_index: int):
    """Return a callable f_dxbar0(*params) for evaluating the derivative of x0 with theta"""
    sym = param_syms[param_index]
    dxbar_expr = sp.Matrix.vstack(x0_expr.diff(sym), x0_tilde_expr.diff(sym))  # 4x1
    f = sp.lambdify(param_syms, dxbar_expr, "numpy")
    return lambda *thetas: np.array(f(*thetas), dtype=float).reshape(-1)


def state_traj_ss(A, B, T, U, x0):
    """Solve the system and get the states x"""
    n, m = A.shape[0], B.shape[1]
    sys = ct.ss(A, B, np.eye(n), np.zeros((n, m)))
    tout, yout, xout = ct.forced_response(sys, T, U, X0=x0, return_x=True)
    return xout.T  


def Qo_finite_horizon(A, C, N, steps=1200):
    """Numerical finite-horizon observability Gramian ∫0^N e^{A^T s} C^T C e^{A s} ds."""
    ts = np.linspace(0.0, N, steps)
    dt = ts[1] - ts[0]
    Q = np.zeros((A.shape[0], A.shape[0]))
    G_prev = expm(A.T*0) @ (C.T @ C) @ expm(A*0)
    for k in range(1, steps):
        Ek = expm(A*ts[k])
        Gk = Ek.T @ (C.T @ C) @ Ek
        Q += 0.5*(G_prev + Gk)*dt
        G_prev = Gk
    return Q


def gramian_bound_trace(aug_sys, f_dAbar, theta_tuple, Tvec, Uvec, x0_true, x0_est):
    """
    See Appendix B for the proof of the gramian bound.
    It uses simulate_ct_system to obtain x and x̃.
    """
    # augmented numeric
    Abar, Bbar, Cbar, Dbar = aug_sys.get_numeric(theta_tuple)
    dAbar = np.array(f_dAbar(*theta_tuple), dtype=float)

    # get the state space of the true and the tilde (estimated) system from aug_sys
    ss_true  = aug_sys.sys1.get_ss(theta_tuple[:len(aug_sys.sys1.param_syms)])
    ss_tilde = aug_sys.sys2.get_ss(theta_tuple[:len(aug_sys.sys2.param_syms)])
    # solve for X
    _, _, Xtrue = simulate_ct_system(ss_true,  Tvec, Uvec, x0_true)
    _, _, Xtil  = simulate_ct_system(ss_tilde, Tvec, Uvec, x0_est)
    
    Xbar = np.vstack([Xtrue, Xtil])

    # Compute w_bar(t) as the multiplication of the symbolic derivative of Ā with theta and X_bar
    Wbar = dAbar @ Xbar
    # integrate over finite horizon
    w_l2_sq = float(np.trapz(np.sum(Wbar**2, axis=0), Tvec))

    # Compute gramian (steps sets the accuracy of the integration for the Gramian)
    Qo = Qo_finite_horizon(Abar, Cbar, Tvec[-1], steps=1200)
    trace_Qo = float(np.trace(Qo))

    return Tvec[-1] * trace_Qo * w_l2_sq


def exact_sensitivity_ct_sq(sys: System, f_dA, theta_tuple, Tvec, Uvec, x0):
    """
    Compute exact sensitivity to parameter for system sys
    Use sensitivity equation: ż = A z + (∂A/∂θ) x, y_s = C z. 
    Here z is the sensitivity coefficient for each state.
    Returns (trace, ∫|·|² dt).
    """
    A, B, C, D = sys.get_numeric(theta_tuple)
    X = state_traj_ss(A, B, Tvec, Uvec, x0)
    dA = np.array(f_dA(*theta_tuple), dtype=float)
    W = dA @ X.T
    n = A.shape[0]
    # create sensitivity system
    # initial sensitivity coeff values are 0 (see X0)
    sysz = ct.ss(A, np.eye(n), C, np.zeros((C.shape[0], n)))
    # changing
    _, ys, _ = ct.forced_response(sysz, Tvec, W, X0=np.zeros(n), return_x=True)
    ys = np.squeeze(np.asarray(ys).T)
    # integrate using trapz to get the norm
    l2_sq = float(np.trapz(ys**2, Tvec))
    return ys, l2_sq

def exact_error_sensitivity_ct_sq(sys_true: System, sys_tilde: System,
                                  f_dA_true, f_dA_tilde,
                                  theta_tuple, Tvec, Uvec, x0_true, x0_est):
    """Compute exact sensitivity to parameter for the estimation error"""
    dy_true, _  = exact_sensitivity_ct_sq(sys_true,  f_dA_true,  theta_tuple, Tvec, Uvec, x0_true)
    dy_tilde, _ = exact_sensitivity_ct_sq(sys_tilde, f_dA_tilde, theta_tuple, Tvec, Uvec, x0_est)
    dy_bar = dy_true - dy_tilde
    lhs_ct_sq = float(np.trapz(dy_bar**2, Tvec))
    return dy_bar, lhs_ct_sq

def finite_horizon_gramian(A, N, steps=1200):
    """Solve the Gramian (see paper) Wc(N) using trapezoid for integration."""
    ts = np.linspace(0.0, N, steps)
    dt = ts[1] - ts[0]
    W = np.zeros_like(A)
    G_prev = np.eye(A.shape[0])
    for k in range(1, steps):
        Ek = expm(A*ts[k])
        Gk = Ek @ Ek.T
        W += 0.5*(G_prev + Gk)*dt
        G_prev = Gk
    return W

def gramian_bound_augmented_ct_sq(sys_true: System, sys_tilde: System,
                                  aug_sys: AugmentedSystem, f_dAbar,
                                  theta_tuple, Tvec, Uvec, x0_true, x0_est):
    """
    Write the following in words:
    Bound: ||∂ȳ/∂θ||₂,[0,N] ≤ √λ_max(C̄ Wc C̄ᵀ) · || w̄ ||₂,[0,N],
    with w̄(t) = (∂Ā/∂θ) x̄(t),  x̄ = [x; x̃].
    Returns the squared bound.
    
    This function computes the bound on sensitivity of estimation error wrt parameters
    the bound is computed using the max eigen value of the Gramian 
    
    """
    A1, B1, C1, D1 = sys_true.get_numeric(theta_tuple)
    A2, B2, C2, D2 = sys_tilde.get_numeric(theta_tuple)
    Abar = np.block([[A1, np.zeros_like(A1)], [np.zeros_like(A2), A2]])
    dAbar = np.array(f_dAbar(*theta_tuple), dtype=float)

    X1 = state_traj_ss(A1, B1, Tvec, Uvec, x0_true)
    X2 = state_traj_ss(A2, B2, Tvec, Uvec, x0_est)
    Xbar = np.concatenate([X1, X2], axis=1)
    wbar = (Xbar @ dAbar.T)
    w_energy = np.trapz(np.sum(wbar**2, axis=1), Tvec)
    w_l2 = math.sqrt(max(w_energy, 0.0))

    Cbar = np.hstack([C1, -C2])
    Wc = finite_horizon_gramian(Abar, Tvec[-1], steps=1200)
    S = Cbar @ Wc @ Cbar.T
    lam_max = float(np.array(S).squeeze()) if S.size == 1 else float(eigh((S+S.T)/2, eigvals_only=True).max())
    gain = math.sqrt(max(lam_max, 0.0))
    return (gain * w_l2)**2

def simulate_ct_system(system, T, U, x0=None):
    """
    Simulate continuous-time LTI system using python-control
    under input U(t) over T vector.
    """
    if x0 is None:
        x0 = np.zeros((system.A.shape[0],))
    # If ``system`` is already a StateSpace instance, use it directly
    sys = system if isinstance(system, ct.StateSpace) else ct.ss(
        system.A, system.B, system.C, system.D
    )
    tout, yout, xout = ct.forced_response(sys, T, U, X0=x0, return_x=True)
    yout = np.squeeze(np.array(yout, dtype=float))
    return tout, yout, xout

def compute_theorem1_bound(system, augmented_sys, theta_val, xbar0, Uvec, T_max, f_dAbar):
    """Compute the bound in Theorem 1 for a given parameter value.

    Parameters
    ----------
    system : System
        The nominal system.
    augmented_sys : AugmentedSystem
        Augmented system capturing the difference between the true and
        estimated models.
    theta_val : float or sequence
        Parameter value at which to evaluate the bound.
    xbar0 : np.ndarray
        Initial augmented state ``[x(0); x_tilde(0)]``.
    Uvec : np.ndarray
        Input sequence used for simulation (shape ``(T,)`` or ``(T, m)``).
    T_max : float
        Time horizon of interest.
    f_dAbar : callable
        Function returning 
        ``\\frac{\partial \\bar{y}}{\partial \\theta}`` 
        evaluated at ``theta_val``.
    """

    # Numeric matrices from the augmented system
    param_vals = (theta_val,) if np.isscalar(theta_val) else tuple(theta_val)
    Bbar_np = np.array(augmented_sys.f_B_bar(*param_vals), dtype=float)
    Cbar_np = np.array(augmented_sys.f_C_bar(*param_vals), dtype=float)
    Abar = np.array(augmented_sys.f_A_bar(*param_vals), dtype=float)
    dAbar = np.array(f_dAbar(*param_vals), dtype=float)

    dA_norm = spectral_norm(dAbar)  # ||∂Abar/∂θ||_2

    if Uvec.ndim == 1:
        U = Uvec.reshape(-1, 1)
    else: 
        U = Uvec

    Bu_inf = float(np.max(np.max(np.abs(Bbar_np @ U.T), axis=0)))

    mu = log_norm_2(Abar)
    eps = 1e-9
    if abs(mu) < eps:
        mu = eps if mu >= 0 else -eps

    CbarT_Cbar_norm = spectral_norm(Cbar_np.T @ Cbar_np)
    Cbar_norm = spectral_norm(Cbar_np)

    K1 = (1.0 / (4.0 * abs(mu) ** 3)) * CbarT_Cbar_norm * (np.linalg.norm(xbar0, 2) ** 2)
    K2 = (2.0 / (abs(mu) ** 5)) * (np.linalg.norm(xbar0, 2)) * CbarT_Cbar_norm
    K3 = (1.0 / abs(mu)) * Cbar_norm

    rhs = (
        K1 * (dA_norm ** 2)
        + K2 * (dA_norm ** 3) * Bu_inf
        + K3 * (T_max ** 2) * (dA_norm ** 2) * Bu_inf
    )
    return rhs, dict(K1=K1, K2=K2, K3=K3, mu=mu, dA_norm=dA_norm, Bu_inf=Bu_inf)

def compute_theorem2_bound(system, augmented_sys, theta_val, f_dxbar0):
    """
    See equation (22) in the paper.
    Parameters
    ----------
    system : System
        The nominal system 
    augmented_sys : AugmentedSystem
    theta_val : float or sequence
        Parameter value(s) at which to evaluate the bound.
    f_dxbar0 : callable
        Function returning dx_bar/d_theta evaluated at theta_val as a flat np.array of shape (4,).

    Returns
    -------
    rhs_sq : float
        Right-hand side bound value (squared L2 norm).
    info : dict with meta data
    """
    param_vals = (theta_val,) if np.isscalar(theta_val) else tuple(theta_val)
    Abar = np.array(augmented_sys.f_A_bar(*param_vals), dtype=float)
    Cbar = np.array(augmented_sys.f_C_bar(*param_vals), dtype=float)
    Q = Cbar.T @ Cbar
    P = solve_continuous_lyapunov(Abar.T, -Q)
    lam_max_P = float(np.max(np.real(eigvals(P))))

    # get x0 derivative with theta
    dxbar0 = np.array(f_dxbar0(*param_vals), dtype=float).reshape(-1)
    dx_norm2 = float(dxbar0 @ dxbar0)
    rhs_sq = lam_max_P * dx_norm2
    return rhs_sq, dict(lam_max_P=lam_max_P, dx_norm2=dx_norm2)

def central_diff_ybar_x0(sys_true, sys_tilde, theta_tuple, param_index, delta, Tvec, Uvec, f_xbar0):
    thetas = list(theta_tuple)
    thetas_p = thetas.copy()
    thetas_p[param_index] += delta
    thetas_m = thetas.copy()
    thetas_m[param_index] -= delta

    xbar0_p = f_xbar0(*thetas_p)
    xbar0_m = f_xbar0(*thetas_m)
    x0_true_p, x0_tilde_p = xbar0_p[:2], xbar0_p[2:]
    x0_true_m, x0_tilde_m = xbar0_m[:2], xbar0_m[2:]

    _, y_true_p,  _ = simulate_ct_system(sys_true.get_ss(tuple(thetas_p)),  Tvec, Uvec, x0_true_p)
    _, y_tilde_p, _ = simulate_ct_system(sys_tilde.get_ss(tuple(thetas_p)), Tvec, Uvec, x0_tilde_p)
    ybar_p = y_true_p - y_tilde_p

    _, y_true_m,  _ = simulate_ct_system(sys_true.get_ss(tuple(thetas_m)),  Tvec, Uvec, x0_true_m)
    _, y_tilde_m, _ = simulate_ct_system(sys_tilde.get_ss(tuple(thetas_m)), Tvec, Uvec, x0_tilde_m)
    ybar_m = y_true_m - y_tilde_m

    # Central difference and L2^2 norm
    dybar_dtheta = (ybar_p - ybar_m) / (2.0 * delta)
    lhs_ct_sq = float(np.trapz(dybar_dtheta**2, Tvec))
    return lhs_ct_sq


def central_diff_ybar(sys_true, sys_tilde, theta_tuple, Tvec,
                      Uvec, delta, x0_true, x0_est, param_index):
    """The central difference method to compute the norm of
    dybar
    """
    th_plus = list(theta_tuple)
    th_minus = list(theta_tuple)
    th_plus[param_index]  += delta
    th_minus[param_index] -= delta
    th_plus, th_minus = tuple(th_plus), tuple(th_minus)

    _, y_true_p, _ = simulate_ct_system(sys_true.get_ss(th_plus),  Tvec, Uvec, x0_true)
    _, y_til_p,  _ = simulate_ct_system(sys_tilde.get_ss(th_plus), Tvec, Uvec, x0_est)
    ybar_p = y_true_p - y_til_p

    _, y_true_m, _ = simulate_ct_system(sys_true.get_ss(th_minus),  Tvec, Uvec, x0_true)
    _, y_til_m,  _ = simulate_ct_system(sys_tilde.get_ss(th_minus), Tvec, Uvec, x0_est)
    ybar_m = y_true_m - y_til_m

    dybar = (ybar_p - ybar_m) / (2.0*delta) # the central diff formula
    lhs_ct_sq = float(np.trapz(dybar**2, Tvec)) # integrate over time horizon
    return lhs_ct_sq

def robustness_R(theta_tuple, dy_l2_by_param, ybar_l2, param_order, eps=1e-12):
    """
    Compute R from equation 10 in the paper
    - theta_tuple : tuple of parameter values (order must match param_order)
    - dy_l2_by_param : dict, e.g. {"theta1": 0.12, "theta2": 0.09}  (L2 norms, not squared)
    - ybar_l2 : scalar L2 norm of ȳ(t) over [0,N]
    - param_order : list like ["theta1"] or ["theta1","theta2"] (defines mapping to theta_tuple)
    """
    if len(theta_tuple) != len(param_order):
        raise ValueError(f"len(theta_tuple)={len(theta_tuple)} != len(param_order)={len(param_order)}")
    for name in param_order:
        if name not in dy_l2_by_param:
            raise KeyError(f"Missing L2 entry for parameter '{name}'")
        val = float(dy_l2_by_param[name])
        if not np.isfinite(val) or val < 0:
            raise ValueError(f"L2 value for '{name}' must be finite and >= 0 (got {val})")
    if not np.isfinite(ybar_l2) or ybar_l2 <= 0:
        raise ValueError(f"L2 norm of y_bar must be finite and > 0 (got {ybar_l2})")

    theta_map = dict(zip(param_order, map(float, theta_tuple)))
    denom = float(ybar_l2)
    s = 0.0
    for name in param_order:
        s += theta_map[name] * (float(dy_l2_by_param[name]) / denom)
    return 1.0 / (1.0 + s)
