import numpy as np
import pandas as pd
import sympy as sp
import plotly.graph_objects as go

from main import (
    System,
    AugmentedSystem,
    simulate_ct_system,
    compute_theorem1_bound,
)


def main():
    theta1 = sp.symbols('theta1', real=True)
    param_syms = [theta1]

    # True system symbolic matrices
    A0 = sp.Matrix([[0, 1],
                    [-20, -2]])
    A1 = sp.Matrix([[0, 0],
                    [-5, -sp.Rational(1, 2)]])
    A_theta = A0 + A1 * theta1
    B = sp.Matrix([[0], [1]])
    C = sp.Matrix([[1, 0]])
    D = sp.Matrix([[0]])
    sys_true = System(A_theta, B, C, D, param_syms)

    # Estimated system symbolic matrices
    A0_t = sp.Matrix([[0, 1],
                      [-19.80, -2.05]])
    A1_t = sp.Matrix([[0, 0],
                      [-5.10, -0.48]])
    Atilde_theta = A0_t + A1_t * theta1
    sys_tilde = System(Atilde_theta, B, C, D, param_syms)

    # Augmented system
    aug_sys = AugmentedSystem(sys_true, sys_tilde)

    # Derivative of Abar w.r.t theta
    dAbar_dtheta_sym = sp.diff(aug_sys.A_sym, theta1)
    f_dAbar = sp.lambdify(aug_sys.param_syms, dAbar_dtheta_sym, 'numpy')

    # Experiment setup
    T_final = 1.0
    dt = 0.01
    Tvec = np.arange(0.0, T_final + dt, dt)

    u_amp = 0.1
    u_freq = 20
    Uvec = u_amp * np.sin(2 * np.pi * u_freq * Tvec)

    x0_true = np.array([0.2, 0.0])
    x0_est = np.array([0.2, 0.0])
    xbar0 = np.concatenate([x0_true, x0_est])

    theta_grid = np.linspace(0.0, 1, 6)
    delta_theta = 1e-2
    data = []

    for theta in theta_grid:
        # Empirical derivative via finite difference
        ss_true = sys_true.get_ss((theta,))
        ss_tilde = sys_tilde.get_ss((theta,))
        _, y_true = simulate_ct_system(ss_true, Tvec, Uvec)
        _, y_tilde = simulate_ct_system(ss_tilde, Tvec, Uvec)
        ybar = y_true - y_tilde

        ss_true2 = sys_true.get_ss((theta + delta_theta,))
        ss_tilde2 = sys_tilde.get_ss((theta + delta_theta,))
        _, y_true2 = simulate_ct_system(ss_true2, Tvec, Uvec)
        _, y_tilde2 = simulate_ct_system(ss_tilde2, Tvec, Uvec)
        ybar2 = y_true2 - y_tilde2

        dybar_dtheta = (ybar2 - ybar) / delta_theta
        d_ybar_inf = np.max(np.abs(dybar_dtheta))

        bound_val, _ = compute_theorem1_bound(
            sys_true, aug_sys, theta, xbar0, Uvec, T_final, f_dAbar
        )

        data.append(
            {
                'theta': theta,
                'd_ybar_dtheta_inf': d_ybar_inf,
                'bound_rhs': bound_val,
            }
        )

    df = pd.DataFrame(data)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df['theta'],
            y=df['d_ybar_dtheta_inf'],
            mode='lines+markers',
            marker=dict(symbol='circle'),
            name='|∂ȳ/∂θ|∞ (empirical)',
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df['theta'],
            y=df['bound_rhs'],
            mode='lines+markers',
            marker=dict(symbol='square'),
            name='Theorem 1 bound',
        )
    )
    fig.update_layout(
        xaxis_title='theta',
        yaxis_title='Sensitivity of estimation error',
        title='Bound vs. empirical norms across theta',
        yaxis_type='log',
    )
    fig.show()


if __name__ == '__main__':
    main()
