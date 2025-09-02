"""Helper classes and functions for robustness estimates."""

import control as ct
import numpy as np
import sympy as sp

from norms import spectral_norm, log_norm_2


class System(ct.StateSpace):
    """ A System class to manage the system objects
    used in this paper. It is a child of python-control
    statespace class and additionally provides symbolic
    handling of matrices.
    """
    def __init__(self, A_sym, B_sym, C_sym, D_sym, param_syms=None):
        """
        Initialize with symbolic matrices and optional parameter symbols.
        """
        self.A_sym = A_sym
        self.B_sym = B_sym
        self.C_sym = C_sym
        self.D_sym = D_sym
        self.param_syms = param_syms if param_syms is not None else []
        # Lambdify for numeric evaluation
        self.f_A = sp.lambdify(self.param_syms, self.A_sym, 'numpy')
        self.f_B = sp.lambdify(self.param_syms, self.B_sym, 'numpy')
        self.f_C = sp.lambdify(self.param_syms, self.C_sym, 'numpy')
        self.f_D = sp.lambdify(self.param_syms, self.D_sym, 'numpy')
        # Default numeric matrices (at zeros)
        param_zeros = [0.0]*len(self.param_syms)
        self.A = np.array(self.f_A(*param_zeros), dtype=float)
        self.B = np.array(self.f_B(*param_zeros), dtype=float)
        self.C = np.array(self.f_C(*param_zeros), dtype=float)
        self.D = np.array(self.f_D(*param_zeros), dtype=float)
        super().__init__(self.A, self.B, self.C, self.D)

    def get_numeric(self, param_vals):
        """Return numeric matrices for given parameter values."""
        A = np.array(self.f_A(*param_vals), dtype=float)
        B = np.array(self.f_B(*param_vals), dtype=float)
        C = np.array(self.f_C(*param_vals), dtype=float)
        D = np.array(self.f_D(*param_vals), dtype=float)
        return A, B, C, D

    def get_ss(self, param_vals):
        """Return a python-control StateSpace object for given parameter values."""
        A, B, C, D = self.get_numeric(param_vals)
        return ct.ss(A, B, C, D)




class AugmentedSystem(ct.StateSpace):
    """Augment two :class:`System` objects into a single system.

    The augmented system has a block-diagonal ``A`` matrix, the ``B``
    matrices stacked vertically, and the ``C`` matrices stacked
    horizontally such that the output corresponds to the difference of
    the individual system outputs.  ``D`` is set to the difference of the
    original feedthrough terms so that the resulting state-space model has
    a single input and a single output.
    """

    def __init__(self, sys1, sys2):
        self.sys1 = sys1
        self.sys2 = sys2

        # Dimension checks for compatibility
        if sys1.param_syms != sys2.param_syms:
            raise ValueError("Systems must share the same parameter symbols.")

        A1, B1, C1, D1 = sys1.A_sym, sys1.B_sym, sys1.C_sym, sys1.D_sym
        A2, B2, C2, D2 = sys2.A_sym, sys2.B_sym, sys2.C_sym, sys2.D_sym

        n1, n2 = A1.shape[0], A2.shape[0]
        if A1.shape[1] != n1 or A2.shape[1] != n2:
            raise ValueError("A matrices must be square.")

        if B1.shape[0] != n1 or B2.shape[0] != n2:
            raise ValueError("B rows must match corresponding A dimensions.")

        if C1.shape[1] != n1 or C2.shape[1] != n2:
            raise ValueError("C columns must match corresponding A dimensions.")

        m1, m2 = B1.shape[1], B2.shape[1]
        p1, p2 = C1.shape[0], C2.shape[0]
        if m1 != m2:
            raise ValueError("Systems must have the same number of inputs.")
        if p1 != p2:
            raise ValueError("Systems must have the same number of outputs.")
        if D1.shape != (p1, m1) or D2.shape != (p2, m2):
            raise ValueError("D matrices must have shape (outputs, inputs).")

        # Symbolic block matrices describing the augmented dynamics
        self.A_sym = self.augment_matrices(A1, A2)
        self.B_sym = self.stack_matrices(B1, B2, axis=0)
        self.C_sym = self.stack_matrices(C1, -C2, axis=1)
        # Feedthrough term corresponds to y - y_tilde
        self.D_sym = D1 - D2

        # Provide aliases used elsewhere in the notebook
        self.Bbar_sym = self.B_sym
        self.Cbar_sym = self.C_sym

        # Lambdify for numeric evaluation
        self.param_syms = sys1.param_syms
        self.f_A = sp.lambdify(self.param_syms, self.A_sym, "numpy")
        self.f_B = sp.lambdify(self.param_syms, self.B_sym, "numpy")
        self.f_C = sp.lambdify(self.param_syms, self.C_sym, "numpy")
        self.f_D = sp.lambdify(self.param_syms, self.D_sym, "numpy")

        # Default numeric matrices (evaluated at zeros)
        param_zeros = [0.0] * len(self.param_syms)
        self.A = np.array(self.f_A(*param_zeros), dtype=float)
        self.B = np.array(self.f_B(*param_zeros), dtype=float)
        self.C = np.array(self.f_C(*param_zeros), dtype=float)
        self.D = np.array(self.f_D(*param_zeros), dtype=float)

        super().__init__(self.A, self.B, self.C, self.D)

    def augment_matrices(self, *blocks):
        if len(blocks) == 1 and isinstance(blocks[0], (list, tuple)):
            blocks = tuple(blocks[0])
        # SymPy path
        if all(isinstance(b, sp.MatrixBase) for b in blocks):
            return sp.BlockDiagMatrix(*blocks).as_explicit()
        # NumPy path
        if all(isinstance(b, np.ndarray) for b in blocks):
            try:
                from scipy.linalg import block_diag as scipy_blkdiag
                return scipy_blkdiag(*blocks)
            except Exception:
                # minimal fallback
                m = sum(b.shape[0] for b in blocks)
                n = sum(b.shape[1] for b in blocks)
                out = np.zeros((m, n), dtype=np.result_type(*blocks))
                i = j = 0
                for b in blocks:
                    r, c = b.shape
                    out[i:i+r, j:j+c] = b
                    i += r; j += c
                return out
        raise TypeError("All blocks must be all SymPy \
            matrices or all NumPy arrays.")

    def stack_matrices(self, M1, M2, axis=0):
        """Stack two matrices along a specified axis."""
        if axis == 0:
            return sp.BlockMatrix([[M1], [M2]]).as_explicit()
        elif axis == 1:
            return sp.BlockMatrix([[M1, M2]]).as_explicit()
        else:
            raise ValueError("Invalid axis. Use 0 for vertical stacking or 1 for horizontal stacking.")

    def get_numeric(self, param_vals):
        """Return numeric matrices for given parameter values."""
        A = np.array(self.f_A(*param_vals), dtype=float)
        B = np.array(self.f_B(*param_vals), dtype=float)
        C = np.array(self.f_C(*param_vals), dtype=float)
        D = np.array(self.f_D(*param_vals), dtype=float)
        return A, B, C, D

    def get_ss(self, param_vals):
        """Return a python-control StateSpace object for given parameter values."""
        A, B, C, D = self.get_numeric(param_vals)
        return ct.ss(A, B, C, D)

### functions below can be moved around 

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
    tout, yout, _ = ct.forced_response(sys, T, U, X0=x0, return_x=True)
    yout = np.squeeze(np.array(yout, dtype=float))
    return tout, yout

def compute_theorem1_bound(system, augmented_sys, theta_val, xbar0, Uvec, T_max, f_dAbar):
    """Compute the bound in Theorem 1 for a given parameter value.

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
        Function returning ``∂Ā/∂θ`` evaluated at ``theta_val``.
    """

    # Numeric matrices from the augmented system
    param_vals = (theta_val,) if np.isscalar(theta_val) else tuple(theta_val)
    Bbar_np = np.array(augmented_sys.f_B(*param_vals), dtype=float)
    Cbar_np = np.array(augmented_sys.f_C(*param_vals), dtype=float)
    Abar = np.array(augmented_sys.f_A(*param_vals), dtype=float)
    dAbar = np.array(f_dAbar(*param_vals), dtype=float)

    # Norms
    dA_norm = spectral_norm(dAbar)  # ||∂Abar/∂θ||_2

    U = Uvec.reshape(-1, 1) if Uvec.ndim == 1 else Uvec
    Bu_inf = float(np.max(np.max(np.abs(Bbar_np @ U.T), axis=0)))

    # Log-norm (matrix measure)
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

### plotly plotting functions can be added
