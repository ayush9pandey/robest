from norms import spectral_norm
import control as ct
import numpy as np
import sympy as sp


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
    """ An augmented system object that 
    augments two systems (sys and sys_tilde) by collecting their 
    A matrices in a diagonal block matrix.
    B and B_tilde are stacked vertically.
    C and -C_tilde are stacked horizontally.
    D and D_tilde are in a block diagonal
    """
    def __init__(self, sys1, sys2):
        self.sys1 = sys1
        self.sys2 = sys2
        # Symbolic block matrices
        self.A_sym = self.augment_matrices(sys1.A_sym, sys2.A_sym)
        self.B_sym = self.stack_matrices(sys1.B_sym, sys2.B_sym, axis=0)
        self.C_sym = self.stack_matrices(sys1.C_sym, -sys2.C_sym, axis=1)
        self.D_sym = self.augment_matrices(sys1.D_sym, sys2.D_sym)
        print(self.D_sym)
        # Lambdify for numeric evaluation
        self.param_syms = sys1.param_syms
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
    x0 = np.zeros((system.A.shape[0],))
    sys = ct.ss(system.A, system.B,
                system.C, system.D)
    tout, yout, xout = ct.forced_response(sys, T, U,
                                          X0=x0, return_x=True)
    yout = np.squeeze(np.array(yout, dtype=float))
    return tout, yout

def compute_theorem1_bound(system, augmented_sys, theta_val, xbar0, u_amp, T_max):
    """See Theorem 1."""
    B_np = np.array(system.B, dtype=float)
    C_np = np.array(system.C, dtype=float)
    D_np = np.array(system.D, dtype=float)
    Bbar_np = np.array(augmented_sys.Bbar_sym, dtype=float)
    Cbar_np = np.array(augmented_sys.Cbar_sym, dtype=float)

    Abar = np.array(f_Abar(theta_val), dtype=float)
    dAbar = np.array(f_dAbar(theta_val), dtype=float)

    # Norms
    dA_norm = spectral_norm(dAbar) # ||∂Abar/∂θ||_2
    Bu_inf = np.linalg.norm(Bbar_np*u_amp, ord=np.inf) # ||Bbar * u||_∞ for scalar u, worst-case amplitude
    if Uvec.ndim == 1:
        U = Uvec.reshape(-1, 1)  # (T, 1)
    else:
        U = Uvec  # (T, m)
    # computation of Bu_infty norm:
    # For each t, find max of 
    # v_t = Bbar @ u_t. Then ||v_t||_\infty = max_i |(v_t)_i|
    Bu_inf = float(np.max(np.max(np.abs(Bbar_np @ U.T), axis=0)))

    # Log-norm (matrix measure)
    mu = log_norm_2(Abar)
    # Small epsilon safeguard 
    # if mu is too close to zero
    eps = 1e-9
    if abs(mu) < eps:
        mu = eps if mu >= 0 else -eps

    CbarT_Cbar_norm = spectral_norm(Cbar_np.T @ Cbar_np)
    Cbar_norm = spectral_norm(Cbar_np)
    K1 = (1.0/(4.0*abs(mu)**3)) * CbarT_Cbar_norm * (np.linalg.norm(xbar0, 2)**2)
    K2 = (2.0/(abs(mu)**5)) * (np.linalg.norm(xbar0, 2)) * CbarT_Cbar_norm
    K3 = (1.0/abs(mu)) * Cbar_norm

    # RHS of inequality
    rhs = (K1 * (dA_norm**2)
           + K2 * (dA_norm**3) * Bu_inf
           + K3 * (T_max**2) * (dA_norm**2) * Bu_inf)
    return rhs, dict(K1=K1, K2=K2, K3=K3, mu=mu, dA_norm=dA_norm, Bu_inf=Bu_inf)

### plotly plotting functions can be added
