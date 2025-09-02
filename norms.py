import numpy as np

# Helper functions
def spectral_norm(M: np.ndarray) -> float:
    """Matrix 2-norm (largest singular value)."""
    return np.linalg.svd(M, compute_uv=False)[0]

def log_norm_2(A: np.ndarray) -> float:
    """
    Log norm induced by 2-norm: mu_2(A) = lambda_max(A + A^T)/2. 
    See Lemma 1 in the paper.
    """
    S = (A + A.T)
    w = np.linalg.eigvalsh(S)  # symmetric eigenvalues
    return 0.5*float(np.max(w))

