import numpy as np
import scipy.sparse.linalg as spla
import scipy.sparse as sp

def diagnostics(A, x, b):
    # A: csr_matrix or LinearOperator
    # x: current solution vector
    # b: rhs
    # Ensure numpy arrays
    x = np.asarray(x)
    b = np.asarray(b)

    # residuals
    r = A.dot(x) - b
    rnorm = np.linalg.norm(r)
    bnorm = np.linalg.norm(b)
    xnorm = np.linalg.norm(x)
    rel_r = rnorm / (bnorm + 1e-30)
    print("||r|| =", rnorm, "||b|| =", bnorm, "relative residual =", rel_r)
    print("||dx|| (you reported) ~", np.linalg.norm(x - x))  # placeholder

    # dtype check
    dtype = None
    try:
        dtype = A.dtype
    except AttributeError:
        dtype = x.dtype
    print("dtype (matrix/vec):", dtype)

    # estimate condition number lower bound via Lanczos (approximate)
    try:
        # compute a couple largest/smallest singular values (symmetric A assumed -> use eigsh)
        if sp.issparse(A) or hasattr(A,'shape'):
            n = A.shape[0]
            # estimate largest eigenvalue
            lam_max, _ = spla.eigsh(A, k=1, which='LA', tol=1e-2, maxiter=500)
            lam_min, _ = spla.eigsh(A, k=1, which='SA', tol=1e-2, maxiter=500)
            lam_max = float(lam_max[0]); lam_min = float(lam_min[0])
            print("Approx eigenvalues: lambda_max =", lam_max, "lambda_min =", lam_min)
            if lam_min > 0:
                print("Est cond(A) ~", lam_max / lam_min)
            else:
                print("lambda_min <= 0 -> matrix indefinite or singular/near-singular")
    except Exception as e:
        print("Eigen estimate failed:", e)

    # estimated achievable relative residual ~ kappa * eps
    eps = np.finfo(np.float64).eps
    print("machine eps (float64) ~", eps)
    # if cond estimated:
    try:
        kappa = lam_max / lam_min
        min_rel = kappa * eps
        print("Estimated achievable relative residual ~ kappa*eps =", min_rel)
    except Exception:
        pass

# Example usage:
# diagnostics(A, x_current, b)
