import numpy as np

def matrnr():
    # set your matriculation number here
    matrnr = 23423928
    return matrnr


def projectedBacktrackingSearch(f, P, x: np.array, d: np.array, sigma=1.0e-4, verbose=0):
    xp = P.project(x)
    gradx = f.gradient(xp)
    decrease = gradx.T @ d

    if decrease >= 0:
        raise TypeError('descent direction check failed!')

    if sigma <= 0 or sigma >= 1:
        raise TypeError('range of sigma is wrong!')

    if verbose:
        print('Start projectedBacktrackingSearch...')

def backtracking_line_search(f, grad_f, P, xk, dk, sigma=0.1):
    if np.dot(grad_f(xk), dk) >= 0:
        raise ValueError("Descent direction check fails: ∇f(xk)⊤dk ≥ 0")
    
    t = 1.0
    while f(xk + t * dk) >= f(xk) - sigma / t * np.linalg.norm(xk - P(xk - t * grad_f(xk)))**2:
        t *= 0.5
    
    return t

# Example functions for testing (replace these with actual implementations)
def f(x):
    return np.sum(x**2)

def grad_f(x):
    return 2 * x

def P(x):
    # Projection function (identity for simplicity)
    return x

# Example usage
xk = np.array([1.0, 1.0])
dk = np.array([-1.0, -1.0])

t_star = backtracking_line_search(f, grad_f, P, xk, dk, sigma=0.1)
print(f"Optimal step size t*: {t_star}")
