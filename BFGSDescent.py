# Optimization for Engineers - Dr.Johannes Hild
# global BFGS descent

# Purpose: Find xmin to satisfy norm(gradf(xmin))<=eps
# Iteration: x_k = x_k + t_k * d_k
# d_k is the BFGS direction. If a descent direction check fails, d_k is set to steepest descent and the inverse BFGS matrix is reset.
# t_k results from Wolfe-Powell

# Input Definition:
# f: objective class with methods .objective() and .gradient()
# x0: column vector in R ** n(domain point)
# eps: tolerance for termination. Default value: 1.0e-3
# verbose: bool, if set to true, verbose information is displayed

# Output Definition:
# xmin: column vector in R ** n(domain point)

# Required files:
# t = WolfePowellSearch(f, x, d) from WolfePowellSearch.py

# Test cases:
# myObjective = noHessianObjective()
# x0 = np.array([[-0.01], [0.01]])
# xmin = BFGSDescent(myObjective, x0, 1.0e-6, 1)
# should return
# xmin close to [[0.26],[-0.21]] with the inverse BFGS matrix being close to [[0.0078, 0.0005], [0.0005, 0.0080]]


import numpy as np
import WolfePowellSearch as WP


def matrnr():
    # set your matriculation number here
    matrnr = 23423928
    return matrnr


def BFGSDescent(f, x0: np.array, eps=1.0e-3, verbose=0):
    if eps <= 0:
        raise TypeError('range of eps is wrong!')

    if verbose:
        print('Start BFGSDescent...')

    countIter = 0
    x = x0
    n = x0.shape[0]
    E = np.eye(n)
    B = E


    while np.linalg.norm(f.gradient(x)) > eps :
        d = -B @ f.gradient(x)
        if f.gradient(x).T @ d >= 0:
            d = -f.gradient(x)
            B = E
        t_k = WP.WolfePowellSearch(f,x,d)
        delta_g = f.gradient(x+t_k * d) - f.gradient(x)
        delta_x = t_k * d
        x = x + t_k * d
        if delta_g.T @ delta_x <= 0:
            B = E
        else :
            rk = delta_x -B @ delta_g
            B += ((rk @ delta_x.T + delta_x @ rk.T) / (delta_g.T @ delta_x)) - ((rk.T @ delta_g) / (np.square(delta_g.T @ delta_x))) * (delta_x @ delta_x.T)
 
        countIter = countIter + 1

        
    if verbose:
        gradx = f.gradient(x)
        print('BFGSDescent terminated after ', countIter, ' steps with norm of gradient =', np.linalg.norm(gradx), 'and the inverse BFGS matrix is')
        print(B)

    return x

