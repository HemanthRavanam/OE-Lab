# Optimization for Engineers - Dr.Johannes Hild
# projected inexact Newton descent

# Purpose: Find xmin to satisfy norm(xmin - P(xmin - gradf(xmin)))<=eps
# Iteration: x_k = P(x_k + t_k * d_k)
# d_k starts as a steepest descent step and then CG steps are used to improve the descent direction until negative curvature is detected or a full Newton step is made.
# t_k results from projected backtracking

# Input Definition:
# f: objective class with methods .objective() and .gradient()
# P: box projection class with method .project() and .activeIndexSet()
# x0: column vector in R ** n(domain point)
# eps: tolerance for termination. Default value: 1.0e-3
# verbose: bool, if set to true, verbose information is displayed

# Output Definition:
# xmin: column vector in R ** n(domain point)

# Required files:
# dH = projectedHessApprox(f, P, x, d) from projectedHessApprox.py
# t = projectedBacktrackingSearch(f, P, x, d) from projectedBacktrackingSearch.py

# Test cases:
# p = np.array([[1], [1]])
# myObjective = simpleValleyObjective(p)
# a = np.array([[1], [1]])
# b = np.array([[2], [2]])
# myBox = projectionInBox(a, b)
# x0 = np.array([[2], [2]], dtype=float)
# eps = 1.0e-3
# xmin = projectedInexactNewtonCG(myObjective, myBox, x0, eps, 1)
# should return xmin close to [[1],[1]]

import numpy as np
import projectedBacktrackingSearch as PB
import projectedHessApprox as PHA

def matrnr():
    # set your matriculation number here
    matrnr = 23423928
    return matrnr


def projectedInexactNewtonCG(f, P, x0: np.array, eps=1.0e-3, verbose=0):

    if eps <= 0:
        raise TypeError('range of eps is wrong!')

    if verbose:
        print('Start projectedInexactNewtonCG...')

    countIter = 0
    xp = P.project(x0)
    # INCOMPLETE CODE STARTS
    cfail = False

    intterm = xp - P.project(xp - f.gradient(xp))

    minval = np.min((0.5,(np.sqrt(np.linalg.norm(intterm)))))
    n = (minval) * np.linalg.norm(intterm)

    while np.linalg.norm(intterm) > eps :
        xj = xp
        rj = f.gradient(xj)
        dj = -rj

        while np.linalg.norm(rj) > n :
            dH = PHA.projectedHessApprox(f, P, xp, dj)
            rho_j = dj.T @ dH
            if rho_j <= eps * np.square(np.linalg.norm(dj)):
                cfail = True
                break
            tj = np.square(np.linalg.norm(rj))/rho_j
            xj = xj + (tj * dj)
            r_old = rj
            rj = r_old + (tj * dH)
            beta_j = np.square((np.linalg.norm(rj))/(np.linalg.norm(r_old)))
            dj = -rj + (beta_j * dj)

        if not cfail:
            dk = xj - xp
        else:
            dk = -f.gradient(xp)

        tk = PB.projectedBacktrackingSearch(f, P, xp, dk)
        xp = P.project(xp + (tk * dk))

        intterm = xp - P.project(xp - f.gradient(xp))

        minval = np.min((0.5,(np.sqrt(np.linalg.norm(intterm)))))
        n = (minval) * np.linalg.norm(intterm)

        countIter = countIter +1
    # INCOMPLETE CODE ENDS
    if verbose:
        gradx = f.gradient(xp)
        stationarity = np.linalg.norm(xp - P.project(xp - gradx))
        print('projectedInexactNewtonCG terminated after ', countIter, ' steps with stationarity =', np.linalg.norm(stationarity))

    return xp

