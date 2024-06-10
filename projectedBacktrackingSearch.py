# Optimization for Engineers - Dr.Johannes Hild
# projected backtracking line search

# Purpose: Find t to satisfy f(x+t*d)< f(x) - sigma/t*norm(x-P(x - t*gradient))**2

# Input Definition:
# f: objective class with methods .objective() and .gradient() and .hessian()
# P: box projection class with method .project()
# x: column vector in R**n (domain point)
# d: column vector in R**n (search direction)
# sigma: value in (0,1), marks quality of decrease. Default value: 1.0e-4.
# verbose: bool, if set to true, verbose information is displayed

# Output Definition:
# t: t is set to the biggest 2**m, such that 2**m satisfies the projected sufficient decrease condition

# Required files:
# <none>

# Test cases:
# p = np.array([[0], [1]])
# myObjective = simpleValleyObjective(p)
# a = np.array([[-2], [1]])
# b = np.array([[2], [2]])
# eps = 1.0e-6
# myBox = projectionInBox(a, b, eps)
# x = np.array([[1], [1]])
# d = np.array([[-1.99], [0]])
# sigma = 0.5
# t = projectedBacktrackingSearch(myObjective, myBox, x, d, sigma, 1)
# should return t = 0.5

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

    beta = 0.5
    t = 1

    def WP1(ft, fx, P, s):
        isWP1 = ft <= fx - s * sigma * np.square(np.linalg.norm(x - P.project(x - t * f.gradient(x))))
        return isWP1

    fx = f.objective(xp)
    
    while not WP1(f.objective(P.project(x + t * d)), fx, P, t):
        t = t / 2
    

    if verbose:
        print('projectedBacktrackingSearch terminated with t=', t) 

    return t

'''
(sigma*np.linalg.norm(x-P(x-t*f.gradient(x)))^^2*descent)
'''

'''
 def WP1(ft, fx, P, s):
        isWP1 = ft <= fx - s * sigma * 2^np.linalg.norm(x-P(x-t * f.gradient(x))) * descent
        return isWP1

    while WP1(f.objective(P * (x + (t * d)) , t) == False:
        t = t / 2
        
'''
'''
def WP1(ft, fx, P, s):
        isWP1 = ft <= fx - s * sigma * 2^np.linalg.norm(x-P(x-t * f.gradient(x)))
        return isWP1

    while WP1(f.objective(int(P * (x + (t * d)) , t)) == False:
        t = t / 2

'''
'''
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
            dk = -f.gradient(xj)

        tk = PB.projectedBacktrackingSearch(f, P, xp, dk)
        xp = P.project(xp + (tk * dk))

        intterm = xp - P.project(xp - f.gradient(xp))

        minval = np.min((0.5,(np.sqrt(np.linalg.norm(intterm)))))
        n = (minval) * np.linalg.norm(intterm)
'''