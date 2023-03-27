"""
Example script that creates the model using create_symbolic without symplification.
"""

# Author: Fabricio Olivetti de Franca <folivetti@ufabc.edu.br>
#         Gabriel Kronberger <Gabriel.Kronberger@fh-hagenberg.at>
#
# License: MIT


import numpy as np
from scipy.optimize import minimize

x = np.array([0.02, 0.02, 0.06, 0.06, 0.11, 0.11, 0.22, 0.22, 0.56, 0.56, 1.10, 1.10])
y = np.array([76, 47, 97, 107, 123, 139, 159, 152, 191, 201, 207, 200])
t0 = np.array([205.1, 0.08])
t0 = np.array([200, 0.05])

def negLogLike(theta):
    y_pred = theta[0] * x / (theta[1] + x)
    print(theta, y_pred)
    residue = y - y_pred
    return 0.5 * np.square(residue).sum()

def jac(theta):
    y_pred = theta[0] * x / (theta[1] + x)
    residue = y - y_pred
    j = np.zeros(2)
    j[0] = np.sum(-residue*(x / (theta[1] + x)))
    j[1] = np.sum(residue*(theta[0] * x / (theta[1] + x)**2))
    return j

'''
0.5 * sum( (y - f)^2 )

dL/di = -sum( (y - f) * df/di )
d^2L/di dj = sum( df/dj * df/di - (y - f) * d^2 f/didj )
'''
def hess(theta):
    y_pred = theta[0] * x / (theta[1] + x)
    residue = y - y_pred
    h = np.zeros((2, 2))
    h[0][0] = j[0]*j[0]/residue**2 # np.sum(-residue*(x / (theta[1] + x)))
    h[0][1] = j[1]*j[0]/residue**2 
    h[1] = np.sum(residue*(theta[0] * x / (theta[1] + x)**2))
    return j

if __name__ == '__main__':
    opt = minimize(negLogLike, t0, jac=jac, method='CG')
    print(negLogLike(t0))
    print(opt)
    theta = opt.x
    y_pred = theta[0] * x / (theta[1] + x)
    print(y - y_pred)
    print(np.square(x / (theta[1] + x)).sum())
