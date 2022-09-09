"""
Example script that creates the model manually using sympy.
"""

# Author: Fabricio Olivetti de Franca <folivetti@ufabc.edu.br>
#         Gabriel Kronberger <Gabriel.Kronberger@fh-hagenberg.at>
#
# License: MIT

import numpy as np
import sympy as sym

import matplotlib

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

matplotlib.rc('font', **font)

from symbolic_expression import SymExpr
from profile_t import ProfileT
from plots import plot_all_theta_theta, plot_all_tau_theta

def DemoBOD():
    # data points
    x = np.array([1,2,3,4,5,7])
    y = np.array([8.3, 10.3, 19.0, 16.0, 15.6, 19.8])

    # sympy symbols
    x_symb = sym.symbols('x')
    t_symb = sym.symbols('theta0 theta1')

    # initial values of theta
    theta = [20, 0.24]

    # regression model
    expr = t_symb[0] * (1 - sym.exp(-x_symb * t_symb[1]))

    # create profile object and calculate the ci with linear approx. and without
    profile = ProfileT(SymExpr(expr, t_symb, x_symb, x, y), theta)
    profile.report_parameters_ci(0.01, True)
    profile.report_parameters_ci(0.01)
    print("\nPrediction intervals (linear):")
    profile.report_prediction_interval(np.arange(0,6,1), 0.01, True)
    print("\nPrediction intervals (profile):")
    profile.report_prediction_interval(np.arange(0,6,1), 0.01)

    # create the plots
    plot_all_theta_theta(profile, "BOD", 0.01, font)
    plot_all_tau_theta(profile, "BOD", font)


if __name__ == '__main__':
    DemoBOD()