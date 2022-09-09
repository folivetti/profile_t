"""
Example script that creates the model using create_symbolic without symplification.
"""

# Author: Fabricio Olivetti de Franca <folivetti@ufabc.edu.br>
#         Gabriel Kronberger <Gabriel.Kronberger@fh-hagenberg.at>
#
# License: MIT


import numpy as np
import sympy as sym

from profile_t.symbolic_expression import create_symbolic
from profile_t import ProfileT
from profile_t.plots import plot_all_theta_theta, plot_all_tau_theta

def DemoPuromycin():
    p = 2
    t_syms = sym.symbols(f"theta0:{p}")
    x_syms = sym.symbols("x")
    x = np.array([0.02, 0.02, 0.06, 0.06, 0.11, 0.11, 0.22, 0.22, 0.56, 0.56, 1.10, 1.10])
    y = np.array([76, 47, 97, 107, 123, 139, 159, 152, 191, 201, 207, 200])  
    expr, theta = create_symbolic('205.1*x0 / (0.08 + x0)', x, y, False)

    # create profile object and calculate the ci with linear approx. and without
    profile = ProfileT(expr, theta)
    profile.report_parameters_ci(0.01, True)
    profile.report_parameters_ci(0.01)
    print("\nPrediction intervals (linear):")
    profile.report_prediction_interval(np.arange(0,12,1), 0.01, True)
    print("\nPrediction intervals (profile):")
    profile.report_prediction_interval(np.arange(0,12,1), 0.01)

    # create the plots
    plot_all_theta_theta(profile, "Puromycin", 0.01)
    plot_all_tau_theta(profile, "Puromycin")


if __name__ == '__main__':
    DemoPuromycin()