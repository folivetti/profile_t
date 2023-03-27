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
from profile_t import ProfileT, GaussianLikelihood
from profile_t.plots import plot_all_theta_theta, plot_all_tau_theta

def DemoPuromycin():
    x = np.array([0.02, 0.02, 0.06, 0.06, 0.11, 0.11, 0.22, 0.22, 0.56, 0.56, 1.10, 1.10])
    y = np.array([76, 47, 97, 107, 123, 139, 159, 152, 191, 201, 207, 200])
    expr, theta = create_symbolic('205.1*x0 / (0.08 + x0)', 1, False)

    # create profile object and calculate the ci with linear approx. and without
    profile = ProfileT(expr, np.array([200, 0.05]), GaussianLikelihood(expr))
    profile.fit(x, y)
    profile.fit_all_profiles()

    profile.report_parameters_ci(0.05, True)
    profile.report_parameters_ci(0.05)
    print("\nPrediction intervals (Laplace):")
    profile.report_prediction_ci(x, 0.01, True)
    '''
    print("\nPrediction intervals (profile):")
    #profile.report_prediction_interval(x, 0.01)
    print("\nConfidence intervals of new points (linear):")
    profile.report_prediction_interval(np.array([0.04, 0.15, 0.6]), 0.1, True)
    print("\nConfidence intervals of new points (profile):")
    profile.report_prediction_interval(np.array([0.04, 0.15, 0.6]), 0.1, False)
    print("\nPrediction intervals of new points (linear):")
    profile.report_prediction_interval(np.array([0.04, 0.15, 0.6]), 0.1, True, True)
    print("\nPrediction intervals of new points (profile):")
    profile.report_prediction_interval(np.array([0.04, 0.15, 0.6]), 0.1, False, True)
    # create the plots
    plot_all_theta_theta(profile, "Puromycin", 0.01)
    plot_all_tau_theta(profile, "Puromycin")
    '''

if __name__ == '__main__':
    DemoPuromycin()
