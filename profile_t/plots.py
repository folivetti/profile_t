"""
The :mod:`plots` implements auxiliary functions to plot the
confidence intervals.
"""

# Author: Fabricio Olivetti de Franca <folivetti@ufabc.edu.br>
#         Gabriel Kronberger <Gabriel.Kronberger@fh-hagenberg.at>
#
# License: MIT

import numpy as np
from scipy import stats
from scipy.interpolate import CubicSpline
import matplotlib
import matplotlib.pyplot as plt


def plot_theta_theta(profile, i, j, alpha, ax):
    '''
    Creates the pairwise plot of two parameters using the
    confidence alpha.

    Parameters
    ----------
    profile : ProfileT object
               the profile t with the pre-calculated splines
    i : int
          index of first parameter
    j : int
          index of second parameter
    alpha : float
             significance level
    ax : matplotlib axis
          axis to insert plot
    '''
    p, q = profile.approximateProfileContour(i, j, alpha)
    ax.plot(p, q, label='data', color='k')


def plot_tau_theta(profile, i, ax):
    '''
    Creates the linear and profile plot for the i-th parameter.

    Parameters
    ----------
    profile : ProfileT object
               the profile t with the pre-calculated splines
    i : int
          index of the parameter
    ax : matplotlib axis
          axis to insert plot
    '''
    t_dist = stats.t.ppf(1 - 0.01, profile.model.m - profile.model.n)
    taus = np.arange(-t_dist, t_dist, 0.01)
    thetas = profile.spline_tau2theta[i](taus)
    deltas = profile.theta[i] + profile.se[i] * taus
    ax.plot(thetas, taus, label='profile-t', color='blue')
    ax.plot(deltas, taus, '--', label='linear', color='orange')
    ax.legend()
    ax.set_xlabel(r'$\theta_{}$'.format(i))
    ax.set_ylabel(r'$\tau$')


def plot_errorbar(profile, alpha, ixs, ax1, ax2, ix=0):
    '''
    Creates the linear and profile plot for prediction
    intervals of data points with indices ixs.

    Parameters
    ----------
    profile : ProfileT object
               the profile t with the pre-calculated splines
    alpha : float
             significance level
    ixs : array_like
           indeces of the data points
    ax1 : matplotlib axis
           axis to insert linear plot
    ax2 : matplotlib axis
           axis to insert profile plot
    ix : int
         dimension to plot (default=0, only for multi-dimension)
    '''
    t_dist = stats.t.ppf(1 - alpha/2.0, profile.model.m - profile.model.n)
    lower, upper = profile.getPredictionsIntervals(alpha)
    ypred = profile.model.trueF(profile.theta)
    if profile.model.multivar:
        x = [profile.model.x[ix][i] for i in ixs]
    else:
        x = profile.model.x[ixs]

    ax1.plot(x, ypred[ixs], label="linear", color="orange", linestyle="--")
    ax1.fill_between(x, lower[ixs], upper[ixs], color="orange", alpha=0.2)

    lowers, uppers = np.zeros(profile.model.m), np.zeros(profile.model.m)
    for i in ixs:
        profile_t = profile.calculate_points_param_pred(i).right
        spline = CubicSpline(profile_t[0], profile_t[1][0, :])
        lowers[i], uppers[i] = spline(-t_dist), spline(t_dist)

    ax2.plot(x, ypred[ixs], label="profile-t", color="blue")
    ax2.fill_between(x, lowers[ixs], uppers[ixs], color="blue", alpha=0.2)

    ax1.set_xlabel(r"$x_{}$".format(ix))
    ax1.set_ylabel(r"$y$")
    ax1.legend()
    ax2.set_xlabel(r"$x_{}$".format(ix))
    ax2.set_ylabel(r"$y$")
    ax2.legend()


def plot_all_tau_theta(profile, model_name, font=None, hide_ticks=False):
    '''
    Creates all the tau x theta plots.

    Parameters
    ----------
    profile : ProfileT object
               the profile t with the pre-calculated splines
    model_name : str
                  name of the model, this will be prepended
                  into the figure filename.
    font : dict
            matplotlib font information (optional)
    hide_ticks : bool
                  whether to hide the ticks (optional)
    '''
    if font is not None:
        matplotlib.rc('font', **font)

    n_vars = len(profile.theta)
    for i in range(n_vars):
        _, ax = plt.subplots(figsize=(12, 12))
        plot_tau_theta(profile, i, ax)
        if hide_ticks:
            plt.xticks([])
            plt.yticks([])
        plt.savefig(f"{model_name}_tau_theta_{i}.pdf")


def plot_all_theta_theta(profile, model_name, alpha,
                         font=None, hide_ticks=False):
    '''
    Creates the pairwise plots of each pair of parameters.

    Parameters
    ----------
    profile : ProfileT object
               the profile t with the pre-calculated splines
    model_name : str
                  name of the model, this will be prepended
                  into the figure filename.
    alpha : float
              significance level
    font : dict
            matplotlib font information (optional)
    hide_ticks : bool
                  whether to hide the ticks (optional)
    '''
    if font is not None:
        matplotlib.rc('font', **font)

    n_vars = len(profile.theta)
    plt.subplots(n_vars-1, n_vars-1, figsize=(15, 15),
                 constrained_layout=True)
    for i in range(n_vars):
        for j in range(n_vars):
            if j > i:
                upper = (n_vars-1)*i + j
                lower = (n_vars-1)*j + i + 1
                ax = plt.subplot(n_vars-1, n_vars-1, upper)
                profile.plot_theta_theta(i, j, alpha, ax)
                ax.set_xlabel(r"$\theta_{}$".format(i))
                ax.set_ylabel(r"$\theta_{}$".format(j))
                if upper != lower and lower <= (n_vars-1)*(n_vars-1):
                    ax = plt.subplot(n_vars-1, n_vars-1, lower)
                    ax.set_axis_off()
    if hide_ticks:
        plt.xticks([])
        plt.yticks([])

    plt.savefig(f"{model_name}_theta_theta.pdf")
