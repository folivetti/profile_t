"""
The :mod:`profile_t` module implements the profile t likelihood algorithm. This
calculates the confidence interval and prediction interval for any nonlinear
regression models described as SymExpr or SymExprMultivar object.
"""

# Author: Fabricio Olivetti de Franca <folivetti@ufabc.edu.br>
#         Gabriel Kronberger <Gabriel.Kronberger@fh-hagenberg.at>
#
# License: MIT

import sys
import warnings
import numpy as np
import scipy.linalg as la

from scipy.optimize import minimize
from scipy.interpolate import CubicSpline
from scipy import stats
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

__all__ = [
    "ProfileT"
]

class ProfileT:
    """Class for profile t calculation"""

    def __init__(self, likelihood):
        '''
        Compute the profile t function from the symbolic nonlinear
        regression model.

        Parameters
        ----------
        start_theta : array_like
                      initial values of numerical parameters.
        likelihood : a likelihood object.

        Examples
        --------
        >>> TODO.
        OUTPUT TODO
        '''
        self.likelihood = likelihood

        # maximum value of tau for 99% confidence
        # using F-statistics with n, m-n degrees of freedom
        self.tau_max = np.sqrt(stats.f.ppf(1 - 0.01,
                                           self.likelihood.n,
                                           self.likelihood.m - self.likelihood.n)
                               )

        self.profiles = {}
        self.spline_tau2theta = {}
        self.spline_theta2tau = {}
        self.calculate_statistics()

    def fit_all_profiles(self):
        for ix in range(self.likelihood.n):
            self.fit_profile(ix)

    def fit_profile(self, ix):
        if ix in self.profiles:
            return

        # the either object will contain None in left field
        # only when the process returned without error.
        self.profiles[ix] = self.calculate_points_param(ix, 300, 8)
        self.create_splines(ix)

    def calculate_points_param(self, idx, kmax, step):
        '''
        Calculates the data points for numeric parameter idx.
        '''
        delta = -self.se[idx]/step

        # First we generate the data points starting from -delta,
        # after that we generate the points starting from delta.
        # Finally, we concatenate the data points together
        # with the trivial point where tau = 0.

        tau_1, m_1, deltas_1 = self.points_from_delta(delta, idx, kmax)
        tau_2, m_2, deltas_2 = self.points_from_delta(-delta, idx, kmax)

        # inserts the (tau, theta) = (0, theta_opt) point
        tau = np.array(tau_1 + [0] + tau_2)
        ix_tau = tau.argsort()

        return (tau[ix_tau],
                 np.array(m_1 + [self.likelihood.theta] + m_2)[ix_tau, :].T,
                 np.array(deltas_1 + [0] + deltas_2)[ix_tau])

    def points_from_delta(self, delta, idx, kmax):
        '''
        Calculates the data points for an initial delta.
        '''
        taus, thetas, deltas = [], [], []
        inv_slope = 1.0
        t = 0.0

        # we disturb the idx-th value of theta by delta*t
        # and re-optimize the other parameters.
        # This will create a series of (tau, theta) values
        # used to calculate the CI.
        for _ in range(kmax):
            t += inv_slope
            theta_cond = self.likelihood.refit(self.likelihood.theta[idx] + delta*t, idx)
            nnlOpt = self.likelihood.minNegLogLikelihood()
            nnl = self.likelihood.negLogLikelihood(theta_cond)

            if nnl < nnlOpt:
                raise Exception(f"A better parameter was found, refit the model with starting theta = {theta_cond}.")

            tau_i = np.sign(delta) * np.sqrt(2*nnl - 2*nnlOpt)
            zv = self.likelihood.negLogLikelihoodGrad(theta_cond)[idx]

            inv_slope = np.abs(tau_i/(self.se[idx] * zv))
            inv_slope = min(4.0, max(np.abs(inv_slope), 1.0/16))

            taus.append(tau_i)
            thetas.append(theta_cond.copy())
            deltas.append((self.likelihood.theta[idx] - theta_cond[idx])/self.se[idx])

            if np.abs(tau_i) > self.tau_max:
                break

        return (taus, thetas, deltas)

    def calculate_statistics(self):
        """
        Calculates the following statistics about the model:

            * se : standard error
            * corr : correlation between coefficients
        """

        H = self.likelihood.hessian()

        # we must use economic mode so R will be a square matrix
        R = la.cholesky(H, lower=True)
        self.invH = la.solve(R @ R.T, np.eye(self.likelihood.n)) # la.inv(H)
        self.se = np.sqrt(np.diag(self.invH))

        L = np.zeros((self.likelihood.n, self.likelihood.n))
        for i in range(self.likelihood.n):
            for j in range(self.likelihood.n):
                L[i, j] = self.invH[j, i] / self.se[i] / self.se[j]

        self.corr = L

    def report_parameters_ci(self, alpha, use_laplace=False):
        '''
        Prints out the model ssr and s^2 and the confidence interval.

        Parameters
        ----------
        alpha : float
                significance level
        use_linear : bool (default False)
                     whether to use linear instead of profile t
        '''

        print(f"theta: {self.likelihood.theta}")

        lower, upper = self.get_params_intervals(alpha, use_laplace)
        print("theta\tEstimate\tStd. Error.\tLower\t\tUpper\t\t"
              "Corr. Matrix")
        for i in range(self.likelihood.n):
            print(f"{i}\t{self.likelihood.theta[i]:e}\t{self.se[i]:e}\t"
                  f"{lower[i]:e}\t{upper[i]:e}\t{np.round(self.corr[i,:],2)}")

    def report_prediction_ci(self, xs, alpha, use_laplace=False):
        '''
        Prints out the prediction intervals for the data points
        ixs.

        Parameters
        ----------
        ixs : array_like
               list of indices to report
        alpha : float
                significance level
        use_linear : bool (default False)
                     whether to use linear instead of profile t
        '''
        lower, upper = self.get_prediction_ci(alpha, xs, use_laplace)
        y_pred = self.likelihood.predict(xs)
        print("y_pred\t\tlow\t\thigh")
        for i, _ in enumerate(xs):
            print(f"{y_pred[i]:e}\t"
                  f"{lower[i]:e}\t{upper[i]:e}")

    def get_params_intervals(self, alpha, use_laplace=False):
        '''
        Returns the lower and upper bounds of the
        parameters intervals with confidence alpha.

        Parameters
        ----------
        alpha : float
                significance level
        use_linear : bool
                      whether to use linear or not (default False)

        Returns
        --------
        lower : array_like
                 lower bounds for each parameter
        upper : array_like
                 upper bounds for each parameter
        '''
        t_dist = stats.t.ppf(1 - alpha/2.0, self.likelihood.m - self.likelihood.n)
        if use_laplace:
            lower = self.likelihood.theta - self.se * t_dist
            upper = self.likelihood.theta + self.se * t_dist
        else:
            lower = np.array(list(self.spline_tau2theta[i](-t_dist) for i in range(self.likelihood.n)))
            upper = np.array(list(self.spline_tau2theta[i](t_dist) for i in range(self.likelihood.n)))
        return lower, upper

    def get_prediction_ci(self, alpha, x, use_laplace):
        '''
        Returns the lower and upper bounds of the
        prediction intervals with confidence alpha.

        Parameters
        ----------
        alpha : float
                significance level
        ixs : array_like
               list of indices to report
        use_linear : bool
                      whether to use linear or not (default False)

        Returns
        --------
        lower : array_like
                 lower bounds for each data point
        upper : array_like
                 upper bounds for each data point
        '''
        y_pred = self.likelihood.predict(x)
        t_dist = stats.t.ppf(1 - alpha/2.0, self.likelihood.m - self.likelihood.n)

        if use_laplace:
            jac = self.likelihood.model.jac(x, self.likelihood.theta)
            res_std_error = np.zeros(len(y_pred))
            for i in range(len(y_pred)):
                row = np.array([(self.invH[j, :] * jac[i, :]).sum()
                               for j in range(self.likelihood.n)])
                res_std_error[i] = np.sqrt((jac[i, :] * row).sum())

            lower = y_pred - res_std_error * t_dist
            upper = y_pred + res_std_error * t_dist
        else:
            lower, upper = np.zeros(np.max(len(x))+1), np.zeros(np.max(len(x))+1)
            for i,xi in enumerate(x):
               theta = self.likelihood.theta.copy()
               theta[0] = y_pred[i]
               new_ll = self.likelihood.rewrite(xi, theta)
               profile_pred = ProfileT(new_ll)
               profile_pred.fit_profile(0)
               lower[i], upper[i] = profile_pred.spline_tau2theta[0](-t_dist), profile_pred.spline_tau2theta[0](t_dist)

        return lower, upper

    def create_splines(self, idx):
        '''
        Creates the profile t cubic splines
        interpolating the calculated data points
        tau and theta.
        '''
        tau = self.profiles[idx][0]
        theta = self.profiles[idx][1][idx, :]
        sorted_idx = theta.argsort()

        if len(tau) < 2:
            self.spline_tau2theta[idx] = CubicSpline([0, 1],
                                                     [-self.se[idx],
                                                      self.se[idx]])
            self.spline_theta2tau[idx] = CubicSpline([-self.se[idx],
                                                      self.se[idx]],
                                                     [0, 1])
        else:
            self.spline_tau2theta[idx] = CubicSpline(tau, theta)
            self.spline_theta2tau[idx] = CubicSpline(theta[sorted_idx],
                                                     tau[sorted_idx])

    def splines_sketches(self, tau_scale, p_idx, q_idx):
        '''
        Creates the pairwise splines
        used to calculate the pairwise plot points.

        Returns
        -------
        spline_g : array_like
                    matrix of spline functions for
                    every pair of parameters.
        '''
        spline_g = [[lambda x: x for _ in range(self.likelihood.n)] for _ in range(self.likelihood.n)]
        for p_idx in range(self.likelihood.n):
            for q_idx in range(self.likelihood.n):
                if p_idx == q_idx:
                    continue
                theta_q = self.profiles[q_idx][1][p_idx, :]
                tau_q = self.profiles[q_idx][0]

                gpq = self.spline_theta2tau[p_idx](theta_q)/tau_scale
                idx = np.abs(gpq) < 1
                gpq = np.arccos(gpq[idx])

                if len(tau_q) < 2:
                    spline_g[p_idx][q_idx] = lambda x: x
                else:
                    spline_g[p_idx][q_idx] = CubicSpline(tau_q[idx], gpq)
        return spline_g

    def approximate_contour(self, ix1, ix2, alpha):
        '''
        Approximates de profile countour plot
        for parameters ix1 and ix2 with confidence alpha.

        Parameters
        ----------
        ix1 : array_like
               index of the first parameter
        ix2 : array_like
               index of the second parameter
        alpha : float
                 significance level

        Returns
        -------
        p : array_like
             points for the first parameter
        q : array_like
             points for the second parameter
        '''
        tau_scale = np.sqrt(self.likelihood.n *
                            stats.f.ppf(1 - alpha,
                                        self.likelihood.n,
                                        self.likelihood.m - self.likelihood.n)
                            )
        spline_g = self.splines_sketches(tau_scale)

        angle_pairs = [(0, spline_g[ix2][ix1](1)),
                       (spline_g[ix1][ix2](1), 0),
                       (np.pi, spline_g[ix2][ix1](-1)),
                       (spline_g[ix1][ix2](-1), np.pi)
                       ]

        a = np.zeros(5)
        d = np.zeros(5)
        for j in range(4):
            a_j = (angle_pairs[j][0] + angle_pairs[j][1]) / 2.0
            d_j = angle_pairs[j][0] - angle_pairs[j][1]
            if d_j < 0:
                d_j = -d_j
                a_j = -a_j
            a[j] = a_j
            d[j] = d_j
        a[4] = a[0] + 2*np.pi
        d[4] = d[0]

        ixs = np.argsort(a)
        a = a[ixs]
        d = d[ixs]

        spline_ad = CubicSpline(a, d)
        n_steps = 100
        taup = np.zeros(n_steps)
        tauq = np.zeros(n_steps)
        p = np.zeros(n_steps)
        q = np.zeros(n_steps)
        for i in range(n_steps):
            a_i = i * np.pi * 2 / (n_steps - 1) - np.pi
            d_i = spline_ad(a_i)
            taup[i] = np.cos(a_i + d_i / 2) * tau_scale
            tauq[i] = np.cos(a_i - d_i / 2) * tau_scale
            p[i] = self.spline_tau2theta[ix1](taup[i])
            q[i] = self.spline_tau2theta[ix2](tauq[i])
        return p, q
