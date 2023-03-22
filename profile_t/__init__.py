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
import numpy as np
import scipy.linalg as la

from scipy.optimize import minimize
from scipy.interpolate import CubicSpline
from scipy import stats
import warnings

from profile_t.either import Either, seq_either

warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

__all__ = [
    "ProfileT",
]

class ProfileT:
    """Class for profile t calculation"""

    def __init__(self, model, likelihood, s_err=1.0):
        '''
        Compute the profile t function from the symbolic nonlinear
        regression model.

        Parameters
        ----------
        model : SymExpr or SymExprMultivar
                representing the nonlinear regression model.
        start_theta : array_like
                      initial values of numerical parameters.
        prediction : whether it is being used for prediction intervals.

        Examples
        --------
        >>> TODO.
        OUTPUT TODO
        '''
        self.model = model
        self.likelihood = likelihood
        self.s_err = s_err

        self.m = None
        self.n = None
        self.theta = None
        self.tau_max = None
        self.is_fitted_ = False
        self.has_profile_ = False

    def fit(self, x, y, start_theta):
        # minimize the negative log-likelihood
        # ensure the current parameters values are the local optima.
        self.m = x.shape[0]
        self.n = len(start_theta)
        f, jac = self.likelihood(self.model.func, x, y)
        opt = minimize(f, start_theta, jac=jac, method='CG')
        self.theta = opt.x
        self.is_fitted_ = True

    def fit_profile(self, x, y, start_theta, ix):
        # fits only once
        if not self.is_fitted_:
            self.fit(x, y, start_theta)

        # maximum value of tau for 99% confidence
        # using F-statistics with n, m-n degrees of freedom
        self.tau_max = np.sqrt(stats.f.ppf(1 - 0.01,
                                           self.model.n,
                                           self.model.m - self.model.n)
                               )

        # the either object will contain None in left field
        # only when the process returned without error.
        either_tp = Either(left=self.theta)
        while either_tp.left is not None:
            self.theta = either_tp.left

            self.calculate_statistics()
            if self.ssr <= 1e-15:
                sys.exit("The model has a perfect fit to the data and thus there is no uncertainties.")

            either_tp = self.calculate_data_points()
        self.proft = either_tp.right
        self.create_splines()

    def calculate_data_points(self, kmax=300, step=8):
        '''
        Calculates the data points to generate the cubic
        splines for profile t function.
        Repeat that for each numeric parameter and returns
        None on the right field in case it finds a better optima.
        '''
        rng = [0] if self.prediction else range(self.model.n)
        return seq_either([self.calculate_points_param(idx, kmax, step)
                           for idx in rng])

    def calculate_points_param(self, idx, kmax, step):
        '''
        Calculates the data points for numeric parameter idx.
        '''

        def fixed_jac(theta):
            J = self.model.jac(theta)
            J[:, idx] = 0.0
            return J

        delta = -self.se[idx]/step

        # First we generate the data points starting from -delta,
        # after that we generate the points starting from delta.
        # Finally, we concatenate the data points together
        # with the trivial point where tau = 0.

        res = self.points_from_delta(delta, idx, fixed_jac, kmax)
        if res.left is None:
            tau_1, m_1, deltas_1 = res.right
            res = self.points_from_delta(-delta, idx, fixed_jac, kmax)
            if res.left is None:
                tau_2, m_2, deltas_2 = res.right
            else:
                return res
        else:
            return res

        # inserts the (tau, theta) = (0, theta_opt) point
        tau = np.array(tau_1 + [0] + tau_2)
        ix_tau = tau.argsort()

        return Either(right=(tau[ix_tau],
                             np.array(m_1 + [self.theta] + m_2)[ix_tau, :].T,
                             np.array(deltas_1 + [0] + deltas_2)[ix_tau]))

    def points_from_delta(self, delta, idx, fixed_jac, kmax):
        '''
        Calculates the data points for an initial delta.
        '''
        taus, thetas, deltas = [], [], []
        inv_slope = 1.0
        t = 0.0
        theta_cond = self.theta.copy()

        # we disturb the idx-th value of theta by delta*t
        # and re-optimize the other parameters.
        # This will create a series of (tau, theta) values
        # used to calculate the CI.
        for _ in range(kmax):
            t += inv_slope
            theta_cond[idx] = self.theta[idx] + delta*t
            opt = least_squares(self.model.func, theta_cond, fixed_jac,
                                method='lm', x_scale=self.se)
            residue = self.model.y - self.model.trueF(opt.x)
            ssr_cond = np.square(residue).sum()

            if (ssr_cond-self.ssr) < 0:
                print("Found a better local model, restarting...")
                return Either(left=opt.x)

            s = self.s if self.correction == 0.0 else self.s # * np.sqrt(self.model.m - self.model.n + 1) 
            tau_i = np.sign(delta) * np.sqrt(ssr_cond - self.ssr) / s

            inv_slope = tau_i * self.s * self.s
            inv_slope = inv_slope/(self.se[idx] *
                                   residue.dot(self.model.jac(opt.x)[:, idx]))
            inv_slope = min(4.0, max(np.abs(inv_slope), 1.0/16))

            taus.append(tau_i)
            thetas.append(opt.x.copy())
            deltas.append((self.theta[idx] - opt.x[idx])/self.se[idx])

            if np.abs(tau_i) > self.tau_max: #+ (self.s if self.correction else 0.0):
                break

        return Either(right=(taus, thetas, deltas))

    def calculate_statistics(self):
        """
        Calculates the following statistics about the model:

            * se : standard error
            * res_std_error : residual standard error
            * ssr : sum of squares residuals
            * s : residual standard error
            * corr : correlation between coefficients
        """
        n = self.model.n
        m = self.model.m

        y_hat = self.model.trueF(self.theta)
        self.ssr = np.square(y_hat - self.model.y).sum()

        J = self.model.jac(self.theta)

        # we must use economic mode so R will be a square matrix
        _, R = la.qr(J, mode='economic')
        self.R = la.inv(R)

        self.s = np.sqrt(self.ssr/(m-n))
        self.se = np.sqrt(np.square(np.triu(self.R)).sum(axis=1))
        L = (np.triu(self.R).T/self.se).T
        self.se = self.s*self.se

        self.corr = L@L.T
        j_r = J@R
        self.res_std_error = self.s * np.sqrt(np.square(j_r).sum(axis=1))

    def calculate_statistics_newpoints(self, x):
        """
        Calculates the following statistics about the model:

            * res_std_error : residual standard error
        """
        n = len(x)
        m = self.model.m
        x = [x[:, i] for i in range(x.shape[1])] if self.model.multivar else x

        J = self.model.jac_new((x, self.theta))

        # we must use economic mode so R will be a square matrix
        #_, R = la.qr(J, mode='economic')
        #R = la.inv(R)

        j_r = J @ self.R
        return self.s * np.sqrt(np.square(j_r).sum(axis=1))

    def report_parameters_ci(self, alpha, use_linear=False):
        '''
        Prints out the model ssr and s^2 and the confidence interval.

        Parameters
        ----------
        alpha : float
                significance level
        use_linear : bool (default False)
                     whether to use linear instead of profile t
        '''

        print(f"theta: {self.theta}")
        print(f"SSR {self.ssr} s^2 {self.s*self.s}")

        lower, upper = self.get_params_intervals(alpha, use_linear)
        print("theta\tEstimate\tStd. Error.\tLower\t\tUpper\t\t"
              "Corr. Matrix")
        for i in range(self.model.n):
            print(f"{i}\t{self.theta[i]:e}\t{self.se[i]:e}\t"
                  f"{lower[i]:e}\t{upper[i]:e}\t{np.round(self.corr[i,:],2)}")

    def report_prediction_interval(self, xs, alpha, use_linear=False, newpoint=False):
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
        lower, upper = self.get_prediction_intervals(alpha, xs, use_linear, newpoint)
        ypred = np.array([self.model.trueF_new((x, self.theta)) for x in xs])
        print("y_pred\t\tlow\t\thigh")
        for i, x in enumerate(xs):
            print(f"{ypred[i]:e}\t"
                  f"{lower[i]:e}\t{upper[i]:e}")

    def get_params_intervals(self, alpha, use_linear=False):
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
        f_dist = np.sqrt(stats.f.ppf(1 - alpha, self.model.n, self.model.m - self.model.n))
        t_dist = stats.t.ppf(1 - alpha/2.0, self.model.m - self.model.n)
        if use_linear:
            lower = self.theta - self.se * t_dist
            upper = self.theta + self.se * t_dist
        else:
            lower = np.array(list(self.spline_tau2theta[i](-f_dist) for i in range(self.model.n)))
            upper = np.array(list(self.spline_tau2theta[i](f_dist) for i in range(self.model.n)))
        return lower, upper

    def get_prediction_intervals(self, alpha, xs, use_linear=False, newpoint=False):
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
        y_pred = np.array([self.model.trueF_new((x, self.theta)) for x in xs])
        f_dist = np.sqrt(stats.f.ppf(1 - alpha/2.0, self.model.n, self.model.m - self.model.n))
        t_dist = stats.t.ppf(1 - alpha/2.0, self.model.m - self.model.n)
        correction = self.s if newpoint else 0

        # if it is a new data point it must be (self.res_std_error + self.s)
        if use_linear:
            res_std_error = self.calculate_statistics_newpoints(xs)
            lower = y_pred - (res_std_error + correction) * t_dist #np.sqrt(f_dist * self.model.n)
            upper = y_pred + (res_std_error + correction) * t_dist #np.sqrt(f_dist * self.model.n)
        else:
            lower, upper = np.zeros(np.max(len(xs))+1), np.zeros(np.max(len(xs))+1)
            for i,x in enumerate(xs):
               theta = self.theta.copy()
               theta[0] = y_pred[i]
               profile_pred = ProfileT(self.model.rewrite(x), theta, True, 0.0)
               lower[i], upper[i] = profile_pred.spline_tau2theta[0](-t_dist) - correction*t_dist, profile_pred.spline_tau2theta[0](t_dist) + correction*t_dist

        return lower, upper

    def create_splines(self):
        '''
        Creates the profile t cubic splines
        interpolating the calculated data points
        tau and theta.
        '''
        self.spline_tau2theta = []
        self.spline_theta2tau = []
        n = 1 if self.prediction else self.model.n
        for idx in range(n):
            tau = self.proft[idx][0]
            theta = self.proft[idx][1][idx, :]
            sorted_idx = theta.argsort()

            if len(tau) < 2:
                self.spline_tau2theta.append(CubicSpline([0, 1],
                                                         [-self.se[idx],
                                                          self.se[idx]]))
                self.spline_theta2tau.append(CubicSpline([-self.se[idx],
                                                          self.se[idx]],
                                                         [0, 1]))
            else:
                self.spline_tau2theta.append(CubicSpline(tau, theta))
                self.spline_theta2tau.append(CubicSpline(theta[sorted_idx],
                                                         tau[sorted_idx]))

    def splines_sketches(self, tau_scale):
        '''
        Creates the pairwise splines
        used to calculate the pairwise plot points.

        Returns
        -------
        spline_g : array_like
                    matrix of spline functions for
                    every pair of parameters.
        '''
        n = 1 if self.prediction else self.model.n
        spline_g = [[lambda x: x for _ in range(n)] for _ in range(n)]
        for p_idx in range(n):
            for q_idx in range(n):
                if p_idx == q_idx:
                    continue
                theta_q = self.proft[q_idx][1][p_idx, :]
                tau_q = self.proft[q_idx][0]

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
        tau_scale = np.sqrt(self.model.n *
                            stats.f.ppf(1 - alpha,
                                        self.model.n,
                                        self.model.m - self.model.n)
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
