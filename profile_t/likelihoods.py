import numpy as np
from scipy.optimize import minimize

__all__ = [
    "GaussianLikelihood"
]

class GaussianLikelihood:
    def __init__(self, model, theta0, x, y, s_noise=None):
        self.model = model
        self.s_noise = 1.0 if s_noise is None else s_noise
        self.theta = theta0
        self.x = x
        self.y = y
        self.m = len(y)
        self.n = len(theta0)

        self.has_s_noise_ = False if s_noise is None else True
        self.fit()

    def rewrite(self, x, theta):
        new_model = self.model.rewrite(x)
        return GaussianLikelihood(new_model, theta, self.x, self.y)

    def fit(self):
        def fj(theta):
            y_pred = self.model.f(self.x, theta)
            residue = self.y - y_pred
            jac = self.model.jac(self.x, theta)
            negLL = 0.5 * np.square(residue).sum() / self.s_noise
            grad = (-residue @ jac) / self.s_noise
            return negLL, grad

        opt = minimize(fj, self.theta, jac=True, method='CG')
        self.theta = opt.x
        if not self.has_s_noise_:
            self.s_noise = np.square(self.y - self.model.f(self.x, self.theta)).sum() / (self.m - self.n)

    def predict(self, x):
        return self.model.f(self.x, self.theta)

    def hessian(self):
        y_pred = self.model.f(self.x, self.theta)
        res = self.y - y_pred
        jac = self.model.jac(self.x, self.theta)
        hess = sum(res[i]*self.model.hess(self.x[i], self.theta) for i in range(self.m))
        for k in range(self.m):
            for i in range(self.n):
                for j in range(self.n):
                    hess[i, j] += jac[k, i] * jac[k, j]
        return hess / self.s_noise

    def negLogLikelihood(self, theta):
        y_pred = self.model.f(self.x, theta)
        residue = self.y - y_pred
        return 0.5 * np.square(residue).sum() / self.s_noise

    def negLogLikelihoodGrad(self, theta):
        y_pred = self.model.f(self.x, theta)
        residue = self.y - y_pred
        jac = self.model.jac(self.x, theta)
        return (-residue @ jac) / self.s_noise

    def minNegLogLikelihood(self):
        return self.negLogLikelihood(self.theta)

    def refit(self, t0, idx):
        def fj(theta):
            y_pred = self.model.f(self.x, theta)
            residue = self.y - y_pred
            jac = self.model.jac(self.x, theta)
            negLL = 0.5 * np.square(residue).sum() / self.s_noise
            grad = (-residue @ jac) / self.s_noise
            grad[idx] = 0.0
            return negLL, grad

        theta0 = self.theta.copy()
        theta0[idx] = t0
        opt = minimize(fj, theta0, jac=True, method='CG')
        return opt.x
