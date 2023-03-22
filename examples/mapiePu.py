import numpy as np
from profile_t.symbolic_expression import create_symbolic
from scipy.optimize import least_squares
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from mapie.regression import MapieRegressor

x = np.array([0.02, 0.02, 0.06, 0.06, 0.11, 0.11, 0.22, 0.22, 0.56, 0.56, 1.10, 1.10])
y = np.array([76, 47, 97, 107, 123, 139, 159, 152, 191, 201, 207, 200])
model_str = '205.1*x0 / (0.08 + x0^2)'
#model_str = '205.1*x0'

class Puro(BaseEstimator, RegressorMixin):
    def __init__(self, model_str):
        self.model_str = model_str
        self.is_fitted_ = False
    def fit(self, x, y):
        self.model, start_theta = create_symbolic(self.model_str, x, y, False)
        opt = least_squares(self.model.func, start_theta, self.model.jac, method='lm')
        self.theta = opt.x
        self.is_fitted_ = True
        return self

    def predict(self, x):
        check_is_fitted(self)
        return self.model.trueF_new((x, self.theta))

alpha = [0.1]
mapie = MapieRegressor(Puro(model_str))
mapie.fit(x,y)
y_pred, y_pis = mapie.predict(np.array([0.04, 0.15, 0.6]), alpha=alpha)

for yi, pii in zip(y_pred, y_pis):
    print(yi, pii)
