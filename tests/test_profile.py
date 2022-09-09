"""
unit tests for the profile calculations.
"""

# Author: Fabricio Olivetti de Franca <folivetti@ufabc.edu.br>
#         Gabriel Kronberger <Gabriel.Kronberger@fh-hagenberg.at>
#
# License: MIT

import unittest
import numpy as np
import sympy as sym

from profile_t.symbolic_expression import SymExpr, create_symbolic
from profile_t import ProfileT


def get_BOD_profile():
    x = np.array([1,2,3,4,5,7])
    y = np.array([8.3, 10.3, 19.0, 16.0, 15.6, 19.8])
    x_symb = sym.symbols('x')
    t_symb = sym.symbols('theta0 theta1')
    theta = [20, 0.24]
    expr = t_symb[0] * (1 - sym.exp(-x_symb * t_symb[1]))

    return ProfileT(SymExpr(expr, t_symb, x_symb, x, y), theta)


def get_Puromycin_profile():
    p = 2
    t_syms = sym.symbols(f"theta0:{p}")
    x_syms = sym.symbols("x")
    x = np.array([0.02, 0.02, 0.06, 0.06, 0.11, 0.11, 0.22, 0.22, 0.56, 0.56, 1.10, 1.10])
    y = np.array([76, 47, 97, 107, 123, 139, 159, 152, 191, 201, 207, 200])  
    expr, theta = create_symbolic('205.1*x0 / (0.08 + x0)', x, y, False)

    return ProfileT(expr, theta)


profile_BOD = get_BOD_profile()
profile_Puromycin = get_Puromycin_profile()


class TestLinearCI(unittest.TestCase):

    def test_BOD(self):
        lower, upper = profile_BOD.get_params_intervals(0.01, True)
        np.testing.assert_almost_equal(lower, [7.651135279699847302e+00, -4.039178892139433374e-01])
        np.testing.assert_almost_equal(upper, [3.063401498086683361e+01, 1.466100672591006848e+00])

    def test_Puromycin(self):
        lower, upper = profile_Puromycin.get_params_intervals(0.01, True)
        np.testing.assert_almost_equal(lower, [1.906662506855847710e+02,3.787661565126348995e-02])
        np.testing.assert_almost_equal(upper, [2.347010819815317859e+02,9.036570837188795391e-02])

class TestProfileCI(unittest.TestCase):

    def test_BOD(self):
        lower, upper = profile_BOD.get_params_intervals(0.01)
        np.testing.assert_almost_equal(lower, [1.152997919966960438e+01, -4.455328104936989719e-02])
        np.testing.assert_almost_equal(upper, [2.541387387297185398e+02, 2.596329911556340875e+01])


    def test_Puromycin(self):
        lower, upper = profile_Puromycin.get_params_intervals(0.01)
        np.testing.assert_almost_equal(lower, [1.911224325595083258e+02,4.083789544736897426e-02])
        np.testing.assert_almost_equal(upper, [2.367349501532269471e+02,9.726489241186034307e-02])

class TestPredictionLinearMethods(unittest.TestCase):


    def test_BOD(self):
        lower, upper = profile_BOD.get_prediction_intervals(0.01, np.arange(0,3,1), True)
        np.testing.assert_almost_equal(lower[:3], [3.312658549805434838e+00,7.819312071902687400e+00,1.149372954040558703e+01])
        np.testing.assert_almost_equal(upper[:3], [1.246224291221497893e+01,1.723064654062804379e+01,1.900961877172801451e+01])


    def test_Puromycin(self):
        lower, upper = profile_Puromycin.get_prediction_intervals(0.01, np.arange(0,3,1), True)
        np.testing.assert_almost_equal(lower[:3], [4.083948796953198013e+01,4.083948796953198013e+01,9.071601406303562953e+01])
        np.testing.assert_almost_equal(upper[:3], [6.029257499664119990e+01,6.029257499664119990e+01,1.149059729245400092e+02])

class TestPredictionProfileMethods(unittest.TestCase):
    def test_BOD(self):
        lower, upper = profile_BOD.get_prediction_intervals(0.01, np.arange(0,3,1))
        np.testing.assert_almost_equal(lower[:3], [2.977829942580835443e+00,6.006339705743294566e+00,8.990493653182912936e+00])
        np.testing.assert_almost_equal(upper[:3], [1.903741342855682817e+01,1.900853336407513794e+01,2.057778883485272559e+01])
    def test_Puromycin(self):
        lower, upper = profile_Puromycin.get_prediction_intervals(0.01, np.arange(0,3,1))
        np.testing.assert_almost_equal(lower[:3], [3.915104579443168120e+01,3.915104579443168120e+01,8.717961527350821882e+01])
        np.testing.assert_almost_equal(upper[:3], [6.499523935813050457e+01,6.499523935813050457e+01,1.186567869865193927e+02])

class TestProfileMethods(unittest.TestCase):

    def test_BOD(self):
        p, q = profile_BOD.approximate_contour(0, 1, 0.01)
        np.testing.assert_almost_equal(p[:4], [1.237313337327464247e+01,1.278312347756369682e+01,1.322249459766705648e+01,1.369201633363284643e+01])
        np.testing.assert_almost_equal(q[:4], [1.318692339955045492e-02,-1.170430564122878873e-02,-3.470685050183443948e-02,-5.589054913204564989e-02])

    def test_Puromycin(self):
        p, q = profile_Puromycin.approximate_contour(0, 1, 0.01)
        np.testing.assert_almost_equal(p[:4], [1.921657513586993957e+02,1.932216963803349472e+02,1.943578603786636734e+02,1.955709125098300660e+02])
        np.testing.assert_almost_equal(q[:4], [4.183244165666275688e-02,4.090116665786396044e-02,4.006254785815906855e-02,3.931610900503655637e-02])


if __name__ == '__main__':
    unittest.main()
