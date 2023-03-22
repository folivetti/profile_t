"""
The :mod:`symbolic_expression` module implements the support
to handle nonlinear regression models described as strings.
"""

# Author: Fabricio Olivetti de Franca <folivetti@ufabc.edu.br>
#         Gabriel Kronberger <Gabriel.Kronberger@fh-hagenberg.at>
#
# License: MIT

import sys
import numpy as np
import sympy as sym

__all__ = [
    "SymExpr",
    "SymExprMultivar",
    "create_symbolic",
]


class SymExpr:
    """Class with support to symbolic manipulation of a
    regression model model described as a string"""

    def __init__(self, expr, theta_symbs, x_symbs, x, y):
        '''
        Creates an object containing support methods to
        evaluate and handle symbolic regression models.
        This class supports only univariate models.

        Parameters
        ----------
        expr : str
                string with the regression model
        theta_symbs : array_like
                       list of sympy symbols with the names of
                       numerical parameters
        x_symbs : array_like
                   list of sympy symbols with the names of the variables
        x : array_like
             data points of the training set
        y : array_like
             target points of the training set
        '''

        self.x = x
        self.y = y
        self.m = x.shape[0]
        self.n = len(theta_symbs)
        self.multivar = False

        self.t = theta_symbs
        self.z = x_symbs
        self.expr = expr
        self.f = sym.lambdify((self.z, *self.t), self.expr, 'numpy')
        self.trueF = lambda p: self.f(self.x, *p)
        self.trueF_new = lambda xp: self.f(xp[0], *xp[1])
        self.J = sym.Matrix([self.expr]).jacobian(sym.Matrix(self.t))
        self.jac = lambda p: fix_matrix(sym.lambdify((self.z, *self.t),
                                                     self.J,
                                                     'numpy')(self.x, *p)[0]).T
        self.jac_new = lambda xp: fix_matrix(sym.lambdify((self.z, *self.t),
                                                     self.J,
                                                     'numpy')(xp[0], *xp[1])[0]).T
        w = sym.symbols("y")
        self.t0 = (sym.solve(sym.Eq(self.expr, w), self.t[0])[0]
                   .subs({w: self.t[0]}))

    def func(self, theta):
        '''
        Calculate the residuals of the regression
        model given an array of parameters values.

        Parameters
        ----------
        theta : array_like
                 parameters values

        Returns
        -------
        residuals of the regression model
        '''
        return self.trueF(theta) - self.y

    def rewrite(self, x):
        '''
        Rewrites the expression replacing the first
        parameter with the evaluation at data point i.

        Parameters
        ----------
        i : int
             index of data point of reference.

        Returns
        -------
        rewritten expression
        '''
        t0 = self.t0.subs({self.z: x})
        return SymExpr(self.expr.subs({self.t[0]: t0}), self.t, self.z, self.x, self.y)

class SymExprMultivar:
    """Class with support to symbolic manipulation of a
    regression model model described as a string"""

    def __init__(self, expr, theta_symbs, x_symbs, x, y):
        '''
        Creates an object containing support methods to
        evaluate and handle symbolic regression models.
        This class supports multivariate models.

        Parameters
        ----------
        expr : str
                string with the regression model
        theta_symbs : array_like
                       list of sympy symbols with the names of
                       numerical parameters
        x_symbs : array_like
                   list of sympy symbols with the names of the variables
        x : array_like
             data points of the training set
        y : array_like
             target points of the training set
        '''
        self.npx = x
        self.x = [x[:, i] for i in range(x.shape[1])]
        self.y = y
        self.m = x.shape[0]
        self.n = len(theta_symbs)
        self.multivar = True

        self.t = theta_symbs
        self.z = x_symbs
        self.expr = expr
        self.f = sym.lambdify((*self.z, *self.t), self.expr, 'numpy')
        self.trueF = lambda p: self.f(*self.x, *p)
        self.trueF_new = lambda xp: self.f(*xp[0], *xp[1])
        self.J = sym.Matrix([self.expr]).jacobian(sym.Matrix(self.t))
        self.jac = lambda p: fix_matrix(
            sym.lambdify((*self.z, *self.t),
                         self.J,
                         'numpy')(*self.x, *p)[0]).T
        self.jac_new = lambda xp: fix_matrix(sym.lambdify((*self.z, *self.t),
                                                     self.J,
                                                     'numpy')(*xp[0], *xp[1])[0]).T
        w = sym.symbols("y")
        self.t0 = (sym.solve(sym.Eq(self.expr, w), self.t[0])[0]
                   .subs({w: self.t[0]}))

    def func(self, theta):
        '''
        Calculate the residuals of the regression
        model given an array of parameters values.

        Parameters
        ----------
        theta : array_like
                 parameters values

        Returns
        -------
        residuals of the regression model
        '''
        return self.trueF(theta) - self.y

    def rewrite(self, x):
        '''
        Rewrites the expression replacing the first
        parameter with the evaluation at data point i.

        Parameters
        ----------
        i : int
             index of data point of reference.

        Returns
        -------
        rewritten expression
        '''
        t0 = self.t0.subs({self.z[j]: x[j]
                           for j in range(len(self.z))})
        return SymExprMultivar(self.expr.subs({self.t[0]: t0}), self.t, self.z, self.npx, self.y)

def is_div_with_add(expr):
    '''
    Returns whether the expression
    follows the pattern (e1 + e2)^(-1).

    Parameters
    ----------
    expr : sympy expression

    Returns
    -------
    True if expr follows the pattern, False otherwise
    '''
    return (expr.is_Pow
            and expr.args[1] == sym.S.NegativeOne
            and expr.args[0].is_Add)


def get_coefs_from_expr(expr, i, can_replace, apply_simpl):
    '''
    Given an expression, an index i,
    and a flag if we can replace it or not,
    it returns the list of parameters values,
    the list of variable names, the expression
    with numerical values replaced with var names,
    the last index value.

    Parameters
    ----------
    expr : sympy expression
            expression to rewrite
    i : int
         next index sequence
    can_replace : bool
                   if we can safely replace the numerical value
    apply_simpl : bool
                   whether to apply simplifications or not (default True)

    Returns
    -------
    parameters values, parameters names, rewritten expression,
    and the next sequential index.
    '''
    if expr.is_Number:
        num = float(expr)
        # if the value is an integer, do nothing.
        # We are assuming that every integer is either a constant value
        # or it was introduced by sympy.
        if not can_replace or num == round(num):
            return [], [], num, i

        # If it is a float number, return the value,
        # the parameter name and increment the count by 1
        return [float(expr)], [], sym.Symbol(f"theta{i}"), i+1

    # if it is a symbol, just return it with everything else unchanged
    if expr.is_Symbol:
        return [], [str(expr)], expr, i

    coefs, symbs, new_args = [], [], []

    # Check if we shouldn't replace the next number:
    #   - if the current node is * AND
    #   - any argument is a number AND
    #   - any argument is + or (e1 + e2)^(-1)
    # in that case the next number argument will have a multicollinearity with
    # the inner numeric parameters, so we just keep it fixed.
    cannot_replace = (expr.is_Mul and
                      (any(a.is_Number for a in expr.args) and
                       any(a.is_Add for a in expr.args) or
                       any(is_div_with_add(a) for a in expr.args)))
    cannot_replace = apply_simpl and cannot_replace

    # For every argument of the current operator
    for arg in expr.args:
        # if it is a number AND cannot replace,
        # call recursively with a False flag
        if arg.is_Number and cannot_replace:
            ts, ss, new_arg, i = get_coefs_from_expr(arg, i, False, apply_simpl)
        # otherwise, just call it with True
        else:
            ts, ss, new_arg, i = get_coefs_from_expr(arg, i, True, apply_simpl)

        # merge all the results and return
        coefs += ts
        symbs += ss
        new_args.append(new_arg)
    return coefs, symbs, expr.func(*new_args), i


def create_symbolic(model, x, y, apply_simpl=True):
    '''
    Gets an string representing the model and
    the data points and returns either SymExpr or
    SymExprMultivar, returning an object compatible
    with ProfileT class.

    The input variables in the expression should
    follow the format xN with N representing the
    index of the variable in the data set (0-indexed).
    E.g., 'x0 + x1'

    Parameters
    ----------
    model : str
             string of the model
    x : array_like
         input variables
    y : array_like
         target values
    apply_simpl : bool
                   whether to apply simplifications or not (default True)

    Returns
    -------
    SymExpr or SymExprMultivar object to be used
    with ProfileT
    '''
    expr = sym.sympify(model)
    theta, x_vars, expr, n_thetas = get_coefs_from_expr(expr, 0, True, apply_simpl)

    ixs = [int(xv[1:]) for xv in np.unique(x_vars)]
    x_vars = [sym.Symbol(xs) for xs in np.unique(x_vars)]
    theta_vars = [sym.Symbol(f"theta{i}") for i in range(n_thetas)]

    if len(x_vars) == 0:
        sys.exit("ERROR: Constant model!")
    elif len(x_vars) == 1:
        # if the model is univariate and the problem is univariate,
        # just create a SymExpr object
        if len(x.shape) == 1:
            expr = SymExpr(expr, theta_vars, x_vars[0], x, y)
        # if the model is univariate and the data is not, create a SymExpr
        # object and fix the data
        else:
            x_z = np.array(list(xi for xi in x[:, ixs[0]]))
            expr = SymExpr(expr, theta_vars, x_vars, x_z, y)
    else:
        # if the model is multivariate, create
        # SymExprMultivar with the appropriate vars
        expr = SymExprMultivar(expr, theta_vars, x_vars, x[:, ixs], y)

    return expr, theta


def fix_matrix(bad_matrix):
    '''
    Takes a numpy or list of numpy arrays and
    transform any columns containing a single element
    to a constant column with the appropriate dimension.

    Parameters
    ----------
    bad_matrix : array_like
                  ill defined array

    Returns
    -------
    Array representing a matrix with correct dimensions
    '''
    max_len = np.max([len(a)
                      if not np.isscalar(a) else 0
                      for a in bad_matrix]
                     )
    return np.asarray([np.ones(max_len)*m
                       if np.isscalar(m) else m
                       for m in bad_matrix]
                      )
