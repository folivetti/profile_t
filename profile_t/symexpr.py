"""
MIT License

Copyright (c) 2022 Fabricio Olivetti de Franca and Gabriel Kronberger

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
"""

import numpy as np
import sympy as sym 
from functools import partial 

def correctMtx(badMtx):
    '''
    Takes a numpy or list of numpy arrays and 
    transform any columns containing a single element
    to a constant column with the appropriate dimension.
    '''
    # find the maximum length of the bad matrix 
    max_len = np.max([len(a) 
                        if not np.isscalar(a) else 0 
                           for a in badMtx])
    return np.asarray([ np.ones(max_len)*m 
                          if np.isscalar(m) else m 
                              for m in badMtx])


class SymExpr:
    '''
    Takes a string expression, the symbols of the parameters, the symbols of 
    the variables, and the numpy arrays for x and y and creates a symbolic
    expression with support to evaluation of the residuals and jacobian.
    It also support the re-parametrization of the expression required for 
    the calculation of prediction intervals.
    '''
    def __init__(self, expr, theta_symbs, x_symbs, x ,y):

      self.x = x 
      self.y = y 
      self.m = x.shape[0]
      self.n = len(theta_symbs)
      self.multivar = False # to identify this is the univar object

      """ sympy """
      self.t     = theta_symbs
      self.z     = x_symbs
      self.expr  = expr
      self.f     = sym.lambdify((self.z, *self.t), self.expr, 'numpy') # evaluation of the expression given x and theta 
      self.trueF = lambda p: self.f(self.x, *p) # evaluation of given only theta 
      self.J     = sym.Matrix([self.expr]).jacobian(sym.Matrix(self.t)) # jacobian 
      self.jac   = lambda p: correctMtx(sym.lambdify((self.z, *self.t), self.J, 'numpy')(self.x, *p)[0]).T # evaluation of the jacobian given theta 

      # rewrite the expression such as theta0 is a function of (x,y,theta)
      w = sym.symbols("y")
      self.t0 = sym.solve(sym.Eq(self.expr, w), self.t[0])[0].subs({w:self.t[0]})

    def func(self, theta):
      ''' calculate the residuals '''
      return self.trueF(theta) - self.y

    def rewrite(self, ix):
      ''' rewrite the expression for data point ix '''
      t0 = self.t0.subs({self.z:self.x[ix]})
      return self.expr.subs({self.t[0]:t0})

    def funcr(self, ix):
      ''' residual calculation for re-parametrized function ''' 
      expr = self.rewrite(ix)
      g    = sym.lambdify((self.z, *self.t), expr, 'numpy')
      return lambda p: g(self.x, *p) - self.y 

    def jacr(self, ix):
      ''' jacobian calculation for re-parametrized function ''' 
      expr = self.rewrite(ix)
      J    = sym.Matrix([expr]).jacobian(sym.Matrix(self.t))
      jac  = lambda p: correctMtx(sym.lambdify((self.z, *self.t), J, 'numpy')(self.x, *p)[0]).T
      return jac 

class SymExprMultivar:
  ''' same as above but for multivariables '''
  def __init__(self, expr, theta_symbs, x_symbs, x ,y):

      self.x = [x[:,ix] for ix in range(x.shape[1])] # column list
      self.y = y 
      self.m = x.shape[0]
      self.n = len(theta_symbs)
      self.multivar = True

      """ sympy """
      self.t     = theta_symbs
      self.z     = x_symbs
      self.expr  = expr
      self.f     = sym.lambdify((*self.z, *self.t), self.expr, 'numpy')
      self.trueF = lambda p: self.f(*self.x, *p) #partial(self.f, *self.x)(*p)
      self.J     = sym.Matrix([self.expr]).jacobian(sym.Matrix(self.t))
      self.jac   = lambda p: correctMtx(sym.lambdify((*self.z, *self.t), self.J, 'numpy')(*self.x, *p)[0]).T

      # pre-solve for theta0 
      w = sym.symbols("y")
      self.t0 = sym.solve(sym.Eq(self.expr, w), self.t[0])[0].subs({w:self.t[0]})

  def func(self, theta):
      return self.trueF(theta) - self.y

  def rewrite(self, ix):
      t0 = self.t0.subs({self.z[iy]:self.x[iy][ix] for iy in range(len(self.z))})
      return self.expr.subs({self.t[0]:t0})

  def funcr(self, ix):
      expr = self.rewrite(ix)
      g    = sym.lambdify((*self.z, *self.t), expr, 'numpy')
      return lambda p: g(*self.x, *p) - self.y

  def jacr(self, ix):
      expr = self.rewrite(ix)
      J    = sym.Matrix([expr]).jacobian(sym.Matrix(self.t))
      jac  = lambda p: correctMtx(sym.lambdify((*self.z, *self.t), J, 'numpy')(*self.x, *p)[0]).T
      return jac 

def is_div_with_add(expr):
    ''' 
    support function to detect that we have an expression 
    following the pattern (e1 + e2)^(-1) 
    '''
    return expr.is_Pow and expr.args[1] == sym.S.NegativeOne and expr.args[0].is_Add

def get_coefs_from_expr(expr, ix, can_replace):
    '''
    Given an expression, the current parameter index, 
    and a flag whether we can replace the current value or not,
    it returns 
      - a list of the original values of the parameters
      - the list of variables 
      - the expression with all the values replaced by a parameter variable 
      - the last parameter index

    This could turn into a nice example of the uses of monads :)
    '''
    if expr.is_Number:
        num = float(expr)
        # if the value is an integer, do nothing. 
        # We are assuming that every integer is either a constant value 
        # or it was introduced by sympy.
        if not can_replace or num == round(num):
            return [], [], num, ix 
        # If it is a float number, return the value, the parameter name and increment the count by 1 
        return [float(expr)], [], sym.Symbol(f"theta{ix}"), ix+1 
    # if it is a symbol, just return it with everything else unchanged 
    if expr.is_Symbol:
        return [], [str(expr)], expr, ix

    coefs, symbs, new_args = [], [], []

    # Check if we shouldn't replace the next number:
    #   - if the current node is * AND
    #   - any argument is a number AND
    #   - any argument is + or (e1 + e2)^(-1)
    # in that case the next number argument will have a multicollinearity with the inner numeric parameters, so we just keep it fixed.
    cannot_replace = (expr.is_Mul and 
                         (any(a.is_Number for a in expr.args) and 
                              any(a.is_Add for a in expr.args) or any(is_div_with_add(a) for a in expr.args)))
    cannot_replace = False and cannot_replace or (expr.is_Pow and  any(a.is_Number for a in expr.args))

    # For every argument of the current operator 
    for arg in expr.args:
        # if it is a number AND cannot replace, call recursively with a False flag  
        if arg.is_Number and cannot_replace:
            ts, ss, new_arg, ix = get_coefs_from_expr(arg, ix, False)
        # otherwise, just call it with True 
        else:
            ts, ss, new_arg, ix = get_coefs_from_expr(arg, ix, True)

        # merge all the results and return
        coefs += ts
        symbs += ss 
        new_args.append(new_arg)
    return coefs, symbs, expr.func(*new_args), ix

def getSymbolicProfile(model, x, y):
    '''
      given a string representing the model and the data points,
      returns a symbolic expression to be used by the t-Profile class.
      It also returns the current values of theta.
    '''
    expr = sym.sympify(model) 
    theta, x_vars, expr, n_thetas = get_coefs_from_expr(expr, 0, True)
    ixs = [int(xv[1:]) for xv in np.unique(x_vars)] # vars should be 0-indexed and in the format xNUM
    x_vars = [sym.Symbol(xs) for xs in np.unique(x_vars)]
    theta_vars = [sym.Symbol(f"theta{i}") for i in range(n_thetas)]

    if len(x_vars) == 0:
        # we cannot do anything if the model is constant 
        print("ERROR: Constant model!")
        exit()
    elif len(x_vars) == 1:
          # if the model is univariate and the problem is univariate, just create a SymExpr object 
          if len(x.shape) == 1:
              print(expr, theta_vars)
              expr = SymExpr(expr, theta_vars, x_vars[0], x, y)
          # if the model is univariate and the data is not, create a SymExpr object with the correct var 
          else:
              xz = np.array([xi for xi in x[:,ixs[0]]])
              expr = SymExpr(expr, theta_vars, x_vars, xz, y)
    # if the model is multivariate, create SymExprMultivar with the appropriate vars 
    else:
        expr = SymExprMultivar(expr, theta_vars, x_vars, x[:,ixs], y)
    
    return expr, theta
