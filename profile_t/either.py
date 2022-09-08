"""
The :mod:`either` module implements the Either monad
with the same behavior as in the Haskell language.
This module makes it easier to sequentially apply a
function that may fail.
"""

# Author: Fabricio Olivetti de Franca <folivetti@ufabc.edu.br>
#         Gabriel Kronberger <Gabriel.Kronberger@fh-hagenberg.at>
#
# License: MIT

__all__ = [
    "Either",
]


class Either:
    '''
    Class Either inspired by Haskell Monad Either.
    This structure can hold one value of either the
    type left or the type right.
    This is used to make it possible for a function to
    return different types depending on the inner state.

    Also, the use of bind and seq_either make it possible to compose
    multiple Eithers and have the correct behavior.
    '''

    def __init__(self, left=None, right=None):
        '''
        Stores a value of left or right. If both
        values are passed, it stores only the left.

        Parameters
        ----------
        left : value of any type
        right : value of any type
        '''
        self.left = left
        self.right = right if left is None else None

    def map(self, f):
        '''
        Applies a function to the value of right, if it exists.

        Parameters
        ----------
        f : function
             function that receives a value of the
             same type as the field 'right'.
        '''
        if self.left is None:
            self.right = f(self.right)

    def bind(self, f):
        '''
        Applies a function that returns an Either to
        the value of right field, if it is not None.

        Parameters
        ----------
        f : function
             function that receives a value of the same
             type as 'right' and return an Either.
        '''
        if self.left is None:
            res = f(self.right)
            self.left = res.left
            self.right = res.right


def seq_either(list_of_xs):
    '''
    Given a list of Eithers, it returns
    the first left value found if there is any,
    otherwise it returns an Either with a list of
    the right values in the right field.

    Parameters
    ----------
    f : function
         function that receives a value of the same
         type as 'right' and return an Either.
    '''
    res = []
    for x in list_of_xs:
        if x.left is None:
            res.append(x.right)
        else:
            return x
    return Either(right=res)
