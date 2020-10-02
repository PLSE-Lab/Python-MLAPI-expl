import datetime
import numpy as np
import pandas as pd

from learntools.core import *

class CreateX(EqualityCheckProblem):
    _var = 'x'
    _hint = "No olvidar la columna correspodiente a w0"
    _solution = CS(
    """
x = [[1., 0., 0.],
     [1., 0., 1.],
     [1., 1., 0.],
     [1., 1., 1.]]
     
    """)

    def check(self, actual):
        expected = [[1., 0., 0.],
                    [1., 0., 1.],
                    [1., 1., 0.],
                    [1., 1., 1.]]
        #assert actual.shape == expected_shape, ("Se esperaba vector con dimensiones {}, "
        #        "pero tiene dimensiones {}").format(expected_shape, actual.shape)
        assert np.array_equal(actual, expected), ("Se esperaban los elementos {}, "
                "pero los elementos son {}").format(expected, actual)   
        
        
        
class CreateY(EqualityCheckProblem):
    _var = 'y'
    _hint = "Mirar la tabla del NAND"
    _solution = CS(
    """
y =[1.,
    1.,
    1.,
    0.]
     
    """)

    def check(self, actual):
        expected = [1.,1.,1.,0.]
        #assert actual.shape == expected_shape, ("Se esperaba vector con dimensiones {}, "
        #        "pero tiene dimensiones {}").format(expected_shape, actual.shape)
        assert np.array_equal(actual, expected), ("Se esperaban los elementos {}, "
                "pero los elementos son {}").format(expected, actual)   


class CreateW(EqualityCheckProblem):
    _var = 'w'
    _hint = "Deben estar inicilizados a 0, numpy tiene el método np.zeros"
    _solution = CS(
    """
w = np.zeros(len(x[0]))

    """)

    def check(self, actual):
        expected = [0., 0., 0.]
        #assert actual.shape == expected_shape, ("Se esperaba vector con dimensiones {}, "
        #        "pero tiene dimensiones {}").format(expected_shape, actual.shape)
        assert np.array_equal(actual, expected), ("Se esperaban los elementos {}, "
                "pero los elementos son {}").format(expected, actual)   
        

        
        
class CreateF(EqualityCheckProblem):
    _var = 'f'
    _hint = "Tiene que calcular el producto esclar entre los pesos y la fila correspondiente del dataset, numpy tiene el método np.dot"
    _solution = CS(
    """
f = np.dot(w, x[n])

    """)

    def check(self, actual):
        expected = 0.0
        #assert actual.shape == expected_shape, ("Se esperaba vector con dimensiones {}, "
        #        "pero tiene dimensiones {}").format(expected_shape, actual.shape)
        assert actual == expected, ("Se esperaban los elementos {}, "
                "pero los elementos son {}").format(expected, actual)   
        
        

class CreateYHat(EqualityCheckProblem):
    _var = 'yhat'
    _hint = "Hay que comprobar si el valor de f es mayor que el umbral z, y devolver yhat correspondiente"
    _solution = CS(
    """
if f > z:
    yhat = 1.
else:
    yhat = 0.

    """)

    def check(self, actual):
        expected = 0.0
        #assert actual.shape == expected_shape, ("Se esperaba vector con dimensiones {}, "
        #        "pero tiene dimensiones {}").format(expected_shape, actual.shape)
        assert actual == expected, ("Se esperaban los elementos {}, "
                "pero los elementos son {}").format(expected, actual)  
              

            
class UpdateW(EqualityCheckProblem):
    _var = 'w'
    _hint = "Hay que utilizar la función de actualización, paso 4 de la transparencia"
    _solution = CS(
    """
w[0] = w[0] + eta*(y[n] - yhat)*x[n][0]
w[1] = w[1] + eta*(y[n] - yhat)*x[n][1]
w[2] = w[2] + eta*(y[n] - yhat)*x[n][2]

    """)

    def check(self, actual):
        expected = [0.1, 0.,  0.]
        #assert actual.shape == expected_shape, ("Se esperaba vector con dimensiones {}, "
        #        "pero tiene dimensiones {}").format(expected_shape, actual.shape)
        assert np.array_equal(actual, expected), ("Se esperaban los elementos {}, "
                "pero los elementos son {}").format(expected, actual)  
        

qvars = bind_exercises(globals(), [
    CreateX,
    CreateY,
    CreateW,
    CreateF,
    CreateYHat,
    UpdateW
    ],
    tutorial_id=119,
    var_format='step{n}',
    )
__all__ = list(qvars)