import datetime
import numpy as np
import pandas as pd

from learntools.core import *

class UpdateW(EqualityCheckProblem):
    _var = 'w'
    _hint = "Utilice gradiente descendente para actualizar los pesos"
    _solution = CS(
    """
En esta práctica no se da la solución.

    """)

    def check(self, actual):
        expected = np.array([1.58876117, 3.17458055, 11.11863105])
        #assert actual.shape == expected_shape, ("Se esperaba vector con dimensiones {}, "
        #        "pero tiene dimensiones {}").format(expected_shape, actual.shape)
        assert np.allclose(actual, expected), ("Se esperaban los elementos {}, "
                "pero los elementos son {}").format(expected, actual)          
 



qvars = bind_exercises(globals(), [
    UpdateW
    ],
    tutorial_id=119,
    var_format='step_{n}',
    )
__all__ = list(qvars)