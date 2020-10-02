import datetime
import numpy as np
import pandas as pd

import numpy.testing as npt
from learntools.core import *

class check_score(EqualityCheckProblem):
    _var = 'score'
    _hint = "Compruebe KNeighborsClassifier, score"
    _solution = CS(
    """
score = knn.score(X_test, y_test)
score

    """)

    def check(self, actual):
        expected = 0.7337662337662337
        #assert actual.shape == expected_shape, ("Se esperaba vector con dimensiones {}, "
        #        "pero tiene dimensiones {}").format(expected_shape, actual.shape)
         
        # assertAlmostEqual(first, second, places=7, msg=None, delta=None)    
        assert abs(actual - expected < 0.0001), ("Se esperaba el valor {}, "
          "pero el valor actual es {}").format(expected, actual)
      # assert npt.assert_almost_equal(actual, expected, decimal=5), ("Se esperaban los elementos {}, "
      #     "pero los elementos son {}").format(expected, actual)          
 

class best_k_value(EqualityCheckProblem):
    _var = 'best_k_value'
    _hint = "Compruebe GridSearchCV, best_params_"
    _solution = CS(
    """
from sklearn.model_selection import GridSearchCV
knn2 = KNeighborsClassifier()
param_grid = {"n_neighbors": np.arange(3, 27, 2)}
knn_gscv = GridSearchCV(knn2, param_grid, cv=5)
knn_gscv.fit(X, y)

best_k_value = knn_gscv.best_params_
    """)

    def check(self, actual):
        expected = 13
        #assert actual.shape == expected_shape, ("Se esperaba vector con dimensiones {}, "
        #        "pero tiene dimensiones {}").format(expected_shape, actual.shape)
       
         assert actual == expected, ("Se esperaba el valor {}, "
          "pero el valor actual es {}").format(expected, actual)
        
class check_best_score(EqualityCheckProblem):
    _var = 'best_score'
    _hint = "Compruebe GridSearchCV, best_score_"
    _solution = CS(
    """
best_score = knn_gscv.best_score_
    """)

    def check(self, actual):
        expected = 0.7552083333333334
        #assert actual.shape == expected_shape, ("Se esperaba vector con dimensiones {}, "
        #        "pero tiene dimensiones {}").format(expected_shape, actual.shape)
        # assert npt.assert_almost_equal(actual, expected, decimal=5), ("Se esperaban los elementos {}, "
        #        "pero los elementos son {}").format(expected, actual)          
 
        assert abs(actual - expected < 0.0001), ("Se esperaba el valor {}, "
            "pero el valor actual es {}").format(expected, actual)
 

qvars = bind_exercises(globals(), [
    check_score,
    best_k_value,
    check_best_score
    ],
    var_format='step_{n}',
    )
__all__ = list(qvars)