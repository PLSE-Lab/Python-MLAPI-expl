import datetime
import numpy as np
import pandas as pd

from learntools.core import *

class CreateVectorAsRow(EqualityCheckProblem):
    _var = 'vector_row'
    _hint = "Use np.array() para crear un vector como una fila"
    _solution = CS(
    """
vector_row = np.array([1, 2, 3])
    """)

    def check(self, actual):
        expected = np.array([1, 2, 3])
        #assert actual.shape == expected_shape, ("Se esperaba vector con dimensiones {}, "
        #        "pero tiene dimensiones {}").format(expected_shape, actual.shape)
        assert np.array_equal(actual, expected), ("Se esperaban los elementos {}, "
                "pero los elementos son {}").format(expected, actual)          
 
class CreateVectorAsColumn(EqualityCheckProblem):
    _var = 'vector_column'
    _hint = "Use np.array() para crear un vector como una columna"
    _solution = CS(
"""
vector_column = np.array([[1],
                          [2],
                          [3]])    
""")

    def check(self, actual):
        expected = np.array([[1],
                             [2],
                             [3]]) 
        #assert actual.shape == expected_shape, ("Se esperaba vector con dimensiones {}, "
        #        "pero tiene dimensiones {}").format(expected_shape, actual.shape)
        assert np.array_equal(actual, expected), ("Se esperaban los elementos {}, "
                "pero los elementos son {}").format(expected, actual)          
    
class CreateMatrixAsArray(EqualityCheckProblem):
    _var = 'matrix_array'
    _hint = "Use np.array() para crear una matriz como un array"
    _solution = CS(
    """
matrix_array = np.array([[1, 2],
                         [1, 2],
                         [1, 2]])
    """)

    def check(self, actual):
        expected = np.array([[1, 2],
                             [1, 2],
                             [1, 2]])
        #assert actual.shape == expected_shape, ("Se esperaba vector con dimensiones {}, "
        #        "pero tiene dimensiones {}").format(expected_shape, actual.shape)
        assert np.array_equal(actual, expected), ("Se esperaban los elementos {}, "
                "pero los elementos son {}").format(expected, actual)    
            
            
class CreateMatrixAsObject(EqualityCheckProblem):
    _var = 'matrix_object'
    _hint = "Use np.mat() para crear una matriz como un objeto"
    _solution = CS(
"""
matrix_object = np.mat([[1, 2],
                        [1, 2],
                        [1, 2]])
""")

    def check(self, actual):
        expected = np.mat([[1, 2],
                           [1, 2],
                           [1, 2]])
        #assert actual.shape == expected_shape, ("Se esperaba vector con dimensiones {}, "
        #        "pero tiene dimensiones {}").format(expected_shape, actual.shape)    
        assert np.array_equal(actual, expected), ("Se esperaban los elementos {}, "
                "pero los elementos son {}").format(expected, actual)  
            
    
class SelectElementFromVector(EqualityCheckProblem):
    _var = 'element_vector'
    _hint = "Use vector[indice] para seleccionar elementos del vector"
    _solution = CS(
    """
element_vector = vector[2]
    """)

    def check(self, actual):
        vector = np.array([1, 2, 3, 4, 5, 6])
        expected = vector[2]
        assert actual == expected, ("Se esperaba el elemento {}, "
                "pero el elemento es {}").format(expected, actual)
 
class SelectElementFromMatrix(EqualityCheckProblem):
    _var = 'element_matrix'
    _hint = "Use matrix[fila, columna] para seleccionar elementos de la matriz"
    _solution = CS(
"""
element_matrix = matrix[0,1]
""")

    def check(self, actual):
        matrix = np.array([[1,   2,  3],
                           [4,   5,  6],
                           [7,   8,  9],
                           [10, 11, 12]])
        expected = matrix[0,1]
        assert actual == expected, ("Se esperaba el elemento {}, "
                "pero el elemento es {}").format(expected, actual)    
 

class SelectAllFromVector(EqualityCheckProblem):
    _var = 'vector_all'
    _hint = "Use vector[indice] para seleccionar elementos del vector"
    _solution = CS(
"""
vector_all = vector[:]
""")

    def check(self, actual):
        vector = np.array([1, 2, 3, 4, 5, 6])
        expected = vector[:]
        
        assert np.array_equal(actual, expected), ("Se esperaban los elementos {}, "
                "pero los elementos son {}").format(expected, actual)  
    
        
class SelectBeforeFromVector(EqualityCheckProblem):
    _var = 'vector_before'
    _hint = "Use vector[indice] para seleccionar elementos del vector"
    _solution = CS(
"""
vector_before = vector[:3]
""")

    def check(self, actual):
        vector = np.array([1, 2, 3, 4, 5, 6])
        expected = vector[:3]
        assert np.array_equal(actual, expected), ("Se esperaban los elementos {}, "
                "pero los elementos son {}").format(expected, actual)   
        
class SelectAfterFromVector(EqualityCheckProblem):
    _var = 'vector_after'
    _hint = "Use vector[indice] para seleccionar elementos del vector"
    _solution = CS(
"""
vector_after = vector[3:]
""")

    def check(self, actual):
        vector = np.array([1, 2, 3, 4, 5, 6])
        expected = vector[3:]
        assert np.array_equal(actual, expected), ("Se esperaban los elementos {}, "
                "pero los elementos son {}").format(expected, actual)   
             
        
class SelectLastFromVector(EqualityCheckProblem):
    _var = 'vector_last'
    _hint = "Use vector[indice] para seleccionar elementos del vector"
    _solution = CS(
"""
vector_last = vector[-1]
""")

    def check(self, actual):
        vector = np.array([1, 2, 3, 4, 5, 6])
        expected = vector[-1]
        assert np.array_equal(actual, expected), ("Se esperaban los elementos {}, "
                "pero los elementos son {}").format(expected, actual)   
  

class SelectColsFromMatrix(EqualityCheckProblem):
    _var = 'matrix_cols'
    _hint = "Use matrix[fila, columna] para seleccionar elementos de la matriz"
    _solution = CS(
"""
matrix_cols = matrix[:2,:]
""")

    def check(self, actual):
        matrix = np.array([[1,   2,  3],
                           [4,   5,  6],
                           [7,   8,  9],
                           [10, 11, 12]])
        expected = matrix[:2,:]
        assert np.array_equal(actual, expected), ("Se esperaban los elementos {}, "
                "pero los elementos son {}").format(expected, actual)   
        
        
class SelectRowsFromMatrix(EqualityCheckProblem):
    _var = 'matrix_rows'
    _hint = "Use matrix[fila, columna] para seleccionar elementos de la matriz"
    _solution = CS(
"""
matrix_rows = matrix[:,1:2]
""")

    def check(self, actual):
        matrix = np.array([[1,   2,  3],
                           [4,   5,  6],
                           [7,   8,  9],
                           [10, 11, 12]])
        expected = matrix[:,1:2]
        assert np.array_equal(actual, expected), ("Se esperaban los elementos {}, "
                "pero los elementos son {}").format(expected, actual)  
        
        
class MatrixShape(EqualityCheckProblem):
    _var = 'matrix_shape'
    _hint = "Use np.array() para crear una matriz como un array y utilice el atributo shape para describirla"
    _solution = CS(
    """
matrix_shape = matrix.shape
    """)

    def check(self, actual):
        matrix = np.array([[1,   2,  3],
                           [4,   5,  6],
                           [7,   8,  9],
                           [10, 11, 12]])
        expected = matrix.shape
        assert actual == expected, ("Se esperaba matriz con dimensiones {}, "
                "pero tiene dimensiones {}").format(expected, actual)
        
        
        
class MatrixSize(EqualityCheckProblem):
    _var = 'matrix_size'
    _hint = "Use np.array() para crear una matriz como un array y utilice el atributo size para describirla"
    _solution = CS(
    """
matrix_size = matrix.size
    """)

    def check(self, actual):
        matrix = np.array([[1,   2,  3],
                           [4,   5,  6],
                           [7,   8,  9],
                           [10, 11, 12]])
        expected = matrix.size
        assert actual == expected, ("Se esperaba matriz de tamaño {}, "
                "pero tiene tamaño {}").format(expected, actual)        
        
        
class MatrixNDim(EqualityCheckProblem):
    _var = 'matrix_ndim'
    _hint = "Use np.array() para crear una matriz como un array y utilice el atributo ndim para describirla"
    _solution = CS(
    """
matrix_ndim = matrix.ndim
    """)

    def check(self, actual):
        matrix = np.array([[1,   2,  3],
                           [4,   5,  6],
                           [7,   8,  9],
                           [10, 11, 12]])
        expected = matrix.ndim
        assert actual == expected, ("Se esperaba matriz con {} dimensiones, "
                "pero tiene {} dimensiones").format(expected, actual)   
        
        
class VectorizeMatrix(EqualityCheckProblem):
    _var = 'new_matrix'
    _hint = "Use np.vectorize() para aplicar la función que suma un valor a los elementos de la matriz"
    _solution = CS(
"""
add_100 = lambda i: i + 100
vectorized_add_100 = np.vectorize(add_100)
new_matrix = vectorized_add_100(matrix)
""")

    def check(self, actual):
        matrix = np.array([[1,   2,  3],
                           [4,   5,  6],
                           [7,   8,  9],
                           [10, 11, 12]])
        add_100 = lambda i: i + 100
        vectorized_add_100 = np.vectorize(add_100)
        expected = vectorized_add_100(matrix)
        assert np.array_equal(actual, expected), ("Se esperaban los elementos {}, "
                "pero los elementos son {}").format(expected, actual)  
        
        

class BroadcastingMatrix(EqualityCheckProblem):
    _var = 'matrix_100'
    _hint = "Use broadcasting para sumar un valor a los elementos de la matriz"
    _solution = CS(
"""
matrix_100 = matrix + 100
""")

    def check(self, actual):
        matrix = np.array([[1,   2,  3],
                           [4,   5,  6],
                           [7,   8,  9],
                           [10, 11, 12]])
        expected = matrix + 100
        assert np.array_equal(actual, expected), ("Se esperaban los elementos {}, "
                "pero los elementos son {}").format(expected, actual)  
        
        
        
class MaxValsMatrixCols(EqualityCheckProblem):
    _var = 'matrix_max_cols'
    _hint = "Use np.max() con el parámetro axis para aplicar la operación sobre una determinada dimensión de la matriz"
    _solution = CS(
"""
matrix_max_cols = np.max(matrix, axis=0)
""")

    def check(self, actual):
        matrix = np.array([[1,   2,  3],
                           [4,   5,  6],
                           [7,   8,  9],
                           [10, 11, 12]])
        expected = np.max(matrix, axis=0)
        assert np.array_equal(actual, expected), ("Se esperaban los elementos {}, "
                "pero los elementos son {}").format(expected, actual)  
        
        

class MinValsMatrixRows(EqualityCheckProblem):
    _var = 'matrix_min_rows'
    _hint = "Use np.min() con el parámetro axis para aplicar la operación sobre una determinada dimensión de la matriz"
    _solution = CS(
"""
matrix_min_rows = np.min(matrix, axis=1)
""")

    def check(self, actual):
        matrix = np.array([[1,   2,  3],
                           [4,   5,  6],
                           [7,   8,  9],
                           [10, 11, 12]])
        expected = np.min(matrix, axis=1)
        assert np.array_equal(actual, expected), ("Se esperaban los elementos {}, "
                "pero los elementos son {}").format(expected, actual)  
        


        
class MatrixMean(EqualityCheckProblem):
    _var = 'matrix_mean'
    _hint = "Use np.mean() para calcular la media de los valores de una matriz"
    _solution = CS(
    """
matrix_mean = np.mean(matrix)
    """)

    def check(self, actual):
        matrix = np.array([[1,   2,  3],
                           [4,   5,  6],
                           [7,   8,  9],
                           [10, 11, 12]])
        expected = np.mean(matrix)
        assert actual == expected, ("Se esperaba una media de {}, "
                "pero la calculada es {}").format(expected, actual)  
        
        
class MatrixVar(EqualityCheckProblem):
    _var = 'matrix_var'
    _hint = "Use np.var() para calcular la varianza de los valores de una matriz"
    _solution = CS(
    """
matrix_var = np.var(matrix)
    """)

    def check(self, actual):
        matrix = np.array([[1,   2,  3],
                           [4,   5,  6],
                           [7,   8,  9],
                           [10, 11, 12]])
        expected = np.var(matrix)
        assert actual == expected, ("Se esperaba una varianza de {}, "
                "pero la calculada es {}").format(expected, actual) 
        
        
class MatrixStd(EqualityCheckProblem):
    _var = 'matrix_std'
    _hint = "Use np.std() para calcular la desviación estándar de los valores de una matriz"
    _solution = CS(
    """
matrix_std = np.std(matrix)
    """)

    def check(self, actual):
        matrix = np.array([[1,   2,  3],
                           [4,   5,  6],
                           [7,   8,  9],
                           [10, 11, 12]])
        expected = np.std(matrix)
        assert actual == expected, ("Se esperaba una desviación estándar de {}, "
                "pero la calculada es {}").format(expected, actual) 
        
        
class ReshapeMatrix(EqualityCheckProblem):
    _var = 'reshaped_matrix'
    _hint = """
Use reshape() para cambiar la forma de una matriz. 
La forma de la matriz original y la nueva deben tener el mismo número de elementos (es decir, el mismo tamaño). 
    """
    _solution = CS(
    """
reshaped_matrix = matrix.reshape(2, 6)
    """)

    def check(self, actual):
        matrix = np.array([[1,   2,  3],
                           [4,   5,  6],
                           [7,   8,  9],
                           [10, 11, 12]])
        expected = matrix.reshape(2, 6)
        assert np.array_equal(actual, expected), ("Se esperaban los elementos {}, "
                "pero los elementos son {}").format(expected, actual)  
        
        
class ReshapeMatrixTo1D(EqualityCheckProblem):
    _var = 'matrix_1d'
    _hint = """
Use reshape() para cambiar la forma de una matriz y el argumento -1 para seleccionar tantos como sea necesario.  
    """
    _solution = CS(
    """
matrix_1d = matrix.reshape(1, -1)
    """)

    def check(self, actual):
        matrix = np.array([[1,   2,  3],
                           [4,   5,  6],
                           [7,   8,  9],
                           [10, 11, 12]])
        expected = matrix.reshape(1, -1)
        assert np.array_equal(actual, expected), ("Se esperaban los elementos {}, "
                "pero los elementos son {}").format(expected, actual)          
        
        
class ReshapeMatrixToVector(EqualityCheckProblem):
    _var = 'vector_12'
    _hint = """
Use reshape() para cambiar la forma de una matriz y un número entero como argumento para devolver un vector de esa longitud. 
    """
    _solution = CS(
    """
vector_12 = matrix.reshape(12)
    """)

    def check(self, actual):
        matrix = np.array([[1,   2,  3],
                           [4,   5,  6],
                           [7,   8,  9],
                           [10, 11, 12]])
        expected = matrix.reshape(12)
        assert np.array_equal(actual, expected), ("Se esperaban los elementos {}, "
                "pero los elementos son {}").format(expected, actual)  
        
        
        
class FlattenMatrix(EqualityCheckProblem):
    _var = 'flattened_matrix'
    _hint = """
Use flatten() para transformar una matriz en un array unidimensional. 
Alternativamente, podemos usar reshape() para cambiar la forma de una matriz y un número entero como argumento para devolver un vector de esa longitud. 
    """
    _solution = CS(
    """
flattened_matrix = matrix.flatten()
    """)

    def check(self, actual):
        matrix = np.array([[1,   2,  3],
                           [4,   5,  6],
                           [7,   8,  9],
                           [10, 11, 12]])
        expected = matrix.flatten()
        assert np.array_equal(actual, expected), ("Se esperaban los elementos {}, "
                "pero los elementos son {}").format(expected, actual)
        
        

        
class RandomInt(EqualityCheckProblem):
    _var = 'random_int'
    _hint = "Use np.random.randint() para generar los números aleatorios"
    _solution = CS(
    """
random_int = np.random.randint(1, 11, 3)
    """)

    def check(self, actual):
        np.random.seed(0) 
        expected = np.random.randint(1, 11, 3)
        assert np.array_equal(actual, expected), ("Se esperaban los elementos {}, "
                "pero los elementos son {}").format(expected, actual)        
     
    
class RandomNormal(EqualityCheckProblem):
    _var = 'random_normal'
    _hint = "Use np.random.normal() para generar los números aleatorios de la distribución normal "
    _solution = CS(
    """
random_normal = np.random.normal(0.0, 1.0, 3)
    """)

    def check(self, actual):
        np.random.seed(0) 
        expected = np.random.normal(0.0, 1.0, 3)
        assert np.array_equal(actual, expected), ("Se esperaban los elementos {}, "
                "pero los elementos son {}").format(expected, actual)   
        
      
    
class RandomLogistic(EqualityCheckProblem):
    _var = 'random_logistic'
    _hint = "Use np.random.logistic() para generar los números aleatorios de la distribución logística "
    _solution = CS(
    """
random_logistic = np.random.logistic(0.0, 1.0, 3)
    """)

    def check(self, actual):
        np.random.seed(0) 
        expected = np.random.logistic(0.0, 1.0, 3)
        assert np.array_equal(actual, expected), ("Se esperaban los elementos {}, "
                "pero los elementos son {}").format(expected, actual) 
        
        
class RandomUniform(EqualityCheckProblem):
    _var = 'random_uniform'
    _hint = "Use np.random.uniform() para generar los números aleatorios de la distribución uniforme "
    _solution = CS(
    """
random_uniform = np.random.uniform(1.0, 2.0, 3)
    """)

    def check(self, actual):
        np.random.seed(0) 
        expected = np.random.uniform(1.0, 2.0, 3)
        assert np.array_equal(actual, expected), ("Se esperaban los elementos {}, "
                "pero los elementos son {}").format(expected, actual) 
        
        
        
        
        
        
        
        
class LoadHomeData(EqualityCheckProblem):
    _var = 'home_data'
    _hint = "Use the `pd.read_csv` function"
    _solution = CS('home_data = pd.read_csv(iowa_file_path)')

    def check(self, df):
        assert isinstance(df, pd.DataFrame), ("`home_data` should be a DataFrame,"
                " not `{}`").format(type(df),)
        expected_shape = (1460, 81)
        assert df.shape == expected_shape, ("Expected {} rows and {} columns, but"
                " got shape {}").format(expected_shape[0], expected_shape[1], df.shape)

class HomeDescription(EqualityCheckProblem):
    _vars = ['avg_lot_size', 'newest_home_age']
    max_year_built = 2010
    min_home_age = datetime.datetime.now().year - max_year_built
    _expected = [10517, min_home_age]
    _hint = 'Run the describe command. Lot size is in the column called LotArea. Also look at YearBuilt. Remember to round lot size '
    _solution = CS(
"""# using data read from home_data.describe()
avg_lot_size = 10517
newest_home_age = 9
""")


qvars = bind_exercises(globals(), [
    CreateVectorAsRow,
    CreateVectorAsColumn,
    CreateMatrixAsArray,
    CreateMatrixAsObject,
    SelectElementFromVector,
    SelectElementFromMatrix,
    SelectAllFromVector,
    SelectBeforeFromVector,
    SelectAfterFromVector,
    SelectLastFromVector,
    SelectColsFromMatrix,
    SelectRowsFromMatrix,
    MatrixShape,
    MatrixSize,
    MatrixNDim,
    VectorizeMatrix,
    BroadcastingMatrix,
    MaxValsMatrixCols,
    MinValsMatrixRows,
    MatrixMean,
    MatrixVar,
    MatrixStd,
    ReshapeMatrix,
    ReshapeMatrixTo1D,
    ReshapeMatrixToVector,
    FlattenMatrix,
    RandomInt,
    RandomNormal,
    RandomLogistic,
    RandomUniform
    ],
    tutorial_id=118,
    var_format='step_{n}',
    )
__all__ = list(qvars)