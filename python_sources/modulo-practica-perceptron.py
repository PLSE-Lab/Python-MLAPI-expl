
from numba import jit
import numpy as np
import pandas as pd 

class PerceptronPractica():

    def __init__(self, eta = 0.1):
        '''
        Parametros de entrada:
            eta: tasa de aprendizaje
            z: umbral de la funcion de activacion 
            max_iter: maximo numero de iteraciones
        '''
                
        self.eta = eta
        
        
    def __dot_product(self, v1,v2):

        out = 0
        return out
    
    def __activation_function(self, f):

        yhat = 0.
        if f >= self.z:
            yhat = 1.

        return yhat


    def predict(self, x, y):
        '''
        Parametros de entrada:
            x: dataset de entrada       
            y: etiquetas 
        '''        
        
        
        yhat_vec = np.zeros(len(y)) 
    
        return yhat_vec                  
        
    def fit(self, x, y):

        '''
        Parametros de entrada:
            x: dataset de entrada
            y: etiquetas        
        '''

        self.w = np.zeros(len(x[0]))                     
        J = []   
        
        return self.w, J        
       




   
        
        