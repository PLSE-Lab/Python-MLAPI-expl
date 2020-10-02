
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



class SimplePerceptron():

    def __init__(self, eta):
        """
        :param eta: tasa de aprendizaje
        """
        self.eta = eta

    def zeta(self, X):
        """
        Calcula el producto de las entradas por sus pesos
        :param X: datos de entrenamiento con las caracteristicas. 
        """
        zeta = np.dot(1, self.weights[0]) + np.dot(X, self.weights[1:])
        return zeta

    def predict(self, X):
        """
        Calcula la salida de la neurona teniendo en cuenta la función de activación
        :param X: datos con los que predecir la salida de la neurona. 
        :return: salida de la neurona
        """
        output = np.where(self.zeta(X) >= 0.0, 1, 0)
        return output

    def fit(self, X, y):
        #Ponemos a cero los pesos
        self.weights = [0] * (X.shape[1] + 1)
        
        self.errors = []
        self.iteraciones = 0
        
        while True:
            errors = 0
            
            for features, expected in zip(X,y):
                delta_weight = self.eta * (expected - self.predict(features))
                self.weights[1:] += delta_weight * features
                self.weights[0] += delta_weight * 1
                errors += int(delta_weight != 0.0)
                
            self.errors.append(errors)
            self.iteraciones += 1
            
            if errors == 0:
                break