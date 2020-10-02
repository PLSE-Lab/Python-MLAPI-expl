# Trabalho Final - SVM - Abalone
# Aluno: Bruno H. Hjort

# Data Set: http://archive.ics.uci.edu/ml/datasets/Abalone
# 1 - Sex / nominal / -- / M, F, and I (infant)
# 2 - Length / continuous / mm / Longest shell measurement
# 3 - Diameter / continuous / mm / perpendicular to length
# 4 - Height / continuous / mm / with meat in shell
# 5 - Whole weight / continuous / grams / whole abalone
# 6 - Shucked weight / continuous / grams / weight of meat
# 7 - Viscera weight / continuous / grams / gut weight (after bleeding)
# 8 - Shell weight / continuous / grams / after being dried
# 9 - Rings / integer / -- / +1.5 gives the age in years 

import numpy as np
import pandas
from numpy import genfromtxt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import uniform
import scipy
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
#import numpy_indexed as npi

# Carrega os dados crus
data_init = genfromtxt('../input/abalone.data', delimiter=',')

# Buscando valores inválidos (NaN) nas colunas de valores contínuos
for c in range(6):
	print(c+1)
	print(data_init[np.isnan(data_init[:,c+1]),:])
# Conclusão: todos valores válidos, que bom!

# Recarregando dados como string
data = genfromtxt('../input/abalone.data', delimiter=',', dtype='str')

# Separa atributos da classe
X_init = data[:,:8]  # selecionando X
y_init = data[:,8]  # selecionando Y
d = pandas.get_dummies(y_init)
y = d.values.argmax(1)

# Binariza a 1a coluna pois é categórica
X1_init = X_init[:,0]
X1 = pandas.get_dummies(X1_init).values

# Agrega colunas contínuas
X2_8 = X_init[:,1:8]
X = np.append(X1, X2_8, axis = 1)

# Faz redução de atributos por variância mínima
variancia_minima = .99 
print("X antes da redução: ", X.shape)
sel = VarianceThreshold(threshold = (variancia_minima * (1 - variancia_minima)))
X_red = sel.fit_transform(X)
print("X depois da redução: ", X_red.shape)
X = X_red

# Separa massa de treino e de testes
random_state = np.random.RandomState(0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5, random_state = random_state)

# Busca e remove elementos cuja classe tem menos que 5 instancias
#samples_mask = npi.multiplicity(y_train) >= 5
#y_train = y_train[samples_mask]
#X_train = X_train[samples_mask]

# Prepara parâmetros randômicos exponenciais
param_random = {'C': scipy.stats.expon(scale=100), 'gamma': ['auto'],'kernel': ['rbf','poly'], 'class_weight':['balanced', None]}

# Gera e treina o modelo
svc = RandomizedSearchCV(SVC(), param_distributions=param_random, cv=5, n_iter=100, random_state = random_state , n_jobs=-1, scoring='f1_micro')
svc.fit(X_train, y_train)

# Grava em arquivo JobLib
joblib.dump(svc, './abalone.joblib')
print("Arquivo 'abalone.joblib' gravado com modelo gerado.")

# Executa um teste
y_true, y_pred = y_test, svc.predict(X_test)

# Confere e imprime exatidão
print("Exatidão")
print(accuracy_score(y_true, y_pred))
print(classification_report(y_true, y_pred))
