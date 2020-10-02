#!/usr/bin/env python
# coding: utf-8

# ## Breast Cancer Analysis - EDA and Machine Learning - Predicting diagnosis

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


df = pd.read_csv('../input/data.csv') #reading the dataset


# In[ ]:


df.head()


# In[ ]:


df.shape #Number of rows and columns


# The ID isn't useful to predict diagnosis - It'll be removed 

# In[ ]:


df.drop(['id','Unnamed: 32'],axis=1,inplace=True)


# Checking counts of diagnosis of tumor

# In[ ]:


plt.figure(figsize=(11,6))
sns.set()
plt.title('Distribution of pokemon types')
k = sns.countplot(x = 'diagnosis',data=df)
k.set_xticklabels(k.get_xticklabels(), rotation=0)
plt.show()


# In[ ]:


plt.figure(figsize=(11,6))
plt.title('Distribution of radius mean in tumor')
sns.distplot(df['radius_mean'],color='r')


# In[ ]:


plt.figure(figsize=(11,6))
plt.title('Distribution of area mean in tumor')
sns.distplot(df['area_mean'],color='g')


# In[ ]:


df.info()


# Let's convert the variables M in number 0 and B in number 1

# In[ ]:


def convert(x):
    if x == 'M':
        return 0
    if x == 'B':
        return 1


# In[ ]:


df['diagnosis'] = df['diagnosis'].apply(lambda x: convert(x))


# Let's compare predictions with no modified data, normalized data and with PCA

# In[ ]:


#Independent variables
X = df.iloc[:,1:].values


# In[ ]:


#Dependent variables
y = df.iloc[:,0].values


# In[ ]:


from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler


# In[ ]:


# Choosing attributes
pca = PCA(n_components = 3)
# Normalizing data
scaler = MinMaxScaler(feature_range = (0, 1))
Xi = scaler.fit_transform(X)
fit = pca.fit(Xi)


print("Variance: %s" % fit.explained_variance_ratio_)
print(np.sum(fit.explained_variance_ratio_))
p = []
x = []
for i in range(1,25):
    pca = PCA(n_components = i)
    fit = pca.fit(Xi)
    x.append(i)
    p.append(np.sum(fit.explained_variance_ratio_))
x_pca = pca.transform(Xi)


# In[ ]:


plt.grid(True)
plt.scatter(x,p)
plt.show()


# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


# Gradient Boosting

# In[ ]:


# Folds
num_folds = 10
seed = 7

# Number of trees
num_trees = 100

# Folds in data
kfold = KFold(num_folds, True, random_state = seed)

# model
modelo = GradientBoostingClassifier(n_estimators = num_trees, random_state = seed)# Cross Validation
resultado = cross_val_score(modelo, X, y, cv = kfold)
resultado1 = cross_val_score(modelo, Xi, y, cv = kfold)
resultado2 = cross_val_score(modelo, x_pca, y, cv = kfold)

# Printing result without modification
print("Accuracy without modification: %.3f" % (resultado.mean() * 100))
# Printing result with normalization
print("Accuracy with normalization: %.3f" % (resultado1.mean() * 100))
# Printing result with PCA
print("Accuracy with PCA: %.3f" % (resultado2.mean() * 100))


# XGBoost

# In[ ]:


# model
modelo = XGBClassifier(n_estimators = num_trees, random_state = seed)# Cross Validation
resultado = cross_val_score(modelo, X, y, cv = kfold)
resultado1 = cross_val_score(modelo, Xi, y, cv = kfold)
resultado2 = cross_val_score(modelo, x_pca, y, cv = kfold)

# Printing result without modification
print("Accuracy without modification: %.3f" % (resultado.mean() * 100))
# Printing result with normalization
print("Accuracy with normalization: %.3f" % (resultado1.mean() * 100))
# Printing result with PCA
print("Accuracy with PCA: %.3f" % (resultado2.mean() * 100))


# Random Forest

# In[ ]:


modelo = RandomForestClassifier(n_estimators = num_trees, random_state = seed)# Cross Validation
resultado = cross_val_score(modelo, X, y, cv = kfold)
resultado1 = cross_val_score(modelo, Xi, y, cv = kfold)
resultado2 = cross_val_score(modelo, x_pca, y, cv = kfold)

# Printing result without modification
print("Accuracy without modification: %.3f" % (resultado.mean() * 100))
# Printing result with normalization
print("Accuracy with normalization: %.3f" % (resultado1.mean() * 100))
# Printing result with PCA
print("Accuracy with PCA: %.3f" % (resultado2.mean() * 100))


# Neural Networks

# In[ ]:


modelo = MLPClassifier(hidden_layer_sizes=500,max_iter=1000,tol=1e-5,solver='adam')
resultado = cross_val_score(modelo, X, y, cv = kfold)
resultado1 = cross_val_score(modelo, Xi, y, cv = kfold)
resultado2 = cross_val_score(modelo, x_pca, y, cv = kfold)

# Printing result without modification
print("Accuracy without modification: %.3f" % (resultado.mean() * 100))
# Printing result with normalization
print("Accuracy with normalization: %.3f" % (resultado1.mean() * 100))
# Printing result with PCA
print("Accuracy with PCA: %.3f" % (resultado2.mean() * 100))


# AdaBoost

# In[ ]:


# model
modelo = AdaBoostClassifier(n_estimators = num_trees, random_state = seed)# Cross Validation
resultado = cross_val_score(modelo, X, y, cv = kfold)
resultado1 = cross_val_score(modelo, Xi, y, cv = kfold)
resultado2 = cross_val_score(modelo, x_pca, y, cv = kfold)

# Printing result without modification
print("Accuracy without modification: %.3f" % (resultado.mean() * 100))
# Printing result with normalization
print("Accuracy with normalization: %.3f" % (resultado1.mean() * 100))
# Printing result with PCA
print("Accuracy with PCA: %.3f" % (resultado2.mean() * 100))


# SVM

# In[ ]:


modelo = SVC()
resultado = cross_val_score(modelo, X, y, cv = kfold)
resultado1 = cross_val_score(modelo, Xi, y, cv = kfold)
resultado2 = cross_val_score(modelo, x_pca, y, cv = kfold)

# Printing result without modification
print("Accuracy without modification: %.3f" % (resultado.mean() * 100))
# Printing result with normalization
print("Accuracy with normalization: %.3f" % (resultado1.mean() * 100))
# Printing result with PCA
print("Accuracy with PCA: %.3f" % (resultado2.mean() * 100))


# Checking accuracy of algorithms, with mean, std dev and a boxplot.
# High std dev may indicate overfitting.
# High median and mean indicates a good algorithm in this situation. We must considerate time to process the algorithm.
# The first one evaluate the variables without modification. The second one evaluate the normalized variables.
# The third with PCA

# In[ ]:


# Preparing models
modelos = []
modelos.append(('LR', LogisticRegression()))
modelos.append(('XGBoost', XGBClassifier(n_estimators = num_trees, random_state = seed)))
modelos.append(('AdaBoost', AdaBoostClassifier(n_estimators = num_trees, random_state = seed)))
modelos.append(('KNN', KNeighborsClassifier()))
modelos.append(('GradientBoosting', GradientBoostingClassifier(n_estimators = num_trees, random_state = seed)))
modelos.append(('SVM', SVC()))
modelos.append(('R.Forest',RandomForestClassifier(n_estimators = num_trees, random_state = seed)))
modelos.append(('Neural',MLPClassifier(hidden_layer_sizes=500,max_iter=1000,tol=1e-5,solver='adam')))

# Each model will be evaluated by loop
resultados = []
nomes = []

for nome, modelo in modelos:
    kfold = KFold(n_splits = num_folds, random_state = seed)
    cv_results = cross_val_score(modelo, X, y, cv = kfold, scoring = 'accuracy')
    resultados.append(cv_results)
    nomes.append(nome)
    msg = "%s: %f (%f)" % (nome, cv_results.mean(), cv_results.std())
    print(msg)

# Boxplot to compare algorithms
fig = plt.figure(figsize=(13,6))
fig.suptitle('Comparing algorithms')
ax = fig.add_subplot(111)
plt.boxplot(resultados)
ax.set_xticklabels(nomes)
plt.show()


# In[ ]:


# Preparing models
modelos = []
modelos.append(('LR', LogisticRegression()))
modelos.append(('XGBoost', XGBClassifier(n_estimators = num_trees, random_state = seed)))
modelos.append(('AdaBoost', AdaBoostClassifier(n_estimators = num_trees, random_state = seed)))
modelos.append(('KNN', KNeighborsClassifier()))
modelos.append(('GradientBoosting', GradientBoostingClassifier(n_estimators = num_trees, random_state = seed)))
modelos.append(('SVM', SVC()))
modelos.append(('R.Forest',RandomForestClassifier(n_estimators = num_trees, random_state = seed)))
modelos.append(('Neural',MLPClassifier(hidden_layer_sizes=500,max_iter=1000,tol=1e-5,solver='adam')))

# Each model will be evaluated by loop
resultados = []
nomes = []

for nome, modelo in modelos:
    kfold = KFold(n_splits = num_folds, random_state = seed)
    cv_results = cross_val_score(modelo, Xi, y, cv = kfold, scoring = 'accuracy')
    resultados.append(cv_results)
    nomes.append(nome)
    msg = "%s: %f (%f)" % (nome, cv_results.mean(), cv_results.std())
    print(msg)

# Boxplot to compare algorithms
fig = plt.figure(figsize=(13,6))
fig.suptitle('Comparing algorithms')
ax = fig.add_subplot(111)
plt.boxplot(resultados)
ax.set_xticklabels(nomes)
plt.show()


# In[ ]:


# Preparing models
modelos = []
modelos.append(('LR', LogisticRegression()))
modelos.append(('XGBoost', XGBClassifier(n_estimators = num_trees, random_state = seed)))
modelos.append(('AdaBoost', AdaBoostClassifier(n_estimators = num_trees, random_state = seed)))
modelos.append(('KNN', KNeighborsClassifier()))
modelos.append(('GradientBoosting', GradientBoostingClassifier(n_estimators = num_trees, random_state = seed)))
modelos.append(('SVM', SVC()))
modelos.append(('R.Forest',RandomForestClassifier(n_estimators = num_trees, random_state = seed)))
modelos.append(('Neural',MLPClassifier(hidden_layer_sizes=500,max_iter=1000,tol=1e-5,solver='adam')))

# Each model will be evaluated by loop
resultados = []
nomes = []

for nome, modelo in modelos:
    kfold = KFold(n_splits = num_folds, random_state = seed)
    cv_results = cross_val_score(modelo, x_pca, y, cv = kfold, scoring = 'accuracy')
    resultados.append(cv_results)
    nomes.append(nome)
    msg = "%s: %f (%f)" % (nome, cv_results.mean(), cv_results.std())
    print(msg)

# Boxplot to compare algorithms
fig = plt.figure(figsize=(13,6))
fig.suptitle('Comparing algorithms')
ax = fig.add_subplot(111)
plt.boxplot(resultados)
ax.set_xticklabels(nomes)
plt.show()


# In[ ]:




