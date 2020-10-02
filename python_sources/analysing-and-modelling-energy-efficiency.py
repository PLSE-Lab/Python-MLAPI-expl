#!/usr/bin/env python
# coding: utf-8
This study looked into assessing the heating load and cooling load requirements of buildings (that is, energy efficiency) as a function of building parameters. So two outputs; 'heating_load'(Y) and 'cooling_load(Y2)
 Data Analysis, Data visualization, Feature Selection, about 10 Machine Learning models/estimators. Multilayer Perceptron(Deep Learning/Artificial Neural Network). Cross Validation was used in order to avoid overfitting.
# In[ ]:


import numpy
import pandas
from sklearn.preprocessing import MinMaxScaler

from sklearn.feature_selection import RFE
from sklearn.ensemble import ExtraTreesRegressor


 
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix

from sklearn import cross_validation
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR



from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
from sklearn.model_selection import StratifiedKFold
from keras.constraints import maxnorm
#from sklearn.metrics import explained_variance_score
#from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)


# In[ ]:


# load dataset
dataframe = pandas.read_csv(r"../input/ENB2012_data.csv")

# Assign names to Columns
dataframe.columns = ['relative_compactness', 'surface_area', 'wall_area', 'roof_area', 'overall_height', 'orientation', 'glazing_area', 'glazing_area_distribution', 'heating_load', 'cooling_load']


# In[ ]:


print("Head:", dataframe.head())


# In[ ]:


print("Statistical Description:", dataframe.describe())


# In[ ]:



print("Shape:", dataframe.shape)


# In[ ]:



print("Data Types:", dataframe.dtypes)


# In[ ]:



print("Correlation:", dataframe.corr(method='pearson'))


# 'overall_height' has the highest correlation with 'heating_load' and 'cooling_load' (which is a positive correlation), followed by 'roof_area' for both outputs
#  which is a negative correlation, 'orientation' has the least correlation 

# In[ ]:


dataset = dataframe.values


X = dataset[:,0:8]
Y = dataset[:,8]
Y2 = dataset[:,9]


# In[ ]:


#Feature Selection
model = ExtraTreesRegressor()
rfe = RFE(model, 3)
fit = rfe.fit(X, Y)

print("Number of Features: ", fit.n_features_)
print("Selected Features: ", fit.support_)
print("Feature Ranking: ", fit.ranking_) 


# 'wall_area', 'roof_area' and 'overall_height' were top 3 selected features/feature combination for predicting 'heating_load'
#  using Recursive Feature Elimination, the 2nd and 3rd selected features were atually among the attributes with the highest correlation with the  'heating_load'

# In[ ]:


#Feature Selection
model = ExtraTreesRegressor()
rfe = RFE(model, 3)
fit = rfe.fit(X, Y2)

print("Number of Features: ", fit.n_features_)
print("Selected Features: ", fit.support_)
print("Feature Ranking: ", fit.ranking_) 


# 'wall_area', 'glazing_area' and 'overall_height' were top 3 selected features/feature combination for predicting 'cooling_load'
# using Recursive Feature Elimination

# In[ ]:


plt.hist((dataframe.heating_load))


# In[ ]:


plt.hist((dataframe.cooling_load))


# In[ ]:


plt.hist((dataframe.heating_load))

plt.hist((dataframe.cooling_load))


# Most of the dataset's samples fall between 10 and 20 of both 'heating_load' and 'cooling_load' regressional output classes, with a positive skew

# In[ ]:


dataframe.plot(kind='density', subplots=True, layout=(3,4), sharex=False, sharey=False)


# Majority of the features have a positive skew except for a few, 'oreintation' and 'overall_height' have quite even distribution

# In[ ]:


axes = plt.subplots(nrows=2, ncols=3, figsize=(6, 6))dataframe.plot(kind='box', subplots=True, layout=(3,4), sharex=False, sharey=False)


# In[ ]:


scatter_matrix(dataframe)


# In[ ]:


fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(dataframe.corr(), vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = numpy.arange(0,9,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(dataframe.columns)
ax.set_yticklabels(dataframe.columns)


# 'overall_height' has the highest positive corelation as expected

# In[ ]:


num_instances = len(X)

models = []
models.append(('LiR', LinearRegression()))
models.append(('Ridge', Ridge()))
models.append(('Lasso', Lasso()))
models.append(('ElasticNet', ElasticNet()))
models.append(('Bag_Re', BaggingRegressor()))
models.append(('RandomForest', RandomForestRegressor()))
models.append(('ExtraTreesRegressor', ExtraTreesRegressor()))
models.append(('KNN', KNeighborsRegressor()))
models.append(('CART', DecisionTreeRegressor()))
models.append(('SVM', SVR()))

# Evaluations
results = []
names = []
scoring = []

for name, model in models:
    # Fit the model
    model.fit(X, Y)
    
    predictions = model.predict(X)
    
    # Evaluate the model
    kfold = cross_validation.KFold(n=num_instances, n_folds=10, random_state=seed)
    cv_results = cross_validation.cross_val_score(model, X, Y, cv=10)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


# In[ ]:


#boxplot algorithm Comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)


# 'ExtraTrees Regressor' and 'Random Forest' are the best estimators/models for 'heating_load'
#  

# In[ ]:


# Evaluations
results = []
names = []
scoring = []

for name, model in models:
    # Fit the model
    model.fit(X, Y2)
    
    predictions = model.predict(X)
    
    # Evaluate the model
    kfold = cross_validation.KFold(n=num_instances, n_folds=10, random_state=seed)
    cv_results = cross_validation.cross_val_score(model, X, Y2, cv=10)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
    
    


# In[ ]:


#boxplot algorithm Comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


# And 'Random Forest' and 'Bagging Regressor' are the best estimators/models for 'cooling_load', they can be further explored and their hyperparameters tuned

# In[ ]:


# Define 10-fold Cross Valdation Test Harness
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
cvscores = []
for train, test in kfold.split(X, Y):

    # create model
    model = Sequential()
    model.add(Dense(15, input_dim=8, init='uniform', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(8, init='uniform', activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(5, init='uniform', activation='relu'))
    model.add(Dense(1, init='uniform', activation='relu'))

    # Compile model
    model.compile(loss='mean_absolute_error', optimizer='sgd')

    # Fit the model
    model.fit(X[train], Y[train], epochs=300, batch_size=10, verbose=0)

    # Evaluate the model
    scores = model.evaluate(X[test], Y[test], verbose=0)
    print("%s: %.2f%%" % ("score", 100-scores))
cvscores.append(100-scores)
print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))


# In[ ]:


# Define 10-fold Cross Valdation Test Harness
kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
cvscores = []
for train, test in kfold.split(X, Y2):

    # create model
    model = Sequential()
    model.add(Dense(15, input_dim=8, init='uniform', activation='relu'))
    #model.add(Dropout(0.2))
    #model.add(Dense(8, init='uniform', activation='relu', kernel_constraint=maxnorm(3)))
    #model.add(Dropout(0.2))
    model.add(Dense(5, init='uniform', activation='relu'))
    model.add(Dense(1, init='uniform', activation='relu'))

    # Compile model
    model.compile(loss='mean_absolute_error', optimizer='sgd')

    # Fit the model
    model.fit(X[train], Y2[train], epochs=300, batch_size=10, verbose=0)

    # Evaluate the model
    scores = model.evaluate(X[test], Y2[test], verbose=0)
    print("%s: %.2f%%" % ("score", 100-scores))
cvscores.append(100-scores)
print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




