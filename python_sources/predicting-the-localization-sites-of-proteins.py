#!/usr/bin/env python
# coding: utf-8

# Predicting the Cellular Localization Sites of Proteins
# Analysing and visualizing dataset and its features,feature selection, various Machine Learning Classification models/estimators including ensembles. And Deep Learning Multilayer Perceptron architecture(Artificial Neural Network) model. All trained/fitted with Cross Validation. 

# In[ ]:


import numpy
import pandas
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier


import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix


from sklearn import cross_validation
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error



from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
#from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from keras.constraints import maxnorm

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)


# In[ ]:


# load dataset
dataframe = pandas.read_csv("../input/ecoli.csv", delim_whitespace=True)

# Assign names to Columns
dataframe.columns = ['seq_name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'site']

dataframe = dataframe.drop('seq_name', axis=1)

# Encode Data
dataframe.site.replace(('cp', 'im', 'pp', 'imU', 'om', 'omL', 'imL', 'imS'),(1,2,3,4,5,6,7,8), inplace=True)


# In[ ]:


print("Head:", dataframe.head())


# In[ ]:


print("Statistical Description:", dataframe.describe())


# In[ ]:


print("Shape:", dataframe.shape)

print("Data Types:", dataframe.dtypes)


# In[ ]:


print("Correlation:", dataframe.corr(method='pearson'))


# 'mcg'(McGeoch's method for signal sequence recognition) has the highest correlation with the 'site'(Protein localization site) (which is a positive correlation), followed by 'gvh'(von Heijne's method for signal sequence recognition)
#  which is also a positive correlation, 'alm2'(score of ALOM program after excluding putative cleavable signal regions from the sequence) has the least correlation 

# In[ ]:


dataset = dataframe.values


X = dataset[:,0:7]
Y = dataset[:,7] 


# In[ ]:


#Feature Selection
model = LogisticRegression()
rfe = RFE(model, 3)
fit = rfe.fit(X, Y)

print("Number of Features: ", fit.n_features_)
print("Selected Features: ", fit.support_)
print("Feature Ranking: ", fit.ranking_) 


#  'mcg', 'gvh' and 'alm1'(score of the ALOM membrane spanning region prediction program) were top 3 selected features/feature combination for predicting 'Income'
#  using Recursive Feature Elimination, the 1st and 2n are atually the two attributes with the highest correlation with the 
#  'site' classes

# In[ ]:


plt.hist((dataframe.site))


# 
# # Most of the dataset's samples fall within the 'cp'(cytoplasm), 'im'(inner membrane without signal sequence) and 'pp'(perisplasm) output classes in that order

# In[ ]:


dataframe.plot(kind='density', subplots=True, layout=(3,4), sharex=False, sharey=False)


# Majority of the attibutes have positive skews except 'mcg' and 'aac' in that order

# In[ ]:


dataframe.plot(kind='box', subplots=True, layout=(3,4), sharex=False, sharey=False)


# In[ ]:


fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(dataframe.corr(), vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = numpy.arange(0,7,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(dataframe.columns)
ax.set_yticklabels(dataframe.columns)


# 'mcg' has the highest positive corelation as expected

# In[ ]:



num_instances = len(X)

models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
models.append(('L_SVM', LinearSVC()))
models.append(('ETC', ExtraTreesClassifier()))
models.append(('RFC', RandomForestClassifier()))

# Evaluations
results = []
names = []

for name, model in models:
    # Fit the model
    model.fit(X, Y)
    
    predictions = model.predict(X)
    
    # Evaluate the model
    kfold = cross_validation.KFold(n=num_instances, n_folds=10, random_state=seed)
    cv_results = cross_validation.cross_val_score(model, X, Y, cv=kfold, scoring='accuracy')
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


# 'Naive Bayes' and 'Linear Discriminant Analysis' are the best estimators/models for this dataset, they can be further explored and their hyperparameters tuned

# In[ ]:


# Define 10-fold Cross Valdation Test Harness
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
cvscores = []
for train, test in kfold.split(X, Y):

    # create model
    model = Sequential()
    model.add(Dense(20, input_dim=7, init='uniform', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10, init='uniform', activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(5, init='uniform', activation='relu'))
    model.add(Dense(1, init='uniform', activation='relu'))

    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    # Fit the model
    model.fit(X[train], Y[train], epochs=200, batch_size=10, verbose=0)

    # Evaluate the model
    scores = model.evaluate(X[test], Y[test], verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
cvscores.append(scores[1] * 100)
print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




