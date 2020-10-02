#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy.stats import zscore
from sklearn.tree import DecisionTreeRegressor
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


concrete_df = pd.read_csv("/kaggle/input/concrete-compressive-strength-data-set/compresive_strength_concrete.csv")


# In[ ]:


len(concrete_df)


# In[ ]:


#renaming columns
concrete_df = concrete_df.rename(columns={'Cement (component 1)(kg in a m^3 mixture)':"cement",
       'Blast Furnace Slag (component 2)(kg in a m^3 mixture)':"furnace_slag",
       'Fly Ash (component 3)(kg in a m^3 mixture)':"fly_ash",
       'Water  (component 4)(kg in a m^3 mixture)':"water",
       'Superplasticizer (component 5)(kg in a m^3 mixture)':"super_plasticizer",
       'Coarse Aggregate  (component 6)(kg in a m^3 mixture)':"coarse_agg",
       'Fine Aggregate (component 7)(kg in a m^3 mixture)':"fine_agg", 'Age (day)':"age",
       'Concrete compressive strength(MPa, megapascals) ':"compressive_strength"})


# In[ ]:


concrete_df.columns


# In[ ]:


concrete_df.describe().transpose()


# In[ ]:


concrete_df.dtypes


# In[ ]:


concrete_df.describe()


# In[ ]:


actual_strength = concrete_df.compressive_strength


# In[ ]:


# Pairplot using sns
import seaborn as sns
sns.pairplot(concrete_df , diag_kind = 'kde')


# In[ ]:


concrete_df_z = concrete_df.apply(zscore)   #to convert values to z score to remove different units


# In[ ]:


concrete_df_z = pd.DataFrame(concrete_df_z , columns  = concrete_df.columns)
concrete_df_z.describe()


# In[ ]:


y = concrete_df_z[['compressive_strength']]
X = concrete_df_z.drop(labels= "compressive_strength" , axis = 1)


# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= .30, random_state=1)


# In[ ]:


dt_model = DecisionTreeRegressor()


# In[ ]:


dt_model.fit(X_train, y_train)


# In[ ]:


print (pd.DataFrame(dt_model.feature_importances_, columns = ["Imp"], index = X_train.columns))


# In[ ]:


dt_model.score(X_test, y_test)


# In[ ]:


dt_model.score(X_train, y_train)


# In[ ]:


############################################## Iter 2... drop useless columns 


# In[ ]:


drop_cols = ['fly_ash' , 'coarse_agg' , 'fine_agg' , 'super_plasticizer' , 'compressive_strength']

X = concrete_df_z.drop(labels= drop_cols , axis = 1)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.30, random_state=1)


# In[ ]:


dt_model = DecisionTreeRegressor()


# In[ ]:


dt_model.fit(X_train, y_train)


# In[ ]:


print (pd.DataFrame(dt_model.feature_importances_, columns = ["Imp"], index = X_train.columns))


# In[ ]:


dt_model.score(X_test, y_test)


# In[ ]:


dt_model.score(X_train, y_train)


# In[ ]:


from IPython.display import Image  
#import pydotplus as pydot
from sklearn import tree
from os import system

Credit_Tree_File = open('d:\concrete_tree.dot','w')
dot_data = tree.export_graphviz(dt_model, out_file=Credit_Tree_File, feature_names = list(X_train))

Credit_Tree_File.close()


# In[ ]:


########## iteration 3 Kmeans clustering


# In[ ]:


from sklearn.cluster import KMeans


cluster_range = range( 1, 15 )
cluster_errors = []
for num_clusters in cluster_range:
  clusters = KMeans( num_clusters, n_init = 10 )
  clusters.fit(concrete_df_z)
  labels = clusters.labels_
  centroids = clusters.cluster_centers_
  cluster_errors.append( clusters.inertia_ )
clusters_df = pd.DataFrame( { "num_clusters":cluster_range, "cluster_errors": cluster_errors } )
clusters_df[0:15]


# In[ ]:


# Elbow plot
import matplotlib.pyplot as plt
plt.figure(figsize=(12,6))
plt.plot( clusters_df.num_clusters, clusters_df.cluster_errors, marker = "o" )


# In[ ]:



kmeans = KMeans(n_clusters= 6)
kmeans.fit(concrete_df_z)


# In[ ]:


labels = kmeans.labels_
counts = np.bincount(labels[labels>=0])
print(counts)


# In[ ]:


## creating a new dataframe only for labels and converting it into categorical variable
cluster_labels = pd.DataFrame(kmeans.labels_ , columns = list(['labels']))
cluster_labels['labels'] = cluster_labels['labels'].astype('category')
concrete_df_labeled = concrete_df.join(cluster_labels)

concrete_df_labeled.boxplot(by = 'labels',  layout=(3,3), figsize=(30, 20))


# In[ ]:


#No distinct clusters are visible at any number of clusters. Looks like the attributes are weak predictors except for cement. 
# The potential of getting better results by breaking data into clusters is unlikely to give the desired result


# In[ ]:


from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor
from sklearn import model_selection
from sklearn.utils import resample
from sklearn.metrics import accuracy_score
from sklearn import metrics


# In[ ]:


gbmTree = GradientBoostingRegressor(n_estimators=50)
gbmTree.fit(X_train,y_train)
print("gbmTree on training" , gbmTree.score(X_train, y_train))
print("gbmTree on test data ",gbmTree.score(X_test,y_test))


# In[ ]:


bgcl = BaggingRegressor(n_estimators=50, oob_score= True)
bgcl = bgcl.fit(X_train,y_train)
print("bgcl on train data ", bgcl.score(X_train,y_train))
print("bgcl on test data ", bgcl.score(X_test,y_test))
print("out of bag score" , bgcl.oob_score_)


# In[ ]:


rfTree = RandomForestRegressor(n_estimators=50)
rfTree.fit(X_train,y_train)
print("rfTree on train data ", rfTree.score(X_train,y_train))
print("rfTree on test data ", rfTree.score(X_test,y_test))


# In[ ]:


concrete_XY = X.join(y)


# In[ ]:


# configure bootstrap



values = concrete_XY.values

n_iterations = 1000        # Number of bootstrap samples to create
n_size = int(len(concrete_df_z) * 1)    # size of a bootstrap sample

# run bootstrap
stats = list()   # empty list that will hold the scores for each bootstrap iteration
for i in range(n_iterations):

    # prepare train and test sets
	train = resample(values, n_samples=n_size)  # Sampling with replacement 
	test = np.array([x for x in values if x.tolist() not in train.tolist()])  # picking rest of the data not considered in sample
    
    
    # fit model
	gbmTree = GradientBoostingRegressor(n_estimators=50)
	gbmTree.fit(train[:,:-1], train[:,-1])   # fit against independent variables and corresponding target values
	y_test = test[:,-1]    # Take the target column for all rows in test set

    # evaluate model
	predictions = gbmTree.predict(test[:, :-1])   # predict based on independent variables in the test data
	score = gbmTree.score(test[:, :-1] , y_test)

	stats.append(score)


# In[ ]:


# plot scores

from matplotlib import pyplot
pyplot.hist(stats)
pyplot.show()
# confidence intervals
alpha = 0.95                             # for 95% confidence 
p = ((1.0-alpha)/2.0) * 100              # tail regions on right and left .25 on each side indicated by P value (border)
lower = max(0.0, np.percentile(stats, p))  
p = (alpha+((1.0-alpha)/2.0)) * 100
upper = min(1.0, np.percentile(stats, p))
print('%.1f confidence interval %.1f%% and %.1f%%' % (alpha*100, lower*100, upper*100))


# In[ ]:


# configure bootstrap




n_iterations = 1000              # Number of bootstrap samples to create
n_size = int(len(concrete_df_z) * 1)    # size of a bootstrap sample

# run bootstrap
stats = list()   # empty list that will hold the scores for each bootstrap iteration

for i in range(n_iterations):

    # prepare train and test sets
	train = resample(values, n_samples=n_size)  # Sampling with replacement
	test = np.array([x for x in values if x.tolist() not in train.tolist()])  # picking rest of the data not considered in sample

    # fit model
	rfTree = RandomForestRegressor(n_estimators=50)  
	rfTree.fit(train[:,:-1], train[:,-1])   # fit against independent variables and corresponding target values

	rfTree.fit(train[:,:-1], train[:,-1])   # fit against independent variables and corresponding target values
	y_test = test[:,-1]    # Take the target column for all rows in test set

    # evaluate model
	predictions = rfTree.predict(test[:, :-1])   # predict based on independent variables in the test data
	score = rfTree.score(test[:, :-1] , y_test)

	stats.append(score)


# In[ ]:


# plot scores

from matplotlib import pyplot
pyplot.hist(stats)
pyplot.show()
# confidence intervals
alpha = 0.95                             # for 95% confidence 
p = ((1.0-alpha)/2.0) * 100              # tail regions on right and left .25 on each side indicated by P value (border)
lower = max(0.0, np.percentile(stats, p))  
p = (alpha+((1.0-alpha)/2.0)) * 100
upper = min(1.0, np.percentile(stats, p))
print('%.1f confidence interval %.1f%% and %.1f%%' % (alpha*100, lower*100, upper*100))


# #                              Model Tuning using hyper parameters
# 

# In[ ]:


from pprint import pprint


# In[ ]:


from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(random_state = 1)


# In[ ]:


print('Parameters currently in use:\n')
pprint(rf.get_params())


# # RandomSearchCV

# In[ ]:


import numpy as np
print(np.linspace(start = 5, stop = 10, num = 2))


# In[ ]:


from sklearn.model_selection import RandomizedSearchCV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 10 , stop = 15, num = 2)]   # returns evenly spaced 10 numbers
# Number of features to consider at every split
max_features = ['auto', 'log2']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 10, num = 2)]  # returns evenly spaced numbers can be changed to any
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

pprint(random_grid)


# In[ ]:


# Use the random grid to search for best hyperparameters

# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid,
                              n_iter = 5, scoring='neg_mean_absolute_error', 
                              cv = 3, verbose=2, random_state=42, n_jobs=-1,
                              return_train_score=True)

# Fit the random search model
rf_random.fit(X_train, y_train);


# In[ ]:


rf_random.best_params_


# In[ ]:


best_random = rf_random.best_estimator_   # best ensemble model (with optimal combination of hyperparameters)


# In[ ]:


best_random.score(X_test , y_test)


# In[ ]:


# This is the best the randomizedsearchCV could do given the range of values we submitted. It probably got stuck in 
# subobtimal combination of hyper parameters and that is why it's result is lesser than the randomforest regressor earlier


# In[ ]:


from sklearn.model_selection import GridSearchCV


# In[ ]:


param_grid = {
    'bootstrap': [True],
    'max_depth': [5,6],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4],
    'min_samples_split': [5,10],
    'n_estimators': [5,6,7]
}   


# In[ ]:


rf = RandomForestRegressor(random_state = 1)


# In[ ]:


grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = 1, verbose = 0, return_train_score=True)


# In[ ]:


# Fit the grid search to the data
grid_search.fit(X_train, y_train);


# In[ ]:


grid_search.best_params_


# In[ ]:


best_grid = grid_search.best_estimator_
best_grid.score(X_test, y_test)


# In[ ]:


# The accuracy is relatively lower as we have severly restricted the hyper parameter ranges. This was done to minimize 
# execution time. The Girdsearch has lower probability of finding the best combination than the randomsearch

