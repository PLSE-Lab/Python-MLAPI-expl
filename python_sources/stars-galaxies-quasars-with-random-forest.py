#!/usr/bin/env python
# coding: utf-8

# # Classification of Stars, Galaxies and Quasars with a simple random forest
# ## Dataset from the Sloan Digital Sky Survey RD14
# ---

# ## Introduction
# 
# In this notebook, a random forest will perform a task of classification (galaxy, quasar or star) based on space observations. The data come from  the **Sloan Digital Sky Survey** (release 14). For more information on this dataset, [here is the link to its "Overview" section on kaggle](https://www.kaggle.com/lucidlenn/sloan-digital-sky-survey/home).  
# You can also visit the SDSS website for even more information: https://www.sdss.org/.
# 
# So, let's get started by importing libraries:

# In[ ]:


import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
import warnings
warnings.filterwarnings("ignore", category = FutureWarning)
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# ## Data exploration
# 
# Let's import the data in a Dataframe in order to have a rough idea of what the dataset looks like.

# In[ ]:


sdss_pd = pd.read_csv("../input/Skyserver_SQL2_27_2018 6_51_39 PM.csv")
sdss_pd.head()


# In[ ]:


sdss_pd.info()


# Ok, so 17 feature columns with various types (**int64**, **float64**) and 1 target column. Fortunately, no missing values. Let's see how many examples we have for each category:

# In[ ]:


sdss_pd["class"].value_counts().sort_index()


# Almost **50%** of the examples are **GALAXY**, whereas **QUASAR** is less than **10%** of the examples. This inequality in the repartition of the labels may be a problem later.

# In[ ]:


sdss_pd.columns.values


# There are already a few features that we can guess are not useful for the classification task. Both **objid** and **specobjid** are just identifers in the original dataset. Moreover, features related to the camera (**run**, **rerun**, **camcol**, **field**) can also be dropped:

# In[ ]:


sdss_pd.drop(["objid","specobjid","run","rerun","camcol","field"], axis = 1, inplace = True)
sdss_pd.head()


# We need to transform the class column in a more convenient way than a list of string. Here, we will transform the 3 strings into 3 integers:

# In[ ]:


print("Mapping: ", dict(enumerate(["GALAXY","QSO","STAR"])))
sdss_pd["class"] = sdss_pd["class"].astype("category")
sdss_pd["class"] = sdss_pd["class"].cat.codes
print(sdss_pd["class"].value_counts().sort_index())


# Now, we can make some computations, for example with the correlation matrix.

# In[ ]:


corr_matrix = sdss_pd.corr()
corr_matrix["class"].sort_values(ascending = False)


# The **correlation coefficient** is a rough representation of how much a variable is **related** to another: it ranges from -1 to 1, extremum mean complete correlation and 0 means no relation.  
# From the output, 2 features seem very correlated: **mjd** and **plate**. With the random forest, we will see if these relations are meaningful.

# We are going to separate the target column from the others and split the dataset into a training set and a test set with a stratified method where the repartition of each label in both sets is the same as the original one to limit bias.

# In[ ]:


sdss_feat = sdss_pd.drop("class", axis = 1)
sdss_labels = sdss_pd["class"].copy()


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(sdss_feat, sdss_labels, test_size=0.2, random_state=42, stratify=sdss_labels)


# For now,  the values will not be scaled, as scaling is not mandatory for the random forest. But it can still give room for possible improvements.

# ## Training the model and evaluate it
# 
# Let's see what a random forest with the default values of scikit can achieve. It is trained on the whole training set and evaluated on the test set.

# In[ ]:


default_forest = RandomForestClassifier(random_state = 42)
default_forest.fit(X_train, y_train)
default_forest.get_params()


# We can evaluate some metrics with the test set:

# In[ ]:


print("Test accuracy for default forest:", default_forest.score(X_test, y_test))


# In[ ]:


y_pred = default_forest.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)
conf_matrix_pd = pd.DataFrame(data = conf_matrix, 
                              index = ["GALAXY","QSO","STAR"],
                              columns = ["GALAXY","QSO","STAR"])
conf_matrix_pd


# Finally, we can plot the importance of each feature:

# In[ ]:


feat_imp_pd = pd.DataFrame(data = default_forest.feature_importances_,
                          index = sdss_feat.columns,
                          columns = ["Importance"])
feat_imp_pd = feat_imp_pd.sort_values(by = 'Importance', ascending = False)
feat_imp_pd


# In[ ]:


feat_imp_pd.plot(kind = "bar", figsize = (10,5), grid = True)
plt.show()


# As predicted before, **mjd** and **plate** are two important features but the most important one is the **redshift**. The correlation matrix did not point out this relation.
# Moreover, 3 features have less than 1%: **dec**, **fiberid** and **ra**. We can consider the possibility to drop these features.
# Finally, with the relative importance of **i**, **g**, **z**, **u** and **r**, dimension reduction can also be an option.

# ## Deeper training
# 
# To improve the model, we perform at the same time a random search for hyperparemeters optimization and a 5-fold cross validation. The hyperparameters that we try to optimize here are:
# * **n_estimators**: Number of trees in the forest
# * **max_features**: Number of features the tree can use at every split
# * **max_depth**: Maximum level of a tree
# * **min_samples_split**: Minimum number of samples required to split a node
# * **min_samples_leaf**: Minimum number of samples in a leaf node
# * **bootstrap**: Method to choose the samples for training each tree  
# 
# We are going to to train 25 models with different hyperparemeters. With the 4-fold CV, the total is 100 models to train.

# In[ ]:


n_estimators = [int(x) for x in np.linspace(start = 10, stop = 1000, num = 10)]
max_features = ['auto', 'log2']
max_depth = [int(x) for x in np.linspace(10, 100, num = 10)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

rf = RandomForestClassifier()
rf_random = RandomizedSearchCV(estimator = rf, 
                               param_distributions = random_grid, 
                               scoring = 'accuracy', 
                               n_iter = 25, 
                               cv = 4, 
                               verbose = 2, 
                               random_state = 42,
                               n_jobs = -1)
rf_random.fit(X_train, y_train)


# In[ ]:


rf_random.best_score_


# In[ ]:


best_forest = rf_random.best_estimator_
best_forest


# In[ ]:


print("Test accuracy for best forest:", best_forest.score(X_test, y_test))


# In[ ]:


y_pred = best_forest.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)
conf_matrix_pd = pd.DataFrame(data = conf_matrix,
                              index = ["GALAXY","QSO","STAR"],
                              columns = ["GALAXY","QSO","STAR"])
conf_matrix_pd


# In[ ]:


feat_imp_pd = pd.DataFrame(data = best_forest.feature_importances_,
                           index = sdss_feat.columns,
                           columns = ["Importance"])
feat_imp_pd = feat_imp_pd.sort_values(by = 'Importance', ascending = False)
feat_imp_pd


# In[ ]:


feat_imp_pd.plot(kind = "bar", figsize = (10,5), grid = True)
plt.show()


# The model obtained with a random search performs better than the default one from scikit-learn: **98.875%** against **99.1%**  which is more than **0.2%** of improvement. The confusion matrix has the same structure as the previous one (e.g. a recall of 100% for the "GALAXY" class). The bar chart for the feature importance confirms what has been stated for the previous one and suggests more processing on the features.

# ## Further developments
# 
# Some ways of improvement have been stated before, let's summarize the ones which are available here.
# * **Scaling** the values of each feature. Generally, scaling helps improving the model performances. *Normalization* or *standardization* can be performed.
# *  **Dropping** more features like *dec*, *fiberid* and *ra*.
# * **Feature engineering** and **Feature extraction** e.g. *PCA* (Principal Component Analysis).
# * **Using different metrics**. Here, we evaluated the performances of our model mostly on the accuray and a bit on the confusion matrix. However, different metrics exist and can be revelant according to what you want to achieve: *recall*, *precision*, *F1 score*, *ROC AUC*.

# ## Conclusion
# 
# During this notebook, I tried to present the workflow of a machine learning problem, from data analysis to testing the model in a simple way. We explored the SDSS RD14 dataset, selected the revelant features at first sight by analyzing them. Then we created a simple random forest classifier which we tried to improve and took some insights from. Finally, we gave some paths to explore in order to enhance the model.  
# As I am not an expert and as I am still learning new concepts constantly on machine learning and trying to apply them to real world problems, I found this dataset interesting to apply some of the techniques I have learned.
