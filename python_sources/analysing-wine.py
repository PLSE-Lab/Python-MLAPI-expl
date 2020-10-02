#!/usr/bin/env python
# coding: utf-8

# # Analysing Wine
# _Started on 10 October 2017_
# 
# _Reviewed on 15 December 2017_
# * The purpose of this kernel is to explore several ML classifiers on the wine datasets.

# In[1]:


import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix

# Modelling Helpers
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, r2_score

# Modelling Algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC


# # Load data

# In[2]:


redWine = pd.read_csv('../input/winequality-red.csv')
whiteWine = pd.read_csv('../input/winequality-white.csv')


# In[3]:


redWine.describe()


# In[4]:


whiteWine.describe()


# In[5]:


#Checking for duplicates
print("Number of duplicates in red wine: "+ str(np.sum(np.array(redWine.duplicated()))))
print("Number of duplicates in white wine:  "+ str(np.sum(np.array(whiteWine.duplicated()))))


# The red wine data has 240 duplicated rows whereas white wine has 937. While duplicated rows may cause biases in analysis and inference, I think the duplicated rows here look more like several wine tasters rating the same wine similarly. Hence, it will be relevant to keep all the observations as this can add more information.

# In[6]:


# Combining the red and white wine data
wine_df = redWine.append(whiteWine)
print(wine_df.shape)


# ### Summary Statistics

# In[7]:


wine_df.describe()


# # Exploratory data analysis (EDA)

# In[8]:


# Features for the wine data
sns.set()
pd.DataFrame.hist(wine_df, figsize = [15,15], color='green')
plt.show()


# #### Range of predictor variables for wine based on the histogram above
# * Free sulfur dioxide from 0 to 120; total sulfur dioxide from 0 to 300.
# * Volatile acidity from 0 to 1.1; sulphates from 0.2 to 1.3; chlorides from 0 to 0.4.
# * The scale of the features are quite different. Hence, using algorithms that are based on distance, e.g. kNN, may focus unfairly on these features. To use such classifiers, the data would have to be normalized.
# * Wine quality rating are discrete ranging from 3 to 9, with large proportion in the category 5, 6 & 7.

# ### Correlation Heatmap
# 
# This is to generate some correlation plots of the features to see how related one feature is to the next. The Seaborn plotting package which allows us to plot heatmaps very conveniently can be used as follow.

# In[9]:


colormap = plt.cm.viridis
plt.figure(figsize=(12,12))
plt.title('Correlation of Features', y=1.05, size=15)
sns.heatmap(wine_df.astype(float).corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)


# #### Observations from the Plots
# One thing that the correlation plot tells us is that there are very few features strongly correlated with one another. From the point of view of feeding these features into our ML models,  this means that there isn't much redundant data as each feature carries with it some unique information. 
# 
# From the plot, the two most correlated features free sulfur dioxide and total sulfur dioxide. Another point is that alcohol correlates most with quality than any of the other features at 0.44.

# ### Trend of the features grouped quality
# Let's examine the plot. We will scaled the features first to have a better visualization.

# In[10]:


cols_to_scale = wine_df.columns.tolist()
cols_to_scale.remove('quality')
scaled_wine_df = wine_df
scaler = MinMaxScaler()
scaled_wine_df[cols_to_scale] = scaler.fit_transform(scaled_wine_df[cols_to_scale])
scaled_wine_df.groupby('quality').mean().plot(kind='bar', figsize=(15,5))
plt.xlim(-1,9)


# #### Observations from the Plots
# From this plot, it looks like alcohol, volatile acidity, citric acid, chloride, density are correlated to wine quality. These all sounds like "french" to me, but perhaps to the wine connoiseurs, the observations from the above plot may mean something.

# # Wine quality analysis

# #### Investigating how different chemical levels affect the quality of wine
# * View quality of wine as a continuous variable from 0 to 10. This view yields a regression problem.
# * Define good wine to be a wine having quality larger than or equal to 6. This yields a binary classification problem of separating good wine from not so good wine.
# 
# #### I will focus on the classification subtask for this notebook
# * In classification, the goal is to minimize classification error, which is (1 - accuracy). 

# In[11]:


y = wine_df['quality']
X = wine_df.drop('quality', axis=1)


# In[12]:


y1 = y > 5 # is the rating > 5? 
# plot histograms of original target variable (quality)
# and aggregated target variable

plt.figure(figsize=(20,5))
plt.subplot(121)
plt.hist(y, color='black')
plt.title('Wine Quality Distribution')
plt.xlabel('original target value')
plt.ylabel('count')

plt.subplot(122)
plt.title('Wine Quality Distribution')
plt.hist(y1, color='black')
plt.xlabel('aggregated target value')
plt.show()


# The second subplot shows the count distribution of good and bad wine.

# ## Algorithm Evaluation
# Procedure:
# * Separate out a validation dataset.
# * Set-up the test to use 10-fold cross validation.
# * Build four common classification models to predict quality from other wine features.
# * Select the best model.

# ### Split into training and test data

# In[13]:


seed = 8 # for reproducibility
X_train,X_test,y_train,y_test = train_test_split(X, y1, test_size=0.2, random_state=seed)


# The split sets aside 20% of the data as test set for evaluating the model.

# In[14]:


X_train.shape, y_train.shape, X_test.shape, y_test.shape 


# ### Evaluating common classification models

# In[15]:


# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('RF', RandomForestClassifier()))
models.append(('DT', DecisionTreeClassifier()))
models.append(('SVM_rbf', SVC()))
models.append(('SVM_linear', SVC(kernel='linear')))


# In[16]:


# Evaluate each model in turn
train_results = []
test_results = []
names = []
for name, model in models:
    cv_train_results = cross_val_score(model, X_train, y_train, 
                                       cv=10, scoring='accuracy')
    train_results.append(cv_train_results)
    clf = model.fit(X_train, y_train)
    cv_test_results = accuracy_score(y_test, clf.predict(X_test))
    test_results.append(cv_test_results)
    names.append(name)
    result = "%s: %f (%f) %f" % (name, cv_train_results.mean(), cv_train_results.std(), 
                                cv_test_results)
    print(result)


# Random Forest Classifier has the highest training & test accuracy of about 80% & 82%, followed by Decision Trees with training & test accuracy of about 76% & 78% accuracy. 
# 
# Let's create a plot of the model evaluation results and compare the spread and the mean accuracy of each model. This is a population of accuracy measures for each algorithm as each algorithm was evaluated 5 times (5-fold cross validation).

# In[17]:


fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(train_results)
ax.set_xticklabels(names)
plt.show()


# Random Forest has the highest cv score of about 80%.

# ### Feature Importances

# I will now focus on the Random Forest Classifier, tune its hyperparameters and see if the accuracy can be improved. But first, let's investigate the feature importance in this model.

# In[18]:


RF = RandomForestClassifier(random_state=seed)
RF.fit(X_train, y_train)


# In[19]:


names = list(X_train.columns.values)
importances = RF.feature_importances_
# Plot the feature importances of the forest
plt.figure(figsize=(10,5))
plt.title("Feature Importances")
y_pos = np.arange(len(names))
plt.bar(y_pos, importances, align='center')
plt.xticks(y_pos, names, rotation=90)
plt.show()


# The features which contribute more to the quality of the wine include alcohol (highest) followed by volatile acidity and density. This is similar to what we observed earlier on.

# ## Declare hyperparameters to tune
# 
# Within each decision tree, the algorithm can empirically decide where to create branches based on the metrics that it is optimising on. Therefore the actual branch locations and where to split are model parameters. However, the algorithm doesn't decide how many trees to include in the forest, what is the max_depth for the trees etc. These are hyperparameters that the user must set & tune. For this purpose, GridSearchCV can be used.

# ## Tune model using GridSearchCV

# In[20]:


clf = RandomForestClassifier()
grid_values = {'max_features':['auto','sqrt','log2'],'max_depth':[None, 10, 5, 3, 1],
              'min_samples_leaf':[1, 5, 10, 20, 50]}
clf


# In[21]:


grid_clf = GridSearchCV(clf, param_grid=grid_values, cv=10, scoring='accuracy')
grid_clf.fit(X_train, y_train) # fit and tune model


# In[22]:


grid_clf.best_params_


# #### It looks like mostly the default hyperparameters are recommended.

# In[23]:


clf = RandomForestClassifier().fit(X_train, y_train)


# ## Evaluate model accuracy on test data

# In[24]:


y_pred = clf.predict(X_test)


# ### Accuracy and confusion matrix

# In[25]:


print('Training Accuracy :: ', accuracy_score(y_train, clf.predict(X_train)))
print('Test Accuracy :: ', accuracy_score(y_test, y_pred))


# In[26]:


print(confusion_matrix(y_test, y_pred))


# # End notes
# Tips and comments are welcomed. Thank you in advance.
# 
# Rhodium Beng...... 
# 

# In[ ]:




