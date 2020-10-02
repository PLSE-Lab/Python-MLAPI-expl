#!/usr/bin/env python
# coding: utf-8

# # Cancer Diagnostics
# 
# Using [set of breast cancer data](http://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Original%29), create a model to predict breast cancer.
# Also, what traits are most indicative of whether or not an individual will be diagnosed?

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# ## Import data

# In[ ]:


# import data
raw_data = pd.read_csv('../input/data.csv')

df_raw = pd.DataFrame(raw_data)

df_raw.head()


# Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image. n the 3-dimensional space is that described in: [K. P. Bennett and O. L. Mangasarian: "Robust Linear Programming Discrimination of Two Linearly Inseparable Sets", Optimization Methods and Software 1, 1992, 23-34].
# 
# This database is also available through the UW CS ftp server: ftp ftp.cs.wisc.edu cd math-prog/cpo-dataset/machine-learn/WDBC/
# 
# Also can be found on UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29
# 
# Attribute Information:
# 
# 1) ID number 2) Diagnosis (M = malignant, B = benign) 3-32)
# 
# Ten real-valued features are computed for each cell nucleus:
# 
# a) radius (mean of distances from center to points on the perimeter) b) texture (standard deviation of gray-scale values) c) perimeter d) area e) smoothness (local variation in radius lengths) f) compactness (perimeter^2 / area - 1.0) g) concavity (severity of concave portions of the contour) h) concave points (number of concave portions of the contour) i) symmetry j) fractal dimension ("coastline approximation" - 1)
# 
# The mean, standard error and "worst" or largest (mean of the three largest values) of these features were computed for each image, resulting in 30 features. For instance, field 3 is Mean Radius, field 13 is Radius SE, field 23 is Worst Radius.
# 
# All feature values are recoded with four significant digits.
# 
# Missing attribute values: none
# 
# Class distribution: 357 benign, 212 malignant

# ## EDA

# In[ ]:


df_raw.info()


# In[ ]:


df_raw.describe()


# ## Data Cleaning

# In[ ]:


# drop 'Unnamed: 32' column as it only contains NaNs
display(df_raw.shape)
df_raw = df_raw.dropna(axis=1)
df_raw.shape


# In[ ]:


# view target values
df_raw['diagnosis'].value_counts()


# In[ ]:


# convert target value to categorical
df_raw['diagnosis'] = np.where(df_raw['diagnosis'] == 'M', 1, 0)
df_raw['diagnosis'].value_counts()


# In[ ]:


# store clean data
df_clean = df_raw


# ## Model Selection

# In[ ]:


# define data and target
data = df_clean.drop('diagnosis', axis=1)
target = df_clean['diagnosis']

# define training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    data, target, test_size=0.2, random_state=42, stratify=target)


# ### Logistic Regression

# In[ ]:


# instantiate Logistic Regression
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression()
# Train the classifier
LR.fit(X_train, y_train)
# Test the classifier and get the prediction
y_pred = LR.predict(X_test)

# metrics
print('Logistic Regression')
print('\nAccuracy test set:')
print(accuracy_score(y_test, y_pred))
print('\nMean absolute error test set: ')
print(mean_absolute_error(y_test, y_pred))
print('\nMean squared error test set: ')
print(mean_squared_error(y_test, y_pred))


# ### Random Forest Classifer

# In[ ]:


# instantiate Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
RFC = RandomForestClassifier(n_estimators=10)
RFC.fit(X_train, y_train)
y_pred = RFC.predict(X_test)

# metrics
print('Random Forest Classifier')
print('\nAccuracy test set:')
print(accuracy_score(y_test, y_pred))
print('\nMean absolute error test set: ')
print(mean_absolute_error(y_test, y_pred))
print('\nMean squared error test set: ')
print(mean_squared_error(y_test, y_pred))


# ### Gradient Boosting Classifier

# In[ ]:


# Instantiate gradient boosting classifier
from sklearn.ensemble import GradientBoostingClassifier
CLF = GradientBoostingClassifier(max_depth=2, n_estimators=120)
CLF.fit(X_train, y_train)

errors = [mean_squared_error(y_test, y_pred)
         for y_pred in CLF.staged_predict(X_test)]
best_n_estimators = np.argmin(errors)

CLF_best = GradientBoostingClassifier(max_depth=2, n_estimators=best_n_estimators)
CLF_best.fit(X_train, y_train)
y_pred = CLF_best.predict(X_test)

# metrics
print('Gradient Boosting Classifier')
print('\nAccuracy test set:')
print(accuracy_score(y_test, y_pred))
print('\nMean absolute error test set: ')
print(mean_absolute_error(y_test, y_pred))
print('\nMean squared error test set: ')
print(mean_squared_error(y_test, y_pred))


# The Gradient Boosting Classifer is our best performing model without overfitting.

# ## Feature Selection
# Drop correlated features and, if necessary, select only top features.

# In[ ]:


from sklearn.feature_selection import RFECV

# The "accuracy" scoring is proportional to the number of correct classifications
model = GradientBoostingClassifier() 
rfecv = RFECV(estimator=model, step=1, cv=5,scoring='accuracy')   #5-fold cross-validation
rfecv = rfecv.fit(X_train, y_train)

print('Optimal number of features :', rfecv.n_features_)
print('Best features :', X_train.columns[rfecv.support_])


# In[ ]:


# Get columns to keep
mask = rfecv.support_ #list of booleans
new_features = data.columns[mask]

#        
data = data[new_features]
display(data.shape)
data.head()


# ### Check for correlation

# In[ ]:


# plot a heatmap
sns.heatmap(data.corr());


# In[ ]:


# Create correlation matrix
corr_matrix = data.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=4).astype(np.bool))

# Find index of feature columns with correlation greater than 0.90
to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]

display(data.shape)

# Drop correlated features 
for i in to_drop:
    data = data.drop(i, axis=1)

data.shape


# Now that our feature set is reduced to only the non-correlated features, we can optimize our model.

# ## Fine Tune Model

# In[ ]:


# define training and test set with new features
X_train, X_test, y_train, y_test = train_test_split(
    data, target, test_size=0.2, random_state=42)

# sample training data
X_train_sample, X_test_sample, y_train_sample, y_test_sample = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)


# In[ ]:


# define our parameter ranges
learning_rate=[0.01]
n_estimators=[int(x) for x in np.linspace(start = 10, stop = 500, num = 4)]
max_depth=[int(x) for x in np.linspace(start = 3, stop = 15, num = 4)]
max_depth.append(None)
min_samples_split=[int(x) for x in np.linspace(start = 2, stop = 5, num = 4)]
min_samples_leaf=[int(x) for x in np.linspace(start = 1, stop = 4, num = 4)]
max_features=['auto', 'sqrt']

# Create the random grid
param_grid = {'learning_rate':learning_rate,
              'n_estimators': n_estimators,
              'max_features': max_features,
              'max_depth': max_depth,
              'min_samples_split': min_samples_split,
              'min_samples_leaf': min_samples_leaf,
             }

print(param_grid)

# Initialize and fit the model.
from sklearn.model_selection import RandomizedSearchCV
model = GradientBoostingClassifier()
model = RandomizedSearchCV(model, param_grid, cv=3)
model.fit(X_train_sample, y_train_sample)

# get the best parameters
best_params = model.best_params_
print(best_params)


# In[ ]:


# refit model with best parameters
model_best = GradientBoostingClassifier(**best_params)
# Train the classifier 
model_best.fit(X_train, y_train)
# Test the classifier and get the prediction
y_pred = model_best.predict(X_test)

# create a dictionary to hold our metrics
metrics_dict = {}


# In[ ]:


# view top features
feature_importance = model_best.feature_importances_

# Make importances relative to max importance.
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + 0.5

plt.subplot(1,2,2)
plt.barh(pos, feature_importance[sorted_idx], align='center')

plt.yticks(pos, data.columns[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()


# In[ ]:


# sort top features
top_features = np.where(feature_importance > 25)
top_features = data.columns[top_features].ravel()
print(top_features)


# In[ ]:


# plot confusion matrix
from sklearn.metrics import confusion_matrix
import itertools

#
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    else:
        1#print('Confusion matrix, without normalization')

    #print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[ ]:


#
y_pred_train = model_best.predict(X_train)
cnf_matrix_tra = confusion_matrix(y_train, y_pred_train)

#
metrics_dict['Train recall metric'] = 100*cnf_matrix_tra[1,1]/(cnf_matrix_tra[1,0]+cnf_matrix_tra[1,1])
print("Recall metric in the train dataset: {}%".format(metrics_dict['Train recall metric'])
     )

#
class_names = [0,1]
plt.figure()
plot_confusion_matrix(cnf_matrix_tra , classes=class_names, title='Confusion matrix')
plt.show()


# In[ ]:


cnf_matrix = confusion_matrix(y_test, y_pred)

metrics_dict['Test recall metric']=100*cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1])
print("Recall metric in the testing dataset: {}%".format(metrics_dict['Test recall metric'])
     )
#print("Precision metric in the testing dataset: {}%".format(
#    100*cnf_matrix[0,0]/(cnf_matrix[0,0]+cnf_matrix[1,0]))
#     )
# Plot non-normalized confusion matrix
class_names = [0,1]
plt.figure()
plot_confusion_matrix(cnf_matrix , classes=class_names, title='Confusion matrix')
plt.show()


# In[ ]:


# metrics
print('Optimized Gradient Boosting Classifier')
print('\nAccuracy test set:')
print(accuracy_score(y_test, y_pred))
print('\nMean absolute error test set: ')
print(mean_absolute_error(y_test, y_pred))
print('\nMean squared error test set: ')
print(mean_squared_error(y_test, y_pred))

# top features
print('\nTop indicators:')
print(top_features)


# ## Conclusion
# We've cleaned our data, selected our top features, and optimized our model.
# Our final model has an accuracy of ~97% and we've determined the top five indicators of a malignant tissue mass.

# In[ ]:




