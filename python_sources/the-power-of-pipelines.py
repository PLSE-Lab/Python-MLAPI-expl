#!/usr/bin/env python
# coding: utf-8

# In this kernel I want to solely focus on the power of pipelines, which is why I have chosen a simple dataset (Iris) that requires no data wrangling. I will use pipelines to test out different classification models with parameter tuning using GridSearch. The best model will be automatically chosen and several metrics calculated. This includes *precision*, *recall*, *F-1 score*, and a *confusion matrix*, which help understanding where the model classifies (in)correctly. Here, the best model is **RandomForest**, which in fact classifies all observations 

# In[ ]:


# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Load Iris data
data = pd.read_csv('../input/Iris.csv')
data.head()


# **EDA**

# In[ ]:


# Plot the pairplots - this is useful when you don't have a lot of features otherwise there would be way too many gra

sns.pairplot(data.drop('Id', axis='columns'), hue='Species', diag_kind='kde', size=3)
plt.title('Pairplot of Iris dimension')
plt.show()


# We can see that Setosa is very distinct but there is some overlap between Versicolor and Virginica and I expect that is where potential misclassifications can happen. 

# **Prepare the Data**

# In[ ]:


# Import the libraries

# Split data into train and test
from sklearn.model_selection import train_test_split

# Get the classification models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.cluster import KMeans

# Get pipeline
from sklearn.pipeline import Pipeline

# Get GridSearch
from sklearn.model_selection import GridSearchCV

# Get scaling
from sklearn.preprocessing import StandardScaler

# Get metrics
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# In[ ]:


# Split into train and test

X_train, X_test, y_train, y_test = train_test_split(data.drop(['Id', 'Species'], axis='columns'), data['Species'],test_size=0.2, random_state=0)


# **Build a Pipeline and Run the Models**

# 1) First, build the high-level pipeline, which includes scaling as well as the classification model.

# In[ ]:


# Build the pipelines

pipe_lr = Pipeline([('scaler', StandardScaler()),
         ('clf', LogisticRegression(random_state=0))])

pipe_knn = Pipeline([('scaler', StandardScaler()),
         ('clf', KNeighborsClassifier())])

pipe_rf = Pipeline([('scaler', StandardScaler()),
         ('clf', RandomForestClassifier(random_state=0))])

pipe_et = Pipeline([('scaler', StandardScaler()),
         ('clf', ExtraTreesClassifier(random_state=0))])

pipe_svm = Pipeline([('scaler', StandardScaler()),
         ('clf', SVC(random_state=0))])


# 2) Second, set the parameters that you want to tune using GridSearch (or alternatively RandomSearch).

# In[ ]:


# Set grid search params
param_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
param_range_fl = [1.0, 0.5, 0.1]

grid_params_lr = [{'clf__penalty': ['l1', 'l2'],
                'clf__C': param_range_fl,
                'clf__solver': ['liblinear']}] 

grid_params_knn = [{'clf__n_neighbors': [3]}]

grid_params_rf = [{'clf__n_estimators': [100],
                'clf__criterion': ['gini', 'entropy'],
                'clf__min_samples_leaf': [2,5,10],
                'clf__max_depth': [2,5,10],
                'clf__min_samples_split': [2,5,10]}]

grid_params_et = [{'clf__n_estimators': [100],
                'clf__criterion': ['gini', 'entropy'],
                'clf__min_samples_leaf': [2,5,10],
                'clf__max_depth': [2,5,10],
                'clf__min_samples_split': [2,5,10]}]

grid_params_svm = [{'clf__kernel': ['linear', 'rbf'], 
                    'clf__C': param_range}]


# 3) Third, build the GridSearch pipeline for each model using the parameters that I set just now - and running cross-validation 10 times.

# In[ ]:


gs_lr = GridSearchCV(estimator=pipe_lr,
                    param_grid=grid_params_lr,
                    scoring='accuracy',
                    cv=10)

gs_knn = GridSearchCV(estimator=pipe_knn,
                    param_grid=grid_params_knn,
                    scoring='accuracy',
                    cv=10)

gs_rf = GridSearchCV(estimator=pipe_rf,
                    param_grid=grid_params_rf,
                    scoring='accuracy',
                    cv=10)

gs_et = GridSearchCV(estimator=pipe_et,
                    param_grid=grid_params_et,
                    scoring='accuracy',
                    cv=10)

gs_svm = GridSearchCV(estimator=pipe_svm,
                    param_grid=grid_params_svm,
                    scoring='accuracy',
                    cv=10)


# 4) Finally, run Gridsearch and save the results. We can then determine the best model to fit the data.

# In[ ]:


# List of pipelines for ease of iteration
grids = [gs_lr, gs_knn, gs_rf, gs_et, gs_svm]

# Dictionary of pipelines and classifier types for ease of reference
grid_dict = {0: 'Logistic Regression', 1: 'K-Nearest Neighbor', 
                2: 'Random Forest', 3: 'Extra Trees', 
                4: 'Support Vector Machine',}

# Fit the grid search objects
print('Performing model optimizations...')
best_acc = 0.0
best_clf = 0
best_gs = ''
for idx, gs in enumerate(grids):
    print('\nEstimator: %s' % grid_dict[idx])
    # Fit grid search
    gs.fit(X_train, y_train)
    # Best params
    print('Best params: %s' % gs.best_params_)
    # Best training data accuracy
    print('Best training accuracy: %.3f' % gs.best_score_)
    # Predict on test data with best params
    y_pred = gs.predict(X_test)
    # Test data accuracy of model with best params
    print('Test set accuracy score for best params: %.3f ' % accuracy_score(y_test, y_pred))
    # Track best (highest test accuracy) model
    if accuracy_score(y_test, y_pred) > best_acc:
        best_acc = accuracy_score(y_test, y_pred)
        best_gs = gs
        best_clf = idx
print('\nClassifier with best test set accuracy: %s' % grid_dict[best_clf])


# **Use the best model to make predictions**

# **Random Forest** and the **Support Vector Machine Classifier** were the best models. They had the same scores and it's possible that they made the same predictions. I will simply use the best model, which in this case is the Random Forest. That way it is a continuous flow and we can take full advantage of the pipeline without intervening in the code.

# In[ ]:


# Fit and predict with the best model
best_gs.best_estimator_.fit(X_train, y_train)
y_pred = best_gs.best_estimator_.predict(X_test)

# Get the test and cross validation score
test_score = accuracy_score(y_test, y_pred)
cv_scores = cross_val_score(pipe_rf, X_train, y_train, cv=5)
print('Test set accuracy score: %.3f ' % test_score)
print('Cross validation score: %.3f ' % np.mean(cv_scores))
print(cv_scores)


# **Evaluate the model and investigate correct and false classifications**

# In[ ]:


# Get the precision, recall and f1-score
print(classification_report(y_test, y_pred))


# * **Precision**: This is the accuracy of the positive predictions --> TP / (TP + FP)
# * **Recall**: The is the True Positive Rate, i.e. how many positives out of all positives were classified correctly --> TP / (TP + FN)
# * **F1 Score**: This combines *Precision* and *Recall* and gives a high weight lower values. The *F1 Score* can only be high if both *Precision* and *Recall* are high 
#     --> TP / (TP + (FN + FP) / 2)

# In[ ]:


# Get the confusion matrix
label= best_gs.classes_
conf = confusion_matrix(y_test, y_pred)
sns.heatmap(conf, annot=True, cbar=False, xticklabels=label, yticklabels=label)
plt.title('Confusion Matrix')
plt.yticks(rotation=0)
plt.show()


# Now let's look at this again but in a slightly different way - let's rescale the errors. Because some classes may have more observations than others we need to rescale so that we can compare like with like and see where as a proportion the most mistakes happen. In this case **Random Forest** classified the observations perfectly, so there is not much to see.

# In[ ]:


# Get the rescaled confusion matrix based on the errors
row_sums = conf.sum(axis=1, keepdims=True)
norm_conf = conf / row_sums
# Fill the diagonals with 0s
np.fill_diagonal(norm_conf, 0)

# Plot the confusion matrix
sns.heatmap(norm_conf, annot=True, cbar=False, xticklabels=label, yticklabels=label)
plt.title('Error Confusion Matrix')
plt.yticks(rotation=0)
plt.show()

