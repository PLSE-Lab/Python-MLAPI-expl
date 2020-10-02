#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_auc_score


# # Import Data
# Firstly we import the training and test data. They are both put into seperate dataframes with pointid and presence removed, so that they can be used for fitting models, etc.

# In[ ]:


DATA_DIR = '/kaggle/input/killer-shrimp-invasion/'
RANDOM_STATE = 0

train = pd.read_csv(DATA_DIR + 'train.csv')
test = pd.read_csv(DATA_DIR + 'test.csv')


# In[ ]:


X_train = train[['Salinity_today', 'Temperature_today', 'Substrate', 'Depth', 'Exposure']]
X_test = test[['Salinity_today', 'Temperature_today', 'Substrate', 'Depth', 'Exposure']]


# # Fill In Missing Values
# As there are many NaN values within the training and test data, they need to be handled before we can create a model. To get values for these, we use the Iterative Imputer from sklearn.

# In[ ]:


# Iterative imputer
imputer = IterativeImputer(max_iter = 10, random_state = RANDOM_STATE)
imputer.fit(X_train)
X_train = pd.DataFrame(imputer.transform(X_train), columns = X_train.columns)
X_test = pd.DataFrame(imputer.transform(X_test), columns = X_test.columns)

# Scaling
scaler = StandardScaler()
scaler.fit(X_train)
X_train = pd.DataFrame(scaler.transform(X_train), columns = X_train.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns = X_test.columns)


# # Five Fold Cross Validation Function
# This cross validation function has been written just to make the rest of the notebook simpler to do. When the function is called, 5-fold stratified cross validation is performed with the given model. Verbose mode will output the ROC-AUC score for each fold as well as the mean score across all folds. The mean score is returned regardless.

# In[ ]:


def five_fold_cv(model, X_train, Y_train, verbose = True):
    skf = StratifiedKFold(n_splits = 5)
    fold = 1
    scores = []
    
    for train_index, test_index in skf.split(X_train, Y_train):
        X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
        Y_train_fold, Y_test_fold = Y_train.iloc[train_index], Y_train.iloc[test_index]

        model.fit(X_train_fold, Y_train_fold)

        preds = model.predict_proba(X_test_fold)
        preds = [x[1] for x in preds]

        score = roc_auc_score(Y_test_fold, preds)
        scores.append(score)
        if verbose:
            print('Fold', fold, '     ', score)
        fold += 1
    
    avg = np.mean(scores)
    if verbose:
        print()
        print('Average:', avg)
    return avg


# # Baselines
# To compare the performance of the models we create in this notebook we create two different baseline models. The first is a logistic regression model using the data given in train.csv. The second is a logistic regression model using only the temperature feature. One important observation is that the temperature model only scores 0.59823 with ROC-AUC (any models in this notebook that use temperature should ideally be beating that score).

# In[ ]:


avg = five_fold_cv(LogisticRegression(random_state = RANDOM_STATE), X_train, train['Presence'])


# In[ ]:


temperatures = X_train[['Temperature_today']]
avg = five_fold_cv(LogisticRegression(random_state = RANDOM_STATE), temperatures, train['Presence'])


# # Different Polynomials For Temperature
# To improve our logistic regression model we create different polynomial features for temperature. These are from powers 1 to 7 as well as 0.5 and 1.5. We could create further polynomial features, or even apply different functions such as a logarithmic function if we wanted to.

# In[ ]:


#temperatures['Temperature 0.5'] = temperatures['Temperature_today'].apply(lambda x: x ** 0.5)
temperatures['Temperature 1'] = temperatures['Temperature_today']
#temperatures['Temperature 1.5'] = temperatures['Temperature_today'].apply(lambda x: x ** 1.5)
temperatures['Temperature 2'] = temperatures['Temperature_today'] ** 2
temperatures['Temperature 3'] = temperatures['Temperature_today'] ** 3
temperatures['Temperature 4'] = temperatures['Temperature_today'] ** 4
temperatures['Temperature 5'] = temperatures['Temperature_today'] ** 5
temperatures['Temperature 6'] = temperatures['Temperature_today'] ** 6
temperatures['Temperature 7'] = temperatures['Temperature_today'] ** 7
temperatures['Temperature 8'] = temperatures['Temperature_today'] ** 8
temperatures['Temperature 9'] = temperatures['Temperature_today'] ** 9
temperatures['Temperature 10'] = temperatures['Temperature_today'] ** 10


# After creating these features, we then create different logistic regression models using these features. Model scores are compared on a simple plot to show how the cross validation score changes as we increase the degrees of freedom (number of polynomial features).

# In[ ]:


dof_temp_scores = []
dof_temp = ['Temperature 1', 'Temperature 2', 'Temperature 3', 'Temperature 4', 'Temperature 5', 'Temperature 6', 'Temperature 7', 'Temperature 8', 'Temperature 9', 'Temperature 10']
for i in range(10):
    dof_temp_scores.append(five_fold_cv(LogisticRegression(random_state = RANDOM_STATE), temperatures[dof_temp[:i + 1]], train['Presence'], verbose = False))
    
plt.plot(range(1, 11), dof_temp_scores)
plt.xlabel('Degrees of Freedom')
plt.ylabel('Cross Validation Score')
plt.title('Cross Validation Scores For Polynomial Temperature Models')
plt.show()


# # Salinity
# The same method can be repeated for salinity. We have not repeated the baseline for salinity, as you will find it is the same as the first point plotted on the graph (where only the 'Salinity 1' feature is used). This time in the graph we have compared the models using salinity and temperature.

# In[ ]:


# Get salinity polynomial features
salinities = X_train[['Salinity_today']]
for i in range(1, 11):
    salinities['Salinity ' + str(i)] = salinities['Salinity_today'] ** i

# Perform cross validation on models
dof_sal_scores = []
dof_sal = ['Salinity 1', 'Salinity 2', 'Salinity 3', 'Salinity 4', 'Salinity 5', 'Salinity 6', 'Salinity 7', 'Salinity 8', 'Salinity 9', 'Salinity 10']
for i in range(10):
    dof_sal_scores.append(five_fold_cv(LogisticRegression(random_state = RANDOM_STATE), salinities[dof_sal[:i + 1]], train['Presence'], verbose = False))

# Create graph of cross validation scores
plt.plot(range(1, 11), dof_sal_scores, label = 'Salinity Model')
plt.plot(range(1, 11), dof_temp_scores, label = 'Temperature Model')
plt.xlabel('Degrees of Freedom')
plt.ylabel('Cross Validation Score')
plt.title('Cross Validation Scores For Polynomial Salinity and Temperature Models')
plt.show()


# From this graph, it is evident that the salinity models do not perform as well as the temperature models. The salinity models achieves an approximate score of 0.58, plateauing at 2 degrees of freedom. The temperature models achieve an approximate score of 0.89 as you approach 10 degrees of freedom, but plateau before that (at around 6 degrees of freedom).
# 
# # Our Chosen Model
# From the models we have created we can choose a model for making predictions. We will choose the temperature model with 6 degrees of freedom as that is where our model begins to plateau. We will perform cross validation again to see how it performs on each fold, and then use a completely fitted model to generate our predictions.

# In[ ]:


degrees = 6

# Cross validation
five_fold_cv(LogisticRegression(random_state = RANDOM_STATE), temperatures[dof_temp[:degrees]], train['Presence'])

# Feature extraction from the test data (test data has already been scaled)
test_temperatures = pd.DataFrame()
for i in range(1, degrees + 1):
    test_temperatures['Temperature ' + str(i)] = X_test['Temperature_today'] ** i

# Building the actual model
model = LogisticRegression(random_state = RANDOM_STATE)
model.fit(temperatures[dof_temp[:degrees]], train['Presence'])

# View coefficients
print()
print('Coefficients:', model.coef_)
print('Intercept:   ', model.intercept_)


# With that we have created our final model. Since we are using logistic regression we can express this model as an equation, given below:
# $$ p(Shrimp) = \frac{1}{1 + e^{-f(t)}}$$  
# 
# $$ f(t) = -10.52910058 + 2.21529641*t - 0.57623745*t^2 - 2.06832485*t^3\\ + 0.30713969*t^4 - 2.21781761*t^5 - 1.85363275*t^6$$
# 
# All there is left to do at this point is to make our predictions on the test set, which is done below.

# In[ ]:


# Make predictions
preds = model.predict_proba(test_temperatures)
preds = [x[1] for x in preds]

# Save preds to file
res = pd.DataFrame()
res['pointid'] = test['pointid']
res['Presence'] = preds
res.to_csv('predictions.csv', index = False)

