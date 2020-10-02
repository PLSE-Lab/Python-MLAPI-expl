#!/usr/bin/env python
# coding: utf-8

# # Churn Prediction
# 
# Regularized logistic regression and random forest models are compared to predict customer churn. To select the best model a 5-Fold Cross Validation with randomised search was used. Since the dataset is unbalanced with respect to Churn the area under the ROC curve (roc_auc) was used as criterium for the selection.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ## Preparing the data for machine learning
# 
# 1. Set empty TotalCharges to 0 and convert it to numeric
# 2. Encode Churn as 0 for 'No' and 1 for 'Yes'
# 3. One-Hot encoding of categorical data
# 4. Standardization of numerical data
# 5. Drop customerID
# 
# Standardization is applied to improve convergence of regularised Logistic Regression and to put the coefficients on the same scale, so that their magnitude can be interpreted as their importance. The scaling will be used for all the models, even though for Random Forest is not required.
# In order to have a genuine evaluation of the models the scaling must be defined only on the training data, otherwise a bias will be introduced.

# In[ ]:


telco_original = pd.read_csv('../input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv')

telco_original['TotalCharges'] = telco_original.TotalCharges.replace({' ': 0})
telco_original['TotalCharges'] = pd.to_numeric(telco_original.TotalCharges, errors='coerce')
# remove the 9 rows with missing values
print(telco_original.info())

telco_original = telco_original.drop('customerID', axis=1)

telco_original['Churn'] = telco_original.Churn.replace({'No': 0, 'Yes':1})

from sklearn.model_selection import train_test_split
X = telco_original.drop('Churn', axis=1)
y = telco_original.Churn

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=telco_original.Churn, 
                                                    test_size=0.2, random_state=123)


# All the object columns contain only few categories (<5) and can be OneHot encoded (this step could have been done before splitting too, but for consistency with the standardization procedure it will be done after).

# In[ ]:


telco_original.nunique()


# In[ ]:


from sklearn.preprocessing import OneHotEncoder, StandardScaler

# generate the list of categorical and numerical variables
categorical_variables = telco_original.nunique()[telco_original.nunique() < 5].keys().to_list()

numerical_variables=list(set(telco_original.columns) - set(categorical_variables))
categorical_variables.remove('Churn')

ohe = OneHotEncoder(drop='first', sparse=False)

X_train_ohe = ohe.fit_transform(X_train[categorical_variables])
X_train_ohe_df = pd.DataFrame(X_train_ohe, columns=ohe.get_feature_names(categorical_variables))

# Transform only without fitting
X_test_ohe = ohe.transform(X_test[categorical_variables])
X_test_ohe_df = pd.DataFrame(X_test_ohe, columns=ohe.get_feature_names(categorical_variables))


scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train[numerical_variables])
X_train_sc_df = pd.DataFrame(X_train_sc, columns=numerical_variables)

# Transform only without fitting
X_test_sc = scaler.transform(X_test[numerical_variables])
X_test_sc_df = pd.DataFrame(X_test_sc, columns=numerical_variables)

# Merging the transformed dataframe togheter
X_train = pd.merge(X_train_ohe_df, X_train_sc_df, left_index=True, right_index=True)
X_test = pd.merge(X_test_ohe_df, X_test_sc_df, left_index=True, right_index=True)


# ## Models fitting
# 

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
#from sklearn.metrics import roc_auc_score
from scipy.stats import loguniform
import numpy as np
import time

log_reg = LogisticRegression(solver='liblinear')

param_grid = {'penalty': ['l1', 'l2'],
              'C': loguniform(10**-4, 10**4)}

random_search = RandomizedSearchCV(log_reg, param_grid, n_jobs=-1, n_iter=40, scoring='roc_auc', verbose=True)

start = time.time()
random_search.fit(X_train, y_train)
end = time.time()

timings = []
timings.append(end -start)


# In[ ]:


lr_best = random_search.best_estimator_
random_search.best_params_


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()

# Number of trees in random forest
n_estimators = list(np.linspace(start = 200, stop = 2000, num = 10, dtype='int'))
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = list(np.linspace(10, 110, num = 11, dtype='int'))
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
param_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap,
               "criterion": ["gini", "entropy"]}

random_search = RandomizedSearchCV(rf, param_grid, n_jobs=-1, n_iter=40, scoring='roc_auc', verbose=True)

start = time.time()
random_search.fit(X_train, y_train)
end = time.time()

timings.append(end - start)


# In[ ]:


rf_best = random_search.best_estimator_
random_search.best_params_


# In[ ]:


# Import roc_curve
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

def add_roc_plot(model, test_x, test_y, legend_text):
    y_pred_prob = model.predict_proba(test_x)[:, 1]
    # Calculate the roc metrics
    fpr, tpr, thresholds = roc_curve(test_y, y_pred_prob)
    # Plot the ROC curve
    plt.plot(fpr, tpr, label=legend_text)
    plt.legend()

    
models_list = [lr_best, rf_best]
model_names = ['Logistic Regression', 'Random Forest']

plt.figure(figsize=(6, 6))
[add_roc_plot(model, X_test, y_test, legend_text) for model, legend_text in zip(models_list, model_names)]

# Add labels and diagonal line
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.xlim((0,1))
plt.ylim((0,1))
plt.plot([0, 1], [0, 1], "k-")
plt.show()


# In[ ]:


from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, precision_score

list_scores = [roc_auc_score, recall_score, precision_score, accuracy_score]
calc_scores = []
def compute_scores(model, x_test, y_test, scores):
    return [round(score(y_test, model.predict(x_test)), 2) for score in scores]
    
[calc_scores.append(compute_scores(model, X_test, y_test, list_scores)) for model in models_list] 

score_names = ['roc_auc', 'recall', 'precision', 'accuracy']
scores_df = pd.DataFrame(calc_scores, columns=score_names, index=model_names)

scores_df['timing (s)'] = [round(t) for t in timings]
scores_df


# In[ ]:


from sklearn.metrics import plot_confusion_matrix

fig, (ax1,ax2) = plt.subplots(1,2, figsize=(12,8))

ax1.title.set_text('Logistic Regression')
plot_confusion_matrix(lr_best, X_test, y_test, cmap=plt.cm.Blues, values_format='d', ax=ax1)
ax2.title.set_text('Random Forest')
plot_confusion_matrix(rf_best, X_test, y_test, cmap=plt.cm.Blues, values_format='d', ax=ax2)
plt.show()


# The two methods give comparable results. Below is showed grafically how the importances of the top 10 features for the two models change. The lines connect the feature with the same rank, the missing dots are not in the top 10.

# In[ ]:


rf_importance=pd.DataFrame({'feature': X_train.columns, 'importance': rf_best.feature_importances_}).sort_values('importance', ascending=False)
rf_importance.index = [i for i in range(1, len(rf_importance)+1)]
rf_importance.reset_index(inplace=True)
rf_importance['model'] = 'Random Forest'

lr_importance = pd.DataFrame({'feature': X_train.columns, 'importance': abs(lr_best.coef_[0,:])}).sort_values('importance', ascending=False)
lr_importance.index = [i for i in range(1, len(lr_importance)+1)]
lr_importance.reset_index(inplace=True)
lr_importance['model'] = 'Logistic Regression'

topN=10

top10_importance = pd.concat([rf_importance.head(topN), lr_importance.head(topN)])


# In[ ]:


from plotnine import ggplot, aes, geom_point, geom_line, theme_minimal, theme, element_blank, ggtitle

top10_common_category = top10_importance.feature[0:10].append(pd.Series(list(set(top10_importance.feature[10:21])- set(top10_importance.feature[0:10]))))
top10_importance['feature'] = pd.Categorical(top10_importance['feature'], categories = top10_common_category[::-1])
top10_importance['model'] = pd.Categorical(top10_importance['model'], categories = ['Random Forest', 'Logistic Regression'])

ggplot(top10_importance, aes('model','feature')) +    geom_point(color='blue') +    geom_line(aes(group='index')) +    theme_minimal() +    theme(axis_title_x = element_blank(),
          axis_title_y = element_blank()) +\
    ggtitle('Top 10 Feature Importance')


# In[ ]:




