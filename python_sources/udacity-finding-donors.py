#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# # Loading Datasets

# In[ ]:


training_data = pd.read_csv("../input/udacity-mlcharity-competition/census.csv")
test_data = pd.read_csv("../input/udacity-mlcharity-competition/test_census.csv", index_col=0)

print("Shape of Training Dataset:", training_data.shape)
print("Shape of Testing Dataset:", test_data.shape)
training_data.head()


# ## Distribution of Target Label

# In[ ]:


print(training_data.income.value_counts(normalize=True))
print()
print(training_data.income.value_counts())


# It can be seen from above result that training data is highly biased. More than 75% of the data is for "<=50K"

# ## Checking null values in data

# In[ ]:


print("Null Values in training data:", training_data.isna().sum().sum())
print("Null Values in Testing data:", test_data.isna().sum().sum())


# There are lots of null values in test data and so we need to fill in some values for these as we cannot delete these records as it is testing data.

# ### Distribution of Null Values

# In[ ]:


test_data.isna().sum()


# As it can be seen from above output, all the fields have null values, which we need to fill with some values

# ### Types of columns

# In[ ]:


training_data.dtypes


# From above output
# 
# * **Numberical Features** : age, education-num, capital-gain, capital-loss, hours-per-week
# * **Categorical Features** : workclass, education_level, marital-status, occupation, relationship, race, sex, native-country
# * **Output label** : income (Categorical)

# To fill in Null values,
# 
# * **Numberical Features**: Need to be filled with relevant values, either mean or interpolled value
# * **Categorical Features**: Null values will be eliminated when converting the data to one-hot-encoding

# ### Filling Numberical Values with mean value

# In[ ]:


X_test_fill_mean = test_data.fillna(test_data.mean())
X_test_fill_mean.isna().sum()


# In[ ]:


X_test_fill_mean.dtypes


# As we can see all numerical NA values are filled and only categoricals are remaining

# ## Splitting Training Features and Label

# In[ ]:


y_train_raw = training_data.income
X_train_raw = training_data.drop('income', axis=1)


# In[ ]:


X_train_raw['capital-gain']


# ## Finding Distribution of Numerical Data

# In[ ]:


plt.hist(X_train_raw['capital-gain'], bins=25)
plt.title('capital-gain')
plt.ylim([0, 2000])
plt.show()
plt.hist(X_train_raw['capital-loss'], bins=25)
plt.title('capital-loss')
plt.ylim([0, 2000])
plt.show()


# ## Log Transform the above data

# In[ ]:


skewed = ['capital-gain','capital-loss']
X_train_log_transform = pd.DataFrame(X_train_raw)
X_test_log_transform = pd.DataFrame(X_test_fill_mean)

X_train_log_transform[skewed] = X_train_log_transform[skewed].apply(lambda x: np.log(x+1))
X_test_log_transform[skewed] = X_test_log_transform[skewed].apply(lambda x: np.log(x+1))


# In[ ]:


plt.hist(X_train_log_transform['capital-gain'], bins=25)
plt.title('capital-gain')
plt.ylim([0, 2000])
plt.show()
plt.hist(X_train_log_transform['capital-loss'], bins=25)
plt.title('capital-loss')
plt.ylim([0, 2000])
plt.show()


# Normalize the Numerical Data

# In[ ]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

X_train_log_minmax_transform = pd.DataFrame(data = X_train_log_transform)
X_train_log_minmax_transform[numerical] = scaler.fit_transform(X_train_log_transform[numerical])

X_test_log_minmax_transform = pd.DataFrame(data = X_test_log_transform)
X_test_log_minmax_transform[numerical] = scaler.transform(X_test_log_transform[numerical])


# In[ ]:


X_train_log_minmax_transform.describe()


# ## One Hot Encoding

# In[ ]:


X_train = pd.get_dummies(X_train_log_minmax_transform)
X_test = pd.get_dummies(X_test_log_minmax_transform)


# ### Convert Target Label to Numerical

# In[ ]:


y_train = (y_train_raw=='>50K').map(int)


# # Training Model

# In[ ]:


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, fbeta_score, accuracy_score
from sklearn.ensemble import AdaBoostClassifier

clf = AdaBoostClassifier(random_state=42, n_estimators=100)

parameters = {
    'learning_rate': [0.001, 0.1, 1, 10]
}

scorer = make_scorer(fbeta_score, beta=0.5)

grid_obj = GridSearchCV(clf, param_grid=parameters, scoring=scorer)

grid_fit = grid_obj.fit(X_train, y_train)

best_clf = grid_fit.best_estimator_

best_predictions = best_clf.predict(X_train)

print("Final accuracy score on the training data: {:.4f}".format(accuracy_score(y_train, best_predictions)))
print("Final F-score on the training data: {:.4f}".format(fbeta_score(y_train, best_predictions, beta = 0.5)))


# In[ ]:


pred_test = best_clf.predict(X_test)


# In[ ]:


pred_df = pd.DataFrame(pred_test, columns=['income'])
pred_df.index.names = ["id"]
pred_df


# In[ ]:


pred_df.to_csv("submission.csv")


# In[ ]:




