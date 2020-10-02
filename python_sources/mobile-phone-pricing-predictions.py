#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# 
# hai kagglers, today i'm going to make a predictions on Mobile phone pricing dataset using random forest, XGboost and SVM algorithm, let's see which one perform the best.

# ## Import Modules

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# ## Quick look
# 
# In this dataset, we got two csv files, train data and test data. both had 21 column each.

# In[ ]:


train = pd.read_csv('../input/mobile-price-classification/train.csv')
test = pd.read_csv('../input/mobile-price-classification/test.csv')
train.head()


# In[ ]:


train.info()


# In[ ]:


test.head()


# In[ ]:


test.info()


# Our target column is price range from train data. this column has 4 unique value, 0 for low cost mobile phone, 1 for medium cost mobile phone. 2 for high cost and 3 for very high cost. Since the data disributed evenly, we are going to use it as it is,

# In[ ]:


# 0 for low cost
# 1 for medium cost
# 2 for high cost
# 3 for very high cost

train.price_range.value_counts()


# In[ ]:


## EDA
# Distribution

train.hist(bins=30, figsize=(15, 15))


# Here we have a distribution plot for every columns, a few column is evenly distributed like dual sim, bluetooth, price range, wifi, and touch screen. The others distributed randomly.

# In[ ]:


# Most important feature

Corr = train.corr()

IF = Corr['price_range'].sort_values(ascending=False).head(10).to_frame()
IF.head(5)


# In[ ]:


f = plt.figure(figsize=(15,12))

# corr with ram
ax = f.add_subplot(221)
ax = sns.scatterplot(x="price_range", y="ram", color='b', data=train)
ax.set_title('Corr with RAM')

# corr with Battery
ax = f.add_subplot(222)
ax = sns.scatterplot(x="price_range", y="battery_power", color='c', data=train)
ax.set_title('Corr with battery')

# corr with px_width
ax = f.add_subplot(223)
ax = sns.scatterplot(x="price_range", y="px_width", color='r', data=train)
ax.set_title('Corr with px width')

# corr with height
ax = f.add_subplot(224)
ax = sns.scatterplot(x="price_range", y="px_height", color='g', data=train)
ax.set_title('Corr with px height')


# The most important feature is ram, battery power, width and height. Ram is strongly correlated with price range. Herewe can conclude that the main factor of the price is the ram itself. The other are not to strong and seem almost distributed evenly.

# ## Modeling

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[ ]:


# Split Data

X = train.drop('price_range', axis=1)
y = train['price_range']


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)


# In[ ]:


print('X_train : ' + str(X_train.shape))
print('X_test : ' + str(X_test.shape))
print('y_train : ' + str(y_train.shape))
print('y_test : ' + str(y_test.shape))


# ### Random Forest Classifier

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators = 200, criterion = 'entropy', random_state = 12)
classifier.fit(X_train, y_train)

# predict
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)


# ### XGBoost

# In[ ]:


import xgboost as xgb

# Instantiate the XGBClassifier: xg_cl
xg_cl = xgb.XGBClassifier(objective='multi:softmax', num_class=3, n_estimators=150, seed=123)
xg_cl.fit(X_train, y_train)
preds = xg_cl.predict(X_test)

# Compute the accuracy: accuracy
accuracy = float(np.sum(preds==y_test))/y_test.shape[0]
print("accuracy: %f" % (accuracy))


# ### SVM

# In[ ]:


from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

svm = SVC()

parameters = {'C':[0.1, 1, 10], 'gamma':[0.00001, 0.0001, 0.001, 0.01, 0.1]}
model = GridSearchCV(svm, param_grid=parameters)
model.fit(X_train, y_train)

# Best parameters
print("Best CV params", model.best_params_)

# accuracy
print("Test accuracy :", model.score(X_test, y_test))


# Here we found that SVM perfrom the better than the other algorithm. We are going to use this model to predict the data on train and test data.

# In[ ]:


## Assign the value

predicted_value = model.predict(X_test)
actual_value = y


# In[ ]:


## COmparison distribution in Train data

sns.distplot(actual_value, hist=False, label="Actual Values")
sns.distplot(predicted_value, hist=False, label="Predicted Values")
plt.title('Distribution Comaprison with SVM')
plt.show()


# ## Predict
# 
# Let's predict the price range in test data

# In[ ]:


test.head()


# In[ ]:


X2 = test.drop('id', axis=1)


# In[ ]:


## Perform predictions

predicted_test_value = model.predict(X2)
pd.value_counts(predicted_test_value)


# Here we have value count on every unique value on predicted test price range. This is almost evenly distributed, but the phone with a very high cost is dominating over the others with 26.5 %. and the lowest is medium cost with 23.2 %.

# In[ ]:


# Here is how distribution look like in test data price range

sns.distplot(predicted_value, hist=False, label="Predicted Values")
plt.show()


# ## End
# 
# That is all for this kernel today, hope you like it.
# Thank you.
# Have a good day.
