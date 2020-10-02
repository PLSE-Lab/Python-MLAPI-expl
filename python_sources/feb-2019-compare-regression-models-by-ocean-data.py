#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing, svm
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

import warnings
warnings.filterwarnings('ignore')


# **Load dataset**

# In[ ]:


# read datasets
bottle = pd.read_csv('../input/bottle.csv')


# Review Full Dataset and Get Overall Stats

# In[ ]:


bottle.head(2)


# In[ ]:


bottle.describe()


# **Feature Extraction**

# In[ ]:


bottle = bottle[['Salnty', 'T_degC']]
bottle.columns = ['Sal', 'Temp']
data = bottle  #copy original data for later use


# Let us assume less than 1% of total data is only used to model by mistake

# In[ ]:


# Limiting amount of entries to speed up regression time
bottle = bottle[:][:500]

print(bottle.head())


# Visualize the data

# Get Stats

# In[ ]:


data.columns


# In[ ]:


data.dtypes


# In[ ]:


data.describe()


# In[ ]:


data.head(2)


# In[ ]:


# This produces a scatter 
sns.lmplot(x="Sal", y="Temp", data=bottle,
           order=2, ci=None);


# **Feature Transformation**

# In[ ]:


# Identify where we are seeing the null values
bottle.isnull().sum()


# Clean the data

# In[ ]:


# Eliminating NaN or missing input numbers
bottle.fillna(method='ffill', inplace=True)


# **Select features and set the target variable**

# In[ ]:


X = np.array(bottle['Sal']).reshape(-1, 1)
y = np.array(bottle['Temp']).reshape(-1, 1)


# Assume validation set is **less than 20%** and seed is not chosen at the first round of modeling

# In[ ]:


# split data
validation_size = 0.011
seed = 0


# **Split** the data

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=validation_size, 
                                                    random_state=seed)


# Assume **Linear Regression model is chosen by default** and used for predictions as part of first round

# In[ ]:


model_UnitTest = LinearRegression()  #unit test one LR model
model_UnitTest.fit(X_train, y_train)


# Let's **predict** with validation data 

# In[ ]:


y_pred = model_UnitTest.predict(X_test)


# **Model accuracy - Evaluate model**

# In[ ]:


accuracy = model_UnitTest.score(X_test, y_test)
print("Basic Linear Regression Model accuracy: " +"{:.1%}".format(accuracy));


# In[ ]:


print("R2 Score: " +"{:.3}".format(r2_score(y_test, y_pred)));


# 

# Let's draw **plots** to verify how model is doing

# In[ ]:


plt.scatter(X_test, y_test, color='b')
plt.plot(X_test, y_pred, color='k')
plt.show()


# **Let's test the model real time** which is completely new and not in the original dataset) - Production simulation

# In[ ]:


# Test the model using the new dataset
X_new = [[30],[36]] #X_test[5].reshape(1, -1)
y_pred = model_UnitTest.predict(X_new)

plt.scatter(X_train, y_train, color='b')

plt.plot(X_new, y_pred, color='r')
plt.show()


# Since the above model is overfitting, we need to tune our model and work on hyper parameters a bit.
# Now let's jump onto the fun part - **Regression Models' Comparison...!!!**

# In[ ]:


# Import regression models
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))


# **Take full set of data** and select the features

# In[ ]:


data.fillna(method='ffill', inplace=True)

X2 = np.array(data['Sal']).reshape(-1, 1)
y2 = np.array(data['Temp']).reshape(-1, 1)


# **Split data** - 80% for training and 20% for validating the model

# In[ ]:


from sklearn.model_selection import train_test_split
validation_size2 = 0.20
seed2 = 5

X_train2, X_validation2, Y_train2, Y_validation2 = train_test_split(
    X2, y2, test_size=validation_size2, random_state=seed2)


# In[ ]:


X_train2[1]


# In[ ]:


Y_train2[1]


# In[ ]:


#lab_enc = preprocessing.LabelEncoder()  #fix ValueError: Unknown label type: continuous
#Y_train_encoded = lab_enc.fit_transform(Y_train)


# In[ ]:


Y_train2_int_type = Y_train2.astype('int')


# **Evaluate** each model in a loop

# In[ ]:


scoring = 'accuracy'    #10 fold cross validation

results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed2)
	cv_results = model_selection.cross_val_score(model, X_train2, 
                                                 Y_train2_int_type, 
                                                 cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)


# **Draw Algorithm Comparison Plots**

# In[ ]:


fig = plt.figure()
fig.suptitle('Algorithm Comparison')
plt.plot([1,2,3])
ax = fig.add_subplot(211)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


# **Choose the final and check on the accuracy of the best model**

# In[ ]:


model = KNeighborsClassifier()
model.fit(X_train2, Y_train2_int_type)


# Make predictions using the Final Model

# In[ ]:


y_pred2 = model.predict(X_train2)


# Evaluate the Final Model performance

# In[ ]:


print("R2 Score: " +"{:.3}".format(r2_score(Y_train2, y_pred2)));


# **Visualize final model with the best fit Regression Model**

# In[ ]:


plt.scatter(X_train2[:10], Y_train2[:10], color='b')
plt.plot(X_train2[2:4], y_pred2[2:4], color='g')
plt.show()


# **Thank you for your time**
