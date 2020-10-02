#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import sklearn


# In[ ]:


ds = pd.read_csv("../input/winequality-red.csv", sep=",", encoding="utf-8")


# In[ ]:


ds.head(5)


# In[ ]:


from pandas.tools.plotting import scatter_matrix
scatter_matrix(ds, alpha=0.05, figsize=(18, 18));


# In[ ]:


# What we want to predict?
predict_label = 'quality'
labels = ds[predict_label]


# In[ ]:


features_labels = [l for l in ds if l != predict_label]
# Remove the last column, that we will take like the labels column


# In[ ]:


features_labels


# In[ ]:


features = ds[features_labels]


# In[ ]:


features.head(10)


# In[ ]:


labels.head(10)


# In[ ]:


import sklearn.cross_validation
features_train, features_test, labels_train, labels_test = sklearn.cross_validation.train_test_split(features, labels, test_size=80)


# In[ ]:


if ((len(features_train) == len(labels_train)) and (len(features_test) == len(labels_test))):
    print("Splitting was correct.")


# In[ ]:


alpha_start = -1
alpha = [-1e1,
        -1e0,
        -1e-1,
        -1e-2,
        -1e-3,
        -1e-4,
        1e-4,
        1e-3,
        1e-2,
        1e-1,
        1e0,
        1e1]
#step = 1e-4
#curr = alpha_start
#while curr <= abs(alpha_start):
#    alpha.append(curr)
#    curr += step


# In[ ]:


import sklearn.linear_model

model = sklearn.linear_model.Ridge(alpha=0.1)


# In[ ]:


model.fit(X=features_train, y=labels_train)


# In[ ]:


model.get_params()


# In[ ]:


y_test = model.predict(features_test)


# In[ ]:


sklearn.metrics.mean_squared_error(labels_test, y_test)


# In[ ]:


err_by_alpha_depend = []

for a in alpha:
    model = sklearn.linear_model.Ridge(a)
    model.fit(X=features_train, y=labels_train)
    y_test = model.predict(features_test)
    err_by_alpha_depend.append(sklearn.metrics.mean_squared_error(labels_test, y_test))


# In[ ]:


plt.title("Ridge")
plt.xlabel("Alpha")
plt.ylabel("MSE")
plt.plot(alpha, err_by_alpha_depend, "r");


# In[ ]:


alpha


# In[ ]:


min_err = min(err_by_alpha_depend)


# In[ ]:


index_min_err = err_by_alpha_depend.index(min_err)


# In[ ]:


print("Best MSE = {:.4f} with Alpha = {:.7f}".format(min_err, alpha[index_min_err]))


# In[ ]:


import sklearn


# In[ ]:


from sklearn.linear_model import Lasso


# In[ ]:


model2 = Lasso()


# In[ ]:


err_by_alpha_depend2 = []

for a in alpha:
    model2 = Lasso(alpha=a)
    model2.fit(X=features_train, y=labels_train)
    y_test2 = model2.predict(features_test)
    err_by_alpha_depend2.append(sklearn.metrics.mean_squared_error(labels_test, y_test2));


# In[ ]:


ds.columns[7]


# In[ ]:


from sklearn.linear_model import Ridge
model2 = Ridge(alpha=0.01)
model2.fit(X=features_train, y=labels_train)
#pd.Series(model2.coef_).plot()

import pickle
f = open('model.bin', 'wb')
pickle.dump(model2, f)
f.close()


# In[ ]:


model2.coef_


# In[ ]:


plt.title("Lasso")
plt.xlabel("Alpha")
plt.ylabel("MSE")
plt.plot(alpha, err_by_alpha_depend2, "r");


# In[ ]:


min_err2 = min(err_by_alpha_depend2)
index_min_err2 = err_by_alpha_depend2.index(min_err2)
print("Best MSE = {:.4f} with Alpha = {:.7f}".format(min_err2, alpha[index_min_err2]))


# In[ ]:


err_by_alpha_depend3 = []

positive_alpha = [a for a in alpha if (a >= 0)] # Accomplishment for this method

for a in positive_alpha:
    model3 = sklearn.linear_model.SGDRegressor(alpha=a, penalty="l1", tol=1e-3)
    model3.fit(X=features_train, y=labels_train)
    y_test3 = model3.predict(features_test)
    err_by_alpha_depend3.append(sklearn.metrics.mean_squared_error(labels_test, y_test3))

        


# In[ ]:


len(positive_alpha)


# In[ ]:


len(err_by_alpha_depend3)


# In[ ]:


plt.title("Stoch Gradient Descent Regression")
plt.xlabel("Alpha")
plt.ylabel("MSE")
plt.plot(positive_alpha, err_by_alpha_depend3, "r")


# In[ ]:


min_err3 = min(err_by_alpha_depend3)
index_min_err3 = err_by_alpha_depend3.index(min_err3)
print("Best MSE = {:.4f} with Alpha = {:.7f}".format(min_err3, alpha[index_min_err3]))


# # bad attempt 

# In[ ]:


err_by_alpha_depend3


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




