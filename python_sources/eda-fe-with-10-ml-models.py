#!/usr/bin/env python
# coding: utf-8

# # Used Car Data Price Prediction
# 
# Build of the 10+ most popular ML regression models to predict the price of used Car in Indian Market.
# 
# 
# 

# <a class="anchor" id="0.1"></a>
# 
# ## Table of Contents
# 
# 1. [Import libraries & dataset](#1)
# 1. [EDA](#2)
# 1. [Preparing to modeling](#3)
# 1. [ML models](#4)
#     -  [Linear Regression](#4.1)
#     -  [Ridge Regression](#4.2)
#     -  [K Neighbors Regressor](#4.3)
#     -  [SVR](#4.4)
#     -  [Stochastic Gradient Descent](#4.5)
#     -  [Decision Tree Regressor](#4.6)
#     -  [Random Forest](#4.7)
#     -  [Gradient Boosting](#4.8)
#     -  [XG Boost](#4.9)
#     -  [ExtraTreesRegressor](#4.10)
#     -  [VotingRegressor](#4.11)

# ## 1. Import libraries & dataset <a class="anchor" id="1"></a>
# 
# [Back to Table of Contents](#0.1)

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


car_details = pd.read_csv('/kaggle/input/vehicle-dataset-from-cardekho/CAR DETAILS FROM CAR DEKHO.csv')
car_details.head()


# In[ ]:


car_details.info()


# ## 2. EDA <a class="anchor" id="2"></a>
# 
# [Back to Table of Contents](#0.1)

# In[ ]:


import pandas_profiling as pp
pp.ProfileReport(car_details)


# In[ ]:


car_details.drop_duplicates(inplace = True)
car_details1 = car_details.reset_index()
len(car_details)


# In[ ]:


car_details1.describe()


# In[ ]:


plt.plot(car_details1.km_driven)


# In[ ]:


car_details1['name'].value_counts()


# Due to many models with very less data, I will consider cars on the basis of their Company & car name irrespective of the model variation.

# In[ ]:


#Taking company name and parent model name, not focused on exact model type
for i in range(len(car_details1)):
    car_details1.loc[i,'name_model'] = ' '.join(car_details1.loc[i,'name'].split()[:2]) #split and join the string


# In[ ]:


y = pd.DataFrame(car_details1.name_model.value_counts())


# We will consider models with 5 or more entries.

# In[ ]:


y = y[y.name_model > 4]


# In[ ]:


y.name_model.tail(10)


# In[ ]:


x = pd.DataFrame()
x['name_model'] = y.index


# In[ ]:


x.head()


# In[ ]:


len(x)


# In[ ]:


car_details1.info()


# In[ ]:


car_details2 = pd.merge(car_details1, x, on=['name_model'])


# In[ ]:


car_details2.info()


# In[ ]:


# Determination categorical features
numerics = ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']
categorical_columns = []
features = car_details2.columns.values.tolist()
for col in features:
    if car_details2[col].dtype in numerics: 
        continue
    categorical_columns.append(col)


# In[ ]:


categorical_columns = categorical_columns[1:-1]
categorical_columns


# In[ ]:


#one-hot encoding
car_details3 = pd.get_dummies(car_details2, columns=categorical_columns)


# In[ ]:


car_details4 = car_details3.iloc[:,2:]


# In[ ]:


car_details4.columns


# In[ ]:


car_details4.info()


# In[ ]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(list(car_details4.name_model))
car_details4.name_model = le.transform(list(car_details4.name_model))


# In[ ]:


car_details4.info()


# In[ ]:


car_details4.describe()


# In[ ]:


car_details4.corr()


# In[ ]:


#For models from Sklearn
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
train = pd.DataFrame(scaler.fit_transform(car_details4), columns = car_details4.columns)


# ## 3. Preparing to modeling <a class="anchor" id="3"></a>
# 
# [Back to Table of Contents](#0.1)

# In[ ]:


target = train.selling_price
features = train.drop(columns = ['selling_price'])


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, target, stratify = features.name_model)


# ### 4.1 Linear Regression <a class="anchor" id="4.1"></a>
# 
# [Back to Table of Contents](#0.1)

# In[ ]:


# Linear Regression
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X_train, y_train)
linreg.score(X_test,y_test)


# In[ ]:


from sklearn.model_selection import cross_val_score
np.mean(cross_val_score(LinearRegression(), X_train, y_train, cv=10))


# ### 4.2 Ridge Regression <a class="anchor" id="4.2"></a>
# 
# [Back to Table of Contents](#0.1)

# In[ ]:


from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
param_grid = {'alpha': np.logspace(-3, 3, 13)}


# In[ ]:


grid = GridSearchCV(Ridge(), param_grid, cv=10, return_train_score=True, iid=False)
grid.fit(X_train, y_train)


# In[ ]:


grid.score(X_test, y_test)


# In[ ]:


np.mean(cross_val_score(Ridge(), X_train, y_train, cv=10))


# ### 4.3 K Neighbors Regressor <a class="anchor" id="4.3"></a>
# 
# [Back to Table of Contents](#0.1)

# In[ ]:


from sklearn.neighbors import KNeighborsRegressor
neighbors = range(1, 30, 2)

training_scores = []
test_scores = []
for n_neighbors in neighbors:
    knn = KNeighborsRegressor(n_neighbors=n_neighbors).fit(X_train, y_train)
    training_scores.append(knn.score(X_train, y_train))
    test_scores.append(knn.score(X_test, y_test))


# In[ ]:


plt.figure()
plt.plot(neighbors, training_scores, label="training scores")
plt.plot(neighbors, test_scores, label="test scores")
plt.ylabel("accuracy")
plt.xlabel("n_neighbors")
plt.legend()


# In[ ]:


knn = KNeighborsRegressor(n_neighbors=7)
score = cross_val_score(knn, X_train, y_train, cv=10)
print(f"best cross-validation score: {np.max(score):.3}")

knn.fit(X_train, y_train)
print(f"test-set score: {knn.score(X_test, y_test):.3f}")


# ### 4.4 SVR <a class="anchor" id="4.4"></a>
# 
# [Back to Table of Contents](#0.1)

# In[ ]:


from sklearn.svm import SVR
svr = SVR()
svr.fit(X_train, y_train)
print(f"test-set score: {svr.score(X_test, y_test):.3f}")


# In[ ]:


svr1 = SVR(kernel='poly')
svr1.fit(X_train, y_train)
print(f"test-set score: {svr1.score(X_test, y_test):.3f}")


# ### 4.5 Stochastic Gradient Descent <a class="anchor" id="4.5"></a>
# 
# [Back to Table of Contents](#0.1)

# In[ ]:


from sklearn.linear_model import SGDRegressor
sgd = SGDRegressor()
sgd.fit(X_train, y_train)
print(f"test-set score: {sgd.score(X_test, y_test):.3f}")


# ### 4.6 Decision Tree Regressor <a class="anchor" id="4.6"></a>
# 
# [Back to Table of Contents](#0.1)

# In[ ]:


from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor()
dtr.fit(X_train, y_train)
print(f"test-set score: {dtr.score(X_test, y_test):.3f}")


# ### 4.7 Random Forest <a class="anchor" id="4.7"></a>
# 
# [Back to Table of Contents](#0.1)

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor()
rfr.fit(X_train, y_train)
print(f"test-set score: {rfr.score(X_test, y_test):.3f}")


# ### 4.8 Gradient Boosting <a class="anchor" id="4.8"></a>
# 
# [Back to Table of Contents](#0.1)

# In[ ]:


from sklearn.ensemble import GradientBoostingRegressor
gbr = GradientBoostingRegressor()
gbr.fit(X_train, y_train)
print(f"test-set score: {gbr.score(X_test, y_test):.3f}")


# ### 4.9 XG Boost <a class="anchor" id="4.9"></a>
# 
# [Back to Table of Contents](#0.1)

# In[ ]:


from xgboost import XGBRegressor
xgb = XGBRegressor()
xgb.fit(X_train,y_train)
print(f"test-set score: {xgb.score(X_test, y_test):.3f}")


# ### 4.10 Extra Tree Regressor <a class="anchor" id="5.7"></a>
# 
# [Back to Table of Contents](#0.1)

# In[ ]:


from sklearn.ensemble import ExtraTreesRegressor
etr = ExtraTreesRegressor()
etr.fit(X_train, y_train)
print(f"test-set score: {etr.score(X_test, y_test):.3f}")


# ### 4.11 Voting Regressor <a class="anchor" id="4.11"></a>
# 
# [Back to Table of Contents](#0.1)

# In[ ]:


from sklearn.ensemble import VotingRegressor
vr = VotingRegressor(estimators=[('rfr', rfr), ('gbr', gbr), ('xgb', xgb)])
vr.fit(X_train,y_train)
print(f"test-set score: {vr.score(X_test, y_test):.3f}")


# In[ ]:


from sklearn.metrics import mean_squared_error

for clf in (rfr, gbr, xgb, vr):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__,  mean_squared_error(y_test, y_pred))

