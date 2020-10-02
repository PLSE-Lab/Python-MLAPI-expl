#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import os


# In[ ]:


DATA_DIR = '/kaggle/input/eval-lab-1-f464-v2/'

train_file = 'train.csv'
test_file = 'test.csv'


# In[ ]:


train = pd.read_csv(os.path.join(DATA_DIR, train_file))


# In[ ]:


train.head()


# In[ ]:


train.info()


# In[ ]:


print(train.isnull().sum(axis = 0))


# In[ ]:


numerical_features = train.columns[1:10].append(train.columns[11:13])
categorical_features = train.columns[10]
rating = train.columns[13]
print(numerical_features)
print(categorical_features)
print(rating)


# In[ ]:


print(train[numerical_features].mean())
print(train[categorical_features].mode())


# In[ ]:


train[numerical_features] = train[numerical_features].fillna(train[numerical_features].mean())
train[categorical_features] = train[categorical_features].fillna(train[categorical_features].mode())
train = pd.get_dummies(data = train, columns = ['type'])


# In[ ]:


train.head()


# In[ ]:


train.isnull().any().any()


# In[ ]:


print(train.isnull().sum(axis = 0))


# In[ ]:


sns.relplot(x = 'feature5', y = 'rating', data = train)


# feature5
# feature11

# In[ ]:


train.head()


# In[ ]:


features = train.columns[1:12].append(train.columns[13:])
x = train[features]
y = train[rating]


# In[ ]:


x['type_new'].unique()


# In[ ]:


print(x.isnull().any().any())
x.head()


# In[ ]:


y.head()


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.005)


# In[ ]:


x_train.head()


# In[ ]:


x_train.shape


# In[ ]:


scale = RobustScaler()

scale_features = features[:-2]
x_train[scale_features] = scale.fit_transform(x_train[scale_features])
x_test[scale_features] = scale.transform(x_test[scale_features])


# In[ ]:


scale_features


# In[ ]:


print(x_train.head())
print(x_test.head())


# In[ ]:


from sklearn.linear_model import LogisticRegression

model = LogisticRegression(penalty = 'none', solver = 'newton-cg', multi_class = 'multinomial', max_iter = 5000)


# In[ ]:


model.fit(x_train.values, y_train.values)


# In[ ]:


print(model.predict(x_test[:10]))
print(y_test.values[:10])


# In[ ]:


model.score(x_train.values, y_train.values)


# In[ ]:


model.score(x_test.values, y_test.values)


# In[ ]:


from sklearn.metrics import mean_squared_error

pred = model.predict(x_test)

print(mean_squared_error(y_test, pred) ** 0.5)


# In[ ]:


test = pd.read_csv(os.path.join(DATA_DIR, test_file))


# In[ ]:


test[numerical_features] = test[numerical_features].fillna(test[numerical_features].mean())
test[categorical_features] = test[categorical_features].fillna(test[categorical_features].mode())
test = pd.get_dummies(data = test, columns = ['type'])


# In[ ]:


print(test.isnull().any().any())
test.head()


# In[ ]:


x_sub = test[features]
x_sub.head()


# In[ ]:


x_sub[scale_features] = scale.transform(x_sub[scale_features])


# In[ ]:


x_sub.head()


# In[ ]:


pred_sub = model.predict(x_sub.values)
print(pred_sub.shape)


# In[ ]:


test['id'].values


# In[ ]:


sub = pd.DataFrame(data = pred_sub, columns = ['rating'], index = test['id'])


# In[ ]:


sub.head()


# In[ ]:


sub.isnull().any().any()


# In[ ]:


sub.to_csv('sub2.csv')


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier(n_neighbors = 10)


# In[ ]:


knn_model.fit(x_train, y_train)


# In[ ]:


knn_model.score(x_test, y_test)


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Initialize and train
dt_model = DecisionTreeClassifier()
rf_model = RandomForestClassifier()


# In[ ]:


dt_model.fit(x_train, y_train)


# In[ ]:


print(dt_model.predict(x_test[:10]))
print(y_test.values[:10])


# In[ ]:


pred_dt = dt_model.predict(x_test)


# In[ ]:


from sklearn.metrics import accuracy_score

print(accuracy_score(y_test.values, pred_dt))


# In[ ]:


rf_model.fit(x_train, y_train)


# In[ ]:


pred_rf = rf_model.predict(x_test)
print(accuracy_score(y_test.values, pred_rf))


# In[ ]:


print(pred_rf[:10])


# In[ ]:


print(pd.get_dummies(y_train).sum(0))


# In[ ]:


print(y_train.shape)
sorted(y_train.unique())


# In[ ]:


from sklearn.utils import class_weight

cw = class_weight.compute_class_weight('balanced', sorted(y_train.unique()), y_train)
print(cw)


# In[ ]:


cw_dict = {i : cw[i] for i in range(7)}
print(cw_dict)


# In[ ]:


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.ensemble import RandomForestClassifier

#TODO
rf_model_2 = RandomForestClassifier(class_weight = cw_dict)        #Initialize the classifier object

parameters = {'n_estimators':[100, 1000]}    #Dictionary of parameters

scorer = make_scorer(accuracy_score)         #Initialize the scorer using make_scorer

grid_obj = GridSearchCV(rf_model_2, parameters, scoring = scorer)         #Initialize a GridSearchCV object with above parameters,scorer and classifier

grid_fit = grid_obj.fit(x_train, y_train)        #Fit the gridsearch object with X_train,y_train

best_rf_model = grid_fit.best_estimator_         #Get the best estimator. For this, check documentation of GridSearchCV object

unoptimized_predictions = (rf_model_2.fit(x_train, y_train)).predict(x_test)      #Using the unoptimized classifiers, generate predictions
optimized_predictions = best_rf_model.predict(x_test)        #Same, but use the best estimator

acc_unop = accuracy_score(y_test, unoptimized_predictions)*100       #Calculate accuracy for unoptimized model
acc_op = accuracy_score(y_test, optimized_predictions)*100         #Calculate accuracy for optimized model

print("Accuracy score on unoptimized model:{}".format(acc_unop))
print("Accuracy score on optimized model:{}".format(acc_op))


# In[ ]:





# In[ ]:


print(x_test.head())


# In[ ]:


print(y_test.head())


# In[ ]:


x_sub.head()


# In[ ]:


pred_rf_sub = best_rf_model.predict(x_sub.values)
pred_last = rf_model_2.predict(x_sub.values)
print(pred_rf_sub[:10])


# In[ ]:


print(pred_rf_sub[:100])


# In[ ]:


sub_rf = pd.DataFrame(data = pred_rf_sub, columns = ['rating'], index = test['id'])
sub_rf_l = pd.DataFrame(data = pred_last, columns = ['rating'], index = test['id'])


# In[ ]:


sub_rf.to_csv('sub_rf_last.csv')


# In[ ]:





# In[ ]:




