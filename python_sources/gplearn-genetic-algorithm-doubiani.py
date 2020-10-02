#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"></ul></div>

# In[ ]:


# ignore warnings 
import warnings
warnings.filterwarnings('ignore')

# import general packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 
get_ipython().run_line_magic('matplotlib', 'inline')
color = sns.color_palette()
from gplearn.genetic import SymbolicRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error,make_scorer
# algorithms
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor


# modeling helper functions
from sklearn.model_selection import GridSearchCV , KFold , cross_val_score, train_test_split


# In[ ]:


fnc_df = pd.read_csv("C:/OpenClassRooms/projet 8/trends-assessment-prediction/fnc.csv")
loading_df = pd.read_csv("C:/OpenClassRooms/projet 8/trends-assessment-prediction/loading.csv")

fnc_features, loading_features = list(fnc_df.columns[1:]), list(loading_df.columns[1:])
df = fnc_df.merge(loading_df, on="Id")


labels_df = pd.read_csv("C:/OpenClassRooms/projet 8/trends-assessment-prediction/train_scores.csv")
labels_df["is_train"] = True

df = df.merge(labels_df, on="Id", how="left")

test_df = df[df["is_train"] != True].copy()
df = df[df["is_train"] == True].copy()


# In[ ]:


target_cols = ['age', 'domain1_var1', 'domain1_var2', 'domain2_var1', 'domain2_var2']
df.drop(['is_train'], axis=1, inplace=True)
test_df = test_df.drop(target_cols + ['is_train'], axis=1)

# Giving less importance to FNC features since they are easier to overfit due to high dimensionality.
FNC_SCALE = 1/500

df[fnc_features] *= FNC_SCALE
test_df[fnc_features] *= FNC_SCALE


# In[ ]:


features = loading_features + fnc_features


# In[ ]:


df


# In[ ]:


test_df


# In[ ]:


#No Missing Values
x = df['age']
plt.figure(figsize=(8,6))
plt.hist(x, bins=25, color='g')
plt.xlabel('Age')
plt.ylabel('Number of patients')
plt.title('Age distribution of patients', fontsize = 16)
plt.show()


# In[ ]:


x = df['domain1_var1']
plt.figure(figsize=(8,6))
plt.hist(x, bins=25, color='c')
plt.xlabel('domain1_var1')
plt.ylabel('Number of patients')
plt.title('domain1_var1 distribution', fontsize = 16)
plt.show()


# In[ ]:


#We can see that domain1_var1 distribution is approximately normal. So, we will fill the missing values with mean.
df['domain1_var1'].fillna(df['domain1_var1'].mean(), inplace=True)


# In[ ]:


x = df['domain1_var2']
plt.figure(figsize=(8,6))
plt.hist(x, bins=25, color='pink')
plt.xlabel('domain1_var2')
plt.ylabel('Number of patients')
plt.title('domain1_var2 distribution', fontsize = 16)
plt.show()


# In[ ]:


#domain1_var2 is skewed. So, we will fill missing values with median.
df['domain1_var2'].fillna(df['domain1_var2'].median(), inplace=True)


# In[ ]:


x = df['domain2_var1']
plt.figure(figsize=(8,6))
plt.hist(x, bins=25, color='y')
plt.xlabel('domain2_var1')
plt.ylabel('Number of patients')
plt.title('domain2_var1 distribution', fontsize = 16)
plt.show()


# In[ ]:


#domain2_var1 is approximately normal. So, we will fill missing values with mean.
df['domain2_var1'].fillna(df['domain2_var1'].mean(), inplace=True)


# In[ ]:


x = df['domain2_var2']
plt.figure(figsize=(8,6))
plt.hist(x, bins=25, color='r')
plt.xlabel('domain2_var2')
plt.ylabel('Number of patients')
plt.title('domain2_var2 distribution', fontsize = 16)
plt.show()


# In[ ]:


#domain2_var2 is approximately normal. So, we will fill missing values with mean.
df['domain2_var2'].fillna(df['domain2_var2'].mean(), inplace=True)


# In[ ]:


df


# In[ ]:


X_train, X_test, y_train_age, y_test_age = train_test_split(test_df, df['age'],
                                                    train_size=0.75, test_size=0.25, random_state=42)


# In[ ]:


#X_train.reset_index(drop=True,inplace=True)
X_train


# In[ ]:


X_test


# In[ ]:


y_train_age


# In[ ]:


y_test_age


# In[ ]:


#symbRegage = SymbolicRegressor(verbose=1, generations=300, population_size=5000,function_set = ('add', 'sub', 'mul', 'div','sqrt','log','inv','max','min','sin','cos','tan'))
symbRegage = SymbolicRegressor(population_size=5000,
                       generations=50,tournament_size=50,stopping_criteria=0.01,function_set=('add', 'sub', 'mul', 'div','sqrt','log','neg','inv','max','min','sin','cos','tan'),
                       p_crossover=0.7, p_subtree_mutation=0.1,
                       p_hoist_mutation=0.05, p_point_mutation=0.1,
                       max_samples=0.9, verbose=1,
                       parsimony_coefficient=0.01,random_state=42,n_jobs=2)
symbRegage.fit(X_train, y_train_age)
y_pred_age_test = symbRegage.predict(X_test)
y_pred_age_train = symbRegage.predict(X_train)
print('SymbRegage MAE test', mean_absolute_error(y_test_age, y_pred_age_test))
print('SymbRegage MAE train', mean_absolute_error(y_train_age, y_pred_age_train))


# In[ ]:


parameters = {'function_set': [('add', 'sub', 'mul', 'div','sqrt','log','neg','inv','max','min','sin','cos','tan')],
             'init_depth': [(2, 6),(3,7)],
             'max_samples': [1.0,0.9],
             'p_crossover': [1,0.5],
             'p_hoist_mutation': [0.01,0.05],
             'p_point_mutation': [0.01,0.02],
             'random_state': [0],
             'tournament_size': [20,50],
             'verbose': [1],
             'population_size': [5000],
             'parsimony_coefficient': ["auto"],
             'generations': [50],
             'warm_start': [False]}


# In[ ]:


#This part sets up the symbolic regressor
clf = GridSearchCV(symbRegage , parameters, cv=5,n_jobs = -1, verbose = 1)
#This part runs it on our data
clf.fit(X_train, y_train_age)


# In[ ]:


clf.best_params_


# In[ ]:


print(clf.best_estimator_._program)
clf.best_estimator_.score(X_train,y_train_age)


# scores = []
# #best_svr = SVR(kernel='rbf')
# cv = KFold(n_splits=10, random_state=42, shuffle=False)
# for train_index, test_index in cv.split(X_train):
#     print("Train Index: ", train_index, "\n")
#     print("Test Index: ", test_index)
#     X_train_fold, X_test_fold, y_train_fold, y_test_fold = X_train[train_index], X_train[test_index], y_train_age[train_index], y_test_age[test_index]
#     symbRegage.fit(X_train_fold, y_train_fold)
#     scores.append(mean_absolute_error(X_test_fold, y_test_fold))

# In[ ]:


mae_score = make_scorer(mean_absolute_error)
cross_val_score(symbRegage, X_train, y_train_age, cv=10, scoring = mae_score)


# In[ ]:


print(symbRegage._program)


# In[ ]:


X_train, X_test, y_train_d1v1, y_test_d1v1 = train_test_split(test_df, df['domain1_var1'],
                                                    train_size=0.75, test_size=0.25, random_state=42)


# In[ ]:


symbRegd1v1 = SymbolicRegressor(verbose=1, generations=300, population_size=5000,function_set = ('add', 'sub', 'mul', 'div','sqrt','log','inv','max','min','sin','cos','tan'))
symbRegd1v1.fit(X_train, y_train_d1v1)
y_pred_d1v1_test = symbRegd1v1.predict(X_test)
y_pred_d1v1_train = symbRegd1v1.predict(X_train)
print('SymbRegd1v1 MAE test', mean_absolute_error(y_test_d1v1, y_pred_d1v1_test))
print('SymbRegd1v1 MAE train', mean_absolute_error(y_train_d1v1, y_pred_d1v1_train))


# In[ ]:


print(symbRegd1v1._program)


# In[ ]:


X_train, X_test, y_train_d1v2, y_test_d1v2 = train_test_split(test_df, df['domain1_var2'],
                                                    train_size=0.75, test_size=0.25, random_state=42)


# In[ ]:


symbRegd1v2 = SymbolicRegressor(verbose=1, generations=300, population_size=5000,function_set = ('add', 'sub', 'mul', 'div','sqrt','log','inv','max','min','sin','cos','tan'))
symbRegd1v2.fit(X_train, y_train_d1v2)
y_pred_d1v2_test = symbRegd1v2.predict(X_test)
y_pred_d1v2_train = symbRegd1v2.predict(X_train)
print('SymbRegd1v2 MAE test', mean_absolute_error(y_test_d1v2, y_pred_d1v2_test))
print('SymbRegd1v2 MAE train', mean_absolute_error(y_train_d1v2, y_pred_d1v2_train))


# In[ ]:


print(symbRegd1v2._program)


# In[ ]:


X_train, X_test, y_train_d2v1, y_test_d2v1 = train_test_split(test_df, df['domain2_var1'],
                                                    train_size=0.75, test_size=0.25, random_state=42)


# In[ ]:


symbRegd2v1 = SymbolicRegressor(verbose=1, generations=300, population_size=5000,function_set = ('add', 'sub', 'mul', 'div','sqrt','log','inv','max','min','sin','cos','tan'))
symbRegd2v1.fit(X_train, y_train_d2v1)
y_pred_d2v1_test = symbRegd2v1.predict(X_test)
y_pred_d2v1_train = symbRegd2v1.predict(X_train)
print('SymbRegd2v1 MAE test', mean_absolute_error(y_test_d2v1, y_pred_d2v1_test))
print('SymbRegd2v1 MAE test', mean_absolute_error(y_train_d2v1, y_pred_d2v1_train))


# In[ ]:


print(symbRegd2v1._program)


# In[ ]:


X_train, X_test, y_train_d2v2, y_test_d2v2 = train_test_split(test_df, df['domain2_var2'],
                                                    train_size=0.75, test_size=0.25, random_state=42)


# In[ ]:


symbRegd2v2 = SymbolicRegressor(verbose=1, generations=300, population_size=5000,function_set = ('add', 'sub', 'mul', 'div','sqrt','log','inv','max','min','sin','cos','tan'))
symbRegd2v2.fit(X_train, y_train_d2v2)
y_pred_d2v2_test = symbRegd2v2.predict(X_test)
y_pred_d2v2_train = symbRegd2v2.predict(X_train)
print('SymbRegd2v2 MAE test', mean_absolute_error(y_test_d2v2, y_pred_d2v2_test))
print('SymbRegd2v2 MAE train', mean_absolute_error(y_train_d2v2, y_pred_d2v2_train))


# In[ ]:


print(symbRegd2v2._program)


# In[ ]:


X_train


# In[ ]:


test_df


# In[ ]:


df


# In[ ]:


y_pred_age_tot_test = symbRegage.predict(test_df)
y_pred_age_tot_train = symbRegage.predict(df.drop(columns=['age','domain1_var1','domain1_var2','domain2_var1','domain2_var2']))
print('SymbRegage MAE test', mean_absolute_error(df['age'], y_pred_age_tot_test))
print('SymbRegage MAE train', mean_absolute_error(df['age'], y_pred_age_tot_train))


# In[ ]:


y_pred_d1v1_tot_test = symbRegd1v1.predict(test_df)
y_pred_d1v1_tot_train = symbRegd1v1.predict(df.drop(columns=['age','domain1_var1','domain1_var2','domain2_var1','domain2_var2']))
print('symbRegd1v1 MAE test', mean_absolute_error(df['domain1_var1'], y_pred_d1v1_tot_test))
print('symbRegd1v1 MAE train', mean_absolute_error(df['domain1_var1'], y_pred_d1v1_tot_train))


# In[ ]:


y_pred_d1v2_tot_test = symbRegd1v2.predict(test_df)
y_pred_d1v2_tot_train = symbRegd1v2.predict(df.drop(columns=['age','domain1_var1','domain1_var2','domain2_var1','domain2_var2']))
print('symbRegd1v2 MAE test', mean_absolute_error(df['domain1_var2'], y_pred_d1v2_tot_test))
print('symbRegd1v2 MAE train', mean_absolute_error(df['domain1_var2'], y_pred_d1v2_tot_train))


# In[ ]:


y_pred_d2v1_tot_test = symbRegd2v1.predict(test_df)
y_pred_d2v1_tot_train = symbRegd2v1.predict(df.drop(columns=['age','domain1_var1','domain1_var2','domain2_var1','domain2_var2']))
print('symbRegd2v1 MAE test', mean_absolute_error(df['domain2_var1'], y_pred_d2v1_tot_test))
print('symbRegd2v1 MAE train', mean_absolute_error(df['domain2_var1'], y_pred_d2v1_tot_train))


# In[ ]:


y_pred_d2v2_tot_test = symbRegd2v2.predict(test_df)
y_pred_d2v2_tot_train = symbRegd2v2.predict(df.drop(columns=['age','domain1_var1','domain1_var2','domain2_var1','domain2_var2']))
print('symbRegd2v2 MAE test', mean_absolute_error(df['domain2_var2'], y_pred_d2v2_tot_test))
print('symbRegd2v2 MAE train', mean_absolute_error(df['domain2_var2'], y_pred_d2v2_tot_train))


# In[ ]:


d = {'Id' : test_df['Id'],'age': y_pred_age_tot_test , 'domain1_var1' : y_pred_d1v1_tot_test, 'domain1_var2' : y_pred_d1v2_tot_test, 'domain2_var1' : y_pred_d2v1_tot_test, 'domain2_var2' : y_pred_d2v2_tot_test }
predictions = pd.DataFrame(data=d)
predictions


# In[ ]:


test_df


# In[ ]:


sub_df = pd.melt(predictions[["Id", "age", "domain1_var1", "domain1_var2", "domain2_var1", "domain2_var2"]], id_vars=["Id"], value_name="Predicted")
sub_df["Id"] = sub_df["Id"].astype("str") + "_" +  sub_df["variable"].astype("str")

sub_df = sub_df.drop("variable", axis=1).sort_values("Id")
assert sub_df.shape[0] == predictions.shape[0]*5

sub_df.to_csv("submission2.csv", index=False)


# In[ ]:




