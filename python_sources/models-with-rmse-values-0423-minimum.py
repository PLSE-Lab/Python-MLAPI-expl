#!/usr/bin/env python
# coding: utf-8

# In this notebook, the following models are used.
# 1. Linear Regression
# 2. Decision Tree
# 3. Random Forest
# 4. Lasso Regression
# 5. Ridge Regression
# 6. ElasticNet Regression
# 7. SVR
# 
# 
# 
# I have also used **Grid Search** for HyperParameter Tuning.
# 
# I have also done scaling of the dataset with **StandardScaler and MinMaxScaler**.
# 
# I have evaluated model thrice.
# 
# 1. Evaluating models without Scaling
# 2. Evaluating models with Standard Scaling
# 3. Evaluating models with MinMax Scaling
# 
# 
# The result of the lowest RMSE values are attached at the end in the form of DataFrame.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv("../input/graduate-admissions/Admission_Predict_Ver1.1.csv")


# In[ ]:


df.shape


# In[ ]:


df.isnull().sum()


# In[ ]:


df.head()


# In[ ]:


df.drop(['Serial No.'], axis = 1, inplace = True)


# Serial Number can be dropped here as it used for indexing and it shows no relationship with 'Chance of Admit '.

# In[ ]:


df.head()


# # DATA ANALYSIS

# In[ ]:


plt.figure(figsize=(25,10))
sns.heatmap(df.corr(), annot=True, linewidth=0.5, cmap='coolwarm')


# **This shows the correlation between several features.**
# 
# 1. Chance of Admit increases with GRE Score, TOEFL Score and CGPA.
# 2. Chance of Admit is also dependent on University Rating, SOP and LOR.Atlhough, the dependency is relatively less than GRE, TOEFL and CGPA.
# 3. Chance of Admit is moderately dependent on Research. 
# 
# **Correlation is positive here for majority of the features. This means as the Value for a certain feature increases, the Chance of Admit also increases. They are lineraly dependent on each other. Generally, when correlation is 0, it means that there is no relationship between the two features you are looking at.**
# 
# 1 shows high dependece. Values near to -1 shows inverse relationship.

# In[ ]:


sns.pairplot(df)


# This shows the relationship between all the features in the form of ScatterPlot.

# In[ ]:


x = df['Chance of Admit ']
sns.distplot(x , kde= True,rug = False, bins = 30)


# Most of the students have an Admit Chance of 0.6-0.8 when they fulfill certain criterias.

# In[ ]:


x = df['GRE Score']
sns.distplot(x , kde= True,rug = False, bins = 30)


# The GRE Scores are distributed across many Scores. They start at 290 and maximizes at around 310-325 and then decreases for majority of the students.

# In[ ]:


x = df['TOEFL Score']
sns.distplot(x , kde= True,rug = False, bins = 30)


# Maximum students have TOEFL Score between 100 and 110 with peak at 105

# In[ ]:


x = df['CGPA']
sns.distplot(x , kde= True,rug = False, bins = 30)


# Maximum of the students have a CPGA between 8 and 9 with peak around 8.5

# In[ ]:


sns.lineplot(x = 'GRE Score', y = 'CGPA', data = df)


# This shows a linear relationship between CPGA and GRE Score.

# In[ ]:


sns.lineplot(x = 'TOEFL Score', y = 'CGPA', data = df)


# This shows a linear relationship between CGPA and TOEFL Score.

# In[ ]:


sns.jointplot(x = 'GRE Score', y = 'CGPA', data=df)


# The students having higher CGPA also have high GRE Scores. Most of the students have a CGPA of 8.5 and GRE Score from the range of 310-330

# In[ ]:


sns.jointplot(x = 'TOEFL Score', y = 'CGPA', data=df)


# Students having good CGPA also have a good score in TOEFL.
# 

# In[ ]:


sns.jointplot(x = 'TOEFL Score', y = 'University Rating', data=df)


# The TOEFL scores are also a bit higher for the students from higher rated Universities. 

# In[ ]:


sns.jointplot(x = 'GRE Score', y = 'University Rating', data=df)


# GRE scores are much better for students from Higher rated Universities.

# In[ ]:


sns.relplot(x ='SOP', y ='Chance of Admit ', col = 'University Rating', data = df, estimator = None,palette = 'ch:r = -0.8, l = 0.95')


# Higher number of admits are there for SOP from Univesitites having rating 4 and 5. This gradually declines as ratings go down.

# In[ ]:


sns.relplot(x ='LOR ', y ='Chance of Admit ', col = 'University Rating', data = df, estimator = None,palette = 'ch:r = -0.8, l = 0.95')


# The higher the University Rating, the better is the LOR and the chances of admit are better.

# In[ ]:


sns.relplot(x ='Research', y ='Chance of Admit ', col = 'University Rating', data = df, estimator = None,palette = 'ch:r = -0.8, l = 0.95')


# As the University Rating increases so does the Research and this increases the Chance of Admit.

# # MODEL SELECTION
# 

# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn import linear_model
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error


from sklearn_pandas import DataFrameMapper
from numpy import asarray
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


# In[ ]:


X = df.drop(['Chance of Admit '], axis = 1)
X


# In[ ]:


y = df['Chance of Admit ']
y


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20,shuffle = False)


# **Shuffle = False means that values are not randomized and used as they are given in the .CSV file. The first 400 values are used for Training and the last 100 for testing/CV.** 
# 

# **1.) LINEAR REGRESSION WITHOUT SCALING**

# In[ ]:


model1 = LinearRegression()
model1.fit(X_train, y_train)

accuracy1 = model1.score(X_test,y_test)
print(accuracy1*100,'%')


# **1.a) RMSE VALUE FOR LINEAR REGRESSION WITHOUT SCALING**

# In[ ]:


y_pred1 = model1.predict(X_test)

val = mean_squared_error(y_test, y_pred1, squared=False)
val1 = str(round(val, 4))

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred1)))


# **2.) DECISION TREE WITHOUT SCALING**

# In[ ]:


model2 = DecisionTreeRegressor()
model2.fit(X_train, y_train)

accuracy2 = model2.score(X_test,y_test)
print(accuracy2*100,'%')


# **2.a) DECISION TREE RMSE WITHOUT SCALING**

# In[ ]:


y_pred2 = model2.predict(X_test)

val = mean_squared_error(y_test, y_pred2, squared=False)
val2 = str(round(val, 4))


print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred2)))


# **3.) GRID SEARCH FOR N_ESTIMATORS VALUE**

# In[ ]:


n_estimators = [10, 40, 70, 100, 130, 160, 190, 220, 250, 270]

RF = RandomForestRegressor()

parameters = {'n_estimators': [10, 40, 70, 100, 130, 160, 190, 220, 250, 270]}

RFR = GridSearchCV(RF, parameters,scoring='neg_mean_squared_error', cv=5)

RFR.fit(X_train, y_train)

RFR.best_params_


# **3.a) RANDOM FOREST WITHOUT SCALING**

# In[ ]:


model3 = RandomForestRegressor(n_estimators = 190)
model3.fit(X_train, y_train)

accuracy3 = model3.score(X_test,y_test)
print(accuracy3*100,'%')


# **3.b) RANDOM FOREST RMSE WITHOUT SCALING**

# In[ ]:


y_pred3 = model3.predict(X_test)

val = mean_squared_error(y_test, y_pred3, squared=False)
val3 = str(round(val, 4))


print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred3)))


# **4.) GRID SEARCH FOR ALPHA FOR LASSO REGRESSION**

# In[ ]:


lasso = Lasso()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]}

lasso_regressor = GridSearchCV(lasso, parameters, scoring='neg_mean_squared_error', cv = 100)

lasso_regressor.fit(X_train, y_train)

lasso_regressor.best_params_


# **4.b) LASSO REGRESSION WITHOUT SCALING**

# In[ ]:


model4 = linear_model.Lasso(alpha=.001)
model4.fit(X_train,y_train)

accuracy4 = model4.score(X_test,y_test)
print(accuracy4*100,'%')


# **4.c) RMSE FOR LASSO REGRESSION**

# In[ ]:


y_pred4 = model4.predict(X_test)

val= mean_squared_error(y_test, y_pred4, squared=False)
val4 = str(round(val, 4))


print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred4)))


# **5.a) GRID SEARCH FOR ALPHA FOR RIDGE REGRESSION**

# In[ ]:


alpha = [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]

ridge = Ridge()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]}

ridge_regressor = GridSearchCV(ridge, parameters,scoring='neg_mean_squared_error', cv=100)

ridge_regressor.fit(X_train, y_train)
ridge_regressor.best_params_


# **5.b) RIDGE REGRESSION WITHOUT SCALING**

# In[ ]:


model5 = linear_model.Ridge(alpha=1)
model5.fit(X_train,y_train)

accuracy5 = model5.score(X_test,y_test)
print(accuracy5*100,'%')


# **5.c) RMSE FOR RIDGE REGRESSION**

# In[ ]:


y_pred5 = model5.predict(X_test)

val = mean_squared_error(y_test, y_pred5, squared=False)
val5 = str(round(val, 4))


print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred5)))


# **6.A) GRID SEARCH FOR ELASTICNET REGRESSION**

# In[ ]:


Elasticnet = ElasticNet()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]}

en_regressor = GridSearchCV(Elasticnet, parameters, scoring='neg_mean_squared_error', cv = 100)

en_regressor.fit(X_train, y_train)
en_regressor.best_params_


# **6.B) ELASTICNET REGRESSION WITHOUT SCALING**

# In[ ]:


model6 = linear_model.ElasticNet(alpha=0.001)
model6.fit(X_train,y_train)

accuracy6 = model6.score(X_test,y_test)
print(accuracy6*100,'%')


# **6.C) RMSE VALUE FOR ELASTICNET REGRESSION**

# In[ ]:


y_pred6 = model6.predict(X_test)

val = mean_squared_error(y_test, y_pred6, squared=False)
val6 = str(round(val, 4))


print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred6)))


# In[ ]:


data1 = [['Linear Regression ',val1],['Decision Tree',val2],['Random Forest',val3],['Lasso Regression',val4],['Ridge Regression',val5],['ElasticNet Regression',val6]]
d1 = pd.DataFrame(data1,columns = ['Without Scaling Models ','RMSE Error'])
Half1RMSE = d1.copy()
Half1RMSE


# # APPLYING STANDARD SCALER TO DATASET

# In[ ]:


mapper = DataFrameMapper([(df.columns, StandardScaler())])
scaled_features = mapper.fit_transform(df.copy(), 4)
data = pd.DataFrame(scaled_features, index=df.index, columns=df.columns)


# In[ ]:


data.head()


# In[ ]:


x = data.drop(['Chance of Admit '], axis = 1)
x


# In[ ]:


Y = data['Chance of Admit ']
Y


# In[ ]:


x_train, x_test, Y_train, Y_test = train_test_split(x, Y, test_size = 0.20,shuffle = False)


# **Shuffle = False means that values are not randomized and used as they are given in the .CSV file. The first 400 values are used for Training and the last 100 for testing/CV.** 

# **7.A) LINEAR REGRESSION WITH SCALING**

# In[ ]:


model7 = LinearRegression()
model7.fit(x_train, Y_train)

accuracy7 = model7.score(x_test,Y_test)
print(accuracy7*100,'%')


# **7.B) RMSE VALUE FOR LINEAR REGRESSION WITH SCALING**

# In[ ]:


y_pred7 = model7.predict(x_test)

val = mean_squared_error(Y_test, y_pred7, squared=False)
val7 = str(round(val, 4))


print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, y_pred7)))


# **8.A) DECISION TREE WITH SCALING**

# In[ ]:


model8 = DecisionTreeRegressor()
model8.fit(x_train, Y_train)

accuracy8 = model8.score(x_test,Y_test)
print(accuracy8*100,'%')


# **8.B) RMSE VALUE FOR DECISION TREE WITH SCALING**

# In[ ]:


y_pred8 = model8.predict(x_test)

val = mean_squared_error(Y_test, y_pred8, squared=False)
val8 = str(round(val, 4))

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, y_pred8)))


# **9.A) GRID SEARCH FOR N_ESTIMATORS FOR RANDOM FOREST WITH SCALING**

# In[ ]:


n_estimators = [10, 40, 70, 100, 130, 160, 190, 220, 250, 270]

rf = RandomForestRegressor()

parameters = {'n_estimators': [10, 40, 70, 100, 130, 160, 190, 220, 250, 270]}

rfr = GridSearchCV(rf, parameters,scoring='neg_mean_squared_error', cv=10)

rfr.fit(x_train, Y_train)

rfr.best_params_


# **9.B) RANDOM FOREST WITH SCALING**

# In[ ]:


model9 = RandomForestRegressor(n_estimators = 220)
model9.fit(x_train, Y_train)

accuracy9 = model9.score(x_test,Y_test)
print(accuracy9*100,'%')


# **9.C) RMSE VALUE FOR RANDOM FOREST WITH SCALING**

# In[ ]:


y_pred9 = model9.predict(x_test)

val = mean_squared_error(Y_test, y_pred9, squared=False)
val9 = str(round(val, 4))

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, y_pred9)))


# **10.A) GRID SEARCH FOR ALPHA FOR LASSO REGRESSION WITH SCALING**

# In[ ]:


L = Lasso()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]}

LR = GridSearchCV(L, parameters, scoring='neg_mean_squared_error', cv = 100)

LR.fit(x_train, Y_train)
LR.best_params_


# **10.B) LASSO REGRESSION WITH SCALING**

# In[ ]:


model10 = linear_model.Lasso(alpha=.01)
model10.fit(x_train,Y_train)

accuracy10 = model10.score(x_test,Y_test)
print(accuracy10*100,'%')


# **10.C) RMSE FOR LASSO REGRESSION WITH SCALING**

# In[ ]:


y_pred10 = model10.predict(x_test)

val = mean_squared_error(Y_test, y_pred10, squared=False)
val10 = str(round(val, 4))

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, y_pred10)))


# **11.A) GRID SEARCH FOR ELASTICNET REGRESSION WITH SCALING**

# In[ ]:


EN = ElasticNet()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]}

ENR = GridSearchCV(Elasticnet, parameters, scoring='neg_mean_squared_error', cv = 100)

ENR.fit(x_train, Y_train)
ENR.best_params_


# **11.B) ELASTICNET REGRESSION WITH SCALING**

# In[ ]:


model11 = linear_model.Lasso(alpha=.01)
model11.fit(x_train,Y_train)

accuracy11 = model11.score(x_test,Y_test)
print(accuracy11*100,'%')


# **11.C) RMSE FOR ELASTICNET REGRESSION WITH SCALING**

# In[ ]:


y_pred11 = model11.predict(x_test)

val = mean_squared_error(Y_test, y_pred11, squared=False)
val11 = str(round(val, 4))


print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, y_pred11)))


# **12.A) GRID SEARCH FOR SVR WITH SCALING**

# In[ ]:


SVR = SVR()

parameters = {'C':[.0001 ,.001 ,0.1, 1, 10, 100, 1000],
              'epsilon':[0.001, 0.01, 0.1, 0.5, 1, 2, 4]
             }

ENR = GridSearchCV(SVR, parameters, scoring='neg_mean_squared_error', cv = 10)

ENR.fit(x_train, Y_train)
ENR.best_params_


# **12.B) SVR WITH SCALING**

# In[ ]:


from sklearn.svm import SVR
model12 = SVR(C=1, epsilon=0.1)
model12.fit(x_train,Y_train)

model12 = model12.score(x_test,Y_test)
print(model12*100,'%')


# **13.A) GRID SEARCH FOR SCALER RIDGE REGRESSION**

# In[ ]:


R = Ridge()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]}

R = GridSearchCV(R, parameters, scoring='neg_mean_squared_error', cv = 100)

R.fit(x_train, Y_train)
R.best_params_


# **13.B)RIDGE REGRESSION WITH STANDARD SCALER**

# In[ ]:


model13 = linear_model.Ridge(alpha=10)
model13.fit(x_train,Y_train)

accuracy13 = model13.score(x_test,Y_test)
print(accuracy13*100,'%')


# **13.C)RMSE VALUE FOR STANDARD SCALER RIDGE REGRESSION**

# In[ ]:


y_pred13 = model13.predict(x_test)

val = mean_squared_error(Y_test, y_pred13, squared=False)
val13 = str(round(val, 4))

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, y_pred13)))


# In[ ]:


data2 = [['Scaled Linear Regression',val7],['Scaled Decision Tree',val8],['Scaled Random Forest',val9],['Scaled Lasso Regression',val10],['Scaled Ridge Regression',val13],['Scaled ElasticNet Regression',val11]]
d2 = pd.DataFrame(data2,columns = ['Standard Scaler - Model ','RMSE Error'])
Half2RMSE = d2.copy()


# In[ ]:


Half2RMSE


# # **MinMax Scaler Transform**

# In[ ]:


from pandas import DataFrame
trans = MinMaxScaler()
dat = trans.fit_transform(df)
dataset = DataFrame(dat)

df.head()


# In[ ]:


dataset.columns = ['GRE Score','TOEFL Score','University Rating','SOP','LOR','CGPA','Research','Chance of Admit']


# In[ ]:


ex = dataset.drop(['Chance of Admit'], axis = 1)
ex


# In[ ]:


ey = dataset['Chance of Admit']
ey


# In[ ]:


x_t, x_es, Y_t, Y_es = train_test_split(ex, ey, test_size = 0.20,shuffle = False)


# **14.A) LINEAR REGRESSION WITH SCALING**

# In[ ]:


model14 = LinearRegression()
model14.fit(x_t, Y_t)

accuracy14 = model14.score(x_es,Y_es)
print(accuracy14*100,'%')


# **14.B) RMSE LINEAR REGRESSION FOR MIN MAX SCALER**

# In[ ]:


y_pred14 = model14.predict(x_es)

val = mean_squared_error(Y_es, y_pred14, squared=False)
val14 = str(round(val, 4))


print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_es, y_pred14)))


# **15.A) GRID SEARCH FOR ALPHA FOR LASSO REGRESSION**

# In[ ]:


l = Lasso()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]}

lr = GridSearchCV(l, parameters, scoring='neg_mean_squared_error', cv = 100)

lr.fit(x_t, Y_t)
lr.best_params_


# **15.B) LASSO REGRESSION FOR MIN MAX SCALER**

# In[ ]:


model15 = linear_model.Lasso(alpha=.001)
model15.fit(x_t,Y_t)

accuracy15 = model15.score(x_es,Y_es)
print(accuracy15*100,'%')


# **15.C) RMSE VALUE FOR SCALED LASSO REGRESSION**

# In[ ]:


y_pred15 = model15.predict(x_es)

val = mean_squared_error(Y_es, y_pred15, squared=False)
val15 = str(round(val, 4))

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_es, y_pred15)))


# **16.A) GRID SEARCH FOR ALPHA FOR RIDGE REGRESSION**

# In[ ]:


r = Ridge()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]}

r = GridSearchCV(r, parameters, scoring='neg_mean_squared_error', cv = 100)

r.fit(x_t, Y_t)
r.best_params_


# **16.B) RIDGE REGRESSION FOR MIN MAX SCALER**

# In[ ]:


model16 = linear_model.Ridge(alpha=0.01)
model16.fit(x_t,Y_t)

accuracy16 = model16.score(x_es,Y_es)
print(accuracy16*100,'%')


# **16.C) RMSE VALUE FOR SCALED MIN MAX SCALER**

# In[ ]:


y_pred16 = model16.predict(x_es)

val = mean_squared_error(Y_es, y_pred16, squared=False)
val16 = str(round(val, 4))


print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_es, y_pred16)))


# **17.A) GRID SEARCH FOR APLHA FOR ELASTICNET**

# In[ ]:


en = ElasticNet()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]}

enr = GridSearchCV(en, parameters, scoring='neg_mean_squared_error', cv = 100)

enr.fit(x_t, Y_t)
enr.best_params_


# **17.B) ELASTICNET REGRESSION FOR MIN MAX SCALER**

# In[ ]:


model17 = linear_model.Lasso(alpha=.001)
model17.fit(x_t,Y_t)

accuracy17 = model17.score(x_es,Y_es)
print(accuracy17*100,'%')


# **17.C) RMSE VALUE FOR ELASTICNET REGRESSION FOR MIN MAX SCALER**

# In[ ]:


y_pred17 = model17.predict(x_es)

val = mean_squared_error(Y_es, y_pred17, squared=False)
val17 = str(round(val, 4))


print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_es, y_pred17)))


# **18.A) SCALED DECISION TREE MIN MAX SCALER**

# In[ ]:


model18 = DecisionTreeRegressor()
model18.fit(x_t, Y_t)

accuracy18 = model18.score(x_es,Y_es)
print(accuracy18*100,'%')


# **18.B) RMSE ERROR FOR SCALED DECISION TREE**
# 

# In[ ]:


y_pred18 = model18.predict(x_es)

val = mean_squared_error(Y_es, y_pred8, squared=False)
val18 = str(round(val, 4))

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_es, y_pred18)))


# **19.A) GRID SEARCH FOR RANDOM FOREST MIN MAX SCALER**

# In[ ]:


n_estimators = [10, 40, 70, 100, 130, 160, 190, 220, 250, 270]

Rf = RandomForestRegressor()

parameters = {'n_estimators': [10, 40, 70, 100, 130, 160, 190, 220, 250, 270]}

Rfr = GridSearchCV(Rf, parameters,scoring='neg_mean_squared_error', cv=10)

Rfr.fit(x_t, Y_t)

Rfr.best_params_


# **19.B) MIN MAX SCALER RANDOM FOREST**

# In[ ]:


model19 = RandomForestRegressor(n_estimators = 100)
model19.fit(x_t, Y_t)

accuracy19 = model19.score(x_es,Y_es)
print(accuracy19*100,'%')


# **19.C) RMSE VALUE FOR MIN MAX SCALER RANDOM FOREST**

# In[ ]:


y_pred19 = model19.predict(x_es)

val = mean_squared_error(Y_es, y_pred19, squared=False)
val19 = str(round(val, 4))

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_es, y_pred19)))


# **20.A) GRID SEARCH FOR SVR WITH MIN MAX SCALER**

# In[ ]:


from sklearn.svm import SVR

Svr = SVR()

parameters = {'C':[.0001 ,.001 ,0.1, 1, 10, 100, 1000],
              'epsilon':[0.001, 0.01, 0.1, 0.5, 1, 2, 4]
             }

Enr = GridSearchCV(Svr, parameters, scoring='neg_mean_squared_error', cv = 10)

Enr.fit(x_t, Y_t)
Enr.best_params_


# **20.B) SVR WITH MIN MAX SCALER**

# In[ ]:


from sklearn.svm import SVR
model20 = SVR(C=1, epsilon=0.1)
model20.fit(x_t,Y_t)

model20 = model20.score(x_es,Y_es)
print(model20*100,'%')


# In[ ]:


data3 = [['Scaled Linear Regression',val14],['Scaled Lasso Regression',val15],['Scaled Ridge Regression',val16],['Scaled ElasticNet Regression',val17],['Scaled Decision Tree',val18],['Scaled Random Forest',val19]]
d3 = pd.DataFrame(data3,columns = ['Min Max Scaler - Model ','RMSE Error'])
Half3RMSE = d3.copy()
Half3RMSE


# In[ ]:


frames = [Half1RMSE,Half2RMSE,Half3RMSE] 
FullRMSE = pd.concat(frames, axis = 1)


# In[ ]:


FullRMSE


# The 1st column represents datasets which have not been scaled and the models are used on them.
# 
# 
# The 3rd column 'Standard Scaler - Model' means that StandardScaler library was used on the dataset. The successive RMSE Error are included in the 4th column for the Models.
# 
# 
# The 5th column 'Min Max Scaler - Model' means that MinMaxScaler library was used on the dataset. The successive RMSE Error are included in the 6th columns for the Models.
# 
# 

# **This shows that LASSO REGRESSION and ELASTICNET REGRESSION has the minimum RMSE for the dataset. Moreover, they both have the same Accuracy which is 90.15 after scaling the dataset**

# # Therefore, ElasticNet Regression without Scaler Transform provides the lowest value for RMSE.

# **If the notebook was helpful, drop an upvote. Much appreciated and thank you!**

# In[ ]:




