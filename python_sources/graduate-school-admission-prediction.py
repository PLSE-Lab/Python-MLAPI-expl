#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pylab as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler

from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge, Lasso
from sklearn.svm import LinearSVR
import xgboost as xgb


# In[ ]:


# load and split the data
df = pd.read_csv('../input/Admission_Predict_Ver1.1.csv')


# In[ ]:


# refining the features names to avoid any sort of column name errors 
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')


# In[ ]:


# dropping the serial number (insignificant feature)
df.drop(columns=['serial_no.'], axis=1, inplace=True)
print(list(df.columns.values))


# In[ ]:


# scatter plot of the features
cols = list(df.columns.values)
sns.pairplot(df[cols], size = 2)
plt.suptitle('Figure - The scatter plot of features ', x=0.5, y=1.01, verticalalignment='center', fontsize= 20)
plt.tight_layout()
plt.show();


# In[ ]:


# printing the columns' information 
df.info()


# In[ ]:


# checking for any NaN value in dataset
df.isnull().values.any()


# In[ ]:


# finding the significant features for predicting chances of admit
pd.DataFrame(df.corr()['chance_of_admit'])


# As we can see that, GRE score, TOEFL score and CGPA are top 3 features which can be useful to decide the output.

# In[ ]:


sns.set_color_codes("pastel")
sns.kdeplot(df['gre_score'], df['toefl_score'], shade=True, cut=5)


# In[ ]:


# creating features and labels
y = df['chance_of_admit']
x = df


# # Linear regression

# 1. Using data normalization

# In[ ]:


scaler = Normalizer().fit(x)
rescaledX = scaler.transform(x)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(rescaledX, y, random_state=42)

print("X train: ", X_train.shape)
print("X test: ", X_test.shape)
print("y train: ", y_train.shape)
print("y test: ", y_test.shape)


# In[ ]:


lr = LinearRegression()
lr.fit(X_train, y_train)


# In[ ]:


y_pred = lr.predict(X_test)
nlr = round(100*lr.score(X_test, y_test),3)
print('Score: ', nlr)
print('Mean-squared error: ', mean_squared_error(y_test, y_pred))


# 2. Using min-max scaling

# In[ ]:


scaler = MinMaxScaler(feature_range=(0, 1))
rescaledX = scaler.fit_transform(x)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(rescaledX, y, random_state=42)

print("X train: ", X_train.shape)
print("X test: ", X_test.shape)
print("y train: ", y_train.shape)
print("y test: ", y_test.shape)


# In[ ]:


lr = LinearRegression()
lr.fit(X_train, y_train)


# In[ ]:


y_pred = lr.predict(X_test)
mmlr = round(100*lr.score(X_test, y_test),3)
print('Score: ', mmlr)
print('Mean-squared error: ', mean_squared_error(y_test, y_pred))


# 3. Using data Standardization

# In[ ]:


scaler = StandardScaler().fit(x)
rescaledX = scaler.transform(x)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(rescaledX, y, random_state=42)

print("X train: ", X_train.shape)
print("X test: ", X_test.shape)
print("y train: ", y_train.shape)
print("y test: ", y_test.shape)


# In[ ]:


lr = LinearRegression()
lr.fit(X_train, y_train)


# In[ ]:


y_pred = lr.predict(X_test)
slr = round(100*lr.score(X_test, y_test),3)
print('Score: ', slr)
print('Mean-squared error: ', mean_squared_error(y_test, y_pred))


# # XGBoost classifier

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=42)

print("X train: ", X_train.shape)
print("X test: ", X_test.shape)
print("y train: ", y_train.shape)
print("y test: ", y_test.shape)


# In[ ]:


model = xgb.XGBClassifier()
model.fit(X_train, y_train)


# In[ ]:


y_pred = model.predict(X_test)
print('Mean-squared error: ', mean_squared_error(y_test, y_pred))
xgbc = round(100*r2_score(y_test, y_pred),2)
print('R2 score for regression: ', xgbc)


# # Ridge regression classifier

# In[ ]:


scaler = StandardScaler().fit(x)
rescaledX = scaler.transform(x)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(rescaledX, y, random_state=42)

print("X train: ", X_train.shape)
print("X test: ", X_test.shape)
print("y train: ", y_train.shape)
print("y test: ", y_test.shape)


# In[ ]:


model = Ridge(alpha=0.5)
model.fit(X_train, y_train)


# In[ ]:


y_pred = model.predict(X_test)
rrc = round(100*model.score(X_test, y_test),3)
print('Score: ', rrc)
print('Mean-squared error: ', mean_squared_error(y_test, y_pred))


# # Lasso regression (least absolute shrinkage and selection operator)

# In[ ]:


scaler = StandardScaler().fit(x)
rescaledX = scaler.transform(x)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(rescaledX, y, random_state=42)

print("X train: ", X_train.shape)
print("X test: ", X_test.shape)
print("y train: ", y_train.shape)
print("y test: ", y_test.shape)


# In[ ]:


model = Lasso(alpha=0.01)
model.fit(X_train, y_train)


# In[ ]:


y_pred = model.predict(X_test)
lrc = round(100*model.score(X_test, y_test),3)
print('Score: ', lrc)
print('Mean-squared error: ', mean_squared_error(y_test, y_pred))


# # Stochastic gradient descent (SGD)

# In[ ]:


standardScaler = StandardScaler()
standardScaler.fit(X_train)


# In[ ]:


X_train_standard = standardScaler.transform(X_train)
X_test_standard = standardScaler.transform(X_test)


# In[ ]:


sgd_reg = SGDRegressor(max_iter=50)
sgd_reg.fit(X_train_standard, y_train)


# In[ ]:


y_pred = sgd_reg.predict(X_test_standard)
sgdc = round(100*sgd_reg.score(X_test_standard, y_test),2)
print('Score: ', sgdc)


# # Random forest

# In[ ]:


rfr = RandomForestRegressor(n_estimators=100,criterion='mse')
rfr.fit(X_train, y_train)


# In[ ]:


y_pred = rfr.predict(X_test)
rfrc = round(100*rfr.score(X_test,y_test),2)
print('Score: ', rfrc)


# # Multi-layer perceptron regressor (MLP)

# In[ ]:


mlp = MLPRegressor(hidden_layer_sizes=(100,),activation='relu',solver='lbfgs',learning_rate='adaptive',max_iter=1000,learning_rate_init=0.01,alpha=0.5,random_state=15)
mlp.fit(X_train, y_train)


# In[ ]:


y_pred = mlp.predict(X_test)
mlpr = round(100*mlp.score(X_test, y_test),2)
print('Score: ', mlpr)
print('Mean-squared error: ', mean_squared_error(y_test, y_pred))


# # Analysing the best algorithm

# In[ ]:


data = {'Normalised Linear Regression':nlr, 
        'Min-max scaling Linear Regression':mmlr, 
        'Data standardization Linear Regression':slr, 
        'XGBoost classifier':xgbc, 
        'Ridge regression':rrc, 
        'Lasso regression':lrc, 
        'Stochastic gradient descent':sgdc, 
        'Random Forest regressor':rfrc, 
        'Multi-layer perceptron regressor':mlpr}


# In[ ]:


import matplotlib.pyplot as plt

sns.barplot(np.array(list(data.values())), np.array(list(data.keys())), palette="rocket")

plt.show()


# In[ ]:




