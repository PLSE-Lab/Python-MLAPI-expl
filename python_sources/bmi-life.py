#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#importing library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.simplefilter("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


data = pd.read_csv("../input/bmi-and-life/bmi_and_life_expectancy.csv")


# In[ ]:


data.head(2)


# In[ ]:


data.shape


# In[ ]:


data.isnull().any()


# In[ ]:


data.describe()


# In[ ]:


plt.figure(figsize=(7,7))
sns.boxplot(x= 'variable', y = 'value', data = pd.melt(data[['Life expectancy', 'BMI']]))


# In[ ]:


# Plotting the heatmap of correlation between features
plt.figure(figsize=(7,7))
sns.heatmap(data.corr(), cbar=True, square= True, fmt='.1f', annot=True, annot_kws={'size':10}, cmap='Greys')


# In[ ]:


data.plot.hist(grid=True, bins=20, rwidth=0.9)


# In[ ]:


sns.distplot(data['BMI'])
plt.title("Histogram of BMI")
plt.xlabel("BMI")
plt.ylabel("Frequency")
plt.show()


# In[ ]:


sns.distplot(data['Life expectancy'])
plt.title("Histogram for Life Expectancy ")
plt.xlabel("Life expectancy")
plt.ylabel("Frequency")
plt.show()


# In[ ]:


#spliting the data
from sklearn.model_selection import train_test_split


# In[ ]:


X = data['Life expectancy']
y = data['BMI']


# In[ ]:


X = X.values.reshape(-1,1)
y = y.values.reshape(-1,1)


# In[ ]:


#scaling the data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_scaled = sc.fit_transform(X)


# #### Regression on Non0-scaled Data

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X ,y, test_size = 0.3, random_state = 100)


# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


# In[ ]:


lr = LinearRegression()
lr.fit(X_train, y_train)


# In[ ]:


y_pred_lr = lr.predict(X_test)


# In[ ]:


dtr = DecisionTreeRegressor()
dtr.fit(X_train, y_train)


# In[ ]:


y_pred_dtr = dtr.predict(X_test)


# In[ ]:


rfr = RandomForestRegressor()
rfr.fit(X_train, y_train)


# In[ ]:


y_pred_rfr = rfr.predict(X_test)


# In[ ]:


xgr = XGBRegressor()
xgr.fit(X_train, y_train)


# In[ ]:


y_pred_xgr = xgr.predict(X_test)


# #### Regression on Scaled Data

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X_scaled,y, test_size = 0.3, random_state = 100)


# In[ ]:


lr = LinearRegression()
lr.fit(X_train, y_train)


# In[ ]:


y_pred_lrs = lr.predict(X_test)


# In[ ]:


rfr = RandomForestRegressor()
rfr.fit(X_train, y_train)


# In[ ]:


y_pred_rfrs = rfr.predict(X_test)


# In[ ]:


dtr = DecisionTreeRegressor()
dtr.fit(X_train, y_train)


# In[ ]:


y_pred_dtrs = dtr.predict(X_test)


# In[ ]:


xgr = XGBRegressor()
xgr.fit(X_train, y_train)


# In[ ]:


y_pred_xgrs = xgr.predict(X_test)


# #### Evaluating Models (Non-scaled Data)

# In[ ]:


from sklearn.metrics import r2_score


# In[ ]:


R2_score_LR = r2_score(y_test, y_pred_lr)


# In[ ]:


R2_score_DTR =  r2_score(y_test, y_pred_dtr)


# In[ ]:


R2_score_RFR = r2_score(y_test, y_pred_rfr)


# In[ ]:


R2_score_xgr = r2_score(y_test, y_pred_xgr)


# #### Evaluating Models (Scaled Data)

# In[ ]:


R2_score_LRs = r2_score(y_test, y_pred_lrs)


# In[ ]:


R2_score_DTRs = r2_score(y_test, y_pred_dtrs)


# In[ ]:


R2_score_RFRs = r2_score(y_test, y_pred_rfrs)


# In[ ]:


R2_score_xgrs = r2_score(y_test, y_pred_xgrs)


# In[ ]:


models = pd.DataFrame({
    'Model': ['Linear Regression', 'Random Forest', 'XGBoost', 'Decision Tree'],
    'R-square(Non-scaled)': [R2_score_LR*100, R2_score_RFR*100, R2_score_xgr*100, R2_score_DTR*100],
    'R-square(Scaled)': [R2_score_LRs*100, R2_score_RFRs*100, R2_score_xgrs*100, R2_score_DTRs*100]})
models.sort_values(by = 'Model',ascending=False)


# ### THE END!!!
