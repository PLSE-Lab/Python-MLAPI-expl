#!/usr/bin/env python
# coding: utf-8

# # Import Python Library

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics


# # Input Data

# In[ ]:


df = pd.read_csv("../input/2015.csv")
df.columns


# # Check out Correlations Between Features

# In[ ]:


plt.figure(figsize=(10,8))
#2015
corr = df.drop(['Country','Region','Happiness Rank','Standard Error'],axis = 1).corr()
sns.heatmap(corr, cbar = True, square = True, annot=True, linewidths = .5, fmt='.2f',annot_kws={'size': 15}) 
sns.plt.title('Heatmap of Correlation Matrix')
plt.show()


# Happiness Score is strongly correlated with Economy, Family & Health.
# 
#  - Happiness Score vs Economy: 0.78
#  - Happiness Score vs Family: 0.74
#  - Happiness vs Health: 0.72

# # Data Preprocessing

# In[ ]:


X = df.drop(['Standard Error','Happiness Score', 'Happiness Rank', 'Country', 'Region'], axis=1)
y = df['Happiness Score']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
#print('Standardized features\n')
#print(str(X_train[:4]))


# # Linear Model

# In[ ]:


from sklearn import linear_model
lm = linear_model.LinearRegression()
lm.fit(X_train,y_train)
y_pred = lm.predict(X_test)

result_lm = pd.DataFrame({
    'Actual':y_test,
    'Predict':y_pred
})
result_lm['Diff'] = y_test - y_pred
result_lm.head()


# In[ ]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % lm.score(X_test, y_test))

sns.regplot(x='Actual',y='Predict',data=result_lm)


# # Random Forest Regressor

# In[ ]:


from sklearn import ensemble
rf = ensemble.RandomForestRegressor()
rf.fit(X_train,y_train)
y_pred = rf.predict(X_test)

result_rf = pd.DataFrame({
    'Actual':y_test,
    'Predict':y_pred
})
result_rf['Diff'] = y_test - y_pred
result_rf.head()


# In[ ]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % rf.score(X_test, y_test))

sns.regplot(x='Actual',y='Predict',data=result_rf)


# # Support Vector Machine - Linear SVR

# In[ ]:


from sklearn import svm
clf = svm.LinearSVR()
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

result_clf = pd.DataFrame({
        'Actual':y_test,
        'Predict':y_pred
})
result_clf['Diff'] = y_test - y_pred
result_clf.head()


# In[ ]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % clf.score(X_test, y_test))

sns.regplot(x='Actual',y='Predict',data=result_clf)


# **Investigation result**
# ----------
#  1. Happiness Ranking basically depends on Economy. Countries having
#     strong GDP rate are happiest countries indeed.
#  2. Happiness score depends on family, health, freedom, trust on govt.
#     etc. All of these terms have a great linear relationship with
#     happiness score.
#  3. World happiness doesn't define generosity or genesrosity has no
#     remarkable impact on Happiness score.
#  4. Regression Model works quite good in our happiness Dataset. Errors is
#     too small to consider. We can easily check differences on above
#     plots.

# # Predict 2016 Happiness Score with linear model trained with 2015 data set

# In[ ]:


df2 = pd.read_csv("../input/2016.csv")
X2 = df2.drop(['Happiness Score', 'Happiness Rank', 'Country', 'Region','Lower Confidence Interval', 'Upper Confidence Interval'], axis=1)
y2 = df2['Happiness Score']
scaler.fit(X2)
X2 = scaler.transform(X2)
y_pred2 = lm.predict(X2)
result_lm2 = pd.DataFrame({
    'Actual':y2,
    'Predict':y_pred2
})
result_lm2['Diff'] = y2 - y_pred2
result_lm2.head()


# In[ ]:


print('Coefficients: \n', lm.coef_)
print('Mean Absolute Error:', metrics.mean_absolute_error(y2, y_pred2))
print('Mean Squared Error:', metrics.mean_squared_error(y2, y_pred2))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y2, y_pred2)))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % lm.score(X2, y2))

sns.regplot(x='Actual',y='Predict',data=result_lm2)


# # Other regression model

# **Huber Regressor**

# In[ ]:


hbr = linear_model.HuberRegressor()
hbr.fit(X_train,y_train)
y_pred = hbr.predict(X_test)

result_hbr = pd.DataFrame({
        'Actual':y_test,
        'Predict':y_pred
})
result_hbr['Diff'] = y_test - y_pred
result_hbr.head()


# In[ ]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % hbr.score(X_test, y_test))

sns.regplot(x='Actual',y='Predict',data=result_hbr)


# **Bayesian Ridge**

# In[ ]:


br = linear_model.BayesianRidge()
br.fit(X_train,y_train)
y_pred = br.predict(X_test)

result_br = pd.DataFrame({
        'Actual':y_test,
        'Predict':y_pred
})
result_br['Diff'] = y_test - y_pred
result_br.head()


# In[ ]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % br.score(X_test, y_test))

sns.regplot(x='Actual',y='Predict',data=result_br)

