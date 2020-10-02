#!/usr/bin/env python
# coding: utf-8

# ![image.png](attachment:image.png)

# In[ ]:


import numpy as np #linear alzebric operations
import pandas as pd #data manipulation
import matplotlib.pyplot as plt  #data visualization
import seaborn as sns  #data visualization
import plotly.express as px  #data visualization
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df_0 = pd.read_csv('../input/graduate-admissions/Admission_Predict_Ver1.1.csv')
df_2 = pd.read_csv('../input/graduate-admissions/Admission_Predict.csv')


# In[ ]:


#combining both the data and dropping duplicates
df_1 = pd.concat([df_0,df_2]).drop_duplicates().reset_index(drop=True)


# In[ ]:


df_1.head()


# In[ ]:


df_1.corr()


# In[ ]:


sns.barplot(x = df_1.corr()['Chance of Admit '], y = df_1.columns)


# In[ ]:


df_1.info()


# In[ ]:


df_1.shape


# In[ ]:


df_1.isna().sum()


# # EDA

# In[ ]:


sns.heatmap(df_1.corr(), annot = True)


# In[ ]:


#Pairplot gives use graphical representation of relation between features.
sns.pairplot(df_1.drop(['Serial No.'], axis = 1))


# In[ ]:


#Let's plot the boxplot. Boxplot gives good idea about presence of outliers in dataset.
plt.figure(figsize = (10,6))
plot = sns.boxplot(data = df_1[['GRE Score', 'TOEFL Score']],orient = 'h')
plt.show()


# In[ ]:


plt.figure(figsize = (10,6))
plot = sns.boxplot(data = df_1.drop(['Serial No.', 'GRE Score', 'TOEFL Score'], axis = 1),orient = 'h')
plt.show()


# In[ ]:


#Let's analyze the relationship between 'GRE Score' and "Chance of Admit " using jointplot.
sns.jointplot(x = 'GRE Score', y = 'Chance of Admit ', data = df_1,  kind = 'reg', joint_kws={'line_kws':{'color':'yellow'}})


# **High GRE score means higher chance of admission.**

# In[ ]:


#Let's analyze the relationship between 'TOEFL Score' and "Chance of Admit " using jointplot.
sns.jointplot(x = 'TOEFL Score', y = 'Chance of Admit ', data = df_1,  kind = 'reg', joint_kws={'line_kws':{'color':'yellow'}})


# **High TOEFL score means high chance of admission.**

# In[ ]:


#Let's analyze the relationship between "CGPA" and "Chance of Admit " using jointplot.
sns.jointplot(x = 'CGPA', y = 'Chance of Admit ', data = df_1,  kind = 'reg', joint_kws={'line_kws':{'color':'yellow'}})


# In[ ]:


sns.swarmplot(x = 'University Rating', y = 'Chance of Admit ', hue = 'Research', data = df_1)


# In[ ]:


#Let's analyze the relationship between 'University Rating' and 'Chance of Admit' using barplot.
l = df_1.groupby('University Rating')['Chance of Admit '].mean().reset_index()
plot = sns.barplot(x = l['University Rating'], y = l['Chance of Admit '])
for p in plot.patches:
        plot.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., 0.5*p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')


# In[ ]:


#Let's analyze the relationship between "SOP", "LOR" and "Chance of Admit".
sns.lineplot(x = 'SOP', y = 'Chance of Admit ', data = df_1)
sns.lineplot(x = 'LOR ', y = 'Chance of Admit ', data = df_1)


# In[ ]:


#Let's analyze the relationship between "Research" and "Chance of Admit".
l = df_1.groupby('Research')['Chance of Admit '].mean().reset_index()
plot = sns.barplot(x = l['Research'], y = l['Chance of Admit '])
for p in plot.patches:
        plot.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., 0.5*p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')


# # Prediction using different models

# In[ ]:


from sklearn.model_selection import train_test_split
train_data = df_1.drop(['Serial No.'], axis = 1)
train_x, test_x, train_y, test_y = train_test_split(train_data.drop(['Chance of Admit '], axis = 1), train_data['Chance of Admit '], test_size = 0.2, shuffle = False, random_state = 8)


# In[ ]:


from sklearn.linear_model import LinearRegression, Lasso, Ridge, RidgeCV
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR

model_list = []
Error = []
Score = []


# # Support Vector Regression

# In[ ]:


model = SVR()
model.fit(train_x,train_y)
prediction = model.predict(test_x)
print(mean_squared_error(test_y,prediction))
print(r2_score(test_y,prediction))
data = pd.DataFrame({"Predicted" : prediction, "True" : test_y})
sns.lineplot(data = data)
model_list.append('Support Vector Machine')
Error.append(mean_squared_error(test_y,prediction))
Score.append(r2_score(test_y,prediction))


# # AdaBoostRegressor

# In[ ]:


model = AdaBoostRegressor()
model.fit(train_x,train_y)
prediction = model.predict(test_x)
print(mean_squared_error(test_y,prediction))
print(r2_score(test_y,prediction))
data = pd.DataFrame({"Predicted" : prediction, "True" : test_y})
sns.lineplot(data = data)
model_list.append('AdaBoostRegressor')
Error.append(mean_squared_error(test_y,prediction))
Score.append(r2_score(test_y,prediction))


# # Linear Regression

# In[ ]:


model = LinearRegression(normalize = True)
model.fit(train_x,train_y)
prediction = model.predict(test_x)
print(mean_squared_error(test_y,prediction))
print(r2_score(test_y,prediction))
data = pd.DataFrame({"Predicted" : prediction, "True" : test_y})
sns.lineplot(data = data)
model_list.append('Linear Regression')
Error.append(mean_squared_error(test_y,prediction))
Score.append(r2_score(test_y,prediction))


# # Random Forest Regressor
# 

# In[ ]:


model = RandomForestRegressor()
model.fit(train_x,train_y)
prediction = model.predict(test_x)
print(mean_squared_error(test_y,prediction))
print(r2_score(test_y,prediction))
data = pd.DataFrame({"Predicted" : prediction, "True" : test_y})
sns.lineplot(data = data)
model_list.append('Random Forest Regressor')
Error.append(mean_squared_error(test_y,prediction))
Score.append(r2_score(test_y,prediction))


# # KNeighborsRegressor

# In[ ]:


model = KNeighborsRegressor(n_neighbors = 28)
model.fit(train_x,train_y)
prediction = model.predict(test_x)
print(mean_squared_error(test_y,prediction))
print(r2_score(test_y,prediction))
data = pd.DataFrame({"Predicted" : prediction, "True" : test_y})
sns.lineplot(data = data)
model_list.append('KNeighborsRegressor')
Error.append(mean_squared_error(test_y,prediction))
Score.append(r2_score(test_y,prediction))


# # XGBoost Regressor

# In[ ]:


model = xgb.XGBRegressor(learning_rate = 0.04)
model.fit(train_x,train_y)
prediction = model.predict(test_x)
print(mean_squared_error(test_y,prediction))
print(r2_score(test_y,prediction))
data = pd.DataFrame({"Predicted" : prediction, "True" : test_y})
sns.lineplot(data = data)
model_list.append('XGBoost Regressor')
Error.append(mean_squared_error(test_y,prediction))
Score.append(r2_score(test_y,prediction))


# # Gradient Boosting Regressor

# In[ ]:


model = GradientBoostingRegressor()
model.fit(train_x,train_y)
prediction = model.predict(test_x)
print(mean_squared_error(test_y,prediction))
print(r2_score(test_y,prediction))
data = pd.DataFrame({"Predicted" : prediction, "True" : test_y})
sns.lineplot(data = data)
model_list.append('Gradient Boosting Regressor')
Error.append(mean_squared_error(test_y,prediction))
Score.append(r2_score(test_y,prediction))


# # Decision Tree Regressor

# In[ ]:


model = DecisionTreeRegressor()
model.fit(train_x,train_y)
prediction = model.predict(test_x)
print(mean_squared_error(test_y,prediction))
print(r2_score(test_y,prediction))
data = pd.DataFrame({"Predicted" : prediction, "True" : test_y})
sns.lineplot(data = data)
model_list.append('Decision Tree Regressor')
Error.append(mean_squared_error(test_y,prediction))
Score.append(r2_score(test_y,prediction))


# # Lasso Regression

# In[ ]:


model = Lasso()
model.fit(train_x,train_y)
prediction = model.predict(test_x)
print(mean_squared_error(test_y,prediction))
print(r2_score(test_y,prediction))
data = pd.DataFrame({"Predicted" : prediction, "True" : test_y})
sns.lineplot(data = data)
model_list.append('Lasso')
Error.append(mean_squared_error(test_y,prediction))
Score.append(r2_score(test_y,prediction))


# # Ridge Regression

# In[ ]:


model = Ridge()
model.fit(train_x,train_y)
prediction = model.predict(test_x)
print(mean_squared_error(test_y,prediction))
print(r2_score(test_y,prediction))
data = pd.DataFrame({"Predicted" : prediction, "True" : test_y})
sns.lineplot(data = data)
model_list.append('Ridge')
Error.append(mean_squared_error(test_y,prediction))
Score.append(r2_score(test_y,prediction))


# # RidgeCV Regressor

# In[ ]:


model = RidgeCV()
model.fit(train_x,train_y)
prediction = model.predict(test_x)
print(mean_squared_error(test_y,prediction))
print(r2_score(test_y,prediction))
data = pd.DataFrame({"Predicted" : prediction, "True" : test_y})
sns.lineplot(data = data)
model_list.append('RidgeCV')
Error.append(mean_squared_error(test_y,prediction))
Score.append(r2_score(test_y,prediction))


# # Bagging Regressor

# In[ ]:


model = BaggingRegressor()
model.fit(train_x,train_y)
prediction = model.predict(test_x)
print(mean_squared_error(test_y,prediction))
print(r2_score(test_y,prediction))
data = pd.DataFrame({"Predicted" : prediction, "True" : test_y})
sns.lineplot(data = data)
model_list.append('BaggingRegressor')
Error.append(mean_squared_error(test_y,prediction))
Score.append(r2_score(test_y,prediction))


# In[ ]:


model_data = pd.DataFrame({'Model' : model_list, 'Error' : Error, 'Score' : Score})


# In[ ]:


model_data.style.background_gradient(cmap='Blues')


# In[ ]:


sns.barplot(y = model_data['Model'], x = model_data['Score'])


# In[ ]:


sns.barplot(y = model_data['Model'], x = model_data['Error'])


# # Conclusion
# 1. Most important feature is the CGPA. Higher the CGPA hihger is the chances of getting admission.
# 2. For this dataset linear regression is most accurate model.
