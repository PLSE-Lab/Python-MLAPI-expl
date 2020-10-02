#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # 100MLProjects - Project #1: Graduate Admissions Predicition
# 
# ![100MLProjects Project1](https://miro.medium.com/max/700/1*QDNlBUhcLLiPrjQV27G2ZA.png)
# 
# Dataset Source: https://www.kaggle.com/mohansacharya/graduate-admissions 
# 
# Date: June 12, 2020
# 
# ### Lakshmanan Meiyappan
# LinkedIn: https://www.linkedin.com/in/lakshmanan-meiyappan/
# 
# Github: https://github.com/laxmena/
# 

# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


data = pd.read_csv('../input/graduate-admissions/Admission_Predict_Ver1.1.csv')
data2 = pd.read_csv('../input/graduate-admissions/Admission_Predict.csv')

dataset = pd.concat([data, data2])

dataset.sample(5)


# In[ ]:


dataset.drop(columns=['Serial No.'], axis=1, inplace=True)

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values


# ## Exploring Dataset:
# 

# 
# This dataset is created by Mohan Acharya to understand graduate admission process from an Indian Perspective.
# 
# Dataset contains features like GRE Scores, TOEFL, CGPA, SOP Rating, LOR Rating, Research Papers and University Ratings.
# 
# These features are used to predict the chance of admit for a given student.
# 
# Note: 'Chance of Admit' is the probability of getting an admission for the given college for masters program.

# In[ ]:


print(dataset.columns)


# In[ ]:


dataset.describe()


# The mean GRE Score is 316, and TOEFL is 107.
# 
# * **Inference:** GRE is really a challenging exam to tackle, and several Indian students have aced it. The mean score 316 is by itself a very good score.*
# 
# **Average CGPA: 8.59** (My CGPA during Undergrad is 8.59)
# 
# * **Inference:** Students who try to pursue masters program have performed incredibly well in their undergrauate programs.*

# In[ ]:


dataset.isnull().sum()


# There are no missing data's in the dataset, the data is already cleaned.

# In[ ]:


import seaborn as sns
sns.pairplot(dataset)
plt.show()


# **Inference:**
# 
# From my countless hours surfing internet and gathering details about masters program and application process, one feature that all the blogs and articles agreed was - A good 'SOP' could compensate for other weak scores(like GRE, CGPA)
# 
# But this visualization tells us a different story. SOP and LOR are required features, but they are not the game changers. An average SOP with 3.0 rating can still get higher chance of admits in graduate colleges.
# 
# **University Rating vs SOP/LOR**:
# - SOP's didnt seem to play a significant role for Graduate colleges with 4.0 rating
# - Universities with 5.0 rating, has no admissions for applicants with SOP's less than 3.0 score.  
# 
# **Research Papers**:
# - Research papers doesnt seem to play a significant role in the outcome of the admissions here.
# 

# In[ ]:


corr = dataset.corr()
fig, ax = plt.subplots(figsize=(8, 8))
colormap = sns.diverging_palette(220, 10, as_cmap=True)
dropSelf = np.zeros_like(corr)
dropSelf[np.triu_indices_from(dropSelf)] = True
colormap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, cmap=colormap, linewidths=.5, annot=True, fmt=".2f", mask=dropSelf)
plt.title('Graduate Admissions - Features Correlations\n#100MLProjects #laxmena')
plt.show()
fig.savefig('Correlation.png')


# ### GRE Score

# In[ ]:


import seaborn as sns

sns.distplot(dataset.iloc[:,0].values)


# In[ ]:


# GRE Scores
gre_median = np.median(dataset.iloc[:,0].values)
gre_mean = np.mean(dataset.iloc[:,0].values)

print("GRE Scores Summary")
print("GRE Median: ", gre_median)
print("GRE Mean: ", gre_mean)


# ### TOEFL Score Distribution
# 

# In[ ]:


import seaborn as sns

sns.distplot(dataset.iloc[:,1].values)


# In[ ]:


# TOEFL Scores
toefl_median = np.median(dataset.iloc[:,1].values)
toefl_mean = np.mean(dataset.iloc[:,1].values)

print("TOEFL Scores Summary")
print("TOEFL Median: ", toefl_median)
print("TOEFL Mean: ", toefl_mean)


# ### CGPA Distribution

# In[ ]:


import seaborn as sns

sns.distplot(dataset.iloc[:,-3].values)


# In[ ]:


# GRE Scores
cgpa_median = np.median(dataset.iloc[:,-3].values)
cgpa_mean = np.mean(dataset.iloc[:,-3].values)

print("CGPA Scores Summary")
print("CGPA Median: ", cgpa_median)
print("CGPA Mean: ", cgpa_mean)


# ### GRE vs Admit Probability

# In[ ]:



threshold = 0.75
plt.scatter(X[:,0][y>threshold], y[y>threshold], color='green', label='>' + str(threshold*100)+' Chance of Admit')
plt.scatter(X[:,0][y<=threshold], y[y<=threshold], color='red', label='<' + str(threshold*100)+' Chance of Admit')
plt.xlabel('GRE Score')
plt.ylabel('Chance of Admit')
plt.title('GRE Score vs Chance of Admit')
plt.legend()
plt.show()


# ### TOEFL vs Chance of Admit

# In[ ]:


threshold = 0.8
plt.scatter(X[:,1][y>threshold], y[y>threshold], color='green', label='>' + str(threshold*100)+' Chance of Admit')
plt.scatter(X[:,1][y<=threshold], y[y<=threshold], color='red', label='<' + str(threshold*100)+' Chance of Admit')
plt.xlabel('TOEFL Score')
plt.ylabel('Chance of Admit')
plt.title('TOEFL Score vs Chance of Admit')
plt.legend()
plt.show()


# ### GRE Score vs TOEFL Score
# 
# To compare How high scoring stuents GRE perform in TOEFL

# In[ ]:


plt.scatter(X[:,0], X[:,1], color='blue')
plt.xlabel('GRE Score')
plt.ylabel('TOEFL Score')
plt.title('GRE Score vs TOEFL Score')
plt.show()


# ### CGPA vs Chance of Admit

# In[ ]:


threshold = 0.8
plt.scatter(X[:,-2][y>threshold], y[y>threshold], color='green', label='>' + str(threshold*100)+' Chance of Admit')
plt.scatter(X[:,-2][y<=threshold], y[y<=threshold], color='red', label='<' + str(threshold*100)+' Chance of Admit')
plt.xlabel('CGPA')
plt.ylabel('Chance of Admit')
plt.title('CGPA vs Chance of Admit')
plt.legend()
plt.show()


# ## Split Dataset into Training and Test Dataset, Feature Scaling
# 

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(X, y)

def feature_scaler(X):
  sc = StandardScaler()
  X[:,:-1] = sc.fit_transform(X[:,:-1])  
  return X

X_train = feature_scaler(X_train)
X_test = feature_scaler(X_test)


# ## Building Machine Learning Models

# In[ ]:


from sklearn.metrics import mean_absolute_error, mean_squared_error


# ### Linear Regression

# In[ ]:


from sklearn.linear_model import LinearRegression

linear_regressor = LinearRegression()
linear_regressor = linear_regressor.fit(X_train, y_train)


# In[ ]:


y_pred_lr = linear_regressor.predict(X_test)

linear_regressor_score = round(linear_regressor.score(X_test, y_test)*100, 2)
linear_regressor_mas = mean_absolute_error(y_test,y_pred_lr)
linear_regressor_rmse = np.sqrt(mean_squared_error(y_test,y_pred_lr))

print('Accuracy Score: ',linear_regressor_score,'%')
print('Mean Absolute Error: ', linear_regressor_mas)
print('RMSE: ', linear_regressor_rmse)


# In[ ]:


print('Intercept: \n', linear_regressor.intercept_)
print('Coefficients: \n', linear_regressor.coef_)


# ### Decision Tree

# In[ ]:


from sklearn.tree import DecisionTreeRegressor

decision_tree_regressor = DecisionTreeRegressor()
decision_tree_regressor = decision_tree_regressor.fit(X_train, y_train)


# In[ ]:


y_pred_dt = decision_tree_regressor.predict(X_test)

decision_tree_score = round(decision_tree_regressor.score(X_test, y_test)*100, 2)
decision_tree_mas = mean_absolute_error(y_test,y_pred_dt)
decision_tree_rmse = np.sqrt(mean_squared_error(y_test,y_pred_dt))

print('Accuracy: ', decision_tree_score,'%')
print('Mean Absolute Error: ', decision_tree_mas)
print('RMSE: ', decision_tree_rmse)


# In[ ]:


### Random Forest Regression Model


# In[ ]:


from sklearn.ensemble import RandomForestRegressor

random_forest_regressor = RandomForestRegressor()
random_forest_regressor = random_forest_regressor.fit(X_train, y_train)


# In[ ]:


y_pred_rf = random_forest_regressor.predict(X_test)

random_forest_score = round(random_forest_regressor.score(X_test, y_test)*100, 2)
random_forest_mas = mean_absolute_error(y_test,y_pred_rf)
random_forest_rmse = np.sqrt(mean_squared_error(y_test,y_pred_rf))

print('Accuracy: ', random_forest_score,'%')
print('Mean Absolute Error: ', random_forest_mas)
print('RMSE: ', random_forest_rmse)


# ### SVR Model

# In[ ]:


from sklearn.svm import SVR

svr_regressor = SVR(kernel='linear')
svr_regressor = svr_regressor.fit(X_train, y_train)


# In[ ]:


y_pred_svr = svr_regressor.predict(X_test)

svr_score = round(svr_regressor.score(X_test, y_test)*100, 2)
svr_mas = mean_absolute_error(y_test,y_pred_svr)
svr_rmse = np.sqrt(mean_squared_error(y_test,y_pred_svr))

print('Accuracy: ', svr_score,'%')
print('Mean Absolute Error: ', svr_mas)
print('RMSE: ', svr_rmse)


# ## Comparing Machine Learning Models

# In[ ]:


df = pd.DataFrame({'Regression Model': ['Linear Regression', 'Decision Tree Regression', 'Random Forest Regression', 'SVR Model'],
                  'Accuracy Score': [linear_regressor_score, decision_tree_score, random_forest_score, svr_score],
                   'Mean Absolute Error': [linear_regressor_mas, decision_tree_mas, random_forest_mas, svr_mas],
                   'Root Mean Squared Error': [linear_regressor_rmse, decision_tree_rmse, random_forest_rmse, svr_rmse]},
                  columns= ['Regression Model', 'Accuracy Score', 'Mean Absolute Error', 'Root Mean Squared Error'])
print(df.to_markdown())


# ## Visualizing Feature Importance

# In[ ]:


importance_frame = pd.DataFrame()
importance_frame['Importance'] = random_forest_regressor.feature_importances_
importance_frame['Features'] = dataset.columns[:-1]


# In[ ]:


plt.barh([1,2,3,4,5,6,7], importance_frame['Importance'], align='center', alpha=0.5)
plt.yticks([1,2,3,4,5,6,7], importance_frame['Features'])
plt.xlabel('Importance')
plt.title('Feature Importances')
plt.show()


# In[ ]:




