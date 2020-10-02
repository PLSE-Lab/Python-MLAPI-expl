#!/usr/bin/env python
# coding: utf-8

# # <center> Analysis of Graduate Admissions </center>

# ### Acknowledgements

# This dataset is inspired by the UCLA Graduate Dataset and is created by Mohan S Acharya to estimate chances of graduate admission from an Indian perspective.

# ### Inspiration

# This dataset was built with the purpose of helping students in shortlisting universities with their profiles. The predicted output gives them a fair idea about their chances for a particular university.

# In[ ]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# # <center>Data cleaning and processing</center>

# In[ ]:


df = pd.read_csv('../input/graduate-admissions/Admission_Predict_Ver1.1.csv')
df.columns = ['Serial No.','GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA', 'Research', 'Chance of Admit']
df.head()


# Let us drop the **"Serial No."** column as it is irrelevent. Let's also check for any null values in our dataset.

# In[ ]:


df = df.drop(['Serial No.'], axis=1)
df.isnull().sum()


# ### Descriptive Statistics

# In[ ]:


df.describe()


# From the Descriptive Statistcs, it can be observed that average GRE Score was 316/340, TOEFL Score was 107/120, and CGPA was 8.57/10. From these statistics, it can be inferred that the students applying are mostly "above average students" in the traditional sense. Moreover, it can also be seen that 56% of the students 

# # <center>Visualizing the Data</center>

# Now it is time to visualize distribution of the variables

# In[ ]:


fig = plt.figure(figsize=(14, 21))
fig.add_subplot(321)
sns.distplot(df['GRE Score'], kde=True)
plt.title("Distribution of GRE Scores")

fig.add_subplot(322)
sns.distplot(df['TOEFL Score'], kde=True)
plt.title("Distribution of TOEFL Scores")

fig.add_subplot(323)
sns.distplot(df['CGPA'], kde=True)
plt.title("Distribution of CGPA")

fig.add_subplot(324)
sns.distplot(df['SOP'], kde=False)
plt.title("Distribution of SOP Ratings")

fig.add_subplot(325)
sns.distplot(df['University Rating'], kde=False)
plt.title("Distribution of University Rating")

plt.show()


# ### Visualizing the relationship between the independent variables and the dependent variable

# Let's visualize the relationship between each of the continuous variables: GRE Score, TOEFL Score, and CGPA, with the Chance of Admit

# In[ ]:


fig = plt.figure(figsize=(14, 14))
fig.add_subplot(221)
sns.regplot(df['GRE Score'],df['Chance of Admit'])
plt.title("GRE Scores vs Chance of Admit")

fig.add_subplot(222)
sns.regplot(df['TOEFL Score'],df['Chance of Admit'])
plt.title("TOEFL Scores vs Chance of Admit")

fig.add_subplot(223)
sns.regplot(df['CGPA'],df['Chance of Admit'])
plt.title("CGPA vs Chance of Admit")

plt.show()


# It can be seen from the graphs above that all of our continuous independent variables have a strong positive correlation with the Chance of Admit. Moreover, we can expect GRE Scores, TOELF Score, and CGPA to have a positive linear relationship with the Chance of Admit.

# ### Checking for Multicollinearity

# In[ ]:


corr = df.corr()
fig, ax = plt.subplots(figsize=(8, 8))
colormap = sns.diverging_palette(220, 10, as_cmap=True)
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(corr, cmap=colormap, linewidths=.5, annot=True, fmt=".2f", mask=mask)
plt.show()


# From the above correlation matrix it can be seen that TOEFL Scores and GRE Scores have a very high correlation. This can be a cause of multicollinearity in the model, therefore, a Linear Regression is not deemed fit for this dataset and we should proceed with other linear modelling techniques.

# # Data Modelling

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(df.iloc[:,:-1],df.iloc[:,-1],test_size = 0.20, shuffle=False)


# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score, mean_squared_error

models =[['Linear Regression', LinearRegression()],
           ['Random Forest',RandomForestRegressor()],
           ['K-Neighbours', KNeighborsRegressor(n_neighbors = 2)],
           ['SVM', SVR()]]

model_output = {}
for name,model in models:
    model = model
    model.fit(X_train, Y_train)
    pred = model.predict(X_test)
    model_output[f'{name}'] = np.sqrt(mean_squared_error(Y_test, pred))

results = pd.DataFrame(model_output.items())
results.columns = ['Model', 'RMSE']
results.index = np.arange(1,len(results)+1)
results = results.sort_values(by=['RMSE'], ascending=True)
print("Models Trained")


# ## Visualising the results

# In[ ]:


fig = plt.figure(figsize=(8, 8))
sns.barplot(results['Model'],results['RMSE'],palette=reversed(sns.color_palette("rocket")))
plt.title("Comparing all the Models")
plt.show()


# From the graph above, it can be seen that the Linear Regression Model has the lowest Root Mean Squared Error (RMSE) while SVM has the largest RMSE. Since we suspect multicollinearity, we cannot use the Linear Regression model as it may be biased. Therefore, we choose the model with the next lowest RMSE and can tackle multicollinearity, which is Random Forest.

# ### Finding important features

# In[ ]:


model = RandomForestRegressor()
X = df.iloc[:,:-1]
Y = df.iloc[:,-1]
model.fit(X,Y)
feature_names = X.columns
features = pd.DataFrame()
features['Features'] = X.columns
features['Importance'] = model.feature_importances_
features = features.sort_values(by=['Importance'], ascending=False)
features.index = np.arange(1,len(X.columns)+1)


# ### Visualising importance of features

# In[ ]:


fig = plt.figure(figsize=(8, 8))
sns.barplot(features['Features'],features['Importance'],palette=sns.color_palette("rocket"))
plt.title("Feature Importance")
plt.show()


# Although we have visualised the feature importance, they cannot be relied upon since multicolinearity affects the feature importance. However, in Random Forest, the existence of multicolinearity does not affect the prediction accuracy. Therefore the model can only be relied upon for the final predictions and not for feature importance. 

# # <center> The End </center>

# Thank you for going through my notebook. Hopefully you have gained valuable insights into the dataset. As this was my first Kaggle notebook I would like if you Upvote as it will keep me boost my motivation! Kindly let me know if I can improve my work in any aspect.
