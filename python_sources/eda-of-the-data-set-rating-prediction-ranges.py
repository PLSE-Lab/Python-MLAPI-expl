#!/usr/bin/env python
# coding: utf-8

# # An analysis of Applications on Google Play Store
# 

# In[ ]:


#Imports and Dependencies 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import seaborn as sns 
color = sns.color_palette()
sns.set(rc={'figure.figsize':(20,10)})

import plotly
plotly.offline.init_notebook_mode(connected=True)
import plotly.plotly as py
import plotly.graph_objs as go

import plotly.figure_factory as ff
import cufflinks as cf
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler  

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


#Data Import
file = r'../input/googleplaystore.csv'
df = pd.read_csv(file)
df.drop_duplicates(subset='App', inplace=True)
print('Number of Applications in the Data Set:', len(df))


# In[ ]:


#Eliminating stray entries in the 'Installs' and 'Android Ver' columns 
df = df[df['Android Ver'] != 'NaN']
df = df[df['Installs'] != 'Free']


# In[ ]:


#Data Cleaning
#Removing , + M K and $ Symbols + All type conversions

#'Installs' Column
df['Installs'] = df['Installs'].apply(lambda x: x.replace('+', '') if '+' in str(x) else x)
df['Installs'] = df['Installs'].apply(lambda x: x.replace(',', '') if ',' in str(x) else x)
df['Installs'] = df['Installs'].apply(lambda x: float(x))
#'Size' Column
df['Size'] = df['Size'].apply(lambda x: str(x).replace('Varies with device', 'NaN') if 'Varies with device' in str(x) else x)
df['Size'] = df['Size'].apply(lambda x: str(x).replace('M', '') if 'M' in str(x) else x)
df['Size'] = df['Size'].apply(lambda x: str(x).replace(',', '') if 'M' in str(x) else x)
df['Size'] = df['Size'].apply(lambda x: float(str(x).replace('k', '')) / 1000 if 'k' in str(x) else x)
df['Size'] = df['Size'].apply(lambda x: float(x))
#'Price' Column
df['Price'] = df['Price'].apply(lambda x: str(x).replace('$', '') if '$' in str(x) else str(x))
df['Price'] = df['Price'].apply(lambda x: float(x))
#'Reviews' Column
df['Reviews'] = df['Reviews'].apply(lambda x: int(x))


# In[ ]:


df.sample(10)


# In[ ]:


#Overview of the data - Pie Plot of Categories and App Count 
categories = df['Category'].value_counts()

pie_chart = [go.Pie(
        labels = categories.index,
        values = categories.values,
        hoverinfo = 'value + label'
    
)]

layout = go.Layout(title = 'Distribution of Apps across Categories')

print('Total number of categories:', categories.count())
print('Count by categories:')
print(categories)
toPlot = go.Figure(data = pie_chart, layout = layout)

#Please refer to the Results section of the Kernel to find the plot
plotly.offline.plot(toPlot, filename='Pie Chart')


# In[ ]:


#Pairwise Plot of Numeric Features 
rating = df['Rating'].dropna()
size = df['Size'].dropna()
installs = df['Installs'][df.Installs!=0].dropna()
reviews = df['Reviews'][df.Reviews!=0].dropna()
type = df['Type'].dropna()
pairplot = sns.pairplot(pd.DataFrame(list(zip(rating, np.log10(installs), size, np.log10(reviews), type)),
                                     columns=['Rating', 'Installs', 'Size','Reviews', 'Type']), hue='Type', palette="Set1")


# In[ ]:


#Further Analysis using Join Plots
sns.set_style("ticks")
plt.figure(figsize = (10,10))
size_vs_rating = sns.jointplot(df['Size'], df['Rating'], kind = 'kde', color = "orange", size = 8)
review_vs_rating = sns.jointplot(df['Reviews'], df['Rating'], kind = 'reg', color = "orange", size = 8)
installs_vs_rating = sns.jointplot(df['Installs'], df['Rating'], kind ='reg', color = "orange", size = 8)


# In[ ]:


#Box Plot of Category and Avergae App Rating
print('Average App Rating = ', np.nanmean(df['Rating']))
print('Category vs. Avg Rating')
data = [{
    'y': df.loc[df.Category==category]['Rating'], 
    'type':'box',
    'name' : category,
    'showlegend' : False
    } for i,category in enumerate(list(set(df['Category'])))]

layout = {'title' : 'App ratings across major categories',
        'xaxis': {'tickangle':-40},
        'yaxis': {'title': 'Rating'}
         }
toPlot = go.Figure(data = data, layout = layout)

#Please refer to the Results section of the Kernel to find the plot
plotly.offline.plot(toPlot,filename = "BoxPlot")


# # A simple Regression Model to define and predict the Range of the Ratings for the Applications(Exploratory only)

# In[ ]:


df.head()


# In[ ]:


#Discard insignificant features
df.drop(['App', 'Last Updated', 'Current Ver'], 1, inplace = True)


# In[ ]:


df.head()


# In[ ]:


#Feature correlation matrix
fig = plt.figure(figsize = (16,8))
#corr = plt.matshow(df.apply(lambda x:pd.factorize(x)[0]).corr(), fignum = 1)
corr = df.corr(method = 'spearman', min_periods = 5)
plot = plt.matshow(corr, fignum = 1)
fig.colorbar(plot)


# In[ ]:


#Conversion of categorical variables to numeric variables - one hot encoding (manual)
new_df = pd.get_dummies(df, columns = ['Category', 'Content Rating', 'Genres', 'Android Ver', 'Type'])
new_df.head()


# In[ ]:


new_df.replace('?', -9999, inplace = True )
new_df.dropna(inplace = True)


# In[ ]:


new_df.head()


# In[ ]:


#Test-Train Split
X = np.array(new_df.drop(['Rating'],1))
y = np.array(new_df['Rating'])
X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size = 0.3)


# In[ ]:


#Feature scaling 
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# In[ ]:


#Model Defn. and Fitting
regressor = LinearRegression()  
regressor.fit(X_train, y_train)  


# In[ ]:


#Prediction 
y_pred = regressor.predict(X_test)


# In[ ]:


result = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
print('Simple Regression Results')
print(result.head())  


# In[ ]:


#Predictions with XGBoost 
import xgboost as xgb
model = xgb.XGBRegressor(n_estimators=70, learning_rate=0.08, gamma=0, subsample=0.70,
                           colsample_bytree=1, max_depth=7)


# In[ ]:


model.fit(X_train, y_train)


# In[ ]:


import math
print("RMSE: %.2f"
      % math.sqrt(np.mean((model.predict(X_test) - y_test) ** 2)))


# In[ ]:


y_pred1 = model.predict(X_test)
result = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred1})  
result.head()  


# Fine tuning the prediction - by introducing a range

# In[ ]:


#Introducing a range for the predicted Ratings

diff_list = abs(result['Predicted']-result['Actual'])
mean_diff = np.nanmean(diff_list)
variance = np.square(diff_list-mean_diff).sum()/len(diff_list)
standard_deviation = np.sqrt(variance)
lower_bound = np.around(y_pred - (standard_deviation),2)
upper_bound = np.around(y_pred + (standard_deviation),2)


# In[ ]:


result_final = pd.DataFrame({'Actual Rating':y_test, 'Lower Bound on Predicted Rating': lower_bound, 'Upper Bound on Predicted Rating': upper_bound })
print(result_final.sample(10))
print('Writng results to Result Set file')
result_final.to_csv("Result Set.csv")


# In[ ]:




