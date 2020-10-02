#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # My tasks to do
# 
# 
#                            # Get Descriptive stats
#                            # Perform correlations
#                            # histograms
#                            # box plots
#                            # scatter plots
#                            # density plots
#                            # linear regression model
# 
# 

# # importing libraries  

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import tree
from ipykernel import kernelapp as app
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams


# #  Data Importing

# In[ ]:


#loading the csv
df=pd.read_csv("/kaggle/input/crop-prediction/AgrcultureDataset.csv",encoding = "ISO-8859-1")
df.dtypes


# # Data preparation and encoding

# In[ ]:


#indian agricultural production dataset
df.head()


# In[ ]:


#converting production to numeric type
df['Production']=pd.to_numeric(df['Production'],errors='coerce')


# In[ ]:


#grouping area and production for each year by mean
data=df.groupby(['Crop_Year'])['Area','Production'].mean()
data=data.reset_index(level=0, inplace=False)
data


# In[ ]:


#calulation cpi(  )


data['CPI']=data['Production']/data['Area']
data.head()


# # discriptive analysis

# In[ ]:


data.describe()


# # Box plots

# In[ ]:


#boxplot plotting
import seaborn as sns
sns.boxplot(x=data['CPI'])


# In[ ]:


data = data[np.isfinite(data['CPI'])]
data=data[data.CPI >43]
data=data[data.CPI <51]
data.set_index('Crop_Year')
data


# # plotting histogram

# In[ ]:


#plotting histogram
data.hist()


# In[ ]:


corrmat = data.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);


# # Scatter plots 

# In[ ]:


#scatterplot
sns.set()
cols = ['Crop_Year', 'Area', 'Production', 'CPI']
sns.pairplot(data[cols], size = 2.5)
plt.show();


# In[ ]:





# In[ ]:


#comparison of production and area for each year
x_axis=data.Crop_Year
y_axis=data.Area

y1_axis=data.Production

plt.plot(x_axis,y_axis)
plt.plot(x_axis,y1_axis,color='r')

plt.title("Production and area ")
plt.legend(["Production ","Area"])
plt.show()


# In[ ]:


#plotting of production
x_axis=data.Crop_Year
y1_axis=data.Production



plt.plot(x_axis,y1_axis)

plt.title("Year vs Production ")
plt.legend(["Year ","Production"])
plt.show()


# # Applying random forest

# In[ ]:


#importing random forest regressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
# from sklearn.cross_validation import train_test_split


# In[ ]:


#splitting and fitting of the model
x=data.iloc[:,0:1].values
y=data.iloc[:,3].values
regressor=RandomForestRegressor(n_estimators=12,random_state=0,n_jobs=1,verbose=13)

regressor.fit(x,y)


# In[ ]:


#predicting for the test values
y_pred=regressor.predict(x)
y_pred


# In[ ]:


#random forest steps plotting
x_grid=np.arange(min(x),max(x),0.001)
x_grid=x_grid.reshape(len(x_grid),1)
plt.scatter(x,y,color='r')
plt.plot(x_grid,regressor.predict(x_grid),color='b')
a=plt.show()
a


# In[ ]:





# 
# #  DENSITY PLOTS

# In[ ]:


sns.distplot(data['CPI'], hist=True, kde=True, 
             bins=int(180/5), color = 'darkblue',
             kde_kws={'linewidth': 4})


# In[ ]:


sns.distplot(data['Area'], hist=True, kde=True, 
             bins=int(180/5), color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})


# In[ ]:


sns.distplot(data['Production'], hist=True, kde=True, 
             bins=int(180/5), color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})




# In[ ]:


sns.distplot(data['Crop_Year'], hist=True, kde=True, 
             bins=int(180/5), color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})


# # regression model

# In[ ]:


#actual and predicted values
dm = pd.DataFrame({'Actual': y, 'Predicted': y_pred}).reset_index()
x_axis=dm.index
y_axis=dm.Actual
y1_axis=dm.Predicted
plt.plot(x_axis,y_axis)
plt.plot(x_axis,y1_axis)
plt.title("Actual vs Predicted")
plt.legend(["actual ","predicted"])
b=plt.show()
b


# In[ ]:




