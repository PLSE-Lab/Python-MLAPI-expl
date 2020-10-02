#!/usr/bin/env python
# coding: utf-8

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

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


#loading the csv
df=pd.read_csv("../input/agricultural-production-india/apy.csv",encoding = "ISO-8859-1")
df.dtypes


# In[ ]:


#indian agricultural production dataset
df.head()


# In[ ]:


#converting production to numeric type
df['Production']=pd.to_numeric(df['Production'],errors='coerce')


# In[ ]:


from mpl_toolkits.basemap import Basemap
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")


# In[ ]:


state=pd.read_csv("../input/indian-states-lat-lon/lat.csv")
state.head()


# In[ ]:


dff=pd.merge(state.set_index("state"),df.set_index("State_Name"), right_index=True, left_index=True).reset_index()
dff.head()


# In[ ]:


from mpl_toolkits.basemap import Basemap
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")


# In[ ]:


m = Basemap(projection='mill',llcrnrlat=5,urcrnrlat=40, llcrnrlon=60,urcrnrlon=110,lat_ts=20,resolution='c')


# In[ ]:


longitudes = dff["lon"].tolist()
latitudes = dff["lat"].tolist()
#m = Basemap(width=12000000,height=9000000,projection='lcc',
            #resolution=None,lat_1=80.,lat_2=55,lat_0=80,lon_0=-107.)
x,y = m(longitudes,latitudes)


# In[ ]:


fig = plt.figure(figsize=(12,10))
plt.title("All affected areas")
m.plot(x, y, "o", markersize = 3, color = 'blue')
m.drawcoastlines()
m.fillcontinents(color='white',lake_color='aqua')
m.drawmapboundary()
m.drawcountries()
plt.show()


# In[ ]:


#grouping area and production for each year by mean
data=df.groupby(['Crop_Year'])['Area','Production'].mean()
data=data.reset_index(level=0, inplace=False)
data


# In[ ]:


#calulation cpi
data['CPI']=data['Production']/data['Area']
data.head()


# In[ ]:


data.describe()


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


# In[ ]:


#plotting histogram
data.hist()


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


# In[ ]:





# In[ ]:


#importing random forest regressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import train_test_split


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





# In[ ]:


#calculation of mse
from sklearn.metrics import mean_squared_error
print('mse:%.2f'%mean_squared_error(y,y_pred))


# In[ ]:




