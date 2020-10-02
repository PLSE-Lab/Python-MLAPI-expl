#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
print(os.listdir('../input'))


# In[ ]:


import pandas as pd
import numpy as n
import seaborn as sns

path_data = os.path.join('..', 'input', "Coastal Data System - Waves (Mooloolaba) 01-2017 to 06 - 2019.csv")


waves = pd.read_csv('../input/waves-measuring-buoys-data-mooloolaba/Coastal Data System - Waves (Mooloolaba) 01-2017 to 06 - 2019.csv')

waves.columns = ["time", "Hs", "Hmax","Tz","Tp","direction", "temp"]
waves["time"] = pd.to_datetime(waves["time"])
waves["year"] = waves["time"].astype(str).str[0:4]
waves["year"] = waves["year"].astype(int)

waves["time1"] = waves["time"].astype(str).str[11:16]
waves["month"] = waves["time"].astype(str).str[5:7]

waves["dir"] = 10* (waves["direction"] // 10)

waves["temp1"] = (waves["temp"] // 1)
waves.info()
waves = waves[waves.Hs > 0]
waves.head()


# In[ ]:


waves[waves.temp1 > 0].pivot_table('Hs', index='year',columns ="temp1" )


# In[ ]:


waves[waves.temp1 > 0].pivot_table('Hs', index="temp1",columns = 'year').plot(figsize=(20,10), kind="bar")


# In[ ]:


waves[waves.temp > 0].pivot_table('Hs', index="temp1").plot(figsize=(20,10), kind = "bar")


# In[ ]:


import matplotlib.pyplot as plt
sns.set() # Seaborn-Stile verwenden

waves[waves.Hs > 0].pivot_table("Hs",index = "time").plot(figsize=(20,10))
plt.ylabel("Height of waves")


# In[ ]:


waves[waves.Hs > 0].pivot_table("Hmax",index = "time").plot(figsize=(20,10))
plt.ylabel("Height of waves")


# In[ ]:


waves[waves.Hs > 0].pivot_table('Hs', index='year',columns = "month")


# In[ ]:


waves.pivot_table('Hs', index="month",columns = 'year').plot(figsize=(20,10))


# # Wave direction correlated to size 

# In[ ]:


waves[waves.direction > 0].pivot_table('Hs', index="dir").plot(figsize=(20,10), kind = "bar")


# In[ ]:


waves[waves.direction > 0].pivot_table('Hs', columns="dir")


# In[ ]:


from math import pi
y1 = waves[waves.direction > 0].Hs.to_list()
x1 = waves[waves.direction > 0].direction.to_list()


categories = ["North","West","South","East"]
N = len(categories)


plt.rcParams['figure.figsize'] = (15, 15)
plt.axes(projection='polar')

angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]


plt.xticks(angles[:-1], categories, color='black', size=14)
plt.yticks(color="black", size=15)
#plt.ylim(0,100)


plt.polar(x1, y1,  'ro', color= "blue")
#plt.text(x, y, '%d, %d')
plt.show()


# # Conclusion in direction families

# In[ ]:


y2 = waves[waves.direction > 0].Hs.to_list()
x2 = waves[waves.direction > 0].dir.to_list()
#print(y)

categories = ["North","West","South","East"]
N = len(categories)


plt.rcParams['figure.figsize'] = (15, 15)
plt.axes(projection='polar')

angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]


plt.xticks(angles[:-1], categories, color='black', size=14)
plt.yticks(color="black", size=15)
#plt.ylim(0,3)


plt.polar(x2, y2,  'ro', color= "blue")
#plt.text(x, y, '%d, %d')
plt.show()


# # Wave size over the year

# In[ ]:


waves.pivot_table('Hs', index="time1").plot(figsize=(20,10))


# # Wave size in January

# In[ ]:


waves[waves.month == "01"].pivot_table('Hs', index="time1").plot(figsize=(20,10))


# # Wave size in August

# In[ ]:


waves[waves.month == "08"].pivot_table('Hs', index="time1").plot(figsize=(20,10))


# # Prediction of wave size

# In[ ]:


import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


waves.head()


# In[ ]:


X_waves = waves.drop("time",axis = 1)
X_waves = X_waves.drop("Hs",axis = 1)
X_waves = X_waves.drop("year",axis = 1)
X_waves = X_waves.drop("time1",axis = 1)
X_waves = X_waves.drop("dir",axis = 1)
#X_waves = X_waves.drop("Hmax",axis = 1)
X = X_waves.drop("temp",axis = 1)
print(X.shape)
#print(X)
y = waves["Hs"]
print(y.shape)


# In[ ]:


plt.figure(figsize=(15,10))
plt.tight_layout()
seabornInstance.distplot(waves['Hs'])


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[ ]:


regressor = LinearRegression()  
regressor.fit(X_train, y_train)


# In[ ]:


coeff_df = pd.DataFrame(regressor.coef_, X.columns, columns=['Coefficient'])  
coeff_df


# In[ ]:


y_pred = regressor.predict(X_test)


# In[ ]:


df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df1 = df.head(50)


# In[ ]:


df1.plot(kind='bar',figsize=(10,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()


# In[ ]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

