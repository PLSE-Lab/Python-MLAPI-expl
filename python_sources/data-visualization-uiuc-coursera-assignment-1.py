#!/usr/bin/env python
# coding: utf-8

# **First Step - Reading the Dataset**
# 
# *Combined Land-Surface Air and Sea-Surface Water Temperature Anomalies (Land-Ocean Temperature Index, LOTI)*
# 
# * As a very first step, let's read the data into the notebook in order to observe the high level format of the dataset.
# * This will help us to identify what kind of data presented and how they are presented in the dataset.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # tools for data visualization 

dataset1 = pd.read_csv("../input/ExcelFormattedGISTEMPDataCSV.csv")

print (dataset1)


# **What's in the dataset ?**
# 
# * Contains 136 rows and 19 columns
# * According to [https://data.giss.nasa.gov/gistemp/](http://) (owner of the dataset), this dataset presents the temperature anomalies from 1880 to 2015.
# * This dataset has the deviations (of temperature) from the corresponding 1951-1980 means.
# * Further the temperature data listed here are the multiples of hundred of actual degrees of Celcius. That means that we have to divide the temperature deviation values in the dataset by 100. (i.e : if table value = 68 then actual deviation = 0.68 deg-C)
# 
# * **J-D** : mean deviation of the tweleve months of a year.
# 
# * **D-N** : mean deviation of the tweleve months; to November of a year from December of the previous year. (e.g : D-N of 2007 is the mean deviation value from December 2006 to November 2007)
# 
# * **DJF** : mean deviation of the three months; to February of a year from December of the previous year. (e.g : DJF of 2003 is the mean deviation value from December 2002 to February 2003)
#     * These are the three months related to the **Winter** in *nothern hemisphere*.
#     
# * **MAM** : mean deviation of the three months March, April and May of a year. 
#     * These are the three months related to the **Spring** in *nothern hemisphere*.    
#     
# * **JJA** : mean deviation of the three months June, July and August of a year.
#     * These are the three months related to the **Summer** in *nothern hemisphere*.
#     
# * **SON** : mean deviation of the three months September, October and November of a year.
#     * These are the three months related to the **Autumn** in *nothern hemisphere*.
#     
# * Missing Data = " \*\*\*\*\* "

# **Data Preprocessing**

# In[ ]:


# Setting missing values to NaN in order to impute them
dataset1 = dataset1.replace(['*****', '****', '***'], np.nan)

# Splitting data into dependent and independent variables
x = dataset1.iloc[:, :-18].values
Y = dataset1.iloc[:, 1:].values

# Replacing the missing values with the mean of their corresponding columns
from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
Y[:, 0:] = imputer.fit_transform(Y[:, 0:])
Y[:, 0:] = Y[:, 0:] / 100 # converting to actual degrees of Celcius values
print(Y)


# **Data Visualization**
# 
# ***Annual Global Temperature Anomalies***
#  
#  * In the 14th column of the dataset (i.e 13th column in the Y array), J-D values are presented. As we discussed above J-D value gives mean temperature deviation for a particular year.     

# In[ ]:


plt.figure(figsize=(25, 10))
plt.plot(x, Y[:, 12], marker = '.')
plt.title('Annual Global Temperature Anomalies (Mean Temperature Deviations) WRT Mean Temperature of 1951-1980 Interval', fontsize = 20)
plt.ylabel('Temperature Deviation in Degree Celcius', fontsize = 14)
plt.xlabel('Year', fontsize = 14)
plt.show()


# ***Seasonal Global Temperature Anomalies***

# In[ ]:


import matplotlib.lines as mlines

plt.figure(figsize=(25, 10))

plt.plot(x, Y[:, 14], marker = '.', color = 'black')
plt.plot(x, Y[:, 15], marker = '.', color = 'red')
plt.plot(x, Y[:, 16], marker = '.', color = 'blue')
plt.plot(x, Y[:, 17], marker = '.', color = 'green')

plt.title('Seasonal Global Temperature Anomalies (Mean Temperature Deviations) WRT Mean Temperature of 1951-1980 Interval', fontsize = 20)
plt.ylabel('Temperature Deviation in Degree Celcius', fontsize = 14)
plt.xlabel('Year', fontsize = 14)

black_line = mlines.Line2D([], [], color = 'black', marker = '.', label = 'Winter')
red_line = mlines.Line2D([], [], color = 'red', marker = '.', label = 'Spring')
blue_line = mlines.Line2D([], [], color = 'blue', marker = '.', label = 'Summer')
green_line = mlines.Line2D([], [], color = 'green', marker = '.', label = 'Autumn')

plt.legend(handles = [black_line, red_line, blue_line, green_line])

plt.show()


# * *Fitting some multiple linear regression models to learn how the seasonal global temperature has deviated over time.*

# In[ ]:


new_X = x

for i in range(0, 6):
    if i != 1:
        new_X = np.append(np.power(x, i), new_X, axis = 1)

from sklearn.linear_model import LinearRegression

regressor1 = LinearRegression()
regressor2 = LinearRegression()
regressor3 = LinearRegression()
regressor4 = LinearRegression()

regressor1.fit(new_X, Y[:, 14])
regressor2.fit(new_X, Y[:, 15])
regressor3.fit(new_X, Y[:, 16])
regressor4.fit(new_X, Y[:, 17])

black_reg = regressor1.predict(new_X)
red_reg = regressor2.predict(new_X)
blue_reg = regressor3.predict(new_X)
green_reg = regressor4.predict(new_X)

plt.figure(figsize=(25, 10))

plt.scatter(x, Y[:, 14], marker = '.', color = 'black')
plt.scatter(x, Y[:, 15], marker = '.', color = 'red')
plt.scatter(x, Y[:, 16], marker = '.', color = 'blue')
plt.scatter(x, Y[:, 17], marker = '.', color = 'green')

plt.plot(x, black_reg, color = 'black')
plt.plot(x, red_reg, color = 'red')
plt.plot(x, blue_reg, color = 'blue')
plt.plot(x, green_reg, color = 'green')

plt.title('Seasonal Global Temperature Anomalies (Mean Temperature Deviations) WRT Mean Temperature of 1951-1980 Interval', fontsize = 20)
plt.ylabel('Temperature Deviation in Degree Celcius', fontsize = 14)
plt.xlabel('Year', fontsize = 14)

black_line = mlines.Line2D([], [], color = 'black', marker = '.', label = 'Winter')
red_line = mlines.Line2D([], [], color = 'red', marker = '.', label = 'Spring')
blue_line = mlines.Line2D([], [], color = 'blue', marker = '.', label = 'Summer')
green_line = mlines.Line2D([], [], color = 'green', marker = '.', label = 'Autumn')

plt.legend(handles = [black_line, red_line, blue_line, green_line])

plt.show()


# **Observing Global Warming**
# 
# * The 3D graph below depicts how the global temperature has increased over the years. (i.e Global Warming)
# * *The unusual surge (spike) in temperature deviation around the 1940s might had happend because of the Second World War and the heavy industrialization followed by that .*

# In[ ]:


Y1 = np.arange(1, 13, 1)
X1, Y1 = np.meshgrid(x, Y1)
Z = np.matrix.transpose(Y[:, :12])

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

fig = plt.figure(figsize = (30, 20))
ax = fig.gca(projection='3d')

surf = ax.plot_surface(Y1, X1, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=0.25, aspect=10)
plt.show()


# * Author : Dinuka Manohara De Zoysa (Undergraduate, Department of Computer Science and Engineering, University of Moratuwa, Sri Lanka)
# * Created on : 21/04/2018
