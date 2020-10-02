#!/usr/bin/env python
# coding: utf-8

# **In this tutorial I will be doing tutorial on the basic stats,plotting,pandas that are needed for carrying out datascienc projects.This is a kernel in process and I will be updating my work in coming days.If you like my work please do vote.**

# In[5]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# **A] Numpy Basics **

# In[15]:


# Creating List 
my_list1=[1,2,3,4]
my_list2=[11,22,33,44]
my_list1


# In[16]:


# Creating an array 
my_array1=np.array(my_list1)
my_array1


# In[18]:


# Creating List of list 
my_lists=[my_list1,my_list2]
my_lists


# In[20]:


# Creating a multidimensional array or Matrix 
my_array2=np.array(my_lists)

my_array2


# In[22]:


# Finding out the dimension of the matrix or array
my_array2.shape


# In[24]:


# Finding out the Data Types of the array 
my_array2.dtype


# In[26]:


# Creating Zero Arrays 
my_zero_array=np.zeros(5)
my_zero_array


# In[28]:


my_zero_array.dtype


# In[29]:


np.ones([5,5])


# **B] Mean,Median,Mode and Introducing Numpy **
# 
# Lets create an income data,centered arounf 27000 with a normal distribution and standard deviation of 15000 with 10000 point.Then we will compute the mean,median and Mode for this data 

# In[ ]:


import numpy as np 
incomes=np.random.normal(27000,15000,10000)
# Median=27000
# Standard Deviation =15000
# Number of data points =10000


# **1.Displaying the data using a histogram **

# In[ ]:


import matplotlib.pyplot as plt 
plt.hist(incomes,50)
plt.ioff()


# The curve shows us that the salary data is following a normal distribution curve.

# **2.Calculating the mean of the data **

# In[ ]:


np.mean(incomes)


# We can see that the mean of the data is 27083.05 which is more than the mean assumed 27000.This is due to randomness incorporated in the data.

# **3.Calculating the median of the data **

# In[ ]:


np.median(incomes)


# Median of 26972.4 means that half of the people's salary is less than median value of 26972.4 and Half of the people have salary more than median value.

# **4.What is the differemce between mean and Median?**
# 
# We can explain this by adding an outlier to this data.Let us assume a rich person has a salary of 1000000000.This will shift the mean of the data but lets see what will happen to the median of the data

# In[ ]:


incomes=np.append(incomes,[1000000000])


# In[ ]:


np.mean(incomes)


# So we can see that the mean of the data has shifted from 27083 to 127070. This is a huge shift in the mean of the data due to an outlier 

# In[ ]:


np.median(incomes)


# We can see that by adding the outlier salary value to the data set there is not much change in the median of the dataset.So if we can to go my mean of the data set then we would be assuming wrong range for income of the dataset.So in this case it is better to use the median value.

# **5.Mode**

# In[ ]:


from scipy import stats
stats.mode(incomes)


# So the most repated Salary value is 1.e+09 and it repeats 2 times.This is not a good data for calculating the mode.We will generate a data of age of people between 18 and 90 for 500 people.

# In[ ]:


ages=np.random.randint(18,high=90,size=500)
ages


# In[ ]:


stats.mode(ages)


# So age of 34 is the mode of the dataset as it is repaeated 16 times which is most in the dataset.

# **C] Standard deviation and variance **

# In[ ]:


incomes=np.random.normal(100,20,10000)
plt.hist(incomes,50)
plt.show()


# **1.Standard Deviation:**

# In[ ]:


incomes.std()


# So we can say that as the standard deviation of the income data is 20.18 the spread of the data is 20.18 about the mean which is 100.

# **2.Variance **

# In[ ]:


incomes.var()


# **D] Matplotlib Basics**

# **1.Line Graph**

# In[ ]:


from scipy.stats import norm 
import matplotlib.pyplot as plt 
x=np.arange(-3,3,0.001)
plt.plot(x,norm.pdf(x))
plt.ioff()
plt.show()


# **2.Multiple plots on the graph **

# In[ ]:


plt.plot(x,norm.pdf(x))
plt.plot(x,norm.pdf(x,1,0.5))
plt.ioff()


# **3.Save it to a file **

# In[ ]:


plt.plot(x,norm.pdf(x))
plt.plot(x,norm.pdf(x,1,0.5))
#plt.savefig('',format='png')   Specify the path where you want to save the plot 
plt.ioff()


# **4.Adjust the axes **

# In[ ]:


plt.plot(x,norm.pdf(x))
plt.plot(x,norm.pdf(x,1,0.5))
axes=plt.axes()
axes.set_xlim([-5,5])
axes.set_ylim(0,1)
axes.set_xticks([-5,-4,-3,-2,-1,0,1,2,3,4,5])
axes.set_yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
plt.plot()
plt.ioff()


# **5.Adding Grid **

# In[ ]:


plt.plot(x,norm.pdf(x))
plt.plot(x,norm.pdf(x,1,0.5))
axes=plt.axes()
axes.set_xlim([-5,5])
axes.set_ylim(0,1)
axes.set_xticks([-5,-4,-3,-2,-1,0,1,2,3,4,5])
axes.set_yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
axes.grid()
plt.plot()
plt.ioff()


# **6.Change the Line types and Colors **

# In[ ]:


plt.plot(x,norm.pdf(x),'b--')
plt.plot(x,norm.pdf(x,1,0.5),'r:')
axes=plt.axes()
axes.set_xlim([-5,5])
axes.set_ylim(0,1)
axes.set_xticks([-5,-4,-3,-2,-1,0,1,2,3,4,5])
axes.set_yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
axes.grid()
plt.plot()
plt.ioff()


# **7.Labelling the axes and adding legend **

# In[ ]:


plt.plot(x,norm.pdf(x),'b--')
plt.plot(x,norm.pdf(x,1,0.5),'r:')
axes=plt.axes()
axes.set_xlim([-5,5])
axes.set_ylim(0,1)
axes.set_xticks([-5,-4,-3,-2,-1,0,1,2,3,4,5])
axes.set_yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
axes.grid()
plt.xlabel('Greebles')
plt.ylabel('Probability')
plt.legend(['Sbeetches','Gacks'],loc=1)
plt.plot()
plt.ioff()


# **8.XKCD Style**

# In[ ]:


plt.xkcd()
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
plt.xticks([])
plt.yticks([])
ax.set_ylim([-30,10])
data=np.ones(100)
data[70:]-=np.arange(30)
plt.annotate(
'THE DAY I REALIZED\nI COULD COOK BACON\nWHENEVER I WANTED',
xy=(70,1),arrowprops=dict(arrowstyle='->'),xytext=(15,-10))
plt.plot(data)
plt.xlabel('time')
plt.ylabel('my overall health')
plt.ioff()
plt.show()


# **9.Pie Chart**

# In[ ]:


#remove xkcd mode:
#plt.rodefaults()
values=[12,55,4,32,14]
colors=['r','g','b','c','m']
explode=[0,0,0.2,0,0]
labels=['India','United States','Russia','China','Europe']
plt.pie(values,colors=colors,labels=labels,explode=explode)
plt.title('Student Locations')
plt.ioff()


# **10.Bar Chart **

# In[ ]:


values=[12,55,4,32,14]
colors=['r','g','b','c','m']
plt.bar(range(0,5),values,color=colors)
plt.show()


# **11.Scatter Plot **

# In[ ]:


X=np.random.randn(500)
Y=np.random.randn(500)
plt.scatter(X,Y)
plt.show()


# **12.Histogram**

# In[ ]:


incomes=np.random.normal(27000,15000,10000)
plt.hist(incomes,50)
plt.show()


# **13.Box and Whisker Plot **

# In[ ]:


uniformSkewed=np.random.rand(100)*100-40
high_outliers=np.random.rand(100)*50+100
low_outliers=np.random.rand(10)*-50-100
data=np.concatenate((uniformSkewed,high_outliers,low_outliers))
plt.boxplot(data)
plt.show()


# In[ ]:




