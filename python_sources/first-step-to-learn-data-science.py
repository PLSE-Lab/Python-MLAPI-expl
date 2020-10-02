#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# **P.S. IN PREPARING THIS DOCUMENT, I MOSTLY USED THE KERNEL CALLED "Data ScienceTutorial for Beginners". https://www.kaggle.com/kanncaa1/data-sciencetutorial-for-beginners**
# 
# **I AM PREPARING THIS KERNEL TO LEARN, NOT TO TEACH. BUT HOPEFULLY, IT IS ALSO HELPFUL FOR YOU.**
# 

# **GETTING STARTED WITH DATA ANALYSIS**
# 
# First of all, we should introduce the csv folder.

# In[ ]:


data = pd.read_csv('../input/weather_madrid_LEMD_1997_2015.csv')


# To become familiar with data, you can get basic informations from data.

# In[ ]:


data.info()


# At this step, i changed and organized the column names.

# In[ ]:


data.columns = [each.split()[0]+"_"+each.split()[1] if (len(each.split()) > 1) else each for each in data.columns]
data.rename(columns = {" CloudCover":"CloudCover"," Events":"Events"}, inplace = True)


# You can examine the correlation between attributes in visual or else.

# In[ ]:


data.corr()


# With Seaborn library, for example you can visualized your data as color-encoded matrix. At this example, i used heatmap to make more understandable the correlations between attributes.
# 
# The means of parameters which i used:
# 
# * annot :  It is provide, write the data value in each cell when it is True.
# * linewidths  : The line width between cells.
# * fmt : String formatting code to use annottations. Of course, when annot is True.
# * ax : Axes in which to draw the plot.
# * cmap : The mapping from data values to color space.
# 

# In[ ]:


f,ax = plt.subplots(figsize = (20,20))
sns.heatmap(data.corr(),annot = True, linewidths = 0.5, fmt = ".1f", ax = ax, cmap = "Blues")
plt.show()


# Again, you can get basic information about data with head( ) function.  This function shows all columns and five rows, as default. You can change the rows number according to your needs. At this example, i choose ten rows.

# In[ ]:


data.head(10)


# In[ ]:


data.columns


# If you want some visual results, you can use line plot, scatter plot or histogram etc.

# In[ ]:


data.CloudCover.plot(kind = 'line', color = "green", label = "Cloud Cover", figsize = (15,8),
                        linewidth = 1, alpha = 0.5, grid = True, linestyle = "-")

data.Precipitationmm.plot(kind = 'line', color = "green", label = "Precipitation (mm)", figsize = (15,8),
                            linewidth = 1, alpha = 1, grid = True, linestyle = "-")
plt.legend()
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Line Plot")
plt.show()


# In[ ]:


data.plot(kind = "scatter", x = "Mean_TemperatureC", y = "Mean_Humidity", color = "red", 
          alpha = .8, figsize = (15,8))
plt.xlabel("Mean Temperature")
plt.ylabel("Mean Humidity")
plt.title("Mean Temperature - Mean Humidity Scatter Plot")
plt.show()


# In[ ]:


data.Dew_PointC.plot(kind = "hist", bins = 10, figsize = (8,8), color = "r", alpha = 0.7)
plt.title("Dew Point")
plt.show()


# In[ ]:


data.CloudCover.plot(kind = "hist", bins = 50)
plt.clf()


# There is some examples about how to use dictionaries:

# In[ ]:


dictionary = {"Feynman" : "Istatistik", "Akkaya" : "RadarTemelleri"}
print(dictionary.keys())
print(dictionary.values())


# In[ ]:


dictionary['Feynman'] = "FizikDersleri"
print(dictionary)
dictionary['Nisanyan'] = "EasternTurkey"
print(dictionary)
del dictionary['Akkaya']
print(dictionary)
print('Feynman' in dictionary)
dictionary.clear()
print(dictionary)


# In[ ]:


series = data["Min_TemperatureC"]
print(type(series))
frame = data[["Min_TemperatureC"]]
print(type(frame))


# There is some logical comparison and filtering for get more information about data:

# In[ ]:


x = data["Max_TemperatureC"] > 30
data[x]


# In[ ]:


data[np.logical_and(data['Max_TemperatureC'] > 30, data['Min_TemperatureC'] > 25)]


# In[ ]:


data[(data["Max_TemperatureC"] > 30) & (data["Min_TemperatureC"] > 25)]


# In[ ]:


for index,value in data[['Dew_PointC']][0:5].iterrows():
    print(index," : ",value)


# **USER DEFINED FUNCTIONS**
# 
# When you define your own function, you should use **def()** keyword.
# And also if you want to create more understandable function you can use *docstrings*. Let's see:

# In[ ]:


n = int(input("Enter the number:"))

def cube():
    """this function calculates the cube of numbers"""    
    s = n**3
    return s
print("Square of number: " , cube())


# **I'm sorry about error. Please try code yourself for see the result. However, I uploaded some images here so you can review the result.**
# 
# [![image](https://i.hizliresim.com/bVkrYb.png)](https://hizliresim.com/bVkrYb)
# 
# [![image](https://i.hizliresim.com/QLvaWZ.png)](https://hizliresim.com/QLvaWZ)

# **SCOPE**
# 
# All variables which you defined in your code have a scope.
# 
# The **local** scope is the variable which you defined in a method or a function.
# 
# The **global** scope is the variable which you defined in main body.
# 
# The **built-in** scope is pre-defined names. Like *print, len, int* etc.
# 

# In[ ]:


x = 5 
def s():
    x = 25 
    return x
print("The global x: " , x)
print("The local x: " , s())


# When you used an unqualified name, Python searches that name in the local scope, then the global scope and then the built-in scope. This can be compared to kind of hierarchy for easier understanding. Let's examine that:

# In[ ]:


x = 5
def g():
    y = x**2 # there is no local scope so it will use the global one.
    return y
print(g())


# There is any rule about that but it is not recommended to use pre-defined names. If you want to learn them, first import **builtins** module then with **dir** examine the built-in scopes.

# In[ ]:


import builtins
dir(builtins)


# In addition, there is **enclosed** variables. Enclosed variables are used in the upper function in structures with **nested functions**.
# 
# If we add the enclosed variables to our imaginary hierarchy, it would be like that: first the local scope, then the enclosed scope, and then the global scope, and finally the built-in scope.

# In[ ]:


from math import sqrt
x = 9 # global scope
def f1():
    x = 4 # enclosed scope
    z = sqrt(x) # enclosed scope
    def f2():
        y = z * x # local scope
        return y
    return x + f2() + z
print(f1())


# **DEFAULT AND FLEXIBLE ARGUMENTS**
# 
# When defining a function, you can assign a default value to a variable. Let's examine that:

# In[ ]:


def g(a,b,c = 4): 
    y = a + b + c
    return y
print(g(9,13)) # it uses c variable's default value
#change default argument
print(g(9,13,6))


# If you want to create a function with flexible arguments, you can use ***args**  and ****kwargs** method. Here, args can be one or more and kwargs is a dictionary.
# 
# Actually args and kwargs are traditional parameters. What is important here is the number of asterisks. When you use the parameter with one asterisk the result would be a tuple. But when you use the parameter with double asterisks, the result would be a dictionary.

# In[ ]:


def h(*args):
    for i in args:
        print(i)
h(5)

print(" ")

h(5,0,5,0)

print("")

def f(**kwargs):
    for key, value in kwargs.items():
        print(key, ":", value)
        
f(harper_lee = "to kill a mockingbird", lutgens_et_al= "essentials of geology")


# **LAMBDA( ) FUNCTION**  
# 
# Usage of lambda function:
# 
# output = **lambda** input:expression
# 
# * The short way of function defining.
# * You can use the lambda function with just one expression.

# In[ ]:


cube = lambda x:x**3
print(cube(4))


# **MAP( ) FUNCTION**  
# 
# Usage of map function:
# 
# output = **map**(function, input)
# 
# * It takes lists and numbers as arguments.
# * It applies the function to all elements of list.

# In[ ]:


list1 = [4,5,7,10]
square = map(lambda x:x**2,list1)
print(list(square))

