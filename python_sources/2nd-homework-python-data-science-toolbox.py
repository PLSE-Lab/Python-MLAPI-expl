#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# In[ ]:


#user defined function
def return_tuple():
    t = (1,2,3)
    return t #returning t tuple
a,s,d = return_tuple() #a defined as 1,b 2, c 3
print(a,s,d)


# In[ ]:


#scope (global,local,builtins)
x = 5
def func():
    x = 10       
    return x
print(x) 
print(func())


# In this case there are 2 xs, global x and local x,
# global x is in the top, local is under the function.
# When you write "return x * without giving an x value under function(local), python looks at global space. if there is an x value at local space, x's value is in local
# 

# In[ ]:


x = 5
def func1():
    return x
print(x)
print(func1())


# when there is no x value in local and global, last choice, python will look at **builtins** 

# In[ ]:


import builtins
dir(builtins)


# In[ ]:


##nested functions are unified(func inside func) functions   example:
def f():
    e=a+2
    def f2():
        e=3
        return e
    return a
print(f())


# In[ ]:


##default and flexible arguments 
def k(a,b,c=4):
    a = b + c
    print(a)
k(1,2,3)    


# In here, **c** has a default value, 4. It means if you do not enter a value for **c** , it automatically make it 4. a and b don't have a default so you should enter a value for 'em.

# In[ ]:


#flexible *args
def k2(*a):
    for i in a:
        print(i)
k2(6,5,6,7)
print("")
k2(2)
print("")
k2()


# We see a multiplication sign near the  **a** . It means you can enter values as much as you want. Or you don't enter.

# In[ ]:


#flexible *kwargs
def f_dict(**a):
    for key,value in a.items(): #-> provides to see dictionaries' key and value
        print(key,value)
f_dict(kind = "cake",flavour = "banana",piece = 12)
print(f_dict)


# Now there are two multiplication sign. It means you can make dictionary.As you can see at line 5.

# In[ ]:


#making lambda function
def sqrt(a):
    return a**0.5 #this is common way when write funcs
print(sqrt(16))
sqrt2 = lambda a: a**0.5 # this is lambda way
print(sqrt2(16))

    


# In[ ]:


#anonymous func. -> map()
l = [1,2,3,4,5]
a = map(lambda a: a**0.5,l)  #this will apply function for all items in the list
print(list(a))


# In[ ]:


#zipping -- zip()  ->it is just like cartesian product in the maths, but a bit different
l1 = [1,2,3,4]
l2 = [5,6,7,8] 
new = list(zip(l1,l2))
print(new)
#unzipping
unzipped = zip(*new)
unlist1,unlist2=list(unzipped) #unzip returned tuple
print(type(unlist1))


# In[ ]:


##LIST COMPREHENSION
liste = [1,2,3,4,5]
liste2 = [i+2 for i in liste] 
print(liste2)


# In here we write for loop in a line. but this time we fristly write result, then loop

# In[ ]:


#another example
liste3 = [i+5 if i>3 else i-2 if i <= 3 else i+3 for i in liste]
print(liste3)         #else i-2 if i <= 3 -> replace elif
                     # elif i<=3:
                    #     i = i-2  


# I'll try it with a dataset 

# In[ ]:


data = pd.read_csv("../input/metal_bands_2017.csv", encoding = "ISO-8859-1")
avrg = sum(data.fans)/len(data.fans)
data["admiration"] = ["high" if i > avrg else "low" for i in data.fans]
print(data.admiration,data.fans)


# That's all for now,
# **Thanks for checking out!**
