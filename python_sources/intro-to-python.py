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


# In[ ]:


3+5


# # Welcome to Python
# ## Chapter
# 
# We are going to have some fun

# In[ ]:


var1 = 35


# In[ ]:


var1


# In[ ]:


3**2


# In[ ]:


my_str="String 1"
my_str_2='String 2'
my_str_3="""""
    String 3
    """


# In[ ]:


# We use the type function
type (my_str)


# In[ ]:


mylist = [1,4, "Wonderful", "This is not an array1"]
mylist


# In[ ]:


mylist[2]


# In[ ]:


mylist[2]+" "+mylist[3]


# In[ ]:


tup1 = (5, 6, 4)


# In[ ]:


tup1[0]


# In[ ]:


person = {
        "name":"Gordon",
        "age":2627
}


# In[ ]:


person ["name"]


# In[ ]:


# Add a new key
person["shirt_colour"]="purple"
person


# In[ ]:


person["shirt_colour"]="Black & White"


# In[ ]:


person


# In[ ]:


person


# In[ ]:


persons = [
        {
        "name":"Gordon",
        "age":2627
        "gender":"Male"
},
    {
        "name":"Mary",
        "age":21
        "gender":"female"
},
    {
        "name":"Sue",
        "age":574
        "gender":"female"
},
]
persons 


# In[ ]:


persons[1]


# In[ ]:


(persons[1])["name"]


# In[ ]:


(persons[1])["age"]


# In[ ]:


if 5 == 3 and True:

    print("5 is equal to 3")
else:
    print("5 is not equal to 3")    


# In[ ]:


if 5 == 3 and True: print("5 is equal to 3")
else: print("5 is not equal to 3")    


# In[ ]:


if 5 == 3 and True: print("5 is equal to 3")
else: print("5 is not equal to 3")


# In[ ]:


for i in range(0,5):
    print(i)


# In[ ]:


for i in range (0, 5, 2): #increase by 2
    print(i)
    


# In[ ]:


# using loops and conditional statements
# Change update all person dictionary items
# in persons
# if their age is over 30, their shirt colour should
# be red otherwise, blue


# In[ ]:


for person in persons:
    print("----")
    #print(person)
    if person["age"] > 30:
        print(person, "Over 30")
        person["shirt_colour"]="red"
    else:
        print(person, "Age under 30")
        person["shirt_colour"]="blue"
    


# In[ ]:


type(int("3"))


# In[ ]:


type(str("3"))


# In[ ]:


def add(num1, num2):
    return num1+num2


# In[ ]:


add(5,6)


# In[ ]:


def add3(num1, num2, num3=9):
    print("Adding -->  ",num1, num2, num3)
    return num1+num2+num3

print("5 + 6? = ", add3(5,6))
print("1 + 4+5 = ", add3 (num3=4, num2=1, num1=5))


# In[ ]:


subtract3 = lambda x : x -3
subtract = lambda x,y: x-y


# In[ ]:


print( subtract3(4),
      subtract(4,6) )


# In[ ]:


print(subtract3 (5),
subtract(12, 21))


# In[ ]:


"True" if 3 > 5 else "False"


# In[ ]:


"True" if 100 > 12 else "False"


# In[ ]:


"True" if 32 > 59 else "False"


# In[ ]:




