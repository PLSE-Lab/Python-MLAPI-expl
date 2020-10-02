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


get_ipython().system('ls')


# In[ ]:


get_ipython().system('pwd # present working directory')


# In[ ]:


sample_cpp_program="""
#include <stdio.h>

int main(){
    printf("I like juice\\n");
    return 0;
}
"""


# In[ ]:


get_ipython().run_line_magic('store', 'sample_cpp_program > sample.cpp')


# In[ ]:


get_ipython().system('cat sample.cpp')


# In[ ]:


get_ipython().system('gcc -o sample.o sample.cpp')


# In[ ]:


get_ipython().system('./sample.o')


# # Functions

# In[ ]:


def add3(num):
    num = num + 3
    return num


# In[ ]:


add3(5)


# 

# In[ ]:


def add_3_numbers(num1,num2=0,num3=0):
    result = num1+ num2 + num3
    print("num1(%d), num2 (%d), num3(%d)" % 
          (num1,num2,num3))
    return result

print("The sum of 5+8+9 is " )
#example use of "keyword-arguments"
print(add_3_numbers(num3=9,num2=8,num1=5))


# In[ ]:


print(1,3,5)


# In[ ]:


def add_several_numbers(num1, *other_numbers):
    result = num1
    for num in other_numbers:
        result = result + num
    return result


# In[ ]:


add_several_numbers(0,[5,6,7])


# In[ ]:


def add_some_numbers(list_of_numbers):
    result = 0
    for num in list_of_numbers:
        result = result + num
    return result
add_some_numbers([4,5,6,7])


# In[ ]:


code = """
def add_several_numbers(num1, *other_numbers):
    result = num1
    for num in other_numbers:
        result = result + num
    return result
"""
get_ipython().run_line_magic('store', 'code > functions.py')


# In[ ]:


get_ipython().system('cat functions.py')


# In[ ]:


import functions


# In[ ]:


functions.add_several_numbers(5,6434,5)


# In[ ]:


from functions import add_several_numbers as addy


# In[ ]:


addy(434,5656)


# In[ ]:


# Challenge 1
# Create a module named workshop that has the foll.
# functions.
# Function 1: accept_numeric_input
# The function should accept input from a user
# and return the input as a number
# Function 2: accept student details
# This function should accept as paramters
# details for one student (name, id, age)
# inside the function, you should store these details
# in a python dictionary. If no id number is entered
# the default id number should be -1
#. The function should print the details of the student
# from the dictionary created if an additional paramter
# named print_details is True. return the details as 
# a dictionary.
#
# Demonstrate how to import and use your code


# In[ ]:


# using assertions to ensure that only floating numbers are
# passed to the function
def add_3_to_float(float_num):
    #assert type(float_num) == float 
    assert isinstance(float_num,float)
    return float_num 

print( add_3_to_float(4.6) ) # will run
print( add_3_to_float(4) ) # will fail


# In[ ]:


# python only passes by reference
menu=['on_off_sleep','ackee and saltfish']
def test_function_1(list_object):
    list_object.append('jack fruit')
    print("List object inside function {0}".format( list_object ) )
    print("Id of list_object %s " % id(list_object))

test_function_1(menu) # calling the function
print("Menu after function call %s " % menu) # after
print("Id of menu %s " % id(menu))


# In[ ]:


id(menu)


# In[ ]:


my_wallet = 30000
def add_money_to_wallet():
    global my_wallet # indicating that we would like to reference a global variable
    my_wallet = my_wallet + 5000

add_money_to_wallet()
print("Money in wallet : $%.2f" % my_wallet)


# In[ ]:


# The beautiful lambda functions
add_3_lambda_function = lambda x : x + 3

print(add_3_lambda_function(40))

product_lambda_function = lambda x,y : x *y

print(product_lambda_function(40,3))


# In[ ]:


#define our own error/exception
class MyWonderfulError(RuntimeError):
    pass
try:
    raise MyWonderfulError("Life is interesting")
except MyWonderfulError as e:
    
    print("We handled our problems",e)
else:
    pass


# # Working With Files

# In[ ]:


# FILE* file_handle = fopen('test_file.txt','w+')   <--- C equivalient
file_handle=open('test_file.txt','w+')
file_handle.write("I like curry chicken and roti")
file_handle.write("\nWhat is teriyaki chicken?")
file_handle.close()

get_ipython().system('cat test_file.txt')


# In[ ]:


try:
    file_handle=open('test_file.txt','r')
    print(file_handle.read(5) ) # Read the first five characters/bytes and print
    print(file_handle.read(5) ) # Read the next five characters/bytes and print
    file_handle.close()
except IOError as e:
    print("Error with file ",e)


# In[ ]:


get_ipython().system('ls -al')


# In[ ]:


with open('sub_folder/test_file.txt','w+') as file_handle:
    file_handle.write("Tea")
    file_handle.write("Vichysuis")

get_ipython().system('cat sub_folder/test_file.txt')


# In[ ]:


data = [ {
    "id":num,
    "name":"Aloysius %s " % num,
    "age" : 20 + num
} for num in range(1,6)]
data


# In[ ]:


with open('students.csv','w+') as file_handle:
    file_handle.write("Id,Name,Age\n")
    for student in data:
        file_handle.write("%d,'%s',%d\n" % (student['id'] , student['name'], student['age']  ) )

get_ipython().system('cat students.csv')


# In[ ]:


import pandas as pd
data_as_data_frame = pd.DataFrame(data)
data_as_data_frame


# In[ ]:


data_as_data_frame.to_csv("other_students.csv",index=False)
get_ipython().system('cat other_students.csv')


# In[ ]:


import json


# In[ ]:


help(json)


# In[ ]:


json.dumps(data) # output object as json


# In[ ]:


file_handle=open('students.json','w+')
json.dump(data,file_handle) # output object as json to a file
get_ipython().system('cat students.json')


# In[ ]:


file_handle=open('students.json','r+')
data_read = json.load(file_handle) # output object as json to a file
data_read


# In[ ]:




