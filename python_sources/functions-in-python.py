#!/usr/bin/env python
# coding: utf-8

# ## For, If, Functions
# 
# Functions in Python. A function is a set of statements that take inputs, do some specific computation and produces output. These functions are called user defined function

# 

# In[ ]:


## If conditions  and for loops

lst=[1,2,3,4,5,6,7,8,9,10,12,13,14,15]

even_sum=0
odd_sum=0
for i in lst:
    if i%2==0:
        even_sum=even_sum+i
    else:
        odd_sum=odd_sum+i
print(even_sum,odd_sum)
        


# In[ ]:


'''
Syntax
def funcname(parameter1,parameter2,...):
    function body

'''


# In[ ]:


def add(num1,num2):
    return num1+num2


# In[ ]:


add(2,4)


# In[ ]:


def add_list(lst):
    sum1 =0
    for i in lst:
        print(i)
        sum1=sum1+int(i)
    return sum1


# In[ ]:


add_list([1,2,3,4,5,6])


# In[ ]:


input_num=list(input("Enter the list"))


# In[ ]:


input_num


# In[ ]:


add_list(input_num)


# In[ ]:





# In[ ]:




