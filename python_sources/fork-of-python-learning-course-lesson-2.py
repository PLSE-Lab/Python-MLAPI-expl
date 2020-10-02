#!/usr/bin/env python
# coding: utf-8

# # Python Learning Course - Lesson 2
# 
# Welcome to the second lesson of Python! Let's get started!

# ## 1. Advanced print()
# Create two variables, the first one called **age** and the other one called **name**.
# Assign the name Clara and the number 21, in the relevant variables.

# In[1]:


age = 21
name = "Clara"


# As we said in the first lesson, we can not print a number, along with some text, without first converting the number with **str()**.<br>
# To reproduce this error, copy and paste the following code in the cell below and press Shift+Enter to execute the command.<br>
# *print("My name is " + name + " and I am " + age + " years old.")*

# In[2]:


#print("My name is " + name + " and I am " + age + " years old.")


# Now execute the following command:<br>
# *print(age)*

# In[3]:


print(age)


# **Since we can print() the *age* variable, why are we getting the error?**<br>
# Hint: print(*temporary variable*)

# ## 1.2 format()
# ### 1.2.1 Basic Syntax and Use
# The **.format()** function is a great tool with which you can print() your outputs in a clean and neat way!<br>
# Type the phrase we used above, but this time:
# > - Add curly brackets {} in the places where the **name** and the **age** should be.
# > - After you type the phrase and close your quotation marks, type **.format(name, age)**.

# In[5]:


"My name is {} and I am {} years old.".format(name,age)


# ### 1.2.2 Using Placeholder Variables

# In[6]:


"My name is {one} and I am {two} years old.".format(two = age, one = name)


# ### 1.2.3 Output vs print()
# Notice the single quotation marks around the sentence.<br>
# Now copy and paste the command you typed in the previous cell and paste it inside a print(), in the cell bellow.<br>
# Run the cell and notice the difference!

# In[7]:


print("My name is {one} and I am {two} years old.".format(two = age, one = name))


# Output: a generic term that refers to the outcome/result, or value of a system, function, or variable. It's not necessarily visible to the user, unless the code **prints** the outcome. It is mainly used as input for further processing.<br>
# print(): explicitly **prints** the output value from a system, function, or variable, to the user. The printed output doesn't serve any other purpose after it produces it's printed output.

# ## 2. Introduction to Data Structures
# ### 2.1 A first look at Lists
# 
# Assign the name Anna to a variable.<br>
# > - Get the first letter of the name.
# > - Get the first three letters of the name.
# > - Get all the letters.
# > - Get the last letter.
# > - Get the last two letters.

# In[8]:


my_name = "Anna"
my_name[0]


# In[10]:


my_name[0:3]


# In[12]:


my_name[:]


# In[13]:


my_name[-1]


# In[14]:


my_name[-2:]


# ### 2.2 A first look at Sets
# Use the **set()** function on the variable you declared before.<br>
# Assign the new output into a new variable.<br>
# Use the **type()** function on your new variable.

# In[15]:


my_set = set(my_name)
my_set


# In[18]:


type(my_set)


# In[16]:


my_list = [1,2,3,4,5]
my_list


# In[21]:


my_list.append(10) #adds an element after the last one, expanding the lenght of the list
my_list


# In[22]:


my_list.pop() #removes the last element in the list


# In[26]:


my_new_list = my_list.copy()
"""
copies the list to a new list so that changes don't affect the original.
python references lists and changes affect the original list as well.
use copy to explicitly say that you want to create a copy of the original
without any reference or relation.
"""


# In[27]:


my_new_list.pop() #this pop only removes the last element in this list, leaving the original as is
print(my_list)
print(my_new_list)


# In[28]:


len(my_list)


# In[29]:


for k in range(0,len(my_list)):
    my_list[k] = my_list[k] + 100
    print("New element is equal to: " + str(my_list[k]))


# In[30]:


my_list


# In[ ]:




