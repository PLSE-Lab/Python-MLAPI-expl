#!/usr/bin/env python
# coding: utf-8

# # Python Course - Lesson 1
# 
# ## 1. Introduction
# Welcome to the first lesson of Python Course. I would like to thank you for your interest in this initiative of mine and I hope you come to like Python as I do!
# Every time we have to do or learn something new, the first step is considered to be one of the hardest. But since you are here, you are already past that point!
# So, give yourself a pat on the back and let's start demystifying everything that kept you from studying programming all these years!
# 
# ## 2. A few words about Python
# The reasons for which I chose Python as the programming language for this course can be summed up below:
# * It's a **high level programming language**, meaning that it's commands and syntax is very close to our natural language, making very easy to learn!
# * Currently is the **language of choice** in Data Science and Machine Learning.
# * It is **used in most industries** no matter their market focus. So regardless of your expertise, chances are you can use Python to your advantage!
# * There is an **extensive collection of libraries** that will likely cover all your needs!
# * Commands can be executed as you write the code, allowing Python to be even more user friendly!
# * It's Object Oriented! That means... Oh well, let's not complicate things :P
# As we delve deeper into Python you'll come to realise even more advantages that the language provides.
# 
# ## 3. Jupyter Notebook
# Jupyter Notebook is special kind of notebook that allows us to combine Rich Text with Code. It is made by [Jupyter](http://jupyter.org/) and it's a very nice way to present, try and learn code with! Jupyter supports other languages as well, like C, Java, Pearl, etc.

# ## 4. Hello World!
# Let's start by following the cliche of every tutorial ever!
# We are going to use the command ***print()*** to have the computer say the phrase "Hello World!"

# In[1]:


print("Hello World!")


# ## 5. Basic Mathematical Operations
# Let's try some basic mathematical operations. After all, computers are just big calculators!
# 

# ### 5.1 Simple Addition
# *Hint: +*

# In[2]:


2 + 4


# ### 5.2 Simple Substraction
# *Hint: -*

# In[4]:


6 - 3


# ### 5.3 Simple Multiplication
# *Hint: **

# In[5]:


2 * 4


# ### 5.4 Simple Division
# *Hint: //*

# In[6]:


7 // 3


# ### 5.5 Advanced Division (only for Experienced Professionals)
# *Hint: /*

# In[7]:


7 / 3


# ### 5.6 Basic Modular Arithmetic
# *Hint: %*

# In[8]:


7 % 2


# ### 5.7 Super Powers!
# *Hint: ***

# In[10]:


2**3


# ### 5.8 Roots~~~~~

# In[11]:


2**(1/2)


# #### Important Note
# All Programming Languages strictly follow the mathematical priority of operations.
# If you are not familiar with this concept, please have a look [**here**](https://en.wikipedia.org/wiki/Order_of_operations).

# ## 6. Variables
# Variables are containers we use in order to store data. The type of data we can store ranges from Strings and Numbers to Multidimensional Arrays!
# Let's go through what we did so far, but his time using variables.

# ### 6.1 Hello World! (with a twist)
# Assign the phrase into a variable and produce the output using the command we learned above.

# In[13]:


x = "Hello World!"
print(x)
type(x)


# ### 6.2 Mathematical Operations!!
# Assign two numbers in two different variables and give those operations a try!

# In[18]:


z = 2
y = 3
print("Variable z is: " + str(z))
print("Variable y is: " + str(y))


# #### 6.2.1 Addition

# In[15]:


z + y


# #### 6.2.2 Substraction

# In[19]:


z - y


# #### 6.2.3 Multiplication

# In[20]:


z * y


# #### 6.2.4 Division

# In[21]:


z / y


# #### 6.2.5 Mod-whatever

# In[22]:


y % z


# #### 6.2.6 More Super Powers!!

# In[23]:


y**y


# #### 6.2.7 More Roots~~~
# *Hint: x**(1/n)*

# In[24]:


z**(1/y)


# #### Important Note
# Pay attention to the numbers you use. Operations among Integers will produce Integers as a result. Floats with Floats, will produce Floats. Integers with Floats, will produce Floats.

# In[28]:


5 + 2.


# ## 7. Truth or Dare
# In Programming we have various kinds of data types. So far we saw three of them:
# * Integer: 5
# * Float: 2.4
# * String: Hello World!
# 
# Now, we are going to learn about a new kind of data type called a Boolean. This kind of data can only have 2 possible values, **True** or **False**.
# Those values are mainly used when we want to perform checks in our code.
# Let's see a few examples!

# ### 7.1 Check if two numbers are equal

# In[35]:


5 == 5.


# ### 7.2 Check if two numbers are NOT equal

# In[36]:


5 != 5


# ### 7.3 Check if a number is bigger than another one and store the result into a variable

# In[37]:


5 > 5


# In[38]:


5 <= 5


# In[41]:


var = z > y
var


# In[42]:


type(var)


# In[ ]:




