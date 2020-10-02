#!/usr/bin/env python
# coding: utf-8

# ## 1. More Print 
# 
# Run these print statement examples to get familiar with the different ways you can run print functions in python with variables. 
# 
# ### Don't forget to hit the blue copy and edit button in the top right corner in order to open the kernel that will let you edit this notebook! 
# 
# 

# ### Example 1:

# In[ ]:


# Run this code to see how to variables can be added together 
# Notice the + added to myString2

myString = "ABC"
myString = myString + "DE"

myString2 = "FGH"
##Short hand for adding more value to the same variable
myString2 += "IJK"

print(myString)
print(myString2)


# ### Exercise 1:
# 
# 1. Create variables that hold Int and Float (as many as you like)
# 2. Add them using the += shorthand from the last example
# 3. Print to see what you are getting

# In[ ]:


# Your Code Goes Here


# ### Example 2: 

# In[ ]:


myString1 = "Hello World!"


# In[ ]:


# Run this code block to see how python can print more than one vairable 
# You got an error?
# Run the code block above before you run this code so that python know what myString1 is

myString2 = "Hello sun!"
myString3 = "Hello moon!"
print(myString1, myString2, myString3)


# ### Exercise 2:
# 1. Create 3 variables holding whatever Type you'd like (Int, String, Float)
# 2. Print them all in one print statement

# In[ ]:


### Your Code Here


# ### Example 3:

# In[ ]:


#Run this code block to see the difference between uppercase and lowercase variables
# Where was myString1 intialized? Hint python keeps track of the code blocks run above 

mystring1 = "lowercase"

print(mystring1)
print(myString1)


# ## Question 1 - Printing Variables 
# 
# 
# 1. Add code to the following cell to swap variables `a` and `b` (so that `a` refers to the value previously referred to by `b` and vice versa).
# 2. Position your print statements so that you print `a` and `b` before AND after the swap.
# 
# HINT: You need 3 variables and 2 print statements to complete this task

# In[ ]:


########### Setup code - don't touch this part ######################
a = "Variable a"
b = "Variable b"

######################################################################

# Your code goes here. Swap the values to which a and b refer.










######################################################################


# ## 2. More Python Syntax - Input and Format
# 
# Along with the print() function in python, there are other useful functions you can use to code with. 
# 
# The **input()** function allows you to get user input and the **format()** function allows you to better customize your print statements. There are many more built in functions that help you code, you will discover those as you practice and code more. 
# 
# Give these examples a try below! 

# ### Example 1:

# In[ ]:


#Run this input function example

myString = input("Give me a line: ")

print(myString)


# ### Example 2:

# In[ ]:


#Run this format function example

formatString = "This is my formatted string"

print("My formatted string goes here: {} - in the middle of my sentence".format(formatString))


# ### Example 3:

# In[ ]:


#Run this double format function example
#HINT you can format more that one variable using the format function i.e. format(x,s)

formatString = "This is my formatted string"
formatString2 = "This is the SECOND formatted string"

print("My formatted string goes here: {} - in the middle of my sentence, and at the end of my sentence {}".format(formatString,formatString2))


# ## Question 2 - Input, Format and Print
# Create a program that takes 2 inputs: a number, and a string.  Assigns it to a variable and prints the following message:
# 
# My number is: [number] and my string is [string]
# 
# **HINT** you can format more that one variable using the format function i.e. format(variable1,variable2)

# In[ ]:


#HINT you can format more that one variable using the format function i.e. format(x,s)

#Your code goes here


# ### [Explore online](https://docs.python.org/3/library/functions.html) all the capabilities of these functions - you learn through simply playing around with the code and building programs. 
# 
# #### Add your own code blocks and play around with more print, input and format functions.
# 
# #### Add your own code blocks and play around with using variables with print, input and format functions. 
