#!/usr/bin/env python
# coding: utf-8

# # Defining functions
# You've already seen and used functions that are builtin to python such as `print()` and `type()`. But Python has many more functions, and defining your own functions is a big part of python programming.
# 
# In this lesson you will learn more about using and defining functions.
# 
# A function is a block of code which only runs when it is called.
# You can pass data, known as parameters, into a function.
# A function can return data as a result.
# It Can be executed multiple times, without needing to constantly rewrite the entire block of code.
# 
# Creating clean repeatable code is a key part of becoming an effective programmer.
# 
# In a nutshell, while a variable stores data, a function stores code
# 
# # Parameters
# Information can be passed to functions as parameter.
# 
# Parameters are specified after the function name, inside the parentheses. 
# 
# You can add as many parameters as you want, just separate them with a comma.
# 
# 
# # Return
# Send back the result of the function, instead of just print it out
# return allows us to assign the output of the function to a new variable

# In[ ]:


#Your code goes here

###Defining a function

###Calling a function  


###Creating a function - traditional way




###Optimized way




# The above example creates a function called least_difference, which takes three arguments, `a`, `b`, and `c`.
# 
# Functions start with a header introduced by the `def` keyword. The indented block of code following the `:` is run when the function is called.
# 
# `return` is another keyword uniquely associated with functions. When Python encounters a return statement, it exits the function immediately, and passes the value on the right hand side to the calling context.
# 
# Is it clear what least_difference() does from the source code? If we're not sure, we can always try it out on a few examples:

# # Calling a function
# 
# To call a function we simple need to write the name of it and give the correct arguements (with the correct types) and it will execute that block of code. 
# 
# Let's try it! 

# In[ ]:


#Your code here test the function you created 



# # Functions that return
# When we use `return` , you specify the value you want the function to give you. That way, we can call a function and store it in a variable. 

# In[ ]:


#Your code here - storing the return value into a variable


# # Functions that don't return
# 
# Sometimes you need a function to do some piece of code but don't need a value to come out from it. 
# 
# Let's see what would happen if we took return out of the function we created:

# In[ ]:


#Your code goes here copy paste the above function you created an take out return


# In[ ]:


#Your code goes here


# 
# In some cases, you may not need to store a value but want a function to execute some block of code as part of your program. Let's try an example of that!
# 
# Along without a return, this function comes with a `default` value. This is what happens when you assign a value to the arguement name in the function's brackets. If we don't pass the function a value, since we gave it the default value "World" the function still works.

# In[ ]:


#Your code goes here


# In[ ]:


#Your code goes here


# # Local Variables x Global Variables
# 
# It is extremely import to understand the difference between them to avoid ambiguity
# 
# A global variable is defined outside of a subroutine or function. The global variable will hold its value throughout the lifetime of a program. They can be accessed within any function defined for the program
# 
# A local variable is either a variable declared within the function or is an argument passed to a function. As you may have encountered in your programming, if we declare variables in a function then we can only use them within that function

# In[ ]:


#Your code goes here


# # Putting it all together
# 
# Let's try to create a function ourselves! Define a function that takes a list of integers as an arguement. It looks through the list for the largest integer and then returns the largest int. Call the function and store it in a variable. Print the variable. 
# 
# BONUS: Functions are responsible for testing the arguement they receive. Check if you actually received a list (otherwise your code shouldn't work) and check if each element in the list is an integer (otherwise you might be checking on the wrong type).

# In[ ]:


#Set Up Code

list_of_numbers = [70,4,102,88]

#Your code goes here start by defining a function that takes 1 arguement

### Without Bonus



### With Bonus 

