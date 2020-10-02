#!/usr/bin/env python
# coding: utf-8

# # Defining functions
# You've already seen and used functions that are builtin to python such as `print()` and `type()`. But Python has many more functions, and defining your own functions is a big part of python programming.
# 
# In this lesson you will learn more about using and defining functions.
# 
# Builtin functions are great, but we can only get so far with them before we need to start defining our own functions. Below is a simple example.

# In[ ]:


#Your code goes here
def least_difference(a, b, c):
    diff1 = abs(a - b)
    diff2 = abs(b - c)
    diff3 = abs(a - c)
    return min(diff1, diff2, diff3)


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

print(least_difference(1, 10, 100))
print(least_difference(1, 10, 10))
print(least_difference(5, 6, 7))


# # Functions that return
# When we use `return` , you specify the value you want the function to give you. That way, we can call a function and store it in a variable. 

# In[ ]:


#Your code here - storing the return value into a variable

answer = least_difference(5,25,100)

print(answer)


# # Functions that don't return
# 
# Sometimes you need a function to do some piece of code but don't need a value to come out from it. 
# 
# Let's see what would happen if we took return out of the function we created:

# In[ ]:


#Your code goes here copy paste the above function you created an take out return

def least_difference_no_return(a, b, c):
    diff1 = abs(a - b)
    diff2 = abs(b - c)
    diff3 = abs(a - c)
    min(diff1, diff2, diff3)


# In[ ]:


#Your code goes here
print(least_difference_no_return(5,25,100))


# 
# In some cases, you may not need to store a value but want a function to execute some block of code as part of your program. Let's try an example of that!
# 
# Along without a return, this function comes with a `default` value. This is what happens when you assign a value to the arguement name in the function's brackets. If we don't pass the function a value, since we gave it the default value "World" the function still works. 
# 
# 

# In[ ]:


#Your code goes here
def greet(who="World"):
    print("Hello,", who)


# In[ ]:


#Your code goes here
greet()
greet('Mars')


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
def largest_element(a_list):

    largest = 0
    
    for i in a_list:
        if i > largest:
            largest = i
            
    return largest


### With Bonus 
def largest_element_bonus(a_list):
    if type(a_list)== list: 
    
        largest = 0
        for i in a_list:
            if type(i) == int: 
                if i > largest:
                    largest = i
            else:
                print("please make sure each element in the list is of type int")
                return
        return largest
                    
    else: 
        print("please input a list")
        return

print(largest_element(list_of_numbers))

