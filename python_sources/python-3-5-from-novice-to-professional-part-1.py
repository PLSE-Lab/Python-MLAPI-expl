#!/usr/bin/env python
# coding: utf-8

# Multiple Skillsets are required to attain experties in Data science and machine learning and I do appreciate the tirless efforts of experts and masters on kaggle who are helping data science and machine learning communities through valuable kernals and knowledge articles. 
# 
# However, I do believe to divide the training material into segments and teach learners and beginers one segment at a time. So they can acquire the expertise in a systematic way. By saying it, I am starting a series of Kaggle kernels for beginers and learners to start their journey of data science and machine learning from noice to professional level. 

# **Topic Coverd in the Series**
# 
# Following is the list of segments, I would cover in this series. 
# 
#     1. Python programming from Novice to professional
#     2. Numpy from Novice to Professional
#     3. Pandas from Novice to Professional
#     4. Data Visualization from Novice to Professional
#     5. Linear Algebra for Machine Learning from Novice to Professional
#     6. Calculus for Machine Learning from Novice to Professional
#     7. Principle Component Analysis from Novice to Professional
#     8. Scikit learn from Novice to Professioanl
# 

# Let me start this journey with learning python programming langugae. I would say,  it is one of the easist language for programming as compare to C/C++ and Java and you will find out the reasons later in this kernal.

# Instead of adding theroy, i shall try to explain each concepts through examples. So lets start with data types. Following are the data types available in python and i have also added an example for each data type

# In[ ]:


#integer declaration
int_number = 1234
#float declaration
float_number = 25.6
#string declaration
str_value = 'Welcome'
str_value_2 = " To Python... The language of Data"
#tuple declaration
tuple_value = (1,2,3,4)
#list declaration
list_values = [1,2,3]
#dictionary declaration
diction_values = {'key1':'value1','key2':'value2'}


# I shall discuss the above data types, once we use them in the code.  

# **Packages and Modules**
# 
# In python, tools are available as a part of language and we don't need to reinvent the wheel. It helps us develop code quickly and focus on the main objective of a program. However, we can override this behavior through declaring our own function. Examples will show later in this tutorial. Following is the list of some famous packages
# 
# 1. Math
# 2. Numpy
# 3. Pandas
# 4. Matplotlib
# 
# following is the command to import theses packages in the program
# 
# ***import [package name] as [identifier]***
# 
# Now its code time.    
# 
# 

# In[ ]:


import math as m
import random as rd
print(m.sin(60))
#random is the most important function and we shall use it a lot during experiments and model testing/validation
print(rd.random())


# **Getting help on functions and variables**
# 
# If  you are unsure about the available methods for a particular variable or function, use dir([object/function name ]) command. It will list down all available function and then you can use help([function name ]) for further help.

# In[ ]:


str_value='welcome'
dir(str_value)


# In[ ]:


#lets get some information about find function of string
help(str_value.isnumeric)


# Its time to add condition into our codes. In Python we use **If else** statement to apply conditions. Lets look at the following example.  

# In[ ]:


int_number = 10
if int_number==10:
    print('value of int_number is 10')
elif int_number>10:
    print('value of int_number is greater than 10')
else:
    print('value of int_number is less than 10')


# **Important Points about If Statment**
# 
# 1. == has been used for equality operation
# 2. we may have many elif based on the complexity of problem. However, we do have only one else and it will be the last part of if statment
# 3. Unlike C/C++, Python uses identation for block statements. We need to be careful, when use nested if into our codes. Because, it will waste time/energy, if the indentation is not correct.

# **Challenge**
# 
# Can you write a code to find whether a string contains numeric valueor not.
# 
# I shall provide the solutions at the end of the kernel.
# 

# **Loops**
# 
# Sometime, we need to repeat a single statement multiple times. To address it, every language provide loops which iterate statements unless the counter reach the upper limits or condition become false. In python, we normally use two loop statements.
# 
# 1. For Loop
# 2. While Loop
# 
# Both loops server the purpose, the usage depends upon individual's preference. However, we shall discuss both in details.

# In[ ]:


#for loop is also called counter loop as it execute number of times based upon the counter value
for counter in range(100):
    print(counter)
#range() is a built in function in Python which return the list of values from 0 till the value provided to it
#range(100) will return values from 0 to 99


# In[ ]:


int_number = 2
int_result = int_number
int_power = 10
while int_power>1:
    int_result = int_result * int_number
    int_power = int_power - 1
print(int_result)


# In above example, we have written a programe which will calculate the power of any number and as you can see instead of running on a counter, it is running based on the condition. The while loop will terminate, once the condition returns false. The formula to calculate the power is very simple.
# 
# power 2 of a number = number * number
# power 3 of a number = number * number * number
# 
# In the above logic, we have assigned the number to our result without any condition and they iterate the loop till the power is greater than one and each time we are multiplying the result variable with the number. 

#  **Challange**
# 
# Find the factorial of anumber using for loop
# 
# Can you write a while loop to find whether a string is palindrome.
# 
# Example of palindrome strings
# 
# 1. anna
# 2. civic
# 
# I shall provide the solutions at the end of the kernel.

# In[ ]:


#Solution for challenges
str_value = '100'

if str_value.isnumeric():
    print('string contains numeric value')
else:
    print('string does not contain numeric value')

# Stay tuned for for/while loop challenge solution


# In[ ]:


# Factorial of a number n = n * n-1 * n-2 .... 1
int_number = 5 
for counter in range(int_number-1):
    int_number = int_number * (counter+1)
print(int_number)    
    


# The formula to calculate factorial is very simple. You need to multiple a number from itself till one. For example, factorial of  5 = 5 * 4 * 3 * 2 * 1 =120. You can run the for loop in the given solution and figure out how it is calculating the factorial of number. 
# 
# **Small Challenge:**
# You can try to re-write the solution using while loop

# In[3]:


str_value = 'awddna'
reverse_value =''
length = len(str_value)-1
while length>=0:
    reverse_value = reverse_value + str_value[length]
    length = length - 1

if str_value == reverse_value:
    print(str_value, " is palindrome")
else:
    print(str_value, " is not palindrome")
print(reverse_value)    


# Again, try to execute the above code on a paper and understand, how it is identifying a string value is palindrome or not. The most important point for learning is translating technical problem into small segments and then solve one at the time. For example, in the above example, i don't think of comparing string to its reverse. Instead, i put focus to identify the lenght of the string and then put a while loop to read the character in reverser order which starts from the end of a string and reach to the starting index of it, 
# 
# Import Note:
# Python string/Array index starts from O and goes upto lenghth-1. 
# 
# Once, i identify the way to find the reverse, i store it in another variable for comparison and when the loop finishes its execution, the comparison of both values are providing the solution. 

# So far we have seen the basic building blocks of Python. However, in order to build a professional software, we need to modularize our code in an efficient way. So, when the number of lines cross thousands or millions, it would still be easy for us to maint it. Like every language, Python also provides functionality to build classes,packages and functions etc. In upcoming sections,we shall discuss it further with example. 

# In[7]:


def printCounter(int_num):
    for counter in range(int_num):
        print(counter)
printCounter(10)        
printCounter(5)


# The above function will recieve a parameter and execute the loop from 0 till the value of parameter. The advantage of writing a function is reuse functionality instead of duplicating the code. For example, in the above case, we are calling function instead of writing loop two times. Next, we shall define the function to calculate factorial of a given number

# In[10]:


def calc_factorial(int_number):
    counter = int_number-1
    while(counter>=1):
        int_number = int_number * counter
        counter = counter - 1
    return int_number    
calc_factorial(6)


# In[ ]:




