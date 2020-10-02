#!/usr/bin/env python
# coding: utf-8

# # Assignment 1: Check Fibonacci sequences
# 
# Your assigment is to use what you learned so far about Python and build a program to check whether a sequence of numbers follows the Fibonacci sequence for any arbitrary starting numbers, \\( F_0 \\) and \\( F_1 \\).
# 
# A Fibonacci sequence is defined as a sequence of numbers whose value is the sum of its two preceding values. For example the standard Fibonacci sequence is:
# 
# \\[ 0, 1, 1, 2, 3, 5, 8, 13, 21, ... \\]
# 
# where the first number is \\( F_0 = 0 \\) and the second number is \\( F_1 = 1 \\)
# 
# The Fibonacci sequence for \\( F_0 = 3 \\) and \\( F_1 = 9 \\) is:
# 
# \\[ 3, 9, 12, 21, 33, 54, 87, ... \\]

# ## Setup Instructions
# 
# 1. Fork this kernel
# 2. Confirm that the `pada1ds` dataset is copied in your new forked kernel as well.
#     * In the right sidebar, navigate to the Workspace section and confirm the folder `pada1ds` exists.
# 3. Use the code in the next cell to import data from `fib1.csv` and confirm that the data has been saved in the variable `seq`
# 4. Confirm whether the sequence is indeed a Fibonacci sequence. Print out a message that says whether the sequence is a proper Fibonacci sequence.
# 5. Check the other sequences in the other files `fib2.csv` and `fib3.csv`.
# 

# In[ ]:


# Import fib1.csv file and save all row data as integers
# Use this code to import data from a Kaggle kernel. Make sure you import the dataset for this assignment first into your Kaggle kernel

with open('../input/fib1.csv','r') as fin:
    seq = [int(r) for r in fin.readlines()[1:]] # Build list using list comprehension: https://medium.com/better-programming/list-comprehension-in-python-8895a785550b

print(seq)


# In[ ]:


# Confirm whether the numbers contained in fib1.csv is a Fibonacci sequence with F0 = 0 and F1 = 1
# If you're familiar with loops from another programming language, feel free to research how you would use loops in Python to make your code more efficient


# In[ ]:


# Confirm whether the numbers contained in fib2.csv is a Fibonacci sequence with F0 = 4 and F1 = 8
# If you're familiar with loops from another programming language, feel free to research how you would use loops in Python to make your code more efficient


# In[ ]:


# Confirm whether the numbers contained in fib3.csv is a Fibonacci sequence with F0 = 2 and F1 = 6
# If you're familiar with loops from another programming language, feel free to research how you would use loops in Python to make your code more efficient

