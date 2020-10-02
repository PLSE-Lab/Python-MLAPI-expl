#!/usr/bin/env python
# coding: utf-8

# **what is 3SAT**
# * The 3SAT problem aims to find solution to a clause that is in the form of :
# *Clause = lateral1 and lateral2 and lateral3 .... lateraln*
# * knowing that a lateral is of the form:
# *lateral = x or y or z*
# * A lateral must contain exactly 3 variables while a clause might contain a big number of laterals and variables.
# * A soluyion to this problem would be to find the value of our variables such as our clause is True

# **Example of a clause of 6 variables and 3 laterals**
# * Clause = (x1 or x2 or x3) and (x4 or x5 or x6) and (-x1 or x3 or x4)
# * PS : -x1 means not(x1)
# * A solution to this 3SAT problem would be the affectaion : x1 = True, x2 = False, x3 = True, x4 =False, x5=False,x6=True
# * With those values our clause will be = to (True or False or True) and (False or False or True) and (False or True or False) = True or True or True = True

# **What do mean by testing the solution?**
# * We have in our hands two files, the first(3SAT_218dataset.txt) contains the whole clause, each line of the file contains the number of three variables of a lateral.
# * In the second file we have an affectation of those variables
# * So we basically need to replace the variables with there values ( just like we do in equations ) to test if the clause is true or false.

# In[1]:


import pandas as pd
import numpy as np
import random
X = []
#Load variable values into the list X 
#we keep only the value not the variables name since they are ordered so the number of the variable 
#is the index in the list
with open('../input/3SAT_Solution.txt','r') as f:
    for line in f:
        line = line.rstrip("\n")
        splited = line.split("=")
        X.append(bool(int(splited[-1])))
#Loading the clause in a matrix of dimension 218*3 
clause = []
with open('../input/3SAT_218dataset.txt','r') as f:
    for line in f:
        splited = line.split(",")
        del splited[-1]
        int_splited = [int(x) for x in splited]
        clause.append(int_splited)


# In[2]:


#The function simple, we basically loop over the lines of our matrix
#For each line we loop over the colomns (the variable numbers) and we test if the variable will kept as it is
#or we will do a not of the variables (if it is of the form -x)
#For each line we do an or of the 3 variables and store the value in  partial_lateral, once we finish the line
#we do an and of the line to the previous lines in the partial_clause variables
#at the end of the loop of the lines we return partial_clause
def threeSatVerification(clause,X):
    partial_clause = True
    for i in range(len(clause)):
        partial_lateral = False
        for j in range(len(clause[i])):
            partial_lateral = partial_lateral or ( bool(X[clause[i][j]-1]) if (clause[i][j] > 0) else bool(not(X[-clause[i][j]-1])) )
        partial_clause = partial_clause and partial_lateral
    return partial_clause


# In[3]:


#Function call
print(threeSatVerification(clause,X))

