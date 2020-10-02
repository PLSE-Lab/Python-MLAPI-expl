#!/usr/bin/env python
# coding: utf-8

# # Objective : To Identify the most memorable number for a Human of a given list

# If you have ever bought a SIM. The company offers you plenty of mobile numbers to pick from. Its very difficult to identify memorable number
# 
# Inorder to solve this I have designed a simple algo to identify the most memorable number of a given a list. 

# Approach : Any mobile number (example -  '9987823233' ) has repeating patterns such as 99 , 2323 , 33 . These patterns help in recollecting and makes it easy to memorize. Every repeating pattern will have 2 characteristics : Variety & Length
# 
# 99 >> Variety = 1 , Length = 2 ;
# 2323 >> Variety = 2 , length = 4
# 
# We can create a score card based on these 2 parameters and find the best score for such a number. 

# Steps
# 1. Use Regex to Identify Patterns
# 2. Create a Function for returning Matched patterns
# 3. Create Score card function
# 4. Make an Iterator function to throw up the best Number

# ### STEP 1 :  Use Regex to Identify Patterns

# In[ ]:


import re
from re import *
import pandas as pd

t = '2221143432453465654'
reg = r'(\d+)\1+'
#This regex identifies patterns such as 222, 4343,6565. 
re.match(reg,t)


# Here Re.Match will only give one output. We need to get  multiple matches for any number

# ### STEP 2 : Create a Function for returning Matched patterns

# In[ ]:


def matching_num(num):
    import re
    regex = r"(\d+)\1+"
    test_str = num
    matches = re.finditer(regex, test_str)
    l = []
    for matchNum, match in enumerate(matches):
        l.append(match.group())
    return l


# Learn > Enumerate Object - Not very intuitive for me. This piece of code will however do the trick. Lets Test Now

# In[ ]:


a= '8888777123'
b ='8830121233'
c='9823321518'


# In[ ]:


print(matching_num(a))
print(matching_num(b))
print(matching_num(c))


# ### STEP 3 : Create Score card function

# In[ ]:


add = '../input/master_mob_score.csv'
mst= pd.read_csv(add,index_col=0)   
print (mst)


# Here the first column titled as div ~ Variety & Column are length of the Match
# Example 9999999999 will get a perfect score of 100 where as 1122233898 will get a score of 10 + 21.25 + 10 = 41.25 

# In[ ]:


def m_score(j):
    # M_score takes a list and calculates the final score for the number
    # setting up the master for score calculation
    add = '../input/master_mob_score.csv'
    mst= pd.read_csv(add,index_col=0)   
    b=[]
    score = 0   
    #Setting up for loop for calculating scores
    for i in j:
        # X parameter here is Variety -- A number
        # Y Parameter is length of repeat -- A string
        variety =len(set(i))
        length = str(len(i))
        sc = mst.loc[variety,length]
        b.append(sc)
        score = score + sc
    return score


# In[ ]:


print(matching_num(a))
print (m_score(matching_num(a)))
print ("----------------")
print(matching_num(b))
print (m_score(matching_num(b)))
print ("----------------")
print(matching_num(c))
print (m_score(matching_num(c)))


# You can calculate from this example the best number to remember is actually the 1st one

# ### STEP 4 : Make an Iterator function to throw up the best Number

# In[ ]:


def final_score(k) :
    # I am too lazy to call a function in a function
    return m_score(matching_num(k))

def best (j):
    #takes in a list and throws "number" & "score"
    k = []
    maxscore=0
    for i in j:
        k.append(final_score(i))
        if final_score(i) > maxscore :
            # this block is invoked when it finds the new KING in the list
            maxscore = final_score(i)
            num = i
    return num,maxscore


# In[ ]:


w = [a,b,c]
print(best(w))


# ## THE END
