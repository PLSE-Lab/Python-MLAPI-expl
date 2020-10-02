#!/usr/bin/env python
# coding: utf-8

# # Python Learning Course - Lesson 6
# ## Recap and work so far
# In this week's lesson we are going to do a recap of what we learned so far.<br>
# We are going to combine what we know, to create a very simple algorithm that creates a **Christmas Tree**!<br>
# I know we are a bit out of schedule but think of it as Christmas in Australia : P<br><br>
# ## Preparatory Work
# For this week's lesson you are required to prepare a few simple things, that we are going to use during the lesson.<br>
# Let's start building them step by step.

# ### Functions
# Create a function called **fa**, the purpose of which is to *print* an asterisk ** * **. The function should also accept a variable **x** as input.<br>
# Inside the print command, multiply * by x in this manner:

# In[ ]:


# example: ("*" * x)


# In[42]:


def fa(x):
    print("*" * x)
    return


# In[32]:


fa(1)


# Expected Outcome: "*"

# Create a function called **fb**, the purpose of which is to *print* a blanc space. The function should also accept a variable **z** as input.<br>
# Multiply the blank space with z, as in the previous example. Feel free to use "-" instead of blank space, to make things easier to see.

# In[45]:


def fb(z):
    print("-" * z)
    return


# In[46]:


fb(3)


# Expected Outcome: " "

# Create a function called **mybranch** that combines the logic of the two previous functions.<br>
# That is, it prints both the * and the blank space in the same row, after multiplying them by their respective variables.<br>
# The function should accept two input variables x and z.<br>
# Make sure you first print the blank space and then the asterisk.

# In[ ]:


# example " " * z + "*" * x


# In[83]:


def mybranch(z, x):
    print(" " * z + "*" * x)
    return


# In[52]:


mybranch(5,3)


# Expected Outcome: *five spaces followed by three asterisks*

# Create a function called **branchloop** that **loops** nine times and each time, it calls the **mybranch** function.

# In[59]:


def branchloop():
    for i in range(0, 9):
        print(mybranch(i, i))
    return


# In[84]:


def branchloop(y, w):
    for k in range(0, 9):
        mybranch(y, w)
    return


# In[85]:


branchloop(5,5)


# Expected Outcome: *nine asterisks, one under the other, in a vertical fashion*

# Create a function called **trunkloop** that will **loop** two times. Each time it will call the **mybranch** function.

# In[117]:


def trunkloop(y):
    for j in range(0, 2):
        mybranch(y - 1, 1)
    return


# In[118]:


trunkloop(9)


# Create a function called **isTrunk** that checks if a number is equal or higher than **9**.<br>
# If the check is True, then it has to call **trunkloop**. Otherwise, do nothing.

# In[119]:


def isTrunk(x):
    if x >= 9:
        trunkloop(x)
    return


# In[120]:


isTrunk(9)


# In[121]:


isTrunk(7)


# ## End of preparatory work

# In[130]:


def branchloop(y):
    spaces = y
    asterisks = -1
    for k in range(0, y):
        spaces = spaces - 1
        asterisks = asterisks + 2
        mybranch(spaces, asterisks)
    k = k + 1
    return k

def isTrunk(k):
    if k >= 9:
        trunkloop(k)
    return


# In[131]:


def mytree(y):
    var = branchloop(y)
    isTrunk(var)
    return


# In[135]:


mytree(20)

