#!/usr/bin/env python
# coding: utf-8

# In this Kernel I will simulate a outcomes of throwing different amount of 6d dicesvisualized with both Matplotlib and Seaborn libraries.

# Part 1: importing libraries and defining a functions:

# In[ ]:


import random
import matplotlib
import matplotlib.pyplot as plt

dict={}

def throw():
    global x    
    x+=random.randint(1,6)

def multi_throw(dice_amount):
    global x
    x=0
    for i in range(dice_amount):
        throw()
    dict[dice_amount].append(x)
    
def multi_times(time_amount, dice_amount):
    dict[dice_amount]=[]
    for i in range(time_amount):
        multi_throw(dice_amount)


# Part 2: generating results for different amount of 6d dices and packing those to 4 different lists:

# In[ ]:


x=0
th_amount=100000

for i in range(20):
    multi_times(th_amount, i)


# Part 3: Having the results in different lists we may visualize all of them on Matplotlib histogram:

# In[ ]:


fig, ax = plt.subplots()
plt.xlabel('score')
plt.ylabel('occurences')
plt.title('Histogram of '+str(th_amount)+' dice throws')

ax.hist(dict[1],
        bins=range(0,8),
        color="green",
        alpha=0.8,
        )

plt.show()


# Distribution of results of one dice throw are an exception: those present the flat distribution. Lets check the distribution of two dices throw:

# In[ ]:


fig, ax = plt.subplots()
plt.xlabel('score')
plt.ylabel('occurences')
plt.title('Histogram of '+str(th_amount)+' dice throws')

ax.hist(dict[2],
        bins=range(0,20),
        color="red",
        alpha=0.8,
        )

plt.show()


# We can observe the Gaussian distribution forming. Lets check the 3 dices throwing results:

# In[ ]:


fig, ax = plt.subplots()
plt.xlabel('score')
plt.ylabel('occurences')
plt.title('Histogram of '+str(th_amount)+' dice throws')

ax.hist(dict[2],
        bins=range(0,15),
        color="blue",
        alpha=0.8,
        )

plt.show()


# In[ ]:


fig, ax = plt.subplots()
plt.xlabel('score')
plt.ylabel('occurences')
plt.title('Histogram of '+str(th_amount)+' dice throws')


ax.hist(dict[3],
        bins=range(0,36),
        color="red",
        alpha=0.8,
        )

plt.show()


# Matplotlib has visualised the results of throwing 5 dices. We can clearly see the Gaussian distribution of results. Lets see the 6 dices results:

# In[ ]:


fig, ax = plt.subplots()
plt.xlabel('score')
plt.ylabel('occurences')
plt.title('Histogram of '+str(th_amount)+' dice throws')

histy='step'

ax.hist(dict[6],
        bins=range(0,36),
        color="red",
        alpha=0.8,
        )

plt.show()


# Okay, Gaussian distribution of results once again. We may realize that distributions of thrown dices are Gaussian distributions no matter the dices amount. I wouldn't be myself if I wouldnt go to an extreme to prove it, lets go into throwing 20 dices for 1 milion times to prove it:

# In[ ]:


fig, ax = plt.subplots()
plt.xlabel('score')
plt.ylabel('occurences')
plt.title('Histogram of '+str(th_amount)+' dice throws')

histy='step'

ax.hist(dict[19],
        color="green",
        bins=range(40,120),
        alpha=0.8,
        )

plt.show()


# Lets put distributions of 3, 4, 5 & 6 dice throws on one plot:

# In[ ]:


fig, ax = plt.subplots()
plt.xlabel('score')
plt.ylabel('occurences')
plt.title('Histogram of '+str(th_amount)+' dice throws')

histy='step'

ax.hist(dict[6],
        bins=range(2,36),
        color="red",
        alpha=1.0,
        )

ax.hist(dict[5],
        bins=range(2,36),
        color="blue",
        alpha=0.8,
        )

ax.hist(dict[4], 
        bins=range(2,36),
        color="green",
        alpha=0.7,
        )

ax.hist(dict[3], 
        bins=range(2,36),
        color="yellow",
        alpha=0.5,
        )

ax.legend(["6 dices", "5 dices", "4 dices", "3 dices"])

plt.show()


# This Matplotlib histogram looks kinda shoddy, lets draw a Seaborn instead as it looks much clearer:

# In[ ]:


import seaborn as sns

sns.distplot(dict[3], kde=False)


# In[ ]:


sns.distplot(dict[3], kde=False)
sns.distplot(dict[4], kde=False)


# In[ ]:


sns.distplot(dict[3], label="3 dices", kde=False)
sns.distplot(dict[4], label="4 dices", kde=False)
sns.distplot(dict[5], label="5 dices", kde=False)
sns.distplot(dict[6], label="6 dices", kde=False)
plt.legend()


# We may observe that with every next added dice throw the distribution is more stretched and due to this fact 'flattened'. This is due to fact that there are more possibilities that results may fall into when there are more dices thrown.

# In[ ]:


for i in range(1,14):
    sns.distplot(dict[i], kde=False)

plt.show()

