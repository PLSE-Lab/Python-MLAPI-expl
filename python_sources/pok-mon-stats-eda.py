#!/usr/bin/env python
# coding: utf-8

# ## Exploratory Data Analysis
# by [Iqrar Agalosi Nureyza](https://www.kaggle.com/iqrar99)
# 
# Dataset : [Pokemon stats](https://www.kaggle.com/abcsds/pokemon)
# 

# Hello Everyone!
# This is my first Kaggle Notebook. I'm a beginner and I try my best to do data analysis. This Pokemon Dataset is a very good dataset to begin with and I hope you can understand my analysis. I am very open in accepting criticism and suggestions for perfecting this kernel.
# 
# If you find this notebook useful, feel free to **Upvote**.
# 

# **Table of Contents**
# 1. [Basic Analysis](#1)
#     * [Data Cleaning](#2)
#     * [Frequency](#3)
#     * [The Strongest and The Weakest](#4)
#     * [The Fastest and The Slowest](#5)
#     * [Summary](#6)
# 2. [Data Visualisation](#7)
#     * [Count Plot](#8)
#     * [Pie Plot](#9)
#     * [Box Plot and Violin Plot](#10)
#     * [Swarm Flot](#11)
#     * [Heat Map](#12)

# <a id = "1"></a>
# ## Basic Analysis

# In[ ]:


#importing all important packages
import numpy as np #linear algebra
import pandas as pd #data processing
import matplotlib.pyplot as plt #data visualisation
import seaborn as sns #data visualisation
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#Input Data
data = pd.read_csv("../input/pokemon/Pokemon.csv") #reading csv file and save it into a variable
data.head(10) #show the first 10 rows in data


# We finally know that our data has 12 columns.
# * *Name*       : Nominal data
# * *Type 1*     : Nominal data
# * *Type 2*     : Nominal data
# * *Total *     : Ratio data
# * *HP *        : Ratio data
# * *Attack*     : Ratio data
# * *Defense*    : Ratio data
# * *Sp Atk*     : Ratio data
# * *Sp Def*     : Ratio data
# * *Speed*      : Ratio data
# * *Generation* : Ordinal data
# * *Legendary*  : Nominal data

# <a id = "2"></a>
# ### Data Cleaning
# I found some unneeded text in *Name* column. For example, "CharizardMega Charizard X" should be "Mega Charizard X". So we need to remove all characters before "Mega".

# In[ ]:


data.Name = data.Name.str.replace(".*(?=Mega)", "")
data.head(10)


# In[ ]:


data = data.set_index('Name') #change and set the index to the name attribute
data = data.drop(['#'],axis=1) #drop the columns with axis=1; axis=0 is for rows
data.head()


# If we look at row 5, there is a NaN type in the *Type 2* row. We can choose to delete or fill in the data. But in this case if we delete rows that has NaN, then it will mess up our data. Then we'll choose to fill it by copying the data from *Type 1* column.

# In[ ]:


data['Type 2'].fillna(data['Type 1'], inplace=True)
data.head(10)


# <a id = "3"></a>
# ### Frequency
# Now, let's see all unique types in *Type 1* and *Type 2*.

# In[ ]:


print("Type 1:",data["Type 1"].unique(), "=", len(data["Type 1"].unique()))
print("Type 2:",data["Type 2"].unique(), "=", len(data["Type 2"].unique()))


# And we get that there are 18 unique types.
# Ok, now we use *value_counts()* to count each unique type in *Type 1 * and * Type 2*

# In[ ]:


print(data["Type 1"].value_counts())
print(data["Type 2"].value_counts())


# We can conclude that the highest frequency in *Type 1* is **Water** and in *Type 2* is **Flying**. On the other hand, the lowest frequency in *Type 1* is **Flying** and in *Type 2* is **Bug**

# <a id = "4"></a>
# ### The Strongest and The Weakest
# **Who is the strongest and the weakest Pokemons by types?** We will find out.

# In[ ]:


strongest = data.sort_values(by='Total', ascending=False) #sorting the rows in descending order
strongest.drop_duplicates(subset=['Type 1'],keep='first')
#since the rows are now sorted in descending order
#thus we take the first row for every new type of pokemon i.e the table will check Type 1 of every pokemon
#The first pokemon of that type is the strongest for that type
#so we just keep the first row


# So, we finally know who is the strongest pokemons by types. And also the strongest of the strongest pokemon is **Mega Rayquaza**, the Dragon type. And also we know that 10/18 Strongest Pokemons by types are Legendary. Let's check who is the weakest by types.

# In[ ]:


weakest = data.sort_values(by='Total') #sorting the rows in ascending order
weakest.drop_duplicates(subset=['Type 1'],keep='first')


# We finally know who is the weakest pokemons by types. The weakest of the weakest pokemon is **Sunkern**, the Grass type. We can't find the Legendary category here.

# <a id = "5"></a>
# ### The Fastest and The Slowest
# **Now, who is the fastest and the slowest Pokemons by types?**

# In[ ]:


fastest = data.sort_values(by='Speed', ascending=False) #sorting the rows in descending order
fastest.drop_duplicates(subset=['Type 1'],keep='first')


# The Fastest pokemon is **DeoxysSpeed Forme** which is a Legendary Psychic pokemon.

# In[ ]:


slowest = data.sort_values(by='Speed') #sorting the rows in ascending order
slowest.drop_duplicates(subset=['Type 1'],keep='first')


# This data shows that Bug type and Normal type have slowest pokemon compared other types.

# <a id = "6"></a>
# ### Summary

# In[ ]:


#now, let's summary the data
data.describe()


# _________________________________________________________

# <a id = "7"></a>
# ## Data Visualisation
# And now we move to the important part where we will get informations from visualizing our data. First, we make count plots to see value counts for each type

# <a id = "8"></a>
# ### Count Plot

# In[ ]:


sns.set(style = 'darkgrid')
f, ax = plt.subplots(2,1, figsize = (18,10)) #making 2 count plots 

sns.countplot(x = 'Type 1', data = data, order = data['Type 1'].value_counts().index ,ax = ax[0])
sns.countplot(x = 'Type 2', data = data, order = data['Type 2'].value_counts().index ,ax = ax[1])

plt.show()


# <a id = "9"></a>
# ### Percentages for each type
# **How about percentages for each type?** We can make a pie plot to get informations about that. But before that we must to count all pokemon types from both columns and avoid double counting, because we copied the elements in the *Type 1* column to the *Type 2* column before. **Why we do this?** Because there are pokemon that don't have Type 2.

# In[ ]:


#we create a dictionary to make process easier
types_count = {'Grass' : 0, 'Fire' : 0, 'Water' : 0, 'Bug' : 0, 'Normal' : 0, 'Poison' : 0, 
               'Electric' : 0, 'Ground' : 0, 'Fairy' : 0, 'Fighting' : 0, 'Psychic' : 0, 
               'Rock' : 0, 'Ghost' : 0, 'Ice' : 0, 'Dragon' : 0, 'Dark' : 0, 'Steel' : 0, 'Flying' : 0}

type1 = data["Type 1"]
type2 = data["Type 2"]

for i in range(len(type1)):
    
    #first, count the Type 1 column
    types_count[type1[i]] += 1
    
    #now we count the Type 2 column and avoid double counting
    if type1[i] != type2[i]:
        types_count[type2[i]] += 1

for t in types_count:
    print("{:10} = {}".format(t,types_count[t]))


# We got overall count from both types, now it's time to get *Type 1* count and *Type 2* count.

# In[ ]:


type1_count = {'Grass' : 0, 'Fire' : 0, 'Water' : 0, 'Bug' : 0, 'Normal' : 0, 'Poison' : 0, 
               'Electric' : 0, 'Ground' : 0, 'Fairy' : 0, 'Fighting' : 0, 'Psychic' : 0, 
               'Rock' : 0, 'Ghost' : 0, 'Ice' : 0, 'Dragon' : 0, 'Dark' : 0, 'Steel' : 0,
               'Flying' : 0}
type2_count = {'Grass' : 0, 'Fire' : 0, 'Water' : 0, 'Bug' : 0, 'Normal' : 0, 'Poison' : 0, 
               'Electric' : 0, 'Ground' : 0, 'Fairy' : 0, 'Fighting' : 0, 'Psychic' : 0, 
               'Rock' : 0, 'Ghost' : 0, 'Ice' : 0, 'Dragon' : 0, 'Dark' : 0, 'Steel' : 0,
               'Flying' : 0}

for i in range(len(type1)):
    type1_count[type1[i]] += 1
    type2_count[type2[i]] += 1

print("TYPE 1")
for t in type1_count:
    print("{:10} = {}".format(t,type1_count[t]))
print("-------------------")
print("TYPE 2")
for t in type2_count:
    print("{:10} = {}".format(t,type2_count[t]))


# Yes! Now finally we can make the pie plot. We'll make 3 pie plots: Type 1 count, Type 2 count. and overall count,

# In[ ]:


f, axs = plt.subplots(2,2, figsize=(20,20))

labels ='Grass', 'Fire', 'Water', 'Bug', 'Normal', 'Poison', 'Electric', 'Ground',         'Fairy', 'Fighting', 'Psychic', 'Rock', 'Ghost' ,'Ice' ,'Dragon' ,'Dark' ,         'Steel','Flying'
    
size0 = [95,64,126,72,102,62,50,67,40,53,90,58,46,38,50,51,49,101] #overall count
size1 = [70,52,112,69,98,28,44,32,17,27,57,44,32,24,32,31,27,4] #Type 1 count
size2 = [58,40,73,20,65,49,33,48,38,46,71,23,24,27,29,30,27,99] #Type 2 count

#Overall pie
axs[0,0].pie(size0, labels = labels, autopct='%1.1f%%' ,startangle = 90)
axs[0,0].axis("equal")
axs[0,0].set_title("Overall", size = 20)

#Type1 pie
axs[0,1].pie(size1, labels = labels, autopct='%1.1f%%' ,startangle = 180)
axs[0,1].axis("equal")
axs[0,1].set_title("Type 1", size = 20)

#Type2 pie
axs[1,0].pie(size2, labels = labels, autopct='%1.1f%%' ,startangle = 90)
axs[1,0].axis("equal")
axs[1,0].set_title("Type 2", size = 20)

f.delaxes(axs[1,1]) #deleting axs[1,1] so it will be white blank
plt.show()


# <a id = "10"></a>
# ### All stats analysis of the pokemons
# let's analyze all ratio data.

# In[ ]:


data2 = data.drop(['Generation', 'Total', 'Legendary'], axis = 1) #we drop some columns that unnecessary
data2.head()


# In[ ]:


#stats for the Attack by Type 1
sns.set_style("whitegrid")
f, ax = plt.subplots(1,1, figsize = (20,10))

ax = sns.boxplot(data = data2, x = 'Type 1', y = 'Attack')
ax.set_title(label='Attack by Type 1', size = 20)

plt.show()


# Take a look at the data. We can conclude that the **Dragon** type pokemon has an advantage over other types because they have a higher attack compared to other types. Let's see the starter pokemon : Fire, Water, and Grass. Fire Pokemons have a higher attack than Water and Grass. So it's very recommended to use it for attacking opponent for every beginner trainer. (If you ever played pokemon, then u can understand what *starter pokemon* is)

# In[ ]:


#stats for the Attack by Type 2
f, ax = plt.subplots(1,1, figsize = (20,10))

ax = sns.boxplot(data = data2, x = 'Type 2', y = 'Attack')
ax.set_title(label='Attack by Type 2', size = 20)

plt.show()


# And from this chart, we can conclude that **Fighting** pokemon have a higher attack than other pokemon types. All pokemons that have Fighting as their second type have higher attack value.

# In[ ]:


#stats for the Defense by Type 1
f, ax = plt.subplots(1,1, figsize = (20,10))

ax = sns.violinplot(data = data2, x = 'Type 1', y = 'Defense')
ax.set_title(label='Defense by Type 1', size = 20)

plt.show()


# This shows that **Steel ** pokemon have the highest defense compared the other types.

# In[ ]:


#stats for the Defense by Type 2
f, ax = plt.subplots(1,1, figsize = (20,10))

ax = sns.violinplot(data = data2, x = 'Type 2', y = 'Defense')
ax.set_title(label='Defense by Type 2', size = 20)

plt.show()


# And this shows that **Rock** pokemon is better than** Steel** pokemon in Defense values.

# <a id = "11"></a>
# ### Strongest Generation

# In[ ]:


f, ax = plt.subplots(0,0, figsize = (20,10))
ax = sns.swarmplot(data = data, x = 'Generation', y= 'Total', hue = 'Legendary')

plt.axhline(data['Total'].mean(), color = 'red', linestyle = 'dashed') #giving a straight line on mean

plt.show()


# This shows that 3rd generation has many strong pokemons. And also this data informed that all Legendary pokemon are strong or even the strongest.

# <a id = "12"></a>
# ### Finding any correlation

# In[ ]:


f, ax = plt.subplots(0,0,figsize=(15,10))
ax = sns.heatmap(data.corr(), annot = True, cmap = 'winter') #data.corr() used to make correlation matrix

plt.show()


# From the heat map above, the correlation between the attributes of the pokemon is not to much. The highest correlation is *Sp. Attack* and *Total*, following *Attack *with *Total *and *Sp. Defense* and *Total*.

# ______________________________________

# Ok, that's all my analysis. If you think I missed something, feel free to comment. Thank you for reading this notebook!
