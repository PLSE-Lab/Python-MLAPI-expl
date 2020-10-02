#!/usr/bin/env python
# coding: utf-8

# Hello people. My name is Stephen and i am from Greece. This is my first attempt on uploading a Kernel to Kaggle. I am new to Data Science world and i just wanted to share with you, some things i 've learned from tutorials around the web,udemy and of course other kernels on Kaggle site. Especially the Titanic one.
# 
# This kernel has some basic data info commands and much exploratory data analysis. Truth is, that creating this kernel is helping me to further understand all seaborn posibilities. It is too exciting.
# 
# Well, i am sorry about my English and i am sorry for the possible mistakes you will probably notice on my Kernel. 

# #### Imports

# In[ ]:


import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style('whitegrid')


# #### Read Data

# In[ ]:


pokemon = pd.read_csv('../input/Pokemon.csv')


# ###### Show me the first 6 rows

# In[ ]:


pokemon.head(6)


# ###### List all column names

# In[ ]:


print(list(pokemon))


# ###### Show me the columns sum

# In[ ]:


print(len(list(pokemon)))


# #### More Info

# In[ ]:


print(pokemon.info())


# #### Even more info

# In[ ]:


print(pokemon.describe())


# #### Missing Values

# In[ ]:


print(pokemon.isnull().sum())


# ###### I dont really like Type 1, Type 2, Sp. Atk, Sp. Def titles. I hate space between letters. I am going to rename those.

# In[ ]:


pokemon.rename(columns={'Type 1': 'TypeA','Type 2':'TypeB', 'Sp. Atk':'Sp.Atk','Sp. Def':'Sp.Def'}, inplace=True)


# #### I see no point having '#' Column. I am going to delete it.

# In[ ]:


pokemon.drop('#', axis=1,inplace=True)


# #### List the sum of pokemons, grouped by their TypeA.

# In[ ]:


print(pokemon.groupby(['TypeA']).size())


# ### Visualize missing values

# As seen above, there are some missing values on TypeB pokemons. There is no really point doing this, as we already know there are missing values, but i like this trick. So let's visualize it. :D

# In[ ]:


sns.set_context('poster',font_scale=1.1)
plt.figure(figsize=(12,9))
sns.heatmap(pokemon.isnull(),yticklabels=False,cbar=False, cmap='viridis')


# #### Let's see the amount of pokemons of each generation. We notice that 6th generation has the less amount of pokemons.

# In[ ]:


plt.figure(figsize=(9,5))
ax = sns.countplot(x='Generation',data=pokemon,palette='viridis')
ax.axes.set_title("Pokemon Generations",fontsize=18)
ax.set_xlabel("Generation", fontsize=16)
ax.set_ylabel("Total", fontsize=16)


# #### I believe the most critical stats of a pokemon, are attack and defense. I wanted to check if there is a correlation betwwen generations and those two stats. I see nothing clear here. It is pretty random

# In[ ]:


sns.set_style('whitegrid')
sns.lmplot('Defense','Attack',data=pokemon, hue='Generation',
           palette='Spectral',size=8,aspect=1.4,fit_reg=False)


# ####  Let's see the top 7 of TypeA pokemons, ordered by Total

# In[ ]:


plt.figure(figsize=(9,5))
a = sns.countplot(y="TypeA", data=pokemon, palette="Blues_d",
              order=pokemon.TypeA.value_counts().iloc[:7].index)
a.axes.set_title("Top 7",fontsize=18)
a.set_xlabel("Total",fontsize=16)
a.set_ylabel("TypeA", fontsize=16)


# #### Let's see how TypeB pokemons are divided by their type . Flying pokemons are clearly the most on our list.

# In[ ]:


plt.figure(figsize=(15,8))
ax = sns.countplot(x='TypeB',data=pokemon,palette='viridis', order=pokemon.TypeB.value_counts().index)
ax.axes.set_title("Type B Pokemons",fontsize=20)
ax.set_xlabel("Type",fontsize=16)
ax.set_ylabel("Total",fontsize=16)
for item in ax.get_xticklabels():
    item.set_rotation(60)


# #### I am interested on checking the distribution of all pokemons, according their total power. I admit that i was expecting to see a more normal one than this.

# In[ ]:


x = pokemon['Total']

bins = np.arange(150, 800, 12)
plt.figure(figsize=(14,8))
sns.set_context('poster')
ax = sns.distplot(x, kde=False, bins = bins, color = 'darkred',hist_kws={"alpha":0.7})
ax.axes.set_title("Total power of pokemons",fontsize=25)
ax.set_xlabel("Total")
ax.set_ylabel("Pokemons")


# #### It is expected to wonder if legendary pokemon's stats are differentiated from normal ones. In this example we are going to check hit points.We see that it tends to be higher, this is kinda normal, because legendary pokemons are supposed to be stronger than normal ones.

# In[ ]:


plt.figure(figsize=(10, 7.5))
sns.boxplot(x='Legendary',y='HP',data=pokemon,palette='winter')


# #### As expected, legendary pokemons have generally much higher total points but not much more attack power. I guess their total power is raised a bit on all their stats, not only attack power.

# In[ ]:


sns.set_style('whitegrid')
sns.lmplot('Total','Attack',data=pokemon, hue='Legendary',
           palette='coolwarm',height=8,aspect=1.3,fit_reg=False)


# #### Let's see it more clear using a regplot. There is no Legednary Pokemon with less than ~580 total points.

# In[ ]:


plt.figure(figsize=(12,8))
sns.set_context('poster')
ax = sns.regplot(x="Legendary", y="Total", data=pokemon, x_jitter=.07, fit_reg=False,color="darkseagreen",)
ax.axes.set_title("Normal VS Legendary",fontsize=25)
ax.set_xlabel("N vs L")
ax.set_ylabel("Total")


# #### Checking legendary vs normal pokemons specific abilities. In this case HP | Attack

# In[ ]:


sns.set_style('whitegrid')
sns.set_context('poster',font_scale=1.1)
sns.lmplot(x="HP", y="Attack", col="Legendary",data=pokemon,
           palette='coolwarm',height=8,aspect=1.4,fit_reg=True)


# #### I was curious about the differences between pokemon generations. I wanted to check their total points, in case newer generations tend to have higher total point score, but it seems random to me.

# In[ ]:


plt.figure(figsize=(12,8))
sns.set_context('poster')
ax = sns.regplot(x="Generation", y="Total", data=pokemon, x_jitter=.07, fit_reg=False,color="darkseagreen",)
ax.axes.set_title("Power per generation",fontsize=25)
ax.set_xlabel("Generations")
ax.set_ylabel("Total")


# I am going to add more into this. Need more numbers and some machine learning on guessing if a pokemon is legendary or perhaps, guessing it's generation, although it seems to me that generation canot be easily guessed.
# 
# Anyway, CU around xD

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




