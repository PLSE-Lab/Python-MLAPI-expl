#!/usr/bin/env python
# coding: utf-8

# # Purpose
# The purpose of this kernel is to take a first look at the Pokemon dataset and create some preliminary  visualizations.

# # Import Packages
# The first step is to import the packages we will be using to read the data. For this kernel, we will be using pandas, matplotlib and seaborn to create our visualizations.

# In[ ]:


#Import Packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# # Load Data and First Look
# Now, we are going to load the pokemon.csv dataset and take our first look at the table. 

# In[ ]:


#Load Pokemon dataset
pokemon = pd.read_csv('../input/Pokemon.csv')

#First look at the data
pokemon.head()


# # Bar Graph: Pokemon Type
# Using mathplotlib, we are going to create a bar graph of the Type 1 column in the pokemon dataset to get a better sense of what kinds of Pokemon there are. 

# In[ ]:


type_one_count = pokemon['Type 1'].value_counts()
type_one_count.plot.bar()
plt.title('Types of Pokemon')
plt.xlabel('Type')
plt.ylabel('Number')
plt.show()


# # Line Graph: Pokemon Stats by Generation
# This line graph contains HP, Attack, Defense, Special Attack, Special Defense and Speed divided into Generations. Based on the line graph, you can tell that the majority of the Pokemons' skills peaked in the 4th Generation.

# In[ ]:


stats_by_generation = pokemon.groupby('Generation').mean()[['HP','Attack','Defense','Sp. Atk','Sp. Def','Speed']]
stats_by_generation.plot.line()
plt.title('Stats by Generation')
plt.xlabel('Generation')
plt.ylabel('Number')
plt.show()


# # Box Plots: Combined 
# This bit of code, using seaborn, will show us all of the box plots for the columns in the pokemon.csv dataset (excluding 'Total', '#', 'Generation' and 'Legendary' columns). Based on the box plots, you can see that there are some potential outliers in the pokemon dataset. 

# In[ ]:


pokemon2 = pokemon.drop(['Total','#','Generation','Legendary'],1)
sns.boxplot(data = pokemon2)
plt.title('Combined Box Plots')
plt.xticks(rotation = 90)
plt.show()


# # Histogram: Hit Points
# If we want to know what kind of distribution the hit points data follows, we can execute the following code using seaborn to create a histogram. We can tell from the histogram that the hit points data may contain several outliers that is causing the data to be skewed right. 

# In[ ]:


dist_hp = pokemon['HP']
sns.distplot(dist_hp)
plt.title('Distribution of Hit Points')
plt.show()


# # Histogram: Attack
# Here is the histogram for the attack values in the dataset. We can tell from the histogram that the attack values follow a slightly more normal distribution. 

# In[ ]:


dist_attack = pokemon['Attack']
sns.distplot(dist_attack)
plt.title('Distribution of Attack')
plt.show()


# # Histogram: Defense
# Here is the histogram for defense values which is shows a right skew. 

# In[ ]:


dist_defense = pokemon['Defense']
sns.distplot(dist_defense)
plt.title('Distribution of Defense')
plt.show()


# # Histogram: Special Attack
# Here is the histogram for special attack which contains a right skew. 

# In[ ]:


dist_spattack = pokemon['Sp. Atk']
sns.distplot(dist_spattack)
plt.title('Distribution of Special Attack')
plt.show()


# # Histogram: Special Defense
# Here is the histogram for special defense values which shows a right skew. 

# In[ ]:


dist_spdefense = pokemon['Sp. Def']
sns.distplot(dist_spdefense)
plt.title('Distribution of Special Defense')
plt.show()


# # Histogram: Speed
# Here is the histogram for speed values which shows a slight right skew. 

# In[ ]:


dist_speed = pokemon['Speed']
sns.distplot(dist_speed)
plt.title('Distribution of Speed')
plt.show()

