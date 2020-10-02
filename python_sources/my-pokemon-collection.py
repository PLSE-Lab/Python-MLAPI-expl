#!/usr/bin/env python
# coding: utf-8

# I have always been a fan of Pokemon. It has been nearly 15 years, but I have not grown bored of it. Each new episode or movie continues to surprise me.
# 
# This kernel will contain some analysis that I have done. I opted to do this to perfect my data visualization and data analysis skills.
# 
# ## CONTENTS:
# 
# ### 1. Missing Values *(complete)*
# * 1.1 Whether missing values are present or not
# * 1.2 Visualize them
# * 1.3 Learn how to impute them
# 
# ### 2. Analyzing Datatypes *(complete)*
# ### 3. Memory Consumption *(complete)*
# ### 4. Data Exploration *(complete)*
# ### 5. Data Analysis *(planned)*
# ### 6. Feature Engineering *(planned)*

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
import seaborn as sns
import missingno as msno

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

df_pokemon = pd.read_csv('../input/pokemon.csv')

print (df_pokemon.shape)


# # 1. Missing values
# 
# In this section we will see:
# * 1.1 Whether missing values are present or not
# * 1.2 Visualize them
# * 1.3 Learn how to impute them
# 
# ## 1.1 Check the presence of missing values

# In[ ]:


df_pokemon.isnull().values.any()


# Missing values are present in the dataframe. Collect columns with missing values into a list:

# In[ ]:


cols_missing_val = df_pokemon.columns[df_pokemon.isnull().any()].tolist()
print(cols_missing_val)


# There are four columns with missing values.
# 
# Let us get the exact count of each of the missing columns:

# In[ ]:


for col in cols_missing_val:
    print("%s : %d" % (col, df_pokemon[col].isnull().sum()))


# ## 1.2 Visualizing missing values

# In[ ]:


msno.bar(df_pokemon[cols_missing_val],figsize=(20,8),color="#32885e",fontsize=18,labels=True,)


# In[ ]:


msno.matrix(df_pokemon[cols_missing_val],width_ratios=(10,1),            figsize=(20,8),color=(0.2,0.2,0.2),fontsize=18,sparkline=True,labels=True)


# In[ ]:


msno.heatmap(df_pokemon[cols_missing_val],figsize=(10,10))


# ## 1.3 Imputing Missing values
# 
# In order to impute missing values in each column, the other values must be seen as well.
# 
# The following snippet shows the number of unique values in each of the columns having missing values:

# In[ ]:


for col in cols_missing_val:
    print("%s : %d" % (col,df_pokemon[col].nunique()))


# **percentage_male**
# 
# percentage_male: The percentage of the species that are male. Blank if the Pokemon is genderless.
# 
# Hence genderless pokemons can be assigned '-1'

# In[ ]:


df_pokemon['percentage_male'].fillna(np.int(-1), inplace=True)


# **type2**
# 
# Let us look at the various string elements present

# In[ ]:


df_pokemon['type2'].unique()


# We will assign missing values with some string not present in this list:

# In[ ]:


df_pokemon['type2'].fillna('HHH', inplace=True)


# **height_m** and **weight_kg**
# 
# We will replace the missing values with 0.

# In[ ]:


df_pokemon['height_m'].fillna(np.int(0), inplace=True)
df_pokemon['weight_kg'].fillna(np.int(0), inplace=True)


# Now let us check if there are missing values remaining

# In[ ]:


df_pokemon.isnull().values.any()


# # Analyzing Datatypes
# 
# In this section we will see:
# * Different datatypes present in the dataframe
# * Visualize the same

# In[ ]:


print(df_pokemon.dtypes.unique())
print(df_pokemon.dtypes.nunique())


# In[ ]:


pp = pd.value_counts(df_pokemon.dtypes)
pp.plot.bar()
plt.show()


# # Memory Consumption
# 
# Memory usage is an important aspect when dealing with hardware of limited capacity. 
# 
# In this section we will see 
# * How we can monitor the memory usage using pandas
# * Reduce the same by reassigning different datatypes

# In[ ]:


mem = df_pokemon.memory_usage(index=True).sum()
print("Memory consumed by training set  :   {} MB" .format(mem/ 1024**2))


# By altering the datatypes we can reduce memory usage.
# 
# First we will reduce the possible integer datatypes

# In[ ]:


def change_datatype(df):
    float_cols = list(df.select_dtypes(include=['int']).columns)
    for col in float_cols:
        if ((np.max(df[col]) <= 127) and(np.min(df[col] >= -128))):
            df[col] = df[col].astype(np.int8)
        elif ((np.max(df[col]) <= 32767) and(np.min(df[col] >= -32768))):
            df[col] = df[col].astype(np.int16)
        elif ((np.max(df[col]) <= 2147483647) and(np.min(df[col] >= -2147483648))):
            df[col] = df[col].astype(np.int32)
        else:
            df[col] = df[col].astype(np.int64)

change_datatype(df_pokemon)


# In[ ]:


mem = df_pokemon.memory_usage(index=True).sum()
print("Memory consumed by training set  :   {} MB" .format(mem/ 1024**2))


# We can see a reduction in memory used to store the dataframe.
# 
# Now let us reduce columns of type **float64** to type **float32**.

# In[ ]:


def change_datatype_float(df):
    float_cols = list(df.select_dtypes(include=['float']).columns)
    for col in float_cols:
        df[col] = df[col].astype(np.float32)
        
change_datatype_float(df_pokemon)

mem = df_pokemon.memory_usage(index=True).sum()
print("Memory consumed by training set  :   {} MB" .format(mem/ 1024**2))


# We can see a further reduction in memory usage. So overall we have reduced the dataframe size by **~50%** from 0.25MB to 0.12MB.

# # Data Exploration
# 
# Let us now explore the data present in different columns based on the various types.
# 

# In[ ]:


object_cols = list(df_pokemon.select_dtypes(include=['object']).columns)


# ## Primary and Secondary types
# 
# Every pokemon is either a primary type (**type1**) or a combination of both primary and secondary type (**type2**). As a result some pokemon do not have a type2 variable, hence they are left to be with missing values. 

# In[ ]:


df_pokemon = df_pokemon.apply(lambda x: x.str.strip() if isinstance(x, str) else x).replace('HHH', np.nan)


# In[ ]:


df_pokemon['type1'].unique()


# In[ ]:


df_pokemon['type2'].unique()


# In[ ]:


#--- Snippet to split pokemon based on whether they are of single type or dual type ---
single_type_pokemon = []
dual_type_pokemon = []

count = 0
for i in df_pokemon.index:
    if(pd.isnull(df_pokemon.type2[i]) == True):
    #if(df_pokemon.type2[i] == np.nan):
        count += 1
        single_type_pokemon.append(df_pokemon.name[i])
    else:
        dual_type_pokemon.append(df_pokemon.name[i])

print(len(dual_type_pokemon))
print(len(single_type_pokemon))


# In[ ]:


data = [417, 384]
colors = ['yellowgreen', 'lightskyblue']

# Create a pie chart
plt.pie(data, 
        labels= ['Dual type', 'Single type'], 
        shadow=True, 
        colors=colors, 
        explode=(0, 0.15), 
        startangle=90, 
        autopct='%1.1f%%')

# View the plot drop above
plt.axis('equal')
plt.title('Dual vs Single type Pokemon')
# View the plot
plt.tight_layout()
plt.show()


# Primary pokemon type distribution:

# In[ ]:


yy = pd.value_counts(df_pokemon['type1'])

fig, ax = plt.subplots()
fig.set_size_inches(11.7, 8.27)
sns.set_style("whitegrid")

ax = sns.barplot(x=yy.index, y=yy, data=df_pokemon)
ax.set_xticklabels(ax.get_xticklabels(), rotation = 75, fontsize = 12)
ax.set(xlabel='Primary types', ylabel='Count')
ax.set_title('Distribution of Primary Pokemon type')


# ## Classfication
# 
# The Classification of the Pokemon as described by the Sun and Moon Pokedex. Apart from classifying Pokemon based n their natural type (rock, water, fire, etc.), they can also be classified by their physical traits which is shown here.

# In[ ]:


df_pokemon['classfication'].nunique()


# There are over 500 types of Pokemon classified based on their physical traits. Let us just see the top 10 most occurring Pokemon types.

# In[ ]:


ss = pd.value_counts(df_pokemon['classfication'])
for i in range(0, 10):
    
    print ("{} : {} ".format(ss.index[i],  ss[i]))


# ## Capture Rate
# 
# Describes the rate at which a Pokemon can be captured into a pokeball after a fight.
# 
# Let is see the list of possible values:

# In[ ]:


df_pokemon['capture_rate'].unique()


# In[ ]:


yy = pd.value_counts(df_pokemon['capture_rate'])

fig, ax = plt.subplots()
fig.set_size_inches(11.7, 8.27)
sns.set_style("whitegrid")

ax = sns.barplot(x=yy.index, y=yy, data=df_pokemon)
ax.set_xticklabels(ax.get_xticklabels(), rotation = 75, fontsize = 12)
ax.set(xlabel='Capture_rate', ylabel='Number of Pokemon')
ax.set_title('Distribution of capture_rate against number of Pokemon')


# ## Abilities
# 
# Each Pokemon can have one or a list of possible powers.
# 
# Let is see the list of possible abilities:

# In[ ]:


df_pokemon['abilities'].nunique()


# In[ ]:


df_pokemon['abilities'].head(20).unique()


# As we can see many Pokemon have more than one ability. 
# 
# We need to drill down to see have of such individual abilities are present and which of them are the most prevalent.

# So up to now we have seen all features of type object (string datatypes).
# 
# Now let us move over to the features of numeric  datatypes.

# ## Pokedex number
# 
# The entry number of the Pokemon in the National Pokedex. Here each Pokemon is assigned a number from 1 - 801.

# In[ ]:


df_pokemon['pokedex_number'].describe()


# ## Height and Weight 
# 
# * **height_m**  : Height of the Pokemon in metres
# * **weight_kg**: The Weight of the Pokemon in kilograms
# 
# Let us visualize the distribution of Pokemon along the range of height and weight:

# In[ ]:


ax_height = sns.distplot(df_pokemon['height_m'], color="y")


# In[ ]:


ax_weight = sns.distplot(df_pokemon['weight_kg'], color="r")


# In[ ]:


#--- Average weight ---
ax = sns.pointplot(df_pokemon['weight_kg'])


# In[ ]:


#---Average height ---
ax = sns.pointplot(df_pokemon['height_m'], color = 'g')


# ***Observations:***
# * Average weight and height is ~59 kg and ~113 m respectively.
# * Most Pokemon have a height of 0.1 - 4 m and weigh between 1 - 200 kg.

# ## base_egg_steps
# 
# The number of steps required to hatch an egg of the Pokemon.

# In[ ]:


df_pokemon['base_egg_steps'].nunique()


# In[ ]:


ax = sns.countplot(df_pokemon['base_egg_steps'])


# ***Observations:***
# * There are only 10 unique steps for an egg to hatch
# * Most of the Pokemon eggs take 5120 steps to hatch

# ## experience_growth
# 
# The Experience Growth of the Pokemon

# In[ ]:


df_pokemon['experience_growth'].nunique()


# In[ ]:


ax = sns.countplot(df_pokemon['experience_growth'])


# ***Observations:***
# * There only 6 unique levels of experience
# * Most Pokemon have experience in the range 1M - 1.25M
# 
# ***What can be done?***
# * We can see which primary type Pokemon has maximum experience growth.
# * Understand  correlation between egg steps and experience.
# To be done in Data Analysis section.

# ## base_happiness
# 
# Base Happiness of the Pokemon.

# In[ ]:


df_pokemon['base_happiness'].nunique()


# In[ ]:


ax = sns.countplot(df_pokemon['base_happiness'])


# *** Observations: ***
# * There are 6 unique levels of happiness
# * Most Pokemon have happiness index of 70
# * Very few are extremely happy!! :)
# * A handful of Pokemon are always grumpy :(
# 
# ***What can be done?***
# * We can see whether this decides their level of strength or not.

# ## hp
# 
# The Base HP of the Pokemon

# In[ ]:


df_pokemon['hp'].nunique()


# In[ ]:


ax = sns.distplot(df_pokemon['hp'], rug=True, hist=False)


# ***Observations: ***
# * Most Pokemon have base hp in the range 25 - 150.
# 
# ***What can be done?***
# 
# We can see whether this decides their:
# * level of strength, 
# * depth in attack
# * resistance to other attacks.

# ## attack and defense of Pokemon
# 
# The Base Attack and Defense of the Pokemon.

# In[ ]:


print(df_pokemon['attack'].nunique())
print(df_pokemon['defense'].nunique())


# In[ ]:


ax_attack = sns.distplot(df_pokemon['attack'], color="r", hist=False)
ax_defense = sns.distplot(df_pokemon['defense'], color="b", hist=False)


# ## sp_attack and sp_defense
# 
# The Base Special Attack and Base Special Defense of the Pokemon.

# In[ ]:


print(df_pokemon['sp_attack'].nunique())
print(df_pokemon['sp_defense'].nunique())


# In[ ]:


ax_attack = sns.distplot(df_pokemon['sp_attack'], color="g", hist=False)
ax_defense = sns.distplot(df_pokemon['sp_defense'], color="y", hist=False)


# ***Observations: ***
# * Most Pokemon have base attack and defense and the related specials in the range 25 - 150.
# * The end range of defense is more than that of attack.
# 
# ***What can be done?***
# * Correlation between attack and special_attack. Same goes for defense as well.
# * Relation with base_hp.
# * Club relation with primary/secondary type of Pokemon.
# 
# ## against_
# 
# There are 18 features that denote the amount of damage taken against an attack of a particular type.
# 
# Let us get the list of columns having string **against_**

# In[ ]:


cols = df_pokemon.columns
against_ = []
for col in cols:
    if ('against_' in str(col)):
        against_.append(col)
        
print(len(against_))        


# We have 18 as required and mentioned.
# 
# Now we can take the average of each of these columns and find the maximum to determine which attack majority of pokemon are susceptible to.
# 
# We would like to have a list of unique values across all these columns.

# In[ ]:


unique_elem = []
for col in against_:
    unique_elem.append(df_pokemon[col].unique().tolist())
    
result = set(x for l in unique_elem for x in l)

result = list(result)
print(type(result))
#print(df_pokemon['against_psychic'].unique())


# We can see that it takes only within the above 6 values.
# 
# Let us see the distribution of these values in each of the columns:

# In[ ]:


import random

for col in range(0, len(against_)):
    print (against_[col])
    print (df_pokemon[against_[col]].unique())
    pp = pd.value_counts(df_pokemon[against_[col]])
    
    color = ['g', 'b', 'r', 'y', 'pink', 'orange', 'brown']
            
    pp.plot.bar(color = random.choice(color))
    plt.show()


# ***Observations: ***
# * Most pokemon suffer from every attack of atleast 1.0.
# * There are some pokemon that can withstand certain attacks.
# 
# ***What can be done?***
# * We can relate which type of pokemon is more susceptible to attacks against certain types of pokemon.
# 
# ## speed
# 
# The Base Speed of the Pokemon

# In[ ]:


print(df_pokemon['speed'].nunique())


# In[ ]:


df_pokemon['speed'].describe()


# In[ ]:


ax_height = sns.distplot(df_pokemon['speed'], color="orange")


# Now what is the fastest and slowest Pokemon?

# In[ ]:


print('Fastest Pokemon: {}'.format(df_pokemon.name[df_pokemon['speed'].idxmax()] ))
print('Slowest Pokemon: {}'.format(df_pokemon.name[df_pokemon['speed'].idxmin()] ))


# We can come up with our our definition of speed.
# 
# * Fast Pokemon > Mean + Standard_dev
# * Slow Pokemon < Mean - Standard_dev
# * Very Fast Pokemon > Mean + 2(Standard_dev)
# * Very Slow Pokemon < Mean - 2(Standard_dev)
# 
# Let us see what we get:

# In[ ]:


speed_statistics = df_pokemon['speed'].describe()

mean = speed_statistics[1]
standard_dev = speed_statistics[2]

#--- Create lists for the four categories mentioned ---
fast_pokemon = []
slow_pokemon = []
v_fast_pokemon = []
v_slow_pokemon = []
normal = []

for i in range(0, len(df_pokemon)):
    if(df_pokemon.speed[i] > mean + (2 * standard_dev)):
        v_fast_pokemon.append(df_pokemon.name[i])
    elif(df_pokemon.speed[i] < mean - (2 * standard_dev)):
        v_slow_pokemon.append(df_pokemon.name[i])
    elif(df_pokemon.speed[i] > mean + standard_dev):
        fast_pokemon.append(df_pokemon.name[i])
    elif(df_pokemon.speed[i] < mean - standard_dev):
        slow_pokemon.append(df_pokemon.name[i])
    else:
        normal.append(df_pokemon.name[i])
    
speed_levels = ['fast_pokemon','slow_pokemon','v_fast_pokemon','v_slow_pokemon','normal']
speed_count = [len(fast_pokemon), len(slow_pokemon), len(v_fast_pokemon),len(v_slow_pokemon),len(normal)]

xlocations = np.array(range(len(speed_count)))
width = 0
plt.bar(xlocations, speed_count, color = 'r')
plt.xticks(xlocations+ width, speed_levels)
#xlim(0, xlocations[-1]+width*2)
plt.title("Count of Pokemon for different Speed levels")


# ***Observations: ***
# * Over 500 pokemon have speed within the mean and single standard deviation which is normal for our standards.
# * A small portion has been labeled as fast and slow.
# * An even smaller portion has been termed very fast and very slow.
# 
# ***What can be done?***
# * We can relate speed to strength of attack and the type of pokemon it is.
# * We can add these features as well if modeling is done at a later time.
# 
# ## generation
# 
# The numbered generation which the Pokemon was first introduced

# In[ ]:


print(df_pokemon['generation'].nunique())


# In[ ]:


ax = sns.countplot(x="generation", data=df_pokemon)


# Let us visualize it in terms of percentage:

# In[ ]:


pp = pd.value_counts(df_pokemon.generation)
pp.plot.pie(startangle=90, autopct='%1.1f%%', shadow=False, explode=(0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05))
plt.axis('equal')
plt.show()


# ***Observations: ***
# * A bulk of the Pokemon are from 1st and 5th generations.
# 
# ***What can be done?***
# * We will see if later generation Pokemon have more hp, are legendary pokemon, or sufficient attck capability.
# 
# ## is_legendary
# 
# Denotes if the Pokemon is legendary.

# In[ ]:


ax = sns.countplot(y=df_pokemon['is_legendary'], data=df_pokemon)


# ***Observations: ***
# * Less than 100 Pokemon are deemed to be legendary.
# 
# 
# ***What can be done?***
# * See whether legendary pokemon have exceptional egg steps, expereince, apart from attack, defense and base hp.
# * What makes them so special to attain status of legend? We will see in the Data Analysis section.

# 

# 

# 

# # STAY TUNED FOR MORE !!!
