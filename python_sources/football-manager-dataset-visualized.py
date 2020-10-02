#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # Data visualization
import matplotlib.pyplot as plt
import scipy.stats as stats

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

# Any results you write to the current directory are saved as output.


# In[ ]:


# The very first look at the dataset
df = pd.read_csv("../input/dataset.csv")
df.head()


# In[ ]:


df.columns


# In[ ]:


df.shape


# In[ ]:


df.info()


# In[ ]:


# Quick statiscial distribution of the age column.
# Rough meaning: 75% of our data includes players <= 27 years old
df['Age'].describe()


# In[ ]:


# Even though the .info method showed us all columns are non-null, let's recheck
df['Age'].isna().unique()
# Output array only holds False values
# This implies we have no null values


# In[ ]:


# List of unique values in the Age column
age_values = df['Age'].unique()
np.sort(age_values)


# In[ ]:


# Split the datframe using groupby method
age = df.groupby(['Age'])
age


# In[ ]:


# Let's pick two features to analyze in detail
# Here we are 'applying' a function to the grouby object 'age'
age[['IntCaps', 'IntGoals']].describe()


# In[ ]:


# Let's extract the max value of international caps and goals for each age group.
age[['IntCaps', 'IntGoals']].max()


# In[ ]:


# Let's make our first data visualization! 
f, axarr = plt.subplots(2, sharex=True, figsize=(8,10))

international_max = age[['IntCaps', 'IntGoals']].max()
axarr[0].plot(international_max)
axarr[0].set_title('Maximum International Caps and Goals by age-group')

international_sum = age[['IntCaps', 'IntGoals']].sum()
axarr[1].plot(international_sum)
axarr[1].set_title('Total International Caps and Goals by age-group')
axarr[1].set_xlabel("Age")
plt.tight_layout()


# In[ ]:


goals_max = international_max['IntGoals']
x = []
for value, dataframe in age:
    if goals_max[value] > 0:
        print(value, dataframe[dataframe.IntGoals == goals_max[value]]['Name'].values)


# In[ ]:


age.mean().head()


# In[ ]:


# Extracting just the physicals attributes
physicals = age[['Height', 'Weight', 'Acceleration', 'Pace', 'Agility', 
                 'Balance', 'Jumping', 'NaturalFitness', 'Stamina', 'Strength']]
# The count for players aged 47+ is too low to feel secure about the mean so let's ignore those ages
physicals_subset = physicals.agg(['mean', 'std']).loc[14:46]

#Setting up the subplot
f, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8), (ax9, ax10)) = plt.subplots(5, 2, figsize=(15, 20))

ax1.plot(physicals_subset['Height'])
ax1.set_title('Height')

ax2.plot(physicals_subset['Weight'])
ax2.set_title('Weight')

ax3.plot(physicals_subset['Acceleration'])
ax3.set_title('Acceleration')

ax4.plot(physicals_subset['Pace'])
ax4.set_title('Pace')

ax5.plot(physicals_subset['Agility'])
ax5.set_title('Agility')

ax6.plot(physicals_subset['Balance'])
ax6.set_title('Balance')

ax7.plot(physicals_subset['Jumping'])
ax7.set_title('Jumping')

ax8.plot(physicals_subset['NaturalFitness'])
ax8.set_title('NaturalFitness')

ax9.plot(physicals_subset['Stamina'])
ax9.set_title('Stamina')

l1, l2 = ax10.plot(physicals_subset['Strength'])
ax10.set_title('Strength')

# Adding legend and title for the subplot
f.legend((l1, l2), ('Mean', 'Std. Dev'), 'upper right')
f.suptitle('All Physicals Analyzed', fontsize=16, y=1.02)

plt.tight_layout()


# In[ ]:


# Distribution of Acceleration throughout the dataset
plt.figure(figsize=(8, 6), dpi=100)
ax = sns.countplot(x="Acceleration",data=df)
ax.set_title("Distribution of Acceleration")


# In[ ]:


# Jointplot showing the correlation between Height and Aerial Ability

sns.set(style="darkgrid", color_codes=True)
j = sns.jointplot("Height", "AerialAbility", data=df, kind="reg", color="m", height=10)
j.annotate(stats.pearsonr)


# In[ ]:


# Jointplot showing the correlation between Pave and Acceleration

sns.set(style="darkgrid", color_codes=True)
j = sns.jointplot("Pace", "Acceleration", data=df, kind="reg", color="m", height=10)
j.annotate(stats.pearsonr)


# In[ ]:


# Plotting a pairplot of all physical attributes to get a quick overview of dependence
pair_df = physicals.mean().loc[14:46]
sns.pairplot(pair_df, kind="reg", plot_kws={'line_kws':{'color':'red'}, 'scatter_kws': {'alpha': 0.5}})


# In[ ]:


# Function to add pearsonr to each x-y plot
def corrfunc(x, y, **kws):
    (r, p) = stats.pearsonr(x, y)
    ax = plt.gca()
    ax.annotate("r = {:.2f} ".format(r),
                xy=(.1, 1.2), xycoords=ax.transAxes)
    ax.annotate("p = {:.3f}".format(p),
                xy=(.6, 1.2), xycoords=ax.transAxes)


# In[ ]:


# Testing correlation between International goals and physical attributes
# Balance stands out!
j = sns.pairplot(data=df.query('IntGoals > 5'),
                  y_vars=['IntGoals'],
                  x_vars=['Balance', 'Jumping', 'NaturalFitness', 'Stamina', 'Strength'],
                kind="reg", plot_kws={'line_kws':{'color':'red'}, 'scatter_kws': {'alpha': 0.5}})
j.map(corrfunc)


# In[ ]:


# Testing the remaining physical attributes
# Notice the weight plot has several players with zero weight
g = sns.pairplot(data=df.query('IntGoals > 5'),
                  y_vars=['IntGoals'],
                  x_vars=['Height', 'Agility', 'Weight', 'Acceleration', 'Pace'],
                kind="reg", plot_kws={'line_kws':{'color':'red'}, 'scatter_kws': {'alpha': 0.5}})

g.map(corrfunc)


# In[ ]:


# Let's query the dataframe to find players who have scored international goals but have null weight
test = df.query('IntGoals > 0')
test[test['Weight']==0]


# In[ ]:


# Map players to age group!
age_values = df['Age'].unique()
age_values = np.sort(age_values)
test = df
bins = [14, 18, 24, 30, 35, np.inf]
names = ['<18', '18-24', '24-30', '30-35', '35+']

test['AgeRange'] = pd.cut(test['Age'], bins, labels=names)

sns.pairplot(data=test.query('IntGoals > 5'),
                  y_vars=['IntGoals'],
                  x_vars=['Balance', 'Strength', 'Jumping'], hue="AgeRange",
            kind="reg", plot_kws={'scatter_kws': {'alpha': 0.7}}, height=10)


# In[ ]:




