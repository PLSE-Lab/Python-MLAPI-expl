#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# This session covers both days 4 and 5 of the data challenge.
# 
# - Data Challenge: Day 4, Visualize Categorical Data
# - Data Challenge: Day 5, Using a Chi-Square Test
# 
# # Day 4 -- Visualize Categorical Data
# 
# The tasks for this date are as follows
# 
# 1. Load the data
# 2. Pick a column with a categorical label
# 3. Plot a bar-chart
# 4. ... with a title
# 5. Extra credit: Try another visualization
# 6. Optional: publically publish this notebook
# 
# ## Load the data
# 

# In[ ]:


data = pd.read_csv('../input/Health_AnimalBites.csv')


# In[ ]:


data.columns


# In[ ]:


data.head()


# ## Pick a categorical label
# 
# SpeciesIDDesc seems interesting.

# In[ ]:


set(data.SpeciesIDDesc)


# ## Plot a bar-chart with a title
# 
# Of course we could go with Matplotlib's `pyplot.bar()` function. I think it would be far simpler to use Seaborn's `countplot()` instead.

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

sns.countplot(x='SpeciesIDDesc', data=data)

plt.title('Species of animals which bit humans') # Sorry, couldn't think of a better name
plt.xlabel('Species')
plt.ylabel('No. of bite cases reported')
plt.xticks(rotation=-45)

plt.show()


# Okay, this makes it looks like the animals from OTHER through FERRET are innocent. It could be because that dogs and cats are more domesticated than the rest.
# 
# Nevertheless, dogs are more likely to bite humans than cat, that is, according to this dataset.
# 
# ## Extra credit: Try another visualization
# 
# Let's toss the defaults and go for a custom sunburst diagram!
# 
# Surnburst diagrams are quite famous in R and D3.js, however there is no popular library in Python that implements it. This routine should do the trick (courtesy of [Sirex's  post](https://stackoverflow.com/a/46790802/4565943))

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

def sunburst(nodes, total=np.pi * 2, offset=0, level=0, ax=None):
    ax = ax or plt.subplot(111, projection='polar')

    if level == 0 and len(nodes) == 1:
        label, value, subnodes = nodes[0]
        ax.bar([0], [0.5], [np.pi * 2])
        ax.text(0, 0, label, ha='center', va='center')
        sunburst(subnodes, total=value, level=level + 1, ax=ax)
    elif nodes:
        d = np.pi * 2 / total
        labels = []
        widths = []
        local_offset = offset
        for label, value, subnodes in nodes:
            labels.append(label)
            widths.append(value * d)
            sunburst(subnodes, total=total, offset=local_offset,
                     level=level + 1, ax=ax)
            local_offset += value
        values = np.cumsum([offset * d] + widths[:-1])
        heights = [1] * len(nodes)
        bottoms = np.zeros(len(nodes)) + level - 0.5
        rects = ax.bar(values, heights, widths, bottoms, linewidth=1,
                       edgecolor='white', align='edge')
        for rect, label in zip(rects, labels):
            x = rect.get_x() + rect.get_width() / 2
            y = rect.get_y() + rect.get_height() / 2
            rotation = (90 + (360 - np.degrees(x) % 180)) % 360
            ax.text(x, y, label, rotation=rotation, ha='center', va='center') 

    if level == 0:
        ax.set_theta_direction(-1)
        ax.set_theta_zero_location('N')
        ax.set_axis_off()


# The data to be depicted here is going to assess which gender bytes the most for a given animal.

# In[ ]:


otherBytes = data[(data.SpeciesIDDesc!='DOG') & (data.SpeciesIDDesc!='CAT')]

bites = [
    ('Bites', len(data), [
        ('Dog', len(data[data.SpeciesIDDesc=='DOG']),[
            ('Male', len(data[(data.SpeciesIDDesc=='DOG') & (data.GenderIDDesc=='MALE')]), []),
            ('Female', len(data[(data.SpeciesIDDesc=='DOG') & (data.GenderIDDesc=='FEMALE')]), [])
        ]),
        ('Cat', len(data[data.SpeciesIDDesc=='CAT']),[
            ('Male', len(data[(data.SpeciesIDDesc=='CAT') & (data.GenderIDDesc=='MALE')]), []),
            ('Female', len(data[(data.SpeciesIDDesc=='CAT') & (data.GenderIDDesc=='FEMALE')]), [])
        ]),
        ('Others', len(otherBytes),[])
    ])
]
sunburst(bites)
plt.title('Gender of Animals which Bit Humans')
plt.show()


# Looks like most of the reported cases are from male animals.

# ## Optional: Publically publish this notebook
# 
# Sure.
# 
# # Day 5 -- Using a Chi-Square test
# 
# The tasks for this date are as follows
# 
# 1. Load the data and stats package
# 2. Pick two columns with categorical variables
# 3. Calculate chi-square
# 4. Extra credit: Visualize the dataset.
# 5. Optional: publically publish this notebook
# 
# 
# ## Load the data and stats package

# In[ ]:


import scipy.stats as stats 

data = pd.read_csv('../input/Health_AnimalBites.csv') # Yes, it was done before


# ## Pick two columns with categorical variables

# In[ ]:


data.head()


# We are going to assess whether the gender of a dog affects its preference on where to bite. So, the columns we are going to need for this analysis are
# 
# - GenderIDDesc
# - WhereBittenIDDesc
# 
# ## Calculate chi-square
# 

# In[ ]:


dogs = data[data.SpeciesIDDesc == 'DOG']
rctable = pd.crosstab(dogs.GenderIDDesc, dogs.WhereBittenIDDesc)
rctable


# Time to establish the null and alternate hypotheses.
# 
# - $H_0$: There is no difference in preference of where to bite between male and female dogs.
# - $H_a$: Male and female dogs likely prefer to bite different places

# In[ ]:


stats.chisquare(rctable.values[0], rctable.values[1])


# We can observe a significantly high $\chi^2$ value and a negligible $p$ value; all the more reason to reject our null hypothesis, $H_0$, and accept our alternative one, $H_a$. 
# 
# So there really *is* a preference depending on the gender of the dogs (according to the reported cases). By the looks of the contingency table, male dogs are more likely to bite the head than female dogs. However, dogs, in general, tend to bite the body than the head (could be because it's easier to grab).
# 
# ## Optional: publish this notebook
# 
# Done!
