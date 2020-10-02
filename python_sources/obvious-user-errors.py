#!/usr/bin/env python
# coding: utf-8

# ## Motivation
# 
# Lea and me, we are currently working on an anomaly detection system for the openfoodfacts community. To play with algorithms and gain a feeling for the data we decided to start on Kaggle. During the next days we will publish some kernels that all belong to our overall analysis and findings. 
# 
# Within this kernel we want to explore and deal with discrete and obvious error sources of the open food facts app:
# 
# * Products exceeding the 100g limit
# * Negative nutrient values 
# * Energy-nutrient mismatches
# * Sugar exceeds carbohydrates
# 
# If you enjoy our analysis you can make us happy with an upvote ;-)

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set_style("darkgrid")

import os
print(os.listdir("../input"))

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# In[ ]:


data = pd.read_csv("../input/nutrition_table.csv", index_col=0)
data.head()


# How many samples are present in our data chunk?

# In[ ]:


data.shape[0]


# ## Sugar exceeds carbohydrates

# Let's peek at some examples:

# In[ ]:


sugar_errors = data[data.sugars_100g > data.carbohydrates_100g].copy()
sugar_errors.head(6)


# In[ ]:


sugar_errors.shape[0] / data.shape[0] * 100


# Ok, around 0.2 % of the data has this kind of error source. Interestingly some salsa products came up with this sugar error. Let's plot the differences of how much the sugar exceeds the carbohydrates:

# In[ ]:


sugar_errors["differences"] = sugar_errors["sugars_100g"].values - sugar_errors["carbohydrates_100g"]


# In[ ]:


plt.figure(figsize=(20,5))
sns.distplot(sugar_errors.differences, color="Fuchsia", kde=False, bins=30)
plt.xlabel("Difference: Sugars - Carbohydrates")
plt.ylabel("Frequency")
plt.title("Error Kind: Sugars exceed Carbohydrates")


# For many samples with this error the difference is close to zero, but there are some as well where the difference is 100 g! That's amazing. What products belong to them? 

# In[ ]:


sugar_errors[sugar_errors.differences > 80]


# Oh sucralose! This is a sweetener and perhaps our analysis can become difficult with this kind of product. Does it contain carbohydrates? I think - yes - as the name already tells us with -ose that is a carbohydrate. But where to but its values? To sugar, to carbohydrates, both? We have to find out, how to correctly deal with sweeteners!

# In[ ]:


sugar_errors[sugar_errors.differences < 1].head(5)


# Perhaps the user flipped over sugars and carbohydrates in these cases?! :-)

# ## Energy-nutrient mismatches
# 
# To find out whether our nutrient energies fit to the energy that was plugged into the database by the user we performed a reconstruction as follows:
# 
# *Reconstruction = 39 kJ times Fat + 17 kJ times Proteins + 17 kJ times Carbohydrates*
# 
# We assume that valid products have a reconstructed energy close to the user provided energy and that both show a linear dependency. Let's explore the scatterplot of both features to examine some energy error sources:

# In[ ]:


energy_errors = np.zeros(data.shape[0])
energy_errors[(data.energy_100g > 3900) | (data.reconstructed_energy > 3900)] = 1
energy_errors[(data.energy_100g == 0) & (data.reconstructed_energy > 0)] = 2
energy_errors[(data.energy_100g > 0) & (data.reconstructed_energy == 0)] = 3


# In[ ]:


fig, ax = plt.subplots(1,2,figsize=(20,5))
ax[0].scatter(data.energy_100g,
              data.reconstructed_energy,
              color="tomato",
              s=0.2)
ax[0].set_xlabel("User provided energy")
ax[0].set_ylabel("Reconstructed energy")
ax[0].set_title("Energy with obvious errors")
ax[1].scatter(data.energy_100g.values[energy_errors == 0],
              data.reconstructed_energy.values[energy_errors == 0],
              color="mediumseagreen",
              s=0.2)
ax[1].set_xlabel("User provided energy")
ax[1].set_ylabel("Reconstructed energy")
ax[1].set_title("Energy without obvious errors")


# * On the **left** you can see that the user provided energy exhibits enormous outliers where the user provived too high numbers of energy. As the energy is limited by $39 kJ/g \cdot 100g = 3900 kJ$ we can easily drop this kind of **too-high-energy-errors**. As you can see by the few outliers of the reconstructed energy there are some errors of this kind as well. 
# * In addition we can see heavy **energy-reconstructed-energy mismatches** on the left hand side. These are all data points that are placed on the axis of the plot. In this case the user hasn't provided the information of protein, fat and carbohydrates but of energy (or the otherway round). Consequently these kind or errors correspond to **incomplete nutrition tables**.
# * On the **right** you can see the scatterplot without these obvious errors. We can clearly see the linear dependency of the reconstructed and the user provided energy. Though we can detect smoother mismatches as well given by the spread of data spots. 
# 
# Now let's have a closer look at the axis errors:
# 
# ### Spots with zero energies

# In[ ]:


data[energy_errors == 2].head(10)


# Wow! That's very cool! Some of these products are marked as sugar free, sweeteners or flavored water. Looking at the products we encounter the problem of: **What is meant by carbohydrates?** Some kind of carbohydrates are not contributing to the energy and consequently its value is zero even though our reconstruction yields high values. We have to find a strategy to work with this kind of products as this is not an error. 
# 
# ### Spots with zero reconstructed energies

# In[ ]:


data[energy_errors == 3].head(10)


# This kind of error is not that obvious. If you search these products in the internet you can find out that these candy cane mints belong to sugar-free pastilles that should not contain any energy at all (zero calories). Consequently some of the products with this kind of errors are like sweeteners or zero-carlorie products again. This is really a difficult kind of product!

# ## Negative nutrient values
# 
# This kind of error may not occur very often as negative values seem to be added wrongly on purpose. Let's see if there are some negative values present in our data chunk:

# In[ ]:


data.min()[data.min() < 0]


# Yes we can see that the minimum values of sugars and proteins are negative. Can we guess that these products were wrong by purpose?

# In[ ]:


data[(data.sugars_100g < 0) | (data.proteins_100g < 0) ]


# In these cases it seems more that the negative values were unintendet. It's really bad that the app does not provide some checks to capture these kind of errors and sends some feedback to the user. 

# ## Products exceeding the 100g limit
# 
# For these kind of errors we build the g_sum feature. It contains the sum of amounts of proteins, carbohydrates and fat. If the obtained value is higher as 100g or lower than 0g there is something wrong with the data entry. Let's find out, how many products exceed the limits:

# In[ ]:


data["exceeded"] = np.where(((data.g_sum > 100) | (data.g_sum < 0)), 1, 0)
data[data.exceeded==1].shape[0] / data.shape[0] * 100


# Around 0.8 % of all data entries exceed the limits. 

# In[ ]:


exceeds = data[(data.g_sum < 0) | (data.g_sum > 100)].g_sum.value_counts()
exceeds = exceeds.iloc[0:10]

plt.figure(figsize=(20,5))
sns.barplot(x=exceeds.index.values, y=exceeds.values, order=exceeds.index, palette="Reds")
plt.title("Common exceeding amouts of proteins, fat, carbohydrates")
plt.xlabel("Summed amounts of proteins, fat, carbohydrates")
plt.ylabel("Frequency")


# Most products only exceeds the 100g by a very small fraction of 0.01 g. This is neglectable and we can easily solve this by rounding the g_sum values. How many errors are left after doing so?

# In[ ]:


data["g_sum"] = np.round(data.g_sum)
g_sum_errors = data[(data.g_sum < 0) | (data.g_sum > 100)].g_sum.value_counts().sum()
g_sum_errors /data.shape[0]


# Ok this looks way better than before. Only 0.005 % of the data points are left with this kind of **exceeding limits** error.

# ## Conclusion
# 
# We have found several sources of errors that are based on wrong user inputs:
# 
# 1. For some products the sum of proteins, carbohydrates and fat exceeds the 100 g limit. But in most cases the total sum is very close to 100 and consequently these products should still be part of the anomaly detection clustering with out mixture model.
# 2. We have found some examples of products were the sugars are higher than the carbohydrates. In some of these cases the product was a sweetener. We should include these kind of products even though it is not obvious how to deal with them.
# 3. The sweeteners were part of the user given and reconstructed energy mismatches as well. Here we found a high reconstructed energy with zero user given energy. This is not an error due to wrong user inputs as you can find nutrition tables declared that way in the internet. Again we should not exclude such products beforehand.
# 4. We found heavy outliers of energy and reconstructed energy that are based on errors and could be excluded. 
# 5. We encountered negative values and in contrast to the other error kinds such products can be excluded from the start. 
