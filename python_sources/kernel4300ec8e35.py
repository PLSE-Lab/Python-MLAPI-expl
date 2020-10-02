#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# Lets find some correlation between diets and mortality rates from COVID-19. Also keep in mind that diets may reflect country's economic status which may influence mortality rates.

# In[ ]:


quantity_kg = pd.read_csv("../input/covid19-healthy-diet-dataset/Food_Supply_Quantity_kg_Data.csv")

# Lets see what our data looks like
quantity_kg


# In[ ]:


# Lets add a few more columns
quantity_kg["Mortality"] = quantity_kg["Deaths"] / quantity_kg["Confirmed"] * 100
quantity_kg["Recovery Rate"] = quantity_kg["Recovered"] / quantity_kg["Confirmed"] * 100 # change recovery to percentage
quantity_kg.dropna(inplace=True)
quantity_kg


# In[ ]:


quantity_kg.columns


# Lets compare the consumption rates and recovery and mortality rates

# In[ ]:


import matplotlib.pyplot as plt
from numpy import cov
for index in quantity_kg.columns[1:25]:
    plt.plot(quantity_kg[index], quantity_kg["Mortality"], 'ro', label="Mortality Rate")
    plt.plot(quantity_kg[index], quantity_kg["Recovery Rate"], 'go', label="Recovery Rate")
    plt.xlabel("{} consumption (%)".format(index))
    plt.ylabel('Rates')
    plt.show()


# Focusing on the data mortality rates and adding line of best fit

# In[ ]:


for index in quantity_kg.columns[1:25]:
    plt.plot(quantity_kg[index], quantity_kg["Mortality"], 'ro', label="Mortality Rate")
    m, b = np.polyfit(quantity_kg[index], quantity_kg["Mortality"], 1)
    plt.plot(quantity_kg[index], m * quantity_kg[index]  + b)
    plt.xlabel("{} consumption (%)".format(index))
    plt.ylabel('Rates')
    plt.show()


# Lets do the same thing with the calories

# In[ ]:


kcal = pd.read_csv("../input/covid19-healthy-diet-dataset/Food_Supply_kcal_Data.csv")
kcal["Mortality"] = kcal["Deaths"] / kcal["Confirmed"] * 100
kcal["Recovery Rate"] = kcal["Recovered"] / kcal["Confirmed"] * 100 # change recovery to percentage
for index in kcal.columns[1:25]:
    plt.plot(kcal[index], kcal["Mortality"], 'ro', label="Mortality Rate")
    plt.plot(kcal[index], kcal["Recovery Rate"], 'go', label="Recovery Rate")
    plt.xlabel("{} consumption (kcal)".format(index))
    plt.ylabel('Rates')
    plt.show()


# In[ ]:


for index in kcal.columns[1:25]:
    plt.plot(kcal[index], kcal["Mortality"], 'ro', label="Mortality Rate")
    plt.xlabel("{} consumption (kcal)".format(index))
    plt.ylabel('Rates')
    plt.show()


# Heatmap of correlations

# In[ ]:


corr = quantity_kg.corr()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,len(quantity_kg.columns),1)
ax.set_xticks(ticks)
plt.xticks(rotation=90)
ax.set_yticks(ticks)
ax.set_xticklabels(quantity_kg.columns)
ax.set_yticklabels(quantity_kg.columns)
plt.show()


# ## Observations
# It looks like notable positive correlations include animal products and meat consumption and negative correlations include cereals, fish and seafoods, fruits, miscellaneous, offals, oilcrops, pulses, spices, starchy roots and treenuts. Lets confirm these suspicions.

# In[ ]:


correlation_rates = pd.Series(data=np.nan, index=quantity_kg.columns[1:24])
for index in quantity_kg.columns[1:24]:
    # could have also used list comprehension here but its a bit ugly
    m, b = np.polyfit(quantity_kg[index], quantity_kg["Mortality"], 1)
    correlation_rates[index] = m
correlation_rates = correlation_rates.sort_values(ascending=False)
correlation_rates.head(10)


# In[ ]:


correlation_rates.tail(10)


# ## Conclusion
# A line of best fit seems to show that **animal fats**, **eggs** and **vegetable oils** have the highest positive correlation and the highest negative correlations correspond with **spices**, **stimulants**  and **aquatic products**.
