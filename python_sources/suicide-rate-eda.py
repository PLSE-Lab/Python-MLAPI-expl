#!/usr/bin/env python
# coding: utf-8

# # EDA for Global Suicide Rates 
# Data source: [Kaggle dataset](https://www.kaggle.com/russellyates88/suicide-rates-overview-1985-to-2016), created by Kaggle user: [Russel Yates](https://www.kaggle.com/russellyates88)   
# This dataset was compiled from datasets sourced from: the [United Nations Development Program](http://hdr.undp.org/en/indicators/137506), the [World Bank](http://databank.worldbank.org/data/source/world-development-indicators#), [Suicide in the Twenty-First Century (Szamil)](https://www.kaggle.com/szamil/suicide-in-the-twenty-first-century/notebook), and the [World Health Organization](http://www.who.int/mental_health/suicide-prevention/en/)
# ## 1) Function Imports and Global Variable Declarations

# In[ ]:


import numpy as np 
import pandas as pd 
from cycler import cycler
import matplotlib.pyplot as plt
from pylab import rcParams
from statistics import mean, mode
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
# MatPlotLib PyPlot parameters 
rcParams['figure.figsize'] = (12, 6)
def get_hex(rgb): 
    return '#%02x%02x%02x' % (int(rgb[0]*255),int(rgb[1]*255),int(rgb[2]*255))
colors = [get_hex(plt.get_cmap("Pastel1")(i)) for i in range(10)]
rcParams['axes.prop_cycle'] = cycler('color', colors)


# ## 2) Data Loading and Cleaning 

# In[ ]:


data = pd.read_csv("../input/master.csv")
print(data.shape)
data.head()


# From the head, the data in *country-year* appears to be contained in the variables *country* and *year* and is less interpretable. 
# The same can be said for the data in *suicides/100k pop*, which can be calculated from the *suicides_no* and *population* variables. 
# Or at least, that's what the DataFrame's head would lead us  to believe. Let's confirm and then remove them if necessary. 

# In[ ]:


print("Is 'country-year' redundant? ", all(data['country-year'] == data['country'] + data['year'].apply(str)))
print("Is 'suicides/100k pop' redundant? ", all(data['suicides/100k pop'] == (100000*data['suicides_no']/data['population']).apply(lambda x: round(x, 2))))
print("    Inconsistent data: ", [round(x, 2) for x in data['suicides/100k pop'] - (100000*data['suicides_no']/data['population']).apply(lambda x: round(x, 2)) if x != 0])
data.drop(columns=['country-year'], inplace=True)
data.rename({'suicides/100k pop': 'rate'}, axis=1, inplace=True)


# Although the data in *suicides/100k pop* was not perfectly replicable with the existing data, it only differed from our forecasts in 4 out of the 17,820 data points by an amount that was likely due to rounding. Since this data is replicable with local data it's not altogether necessary to keep in it. However, this field communicates the suicide rate per-100K people which is a very useful piece of information, so we may want to keep it in just to avoid having to calculate it again. In fact, the suicide rate is so notable, that it seems like an excelent variable to focus the EDA on. So we're going to look into: **What factors influence the suicide rate and how?**
# 
# Are any other columns redundant? Well, the *age* and *generation* columns are likely to go hand in hand since both are determined by your birth date. Though we aren't given the exact age, we are given a range, is this range reprasentative of their generation? 

# In[ ]:


print(all(le.fit_transform(data['age']) == le.fit_transform(data['generation'])))


# And in retrospect one realizes that since the data was recorded over a 31 year period of time, an age-range that corresponded to 'Generation X' in one period of time would correspond to a completely different generation if you look at other dates. These two columns therefore communicate two different things, the *age* column displays the age of the victim when they commited suicide and the *generation* column displays the time-period they were born in, which (when dealing with data that spans multiple decades) are notably different. The categorical nature of the *age* and *generation* columns may later-on require some amount of ordering, so let's do that now. 

# In[ ]:


print(list(data['age'].unique()))
print(list(data['generation'].unique()))
age_order = ['5-14 years', '15-24 years', '25-34 years', '35-54 years', '55-74 years', '75+ years']
generation_order = ['G.I. Generation', 'Silent', 'Boomers', 'Generation X', 'Millenials', 'Generation Z']


# ## 3) Variable Exploration
# Lets do some naive grouping and plotting to see if we can spot any relationship between the suicide rate and the other variables. 

# In[ ]:


rates = []
for age in age_order: 
    age_data = data[data['age'] == age]
    overall_rate = 100000 * age_data['suicides_no'].sum() / age_data['population'].sum() 
    rates.append(overall_rate)
y_pos = np.arange(len(age_order))
plt.figure(figsize=(12,6))
plt.bar(y_pos, rates, align='center')
plt.xticks(y_pos, age_order, size="large")
plt.ylabel('Suicides per 100K people', size="x-large")
plt.title("Suicides per 100K people, sorted by age of victim", size="xx-large", pad=13)
plt.show() 


# Ok, pretty interesting. There's definitely a positive causal link between age and suicide rate. But does this trend hold up when we acount for the gender of the victim? 

# In[ ]:


male_rates = []; female_rates = []
for age in age_order: 
    age_data = data[data['age'] == age]
    male_age_data = age_data[age_data['sex'] == 'male']
    female_age_data = age_data[age_data['sex'] == 'female']
    male_rate = 100000 * male_age_data['suicides_no'].sum() / male_age_data['population'].sum() 
    female_rate = 100000 * female_age_data['suicides_no'].sum() / female_age_data['population'].sum() 
    male_rates.append(male_rate)
    female_rates.append(female_rate)
y_pos = np.arange(len(age_order))
bar_width = 0.35
fig, ax = plt.subplots(figsize=(12,6))
plt.bar(y_pos, female_rates, bar_width, label="Women")
plt.bar(y_pos + bar_width, male_rates, bar_width, label="Men")
plt.xticks(y_pos, age_order, size="large")
plt.ylabel('Suicides per 100K people', size="x-large")
plt.title("Suicides per 100K people, sorted by age & sex of victim", size="xx-large", pad=13)
plt.legend()
plt.show() 


# Wow, there really was is a stark difference between the suicide rates of men and women. Just eyeballing it, it looks as though the male suicide rate is 3 times the female suicide rate! That's very notable. Let us further explore the effect of gender on suicide rates. 

# In[ ]:


female_data = data[data['sex'] == 'female']
male_data = data[data['sex'] == 'male']
gender_rates = [list(female_data['rate']), list(male_data['rate'])]
fig, ax = plt.subplots(figsize=(18,6))
parts = plt.violinplot(gender_rates, showmeans=False, showextrema=False, vert=False)
plt.xlim(left=-2, right=100)
for i, body in enumerate(parts['bodies']): 
    body.set_facecolor(colors[i])
    body.set_edgecolor("black")
plt.title("Suicide rates, by sex", size="xx-large", pad=10)
ax.set_yticks(np.arange(1, 3))
ax.set_yticklabels(["Female", "Male"], size="large")
ax.set_xlabel('Suicides per 100K people', size="x-large") 
plt.show() 
print(f"Average suicides (per 100K people) for men: {mean(male_data['rate']):.2f}, "+      f"for women: {mean(female_data['rate']):.2f}")


# As is clear from the violin pot and average suicide statistics, men have a much higher rate of suicide than women. In fact, across all time-periods and countries, the rate for men is nearly 4 times that of women! That's huge! 

# In[ ]:




