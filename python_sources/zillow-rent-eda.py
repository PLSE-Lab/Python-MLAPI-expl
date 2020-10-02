#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns, os, warnings
color = sns.color_palette()
warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')

price = pd.read_csv("../input/price.csv")


# In[ ]:


price.head()


# In[ ]:


price.describe()


# In[ ]:


price.info()


# Some rent price from Nov, 2010 to March 2012 are missing.

# In[ ]:


plt.figure(figsize = (15, 8))
state_count = price["State"].value_counts()
ax = sns.barplot(state_count.index, state_count.values, order = state_count.index)
plt.xlabel("state")
plt.ylabel("frequency")
plt.title("State Frequency", fontsize = 15)
rects = ax.patches
values = state_count.values
for rect, value in zip(rects, values):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 5, value, ha = "center", va = "bottom")
plt.show()


# We found:
# * Zillow has houses for renting in over 1,000 cities in PA

# In[ ]:


years = list(set([y.split( )[1] for y in price.columns[6:]]))


# In[ ]:


for i, y in enumerate(sorted(years)[1:7]):
    plt.figure(figsize = (15, 8))
    temp = price[price.columns[8+12*i:8+12*(i+1)]]
    temp.boxplot(list(temp.columns))
    plt.title("{} Rent Distribution".format(y), fontsize = 14)
    plt.xticks(rotation = 45)
    plt.show()


# From above, if we take mean on rent, outliers will contribute a strong effect. Take a look with boxplots with no outliers.

# In[ ]:


for i, y in enumerate(sorted(years)[1:7]):
    plt.figure(figsize = (15, 4))
    temp = price[price.columns[8+12*i:8+12*(i+1)]]
    temp.boxplot(list(temp.columns), showfliers = False)
    plt.title("{} Rent Distribution".format(y), fontsize = 14)
    plt.xticks(rotation = 45)
    plt.show()


# From above, we see that median rents increase slightly each year.

# #### A glance of New York area

# In[ ]:


new_york = price[price["State"] == "NY"]
new_york.head()


# In[ ]:


months = price.columns[6:]
new_york_price = new_york[months]

plt.figure(figsize = (15, 5))
plt.plot(months, np.nanmedian(new_york_price, axis = 0))
plt.title("New York Median Rent per Month", fontsize = 14)
plt.xticks(rotation = 90)
plt.show()


# ### Median rent over USA

# In[ ]:


states = list(price["State"].unique())

plt.figure(figsize = (15, 20))
for state in states:
    temp_price = price[price["State"] == state]
    temp_price = temp_price[months]
    
    plt.plot(months, np.nanmedian(temp_price, axis = 0), label = state)
    
plt.title("US States Median Rent per Month", fontsize = 14)
plt.xticks(rotation = 90)
plt.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)
plt.show()


# Note: Top line is Washington D.C. 
# 
# It is not suprice D.C. has the most expensive houses to rent since itself is a "state" which has no other cities to affect its rent (sample size is small). 
# 
# Since population in each state are different, the needs of houses are vary. In this case, we will focus on the first 20 states, which you have seen in the first figure. 

# In[ ]:


some_states = state_count.index[:20]

plt.figure(figsize = (15, 20))
for state in some_states:
    temp_price = price[price["State"] == state]
    temp_price = temp_price[months]
    
    plt.plot(months, np.nanmedian(temp_price, axis = 0), label = state)
    
plt.title("US Top {} States Median Rent per Month (by city)".format(len(some_states)), fontsize = 14)
plt.xticks(rotation = 90)
plt.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)
plt.show()


# Top 3 are 
# * CA, got the first place in 2016
# * NJ, increase smoothly
# * MA, met NJ in 2015
# 
# As California reaches the first after 2016, we will see how it reaches the top.

# ### California

# In[ ]:


california = price[price["State"] == "CA"]
california.head()


# In[ ]:


print("There are {} cities in California".format(len(california["City"].unique())))
print("There are {} metros in California".format(len(california["Metro"].unique())))
print("There are {} counties in California".format(len(california["County"].unique())))


# ### CA Metros

# In[ ]:


metros = california["Metro"].unique()

plt.figure(figsize = (15, 20))
for m in metros:
    temp_price = california[california["Metro"] == m]
    temp_price = temp_price[months]
    
    plt.plot(months, np.nanmedian(temp_price, axis = 0), label = m)
    
plt.title("CA Metro Median Rent per Month", fontsize = 14)
plt.xticks(rotation = 90)
plt.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)
plt.show()


# ### CA Counties

# In[ ]:


counties = california["County"].unique()

plt.figure(figsize = (15, 20))
for c in counties:
    temp_price = california[california["County"] == c]
    temp_price = temp_price[months]
    plt.plot(months, np.nanmedian(temp_price, axis = 0), label = c)
    
plt.title("CA County Median Rent per Month", fontsize = 14)
plt.xticks(rotation = 90)
plt.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)
plt.show()


# ### Top 10 city in CA (by population)

# In[ ]:


plt.figure(figsize = (15, 8))
for c in range(10):
    temp_price = california[california["City"] == california["City"].iloc[c]]
    temp_price = temp_price[months]

    plt.plot(temp_price.columns, np.transpose(temp_price.values), label = california["City"].iloc[c])
    
plt.title("CA Top 10 Cities Rent per Month (by population)", fontsize = 14)
plt.xticks(rotation = 90)
plt.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)
plt.show()


# ### Rent in lagest population city per state

# In[ ]:


plt.figure(figsize = (15, 20))
for s in states:
    temp_price = price[price["State"] == s]
    rank = min(temp_price["Population Rank"])
    temp_price = temp_price[temp_price["Population Rank"] == rank]
    label = "{} {}".format(s, temp_price["City"].unique())
    temp_price = temp_price[months]
    plt.plot(temp_price.columns, np.transpose(temp_price.values), label = label)
    
plt.title("Rent in Largest Population City by State", fontsize = 14)
plt.xticks(rotation = 90)
plt.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)
plt.show()


# In[ ]:




