#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd


# In[ ]:


# Importing the data


# In[ ]:


pop = pd.read_csv('../input/country_population.csv')
fer = pd.read_csv('../input/fertility_rate.csv')
life = pd.read_csv('../input/life_expectancy.csv')


# In[ ]:


# Cleaning the data


# In[ ]:


population = pd.melt(pop, id_vars=['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'], var_name='Year', value_name='pop')
fertility = pd.melt(fer, id_vars=['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'], var_name='Year', value_name='fer_rate')
life_exp = pd.melt(life, id_vars=['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'], var_name='Year', value_name='life_exp')


# In[ ]:


s1 = fertility.iloc[:,5:6].copy()
s1.head()


# In[ ]:


s2 = life_exp.iloc[:,5:6].copy()
s2.head()


# In[ ]:


# Conactenating the columns s1 and s2 


# In[ ]:


new_table = pd.concat([population, s1], axis=1)
new_table.head()


# In[ ]:


final_table = pd.concat([new_table, s2], axis=1)
final_table.head()


# In[ ]:


# Deleting the unwanted columns from the final_table


# In[ ]:


del final_table['Indicator Name']
del final_table['Indicator Code']


# In[ ]:


final_table.head()


# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Comparing the data for INDIA and CHINA


# In[ ]:


india = final_table.loc[final_table['Country Code'] == 'IND']
china = final_table.loc[final_table['Country Code'] == 'CHN']


# In[ ]:


china.head()


# In[ ]:


india.head()


# In[ ]:


# Comapring the population of INDIA and CHINA per year


# In[ ]:


x = range(1960, 2017)
y_india = india.iloc[:,3:4]
y_china = china.iloc[:, 3:4]

plt.title('China vs India')
plt.ylabel('population(billions)')
plt.xlabel('year')

plt.plot(x, y_india, label='India')
plt.plot(x, y_china, label='China')

plt.legend()

plt.show()


# In[ ]:


# Comapring the fertility rate for INDIA and CHINA per year


# In[ ]:


x = range(1960, 2017)
y_india = india.iloc[:,4:5]
y_china = china.iloc[:, 4:5]

plt.title('China vs India')
plt.ylabel('fertility rate(births per woman)')
plt.xlabel('year')

plt.plot(x, y_india, label='India')
plt.plot(x, y_china, label='China')

plt.legend()

plt.show()


# In[ ]:


# Comapring the life expectancy of people in INDIA and CHINA per year


# In[ ]:


x = range(1960, 2017)
y_india = india.iloc[:,5:6]
y_china = china.iloc[:, 5:6]

plt.title('China vs India')
plt.ylabel('life expectancy')
plt.xlabel('year')

plt.plot(x, y_india, label='India')
plt.plot(x, y_china, label='China')

plt.legend()

plt.show()


# In[ ]:




