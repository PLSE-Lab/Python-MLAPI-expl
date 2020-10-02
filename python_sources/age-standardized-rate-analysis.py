#!/usr/bin/env python
# coding: utf-8

# ### Provide code to:
# ### (1) Plot top 20 rank countries in any year between 1985 to 2016 with age-standardized suicide rate for either male, female, or both.
# ### (2) Plot age-standardized suicide rates as a function of gdp_per_capita in all year of a country for either male, female, or both

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import Image
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


data = pd.read_csv('../input/suicide-data/master.csv')


# In[ ]:


data[(data.country == 'Albania')&(data.year==1987)].sort_values('age')


# #### It can be seen from above that "5-14 years" does not sort correctly, so replace it with "05-14 years"

# In[ ]:


data.age = data.age.replace('5-14 years','05-14 years')


# In[ ]:


data[(data.country == 'Albania')&(data.year==1987)].sort_values('age')


# #### Now it sorts correctly!
# #### To facilitate comparative analysis the data is standardized using the weights in the table "WHO World Standard Population Distribution (%), based on world average population between 2000-2025", which is available in the link https://www.who.int/healthinfo/paper31.pdf and is shown in the Table below.

# In[ ]:


Image("../input/suicide-table/weights.png")


# In[ ]:


data.loc[data['age']=='05-14 years',"weights"] = 0.1729
data.loc[data['age']=='15-24 years',"weights"] = 0.1669
data.loc[data['age']=='25-34 years',"weights"] = 0.1554
data.loc[data['age']=='35-54 years',"weights"] = 0.2515
data.loc[data['age']=='55-74 years',"weights"] = 0.1344
data.loc[data['age']=='75+ years',"weights"] = 0.03065
data.head()


# #### Check empty cells in column "weights"

# In[ ]:


data[data['weights'].isnull()]


# #### Add another column "'suicides/100k pop (adjusted)" for age standardized rates

# In[ ]:


data['suicides/100k pop (adjusted)']=data['suicides/100k pop']*data['weights']
data.head()


# #### Check the male standardized rates for Lithuania in 2016 

# In[ ]:


data[(data['country']=='Lithuania')&(data['year']==2016)&(data['sex']=='male')]['suicides/100k pop (adjusted)'] 


# In[ ]:


data[(data['country']=='Lithuania')&(data['year']==2016)&(data['sex']=='male')]['suicides/100k pop (adjusted)'].sum()


# #### Check the same for female

# In[ ]:


data[(data['country']=='Lithuania')&(data['year']==2016)&(data['sex']=='female')]['suicides/100k pop (adjusted)'].sum()


# #### Create 'data_group' with type 'DataFrameGroupBy' for further aggregation 

# In[ ]:


data_group = data.groupby(['country', 'year', 'sex'], as_index=False)


# In[ ]:


data_group


# In[ ]:


data_group.get_group(('Argentina', 2015, 'male'))


# #### Create 'data_agg' of type 'DataFrame' with sum of 'suicides/100k pop (adjusted)' from all age groups

# In[ ]:


data_agg = data_group.agg({'suicides/100k pop (adjusted)':'sum'})
data_agg.head(10)


# In[ ]:


type(data_agg)


# #### Create 'data_group_1' with type 'DataFrameGroupBy' for summing data from both male and female 

# In[ ]:


data_group_1 = data_agg.groupby(['country', 'year'], as_index=False)


# In[ ]:


data_agg_1 = data_group_1.agg({'suicides/100k pop (adjusted)': 'sum'})
data_agg_1.head(10)


# #### Input year (1985-2016) and gender (male, female, or both male & female) of the plot

# In[ ]:


year_spec = 2015
# sex_spec = 'male' 
# sex_spec = 'female' 
sex_spec = 'both male & female'  


# #### Use DataFrame "data_agg" for either male or female and "data_agg_1" for both male and female

# In[ ]:


if sex_spec in ('male', 'female'):
    data_specific = data_agg[(data_agg.year==year_spec)&(data_agg.sex==sex_spec)].sort_values('suicides/100k pop (adjusted)', ascending=False)    
else:
    data_specific = data_agg_1[data_agg_1.year==year_spec].sort_values('suicides/100k pop (adjusted)', ascending=False)


# In[ ]:


data_specific_20 = data_specific.head(20)
data_specific_20


# #### Make the final plot

# In[ ]:


plt.figure(figsize=(16,11))
plt.xticks(rotation=90)
plt.yticks(np.arange(0,70,5))
plt.title('Top 20 rank of age standardized '+sex_spec+' suicide rate per 100,000 people in '+str(year_spec))
sns.barplot(x='country', y='suicides/100k pop (adjusted)', data=data_specific_20)
sns.set(font_scale=3)


# ### The code below plot age-standardized suicides per 100k pop rate as a function of gdp_per_capita in all year of a country for either male, female, or both

# #### Create 'data_agg_2' (add 'gdp_per_capita') of type 'DataFrame' with sum of 'suicides/100k pop (adjusted)' from all age groups

# In[ ]:


data_group_2 = data.groupby(['country', 'year', 'sex', 'gdp_per_capita ($)'], as_index=False)


# In[ ]:


data_agg_2 = data_group_2.agg({'suicides/100k pop (adjusted)':'sum'})
data_agg_2.head(10)


# #### Create 'data_agg_3' (remove 'sex') of type 'DataFrame' with sum of 'suicides/100k pop (adjusted)' from all age groups with both sexes 

# In[ ]:


data_group_3 = data.groupby(['country', 'year', 'gdp_per_capita ($)'], as_index=False)


# In[ ]:


data_agg_3 = data_group_3.agg({'suicides/100k pop (adjusted)':'sum'})
data_agg_3.head(10)


# #### Specify a country 

# In[ ]:


countries =['Lithuania', 'Republic of Korea', 'Russian Federation', 'United States', 'Japan', 
            'Finland', 'Australia','Iceland', 'Austria']
country = countries[0]


# #### Specify sex type

# In[ ]:


# sex_spec = 'Male' 
# sex_spec = 'Female' 
sex_spec = 'Both Male & Female' 

if sex_spec in ('Male', 'Female'):
    data_specific = data_agg_2[(data_agg_2.country==country) & (data_agg_2.sex==sex_spec.lower())]
else:
    data_specific = data_agg_3[data_agg_3.country==country]


# #### Make the scatter plot 

# In[ ]:


sns.set(style='white')


# In[ ]:


sns.lmplot(x='gdp_per_capita ($)', y='suicides/100k pop (adjusted)', data=data_specific, hue='year',
           fit_reg=False, palette='icefire', height=6, aspect=9/6)
plt.title('Suicide Rate Per 100k Population as a Function of GDP per capita for '
          +sex_spec+' in '+country)


# #### The plots below show the results of 8 additional countries: Republic of Korea, Russian Federation, United States, Japan, Finland, Australia, Iceland, and Austria in an order of descending suicide rate in 2015. The only countries that do not show a clear sign of declining suicide rate when gdp_per_capta increases are Korea, Japan, and USA. Korea shows a clear suicide rate increase and USA a bathtub curve while the trend of Japan seems unclear.  

# In[ ]:


Image("../input/gdppics/Korea.png")


# In[ ]:


Image("../input/gdppics/Russia.png")


# In[ ]:


Image("../input/gdppics/Japan.png")


# In[ ]:


Image("../input/gdppics/USA.png")


# In[ ]:


Image("../input/gdppics/Finland.png")


# In[ ]:


Image("../input/gdppics/Australia.png")


# In[ ]:


Image("../input/gdppics/Iceland.png")


# In[ ]:


Image("../input/gdppics/Austria.png")


# In[ ]:




