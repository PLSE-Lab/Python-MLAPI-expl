#!/usr/bin/env python
# coding: utf-8

# ## What crimes category is increasing and what is decreasing in London?
# 

# Tha aim of this notebook is to analyze the London Crimes events  and identify any seasonal patterns.
# 

# In[ ]:


import numpy as np
import pandas as pd
from pandas.io import gbq
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt
import seaborn as sns
import bq_helper
Crimes = bq_helper.BigQueryHelper(active_project="bigquery-public-data", dataset_name="london_crime")


# In[ ]:


QUERY ="""
SELECT major_category, year, sum(value) as number 
FROM  `bigquery-public-data.london_crime.crime_by_lsoa`
group by  major_category,year
order by  number DESC; 
"""
df = Crimes.query_to_pandas_safe(QUERY)
df.head()


# ## First, let's see the percentage of events for each category

# In[ ]:


criminal_summary = df.groupby('major_category').sum()['number']
criminal_summary


# #### Let's visualize.

# In[ ]:


explode_sequence = ('0.1,' * len(criminal_summary)).split(',')
explode_sequence=[float(item) for item in explode_sequence if item !=""]

plt.pie(criminal_summary,explode=explode_sequence, labels=criminal_summary.index, autopct='%1.2f%%',
        shadow=False, startangle=90, );
plt.title("London Criminal Events 2008 -2016");


# ### The metric used to define whether the category crime is increasing or decreasing is the linear regression coefficient of number of crimes over the years.
# If positive, is increasing.
# If negative  is decreasing.
# 
# The absolute value of the coefficient defines the magnitude of the increase/decrease. 

# To compare the linear coefficients for the different sets we need to standardize the data.
# Otherwise, a category with major number of events would have different weight than a category with less events.

# In[ ]:


crimis = (set(df['major_category']))
cat_crime = dict()
min_reg=999
min_item=""
max_reg=0
max_item=""

for crim in crimis:
    temp=df[df['major_category'] == crim];
    X=temp[['year']];
    val_max = temp.number.max()
    val_min = temp.number.min()
    temp['number']=(temp.number-val_min)/(val_max-val_min);
    
    regr = linear_model.LinearRegression()
    regr.fit(X, temp.number.values)
    cat_crime[crim] = regr.coef_
    if regr.coef_ > max_reg:
        max_reg = regr.coef_
        max_item = crim
    if regr.coef_ < min_reg:
        min_reg = regr.coef_
        min_item = crim
        
   

print('coef max {}, for the crime "{}"'.format(max_reg, max_item))
print('coef min {}, for the crime "{}"'.format(min_reg, min_item))


# ### Let's see all the results.

# In[ ]:


mio_dataset=pd.DataFrame.from_dict(cat_crime, orient='index',columns=['coef_regr'])
mio_dataset.sort_values('coef_regr')


# In[ ]:


mio_dataset.hist();


# Most of the coefficients are negative or close to zero.
# Which is a good sign.
# Let's have a look at the worst and the best.

# ### Just out of curiosity... let's find the crime category with most crimes decrease and the one with most crimes increase. 

# In[ ]:


tempmax=df[df['major_category'] == max_item];
val_max = tempmax.number.max()
val_min = tempmax.number.min()
tempmax['number']=(tempmax.number-val_min)/(val_max-val_min)
regrmax = linear_model.LinearRegression()
regrmax.fit(tempmax[['year']], tempmax.number.values)
y_maxPredict=regrmax.predict(tempmax[['year']])

tempmin=df[df['major_category'] == min_item];
val_max = tempmin.number.max()
val_min = tempmin.number.min()
tempmin['number']=(tempmin.number-val_min)/(val_max-val_min)
regrmin = linear_model.LinearRegression()
regrmax.fit(tempmin[['year']], tempmin.number.values)
y_minPredict=regrmax.predict(tempmin[['year']])

X=tempmax[['year']];

plt.figure(figsize=(15, 6))
plt.scatter(tempmax[['year']],tempmax.number.values,color='red')    
plt.plot(tempmax[['year']],y_maxPredict,color='red')

plt.scatter(tempmin[['year']],tempmin.number.values, color='green')    
plt.plot(tempmin[['year']],y_minPredict,color='green')
plt.xlabel('... Time ...')
plt.ylabel("Crime per Day in standizide format")
plt.title('The Crime Category Trend for the increasing "{}" and decresing "{}" Category'.format(max_item,min_item));


# #### The charts of the Crime category with most  decrease crimes (with the real value).

# In[ ]:


tempmax=df[df['major_category'] == max_item];

regrmax = linear_model.LinearRegression()
regrmax.fit(tempmax[['year']], tempmax.number.values)
y_maxPredict=regrmax.predict(tempmax[['year']])

tempmin=df[df['major_category'] == min_item];

regrmin = linear_model.LinearRegression()
regrmax.fit(tempmin[['year']], tempmin.number.values)
y_minPredict=regrmax.predict(tempmin[['year']])

X=tempmax[['year']];

plt.figure(figsize=(15, 6))
plt.scatter(tempmax[['year']],tempmax.number.values,color='red')    
plt.plot(tempmax[['year']],y_maxPredict,color='red')

plt.xlabel('... Time ...')
plt.ylabel("Crime per Year  ")
plt.title('Trend for the most increasing Category Crime "{}"'.format(max_item,min_item));


# #### The charts of the category crimes with most crimes decrease (with the real value).

# In[ ]:


tempmax=df[df['major_category'] == max_item];

regrmax = linear_model.LinearRegression()
regrmax.fit(tempmax[['year']], tempmax.number.values)
y_maxPredict=regrmax.predict(tempmax[['year']])

tempmin=df[df['major_category'] == min_item];
regrmin = linear_model.LinearRegression()
regrmax.fit(tempmin[['year']], tempmin.number.values)
y_minPredict=regrmax.predict(tempmin[['year']])

X=tempmax[['year']];

plt.figure(figsize=(15, 6))

plt.scatter(tempmin[['year']],tempmin.number.values, color='green')    
plt.plot(tempmin[['year']],y_minPredict,color='green')
plt.xlabel('... Time ...')
plt.ylabel("Crime per Year ")
plt.title('Trend for the Decreasing Category Crimes "{}"'.format(min_item));


# From the year 2008 to the year 2016 the number of crimes in Drugs category decreased of more of 25 thousands per year.

# ### Summary:
# 
# Overall, the number of crimes in London is decreasing, there are only two categories of crimes where the number is increasing, which are "Violence Against the Person" and "Other Notifiable Offences".

# In[ ]:




