#!/usr/bin/env python
# coding: utf-8

# # MEI Introduction to Data Science
# # Lesson 3 - Activity 3 (OCR)
# 
# This activity is an opportunity to practise the methods covered in this lesson on the OCR large data set.
# 
# The code you used for grouping was:
# 
# `print(travel_2011_data.groupby(['Region'])['Bicycle percent'].mean())`
# 
# The code you used for filtering was:
# 
# `petrol_data = cars_data[cars_data['PropulsionTypeId'] == 'Petrol']
# print(petrol_data['CO2'].mean())
# print(petrol_data['CO2'].std())`
# 

# ## Problem
# > Is there a difference in the use of any of the modes of travel for different regions?

# ## Getting the data

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt

# importing the data
travel_2011_data=pd.read_csv('../input/ocrlds/OCR-lds-travel-2011.csv', thousands=',')

# inspecting the dataset to check that it has imported correctly
travel_2011_data.head()


# ## Exploring the data

# In[ ]:


# check the datatypes
travel_2011_data.dtypes


# In[ ]:



# use describe for any fields you are going to investigate and filter out or replace any unusable values
travel_2011_data['Bicycle'].describe()


# ## Analysing the data

# Caculate stats for difference methods

# In[ ]:


# find the means and standard deviations for different fields grouped by region
regionsDf = travel_2011_data.groupby('Region',as_index=False).agg(["mean","std", "sum"])
regionsDf


# ### Calc percentage of travel for each region

# In[ ]:


transportTypes = ['Underground, tram', 'Train', 'Bus', 'Taxi', 'Motorcycle', 'Driving a car', 'Passenger in a car', 'Bicycle', 'On foot', 'Other']

regionsPercentDF = regionsDf.xs('sum', level=1,axis=1)
regionsPercentDF = regionsPercentDF[transportTypes].div(regionsPercentDF["In employment"], axis=0)
regionsPercentDF = regionsPercentDF[transportTypes].multiply(100)
regionsPercentDF


# ### Percentage to each region

# In[ ]:


import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

regionsPercentDF.plot.bar(figsize=(12, 8))
regionsPercentDF.transpose().plot.pie(subplots=True, autopct='%1.1f%%', figsize=(100, 100))
plt.savefig("pie.jpg")


# In[ ]:


regionsPercentDF.plot.line(figsize=(20, 20))

rowLables = []
for i,v in regionsPercentDF.reset_index().iterrows():
    rowLables.append(v["Region"])
    for t in transportTypes:
        plt.annotate("{:0.2f}%".format(v[t]), xy=(i,v[t]) )
        
plt.xticks(range(0, len(rowLables)), rowLables, rotation=45)


# ### Percentage line graph minus driving a car

# In[ ]:


regionsPercentDF.drop(columns='Driving a car').plot.line(figsize=(20, 10))

rowLables = []
for i,v in regionsPercentDF.reset_index().iterrows():
    rowLables.append(v["Region"])
    for t in transportTypes:
        plt.annotate("{:0.2f}%".format(v[t]), xy=(i,v[t]) )

plt.xticks(range(0, len(rowLables)), rowLables, rotation=45)


# ### Mean number traveling to each

# In[ ]:


#regionsDf.droplevel(0,axis=1)
del regionsDf["In employment"]
del regionsDf["Work at home"]
del regionsDf["Not in employment"]
regionsDf.xs('mean', level=1,axis=1).plot.bar(figsize=(12, 8), title="Mean activites for each region")


# ##

# In[ ]:


# create box plots for different fields grouped by region

with PdfPages('tranportPerRegion.pdf') as pdf:
    for t in transportTypes:
        plt.figure()
        fig=travel_2011_data.boxplot(column = [t],by='Region', vert=False,figsize=(12, 8)).get_figure()
        pdf.savefig(fig)


# In[ ]:


regionsPercentDF['On foot'].sort_values(ascending = False)


# ## Communicating the result

# In[ ]:


# communicate the result


# In all regions except for London, driving a car is the most common form of transport with over 56% driving a car. In London however this is only 26.34% Driving a car. This is likely due to high public transport usage
# 
# London has a signifcantly higher proportion using Underground/tram compare to any other region with 21.822% using Underground/tram with the next heighest being North East with 2.49%. This is likely due to the tube network in London.
# 
# Train use is a bit higher in London (12.9%), South East (7.0%) and East of England (7.0%) compared to all other region have a train usage percentage under 3% with the lowest being the North East with 1.1%.
# 
# London also has marginally higher bus usage with 13.6% and then the next highest is North East with 9.1%.
# 
# Public transport in london accounts for 48.3% of all transport.
# 
# Bicyle usage varies fairly little between regions with being between 1% and 4% in all regions.  

# In[ ]:




