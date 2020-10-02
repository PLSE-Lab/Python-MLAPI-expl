#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
print("Setup Complete")
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

#import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
 #   for filename in filenames:
  #      print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # Read sf-library Dataset...and Review the data
# 

# In[ ]:


sf_Dataset = pd.read_csv('../input/sf-library-usage-data/Library_Usage.csv')
sf_Dataset


# # show all columns

# In[ ]:


sf_Dataset.columns


# # I will reduce the number of rows from 423K to 20K.

# In[ ]:


new_sf_Dataset = sf_Dataset.sample(20000)
new_sf_Dataset


# In[ ]:





# # *Note_1:
# ## Using Barplot on ['Total Checkouts' and 'Age Range']

# In[ ]:


# Bar chart showing average score for racing games by platform
plt.figure(figsize=(16, 9))

sns.barplot(y=new_sf_Dataset['Total Checkouts'], x=new_sf_Dataset['Age Range'])
# Add label for horizontal axis
#plt.xlabel("")
# Add label for vertical axis
plt.title("Average Score for Age Range, by Total Checkouts")


# # *Note_1:
# ## According to the chart, the higher the average age, the greater the value of Total Checkouts, and the high value of the increase in the age range between 65 and 75, And less Total Checkouts between the Age range between 25 to 34****.

# # *Note_2:
# ## Using Barplot on ['Total Renewals' and 'Age Range']

# In[ ]:


# Bar chart showing average score for racing games by platform
plt.figure(figsize=(16, 9))
sns.barplot(y=new_sf_Dataset['Total Renewals'], x=new_sf_Dataset['Age Range'])
# Add label for horizontal axis
#plt.xlabel("")
# Add label for vertical axis
plt.title("Average Score for Age Range, by Total Checkouts")


# # *Note_2(Total Renewals & Age Range):
# ## According to the chart, the higher the average age, the greater the value of Total Renewals, and the high value of the increase in the age range between 65 and 74, And less Total Renewals between the Age range between 25 to 34.

# # *Note_3:
# 

# In[ ]:


# Bar chart showing average score for racing games by platform
plt.figure(figsize=(20, 9))
sns.barplot(x=new_sf_Dataset['Total Renewals'], y=new_sf_Dataset['Patron Type Definition'])
# Add label for horizontal axis
#plt.xlabel("")
# Add label for vertical axis
plt.title("Average Score for Patron Type Definition, by Total Renewals")


# # *Note_3
# ## The highest results for the Total Renewals for 'staff', 'Retired staff, And The lowest results are for Welcome, Digital Access Card, At User Juvenile, At User Welcome.

# # *Note_4:

# In[ ]:


# Bar chart showing average score for racing games by platform
plt.figure(figsize=(20, 9))
sns.barplot(x=new_sf_Dataset['Total Checkouts'], y=new_sf_Dataset['Patron Type Definition'])
# Add label for horizontal axis
#plt.xlabel("")
# Add label for vertical axis
plt.title("Average Score for Patron Type Definition, by Total Renewals")


# # *Note_4
# ## The highest results for the Total Checkouts for 'staff', 'Retired staff, And The lowest results are for Welcome, Digital Access Card, At User Juvenile, At User Welcome.

# In[ ]:


new_sf_Dataset


# # *Note_5:

# In[ ]:


#new_sf_Dataset = new_sf_Dataset.dropna() 
df = new_sf_Dataset[['Year Patron Registered','Total Checkouts','Total Renewals']]
#df /// values='Year Patron Registered'

ddf = df.set_index(['Year Patron Registered'])
plt.figure(figsize=(16,9))
sns.lineplot(data=ddf)


# # *Note_5
# ## ON > "Year Patron Registered"
# ## 1- The value of total checkouts has been declining significantly over the years, having been at its highest in 2013 and 2015 and being the lowest in value by 2016.
# ## 2- The value of Total Renewals has been declining significantly over the years, having been at its highest in 2013 and being the lowest in value by 2016.
# ## 3- There is also a relationship between 'Total Checkouts' and 'Total Renewals' where the lower the value of checkouts, the lower the value of renewals in the same time period.
# 

# # *Note_6

# In[ ]:


drop_None_Value = new_sf_Dataset.drop(index=new_sf_Dataset[new_sf_Dataset['Circulation Active Year'] == 'None'].index)
new_sf_Dataset = drop_None_Value

df = new_sf_Dataset[['Circulation Active Year','Total Checkouts','Total Renewals']]
#df // values='Year Patron Registered'
ddf = df.set_index(['Circulation Active Year'])
plt.figure(figsize=(16,9))
sns.lineplot(data=ddf)


# # *Note_6
# ## ON > "Circulation Active Year"
# 
# ## 1- Total Checkouts value is increasing significantly from year to year and the highest value was in 2016.
# ## 2- Total Renewals value is increasing significantly from year to year and the highest value was in 2016.
# ## 3- There is a clear relationship between 'Total Checkouts' and 'Total Renewals' as the more 'Total Checkouts' in return, the higher the total renewals.

# # - Test Value in new_sf_Dataset['Circulation Active Year']

# In[ ]:


#lis = new_sf_Dataset['Circulation Active Year']
#for i in lis:
#    print(i)
#new_sf_Dataset['Circulation Active Year'] = new_sf_Dataset["Circulation Active Year"].astype(str).astype(int)
#new_sf_Dataset['Year Patron Registered']


# In[ ]:


#new_sf_Dataset.columns
new_sf_Dataset


# # *Note_7
# ## - To confirm the above, there is a direct relationship between 'Total Checkouts' and 'Total Renewals'.

# In[ ]:


#sns.lmplot(x="Total Checkouts", y="Total Renewals", hue="Outside of County", data=new_sf_Dataset)
sns.regplot(x=new_sf_Dataset['Total Checkouts'], y=new_sf_Dataset['Total Renewals'])


# In[ ]:


new_sf_Dataset


# # *Note_8

# In[ ]:


plt.figure(figsize=(16, 9))

sns.barplot(x=new_sf_Dataset['Circulation Active Month'], y=new_sf_Dataset['Total Checkouts'])


# # *Note_8
# ## July is the highest value in 'Total Checkouts' and September is the lowest month.

# # *Note_9

# In[ ]:


plt.figure(figsize=(16, 9))

sns.barplot(x=new_sf_Dataset['Circulation Active Month'], y=new_sf_Dataset['Total Renewals'])


# # *Note_9
# ## July is the highest value in 'Total Renewals' and October is the lowest month.

# # *Note_10
# ## Rename index from None to id

# In[ ]:


##new_sf_Dataset.index.name= 'id'
new_sf_Dataset.columns


# In[ ]:


#new_sf_Dataset['Circulation Active Year'] = new_sf_Dataset['Circulation Active Year'].astype(str).astype(int)
#sns.lmplot(x='Circulation Active Year', y="Total Renewals", hue="Outside of County",data=new_sf_Dataset)
test_df = new_sf_Dataset.sample(2000)
sns.swarmplot(x='Outside of County',y='Total Checkouts',data=test_df)


# In[ ]:


test_df = new_sf_Dataset.sample(2000)
sns.swarmplot(x='Outside of County',y='Total Renewals',data=test_df)


# # *Note_10
# ## Part 1 - The highest values of Total Checkouts are for people within the country.
# ## Part 2 - The highest values of Total Renewals are for people within the country.
# 

# In[ ]:


new_sf_Dataset


# # *Note_11
# 

# In[ ]:


df = new_sf_Dataset[['Patron Type Definition','Total Checkouts','Age Range']]

heatmap_Data = pd.pivot_table(df, values='Total Checkouts', index=['Patron Type Definition'],columns=['Age Range'])
plt.figure(figsize=(16,9))
sns.heatmap(data=heatmap_Data, annot=True)


# # *Note_11
# ## The highest value of Total Checkouts is in the age period 75 years and over when patron Type Definition = RETIRED STAFF.

# In[ ]:


new_sf_Dataset


# # *Note_12

# In[ ]:


df = new_sf_Dataset[['Patron Type Definition','Total Checkouts','Circulation Active Month']]

heatmap_Data = pd.pivot_table(df, values='Total Checkouts', index=['Patron Type Definition'],columns=['Circulation Active Month'])
plt.figure(figsize=(16,9))
sns.heatmap(data=heatmap_Data, annot=True)


# # *Note_12
# ## THE BEST VALUE FOR 'TOTAL CHECKOUTS' WAS IN July WHEN PATRON TYPE DEFINITION = STAFF.

# In[ ]:


#Outside of County
#sns.lmplot(x="Age Range", y="Provided Email Address", hue="Outside of County", data=new_sf_Dataset)
#sns.swarmplot(x=new_sf_Dataset['Outside of County'],y=new_sf_Dataset['Provided Email Address'])


# In[ ]:


#df = new_sf_Dataset[['Year Patron Registered','Total Checkouts','Total Renewals']]

#df
#heatmap_Data = pd.pivot_table(df, values='Year Patron Registered', index=['Total Checkouts'], 
                             # columns='Total Renewals')

#plt.figure(figsize=(16,9))

#Heatmap showing average arrival delay for each airline by month
#sns.heatmap(data=heatmap_Data, annot=True)

