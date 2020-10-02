#!/usr/bin/env python
# coding: utf-8

# In[2]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[3]:


# get current working directory
os.getcwd()


# In[4]:


# get approximate file sizes 
ten6 = 1000000
dict_filesize = {}

dict_filesize["Resources"] = round(os.path.getsize('../input/Resources.csv')/ten6, 2)
dict_filesize["Schools"] = round(os.path.getsize('../input/Schools.csv')/ten6, 2)
dict_filesize["Donors"] = round(os.path.getsize('../input/Donors.csv')/ten6, 2)
dict_filesize["Donations"] = round(os.path.getsize('../input/Donations.csv')/ten6, 2)
dict_filesize["Teachers"] = round(os.path.getsize('../input/Teachers.csv')/ten6, 2)
dict_filesize["Projects"] = round(os.path.getsize('../input/Projects.csv')/ten6, 2)

# display dict with filesize
dict_filesize

# Largest file is Projects.csv, which is around 2GB 


# In[5]:


# load files using pandas

# schools = pd.read_csv('../input/Donors.csv')

# received the following error while running the above code
# /opt/conda/lib/python3.6/site-packages/IPython/core/interactiveshell.py:2698: DtypeWarning: Columns (4) have mixed types. 
# Specify dtype option on import or set low_memory=False. interactivity=interactivity, compiler=compiler, result=result)

schools = pd.read_csv('../input/Schools.csv', low_memory=False, skiprows=[59987])
teachers = pd.read_csv('../input/Teachers.csv')
donors = pd.read_csv('../input/Donors.csv', low_memory=False)


# In[6]:


schools.head(5)


# In[7]:


schools.tail(5)


# In[8]:


print(schools.loc[[59987]])
# print(schools.loc[[59988]])

# one additional column is present in row no. 59987
# this can be fixed or we can skip and proceed for time being


# In[9]:


teachers.head(5)


# In[10]:


teachers.tail(5)


# In[11]:


donors.head(5)


# In[12]:


donors.tail(5)


# In[13]:


# reading the remaining csv's

donations = pd.read_csv('../input/Donations.csv')


# In[14]:


donations.head(5)


# In[15]:


donations.tail(5)


# In[16]:


# resources = pd.read_csv('../input/Resources.csv', skiprows=[1171,3431,5228,6492,7529,8885,11086,11530], warn_bad_lines=False, error_bad_lines=False)
resources = pd.read_csv('../input/Resources.csv', warn_bad_lines=False, error_bad_lines=False)

# [PENDING] need to handle these skipped rows 


# In[17]:


resources.head(5)


# In[18]:


resources.tail(5)


# In[19]:


projects = pd.read_csv('../input/Projects.csv', error_bad_lines=False, warn_bad_lines=False)

# [PENDING] need to handle these skipped rows 


# In[20]:


projects.head(5)


# In[21]:


projects.tail(5)


# In[22]:


# saving the metadata of the files
# count the number of rows, columns in schools


# In[23]:


schools_rows = schools.shape[0]
schools_cols = schools.shape[1]
teachers_rows = teachers.shape[0]
teachers_cols = teachers.shape[1]
donors_rows = donors.shape[0]
donors_cols = donors.shape[1]
donations_rows = donations.shape[0]
donations_cols = donations.shape[1]
resources_rows = resources.shape[0]
resources_cols = resources.shape[1]
projects_rows = projects.shape[0]
projects_cols = projects.shape[1]

dict_schools = {'name': 'Schools', 'rows_count': schools_rows, 'cols_count': schools_cols, 'cols': list(schools), 'size': dict_filesize['Schools']}
dict_teachers = {'name': 'Teachers', 'rows_count': teachers_rows, 'cols_count': teachers_cols, 'cols': list(teachers), 'size': dict_filesize['Teachers']}
dict_donors = {'name': 'Donors', 'rows_count': donors_rows, 'cols_count': donors_cols, 'cols': list(donors), 'size': dict_filesize['Donors']}
dict_donations = {'name': 'Donations', 'rows_count': donations_rows, 'cols_count': donations_cols, 'cols': list(donations), 'size': dict_filesize['Donations']}
dict_resources = {'name': 'Resources', 'rows_count': resources_rows, 'cols_count': resources_cols, 'cols': list(resources), 'size': dict_filesize['Resources']}
dict_projects = {'name': 'Projects', 'rows_count': projects_rows, 'cols_count': projects_cols, 'cols': list(projects), 'size': dict_filesize['Projects']}


# In[24]:


# dictionary of schools details
dict_metadata = {'Schools': dict_schools, 'Teachers': dict_teachers, 'Donors': dict_donors, 'Donations': dict_donations, 'Resources': dict_resources, 'Projects': dict_projects}


# In[25]:


# display metadata
dict_metadata


# In[26]:


# rows, column counts for each file
print("Type \t   Columns \t Rows")
for k,v in dict_metadata.items():
    print(k, "\t", dict_metadata[k]['cols_count'], "\t", dict_metadata[k]['rows_count'])


# In[27]:


# Relation between Donor City/State and School/Project City/State needs to be found
# Donor ID, Donation ID and Project ID in Donations
# Donor City, Donor State from Donors based on Donor ID in Donations
# School ID from Schools based on Project ID in Projects
# School City, School State based on School ID from Schools


# In[28]:


# merging donations and donors data frames


# In[29]:


donations_donors = donations.merge(donors, on='Donor ID', how='inner')


# In[30]:


donations_donors.head(5)


# In[31]:


# merging projects and schools


# In[32]:


projects_schools = projects.merge(schools, on='School ID', how='inner')


# In[33]:


projects_schools.head(5)


# In[34]:


total = projects_schools.merge(donations_donors, on='Project ID', how='inner')


# In[35]:


total.head(5)


# In[36]:


# adding a new column 'Same City' to find whether the donor city and school city are the same
# to find if there's a relation between donor city and school city


# In[37]:


# if same city then value is set to 1
total.loc[total['School City'] == total['Donor City'], 'Same City'] = 1


# In[38]:


# if not the same city then value is set to 0
total.loc[total['School City'] != total['Donor City'], 'Same City'] = 0


# In[39]:


total.head(5)


# In[40]:


total.groupby('Same City').size()


# In[41]:


# approximately 25% of the donors are from the same city as the school


# In[42]:


# adding a new column 'Same State' to find whether the donor state and school state are the same
# to find if there's a relation between donor state and school state


# In[43]:


# if same state then value is set to 1
total.loc[total['School State'] == total['Donor State'], 'Same State'] = 1


# In[44]:


# if not same state then value is set to 0
total.loc[total['School State'] != total['Donor State'], 'Same State'] = 0


# In[45]:


total.tail(5)


# In[46]:


total.groupby('Same State').size()


# In[47]:


# chances of donors donating are more than double if donor and school belongs to the same state
# approximately 65% of the donations are from donors who belongs to the same state


# In[48]:


# [WORKING] relation between project cost and donation amount


# In[49]:


type(total['Project Cost'])


# In[50]:


total.dtypes


# In[51]:


# donation amount is of type float
# whereas project cost is an object
# as it's in the form $45
# we can create a new column for project cost
# without the dollar symbol


# In[52]:


# total.drop('column name', axis=1, inplace=True)


# In[53]:


total['Project Cost'].unique()


# In[54]:


total['ProjCost'] = total['Project Cost'].apply(lambda x: float(str(x).replace('$','').replace(',','')))


# In[55]:


total['ProjCost'].unique()


# In[56]:


type(total['ProjCost'][5])


# In[57]:


len(total['ProjCost'])


# In[58]:


donations.dtypes


# In[59]:


total['ProjCost'][5]


# In[60]:


total['ProjCost'][445]


# In[61]:


total.tail()


# In[62]:


total['Donation Amount'].max()


# In[63]:


total['ProjCost'].max()


# In[64]:


import matplotlib.pyplot as plt


# In[65]:


plt.scatter(total['ProjCost'], total['Donation Amount'])
plt.show() # Depending on whether you use IPython or interactive mode, etc.


# In[66]:


# p1 - sum of donations by project id


# In[68]:


p1 = total.groupby('Project ID')['Donation Amount'].sum()


# In[73]:


p2 = total.groupby('Project ID', as_index=False)['Donation Amount'].sum()


# In[69]:


p1.count()


# In[70]:


# converting series to frame
p1f = p1.to_frame()


# In[71]:


# not sure why Project ID and Donation Amount are on separate rows
p1f.head()


# In[72]:


p1f.count()


# In[74]:


p2


# In[100]:


p3 = total.groupby(['Project ID','ProjCost'], as_index=False)['Donation Amount'].sum().sort_values(by='ProjCost',ascending=False)


# In[101]:


p3.head()


# In[105]:


p3.head().plot(x='Donation Amount', y='ProjCost', style='o')


# In[110]:


plt.rcParams['agg.path.chunksize'] = 100000


# In[111]:


p3.plot()


# In[112]:


p3.head()


# In[113]:


import seaborn as sns


# In[115]:


sns.jointplot(x='ProjCost',y='Donation Amount',data=p3)


# In[116]:


plt.title("Project Cost vs Donation Amount - by Project", loc='center')


# In[117]:


p3.plot(x='Project ID', y='ProjCost' ,figsize=(12,8), grid=True, label="Project Cost", color="red") 
p3.plot(x='Project ID', y='Donation Amount' ,figsize=(12,8), grid=True, label="Donation Amount", color="blue")


# In[131]:


p3[:500].set_index('Project ID').plot(figsize=(20,10), grid=True, alpha=0.45) 


# In[136]:


list(p3)


# In[137]:


p3.describe


# In[138]:


p3['Donation Percentage'] = round((p3['Donation Amount']/p3['ProjCost'])*100,2)


# In[139]:


p3


# In[140]:


sns.jointplot(x='ProjCost',y='Donation Percentage',data=p3)


# In[ ]:




