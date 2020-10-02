#!/usr/bin/env python
# coding: utf-8

# **QUICK PEEK**

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


data = pd.read_csv('../input/all.csv')
#data.head()
#data.shape
#data.columns
#np.unique(data.State)
#data.isnull().sum() -> todo: deal with these
#data.groupby('State').mean()


# In[ ]:


#how many numeric columns are there? 47?
num_cols = data._get_numeric_data().columns
#print(num_cols.shape)

#what are the categorical columns??
cat_cols = list(set(data.columns)-set(num_cols))
#data[num_cols].isnull().sum()


# Please note that the following plots are for mean value over districts in each state.

# In[ ]:


# let's plot the average data statewise
x = range(len(np.unique(data['State'])))
'''a = data.groupby('State').mean()
a = a.drop('Unnamed: 0',axis=1)
num_cols = num_cols.drop('Unnamed: 0')
for cols in num_cols:
    plt.figure()
    plt.bar(x,a[cols],alpha=0.5)
    plt.xticks(x,a.index,rotation = 90)
    plt.ylabel('mean value per state')
    plt.title('%s'%cols)
    plt.show()'''
#uncomment for the quick peek plots


# **Insights to draw/ Questions to ask from this initial clumsiness:**
# 
#  1. The graph for  mean persons per district per state is highest for WB. This suggests WB has fewer districts than UP. (UP is the most populous state.)
#  2. Household size for Indians seem to have less variance. Most households vary from sizes 4-6
#  3. Sex ratio in years 0-6 is well balanced when compared to the overall sex ratio. This suggests younger Indians have started to love boys and girls equally. Laudable!
#  4. Literacy rates: Female literacy rates are low in some states. Kerala stands tall, all states should look up to Kerala! (PS: I'm not from Kerala, so not a biased model :P, it's the numbers speaking!)
#  5. For rest of the fields, I intend to look at percentages for better insight.
#  6. State wise geographical area can be added to the dataframe, to draw insights on correlations between the size and most of the fields above. for e.g facilities.

# In[ ]:


cols_of_interest = ['State','Persons','Males', 'Females','Persons..literate',
       'Males..Literate', 'Females..Literate','Below.Primary', 'Primary', 'Middle',
       'Matric.Higher.Secondary.Diploma', 'Graduate.and.Above', 'X0...4.years',
       'X5...14.years', 'X15...59.years', 'X60.years.and.above..Incl..A.N.S..',
       'Total.workers','Main.workers', 'Marginal.workers', 'Non.workers',
       'Drinking.water.facilities', 'Safe.Drinking.water',
       'Electricity..Power.Supply.', 'Electricity..domestic.',
       'Electricity..Agriculture.', 'Primary.school', 'Middle.schools',
       'Secondary.Sr.Secondary.schools', 'College', 'Medical.facility',
       'Primary.Health.Centre', 'Primary.Health.Sub.Centre',
       'Post..telegraph.and.telephone.facility', 'Bus.services',
       'Paved.approach.road', 'Mud.approach.road', 'Permanent.House',
       'Semi.permanent.House', 'Temporary.House']

state_wise_data = data[cols_of_interest].groupby('State').sum()
#state_wise_data


# area in sqkm
# 8249
# 275045
# 83743
# 78438
# 94163
# 135192
# 114
# 111
# 491
# 1483
# 3702
# 196244
# 55673
# 44212
# 222236
# 79716
# 191791
# 38852
# 30
# 308252
# 307713
# 22327
# 22429
# 21081
# 16579
# 155707
# 490
# 50362
# 342239
# 7096
# 130060
# 10486
# 240928
# 53483
# 88752
# 

# In[ ]:


state_wise_data['Size'] = [8249,275045,83743,78438,94163,135192,114,111,491,1483,
                           3702,196244,55673,44212,222236,79716,191791,38852,30,
                           308252,307713,22327,22429,21081,16579,155707,490,50362,
                           342239,7096,130060,10486,240928,53483,88752]


# In[ ]:


'''plt.scatter(state_wise_data['Size'],state_wise_data['Drinking.water.facilities'],alpha=0.5)
plt.xlabel('Size')
plt.ylabel('Drinking water facilities')
plt.xticks(rotation=90)'''


# In[ ]:


'''plt.scatter(state_wise_data['Persons'],state_wise_data['Drinking.water.facilities'],alpha=0.5)
plt.xlabel('persons')
plt.ylabel('Drinking water facilities')
plt.xticks(rotation=90)'''


# In[ ]:


'''
#IS there relation between population and area of states?
plt.scatter(state_wise_data['Persons'],state_wise_data['Size'],alpha=0.5)
plt.xlabel('Persons')
plt.ylabel('Area')
plt.xticks(rotation=90)'''


# Let's look at the correlations!

# In[ ]:


import seaborn as sns
corrmat = state_wise_data.corr()
sns.heatmap(corrmat, square=True)


# Moderately high correlation between persons/area and facilities, where stronger correlation was expected

# Fields for relative study:
# males to females ratio;
# percentage literacy;
# percentage workers;

# In[ ]:


state_wise_data.columns


# In[ ]:


lit_ratio = np.divide(state_wise_data['Persons..literate'],state_wise_data['Persons'])
emp_ratio = np.divide(state_wise_data['Total.workers'],state_wise_data['Persons'])
plt.scatter(lit_ratio,emp_ratio)
plt.xlabel('literacy ratio')
plt.ylabel('employment ratio')
for i,j,k in zip(range(35),lit_ratio,emp_ratio):
    plt.annotate(i,xy = (j,k))


# Dadar and Nagar Haveli has a greater employment ratio than Kerala, though its literacy ratio is considerably less :0

# In[ ]:


for i,j in zip(range(35),state_wise_data.index):
    print(i,j)
    


# In[ ]:




