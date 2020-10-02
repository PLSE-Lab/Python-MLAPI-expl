#!/usr/bin/env python
# coding: utf-8

# # This Notebook contains Data Exploration for the used cars Database

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import datetime
import calendar
import seaborn as sns


# In[ ]:


#Setting the datetime format while importing data
fmt = '%Y-%m-%d %H:%M:%S'
dateparse = lambda dates: pd.datetime.strptime(dates, fmt)
rawautodf = pd.read_csv('../input/autos.csv', sep=',',encoding='Latin1',parse_dates = ['dateCrawled','dateCreated','lastSeen'], date_parser = dateparse)


# In[ ]:


rawautodf.head()


# ## Checking the columns
# 
# ### 1. dateCrawled

# In[ ]:


print ("The original dtype of dateCrawled is " + str(rawautodf.dateCrawled.dtype)) # checked datatype
#Let's keep it


# ### 2. name
# I don't think I can extract any information from the name column, I am going to drop it

# In[ ]:


autodf = rawautodf.drop('name',axis = 1)
print( "Size of the dataset - " + str(len(autodf)))


# ### 3. seller

# In[ ]:


print(autodf.groupby('seller').size())
#We can live without the 3 gewerblich. Let's analyze the data with only the private sellers
autodf = autodf[autodf['seller'] != 'gewerblich']
autodf = autodf.drop('seller',axis = 1)
print( "\nSize of the dataset - " + str(len(autodf)))


# ### 4. offerType

# In[ ]:


print(autodf.groupby('offerType').size())
#Same here. Let's drop the 12
autodf = autodf[autodf['offerType'] != 'Gesuch']
autodf = autodf.drop('offerType',axis = 1)
print( "\nSize of the dataset - " + str(len(autodf)))


# ### 5. Price

# In[ ]:


#plt.plot()
#autodf.price.plot(kind='hist',bins=100)
#plt.show()
sns.boxplot(autodf.price)


# This looks wierd. Let's see

# In[ ]:


# are there any cars with 0 value?
print ("How low priced priced cars are there?  " + str(len(autodf[autodf.price < 100 ])))
print ("\nHow many highly price-listed cars are there?  " + str(len(autodf[autodf.price > 150000 ])))
#print autodf[(autodf.price < 0) | (autodf.price > 150000)]
print ("\nWell, that's ridiculous. Let's get rid of them.")
autodf = autodf[autodf.price > 100]
autodf = autodf[autodf.price < 50000]
print( "\nSize of the dataset - " + str(len(autodf)))


# In[ ]:


#plt.plot()
#autodf.price.plot(kind='hist',bins=100)
#plt.legend()
#plt.show()
sns.boxplot(autodf.price)


# In[ ]:


sns.distplot(autodf.price)


# ### 6. abtest
# 
# The column 'abtest' has been hard to interpret. After consulting the original database uploader, 
# we concluded to this interpretation:
# It contains an ebay-intern variable not of interest. 
# You can monitor the conversation and argumentation in the discussion section. - Thanks 

# In[ ]:


autodf = autodf.drop('abtest',1)


# In[ ]:


autodf.info()


# ### 7.vehicleType

# In[ ]:


p = sns.factorplot('vehicleType',data=autodf,kind='count')
p.set_xticklabels(rotation=30) #letitbe


# ### 8.yearOfRegistration

# In[ ]:


print( "\nSize of the dataset - " + str(len(autodf)))
# I only want to consier for 1980 to 2017
autodf = autodf[(autodf.yearOfRegistration >= 1990) & (autodf.yearOfRegistration < 2017)]
print( "\nSize of the dataset - " + str(len(autodf)))


# In[ ]:


p = sns.factorplot('yearOfRegistration',data=autodf,kind='count')
p.set_xticklabels(rotation=90)


# ### 9.gearbox

# In[ ]:


autodf.gearbox = autodf.gearbox.astype('category')
plt.figure(1)
plt.plot()
autodf.gearbox.astype('category').value_counts().plot(kind='bar')
plt.show()

sns.factorplot('gearbox',data=autodf,kind='count',hue='yearOfRegistration')


# #### Correlation in gearbox and year

# In[ ]:


group = autodf.groupby('yearOfRegistration')
temp_df = group.gearbox.value_counts()
#temp_df.head(1)
plt.plot()
temp_df.plot(kind='bar')
plt.show()


# ### 10. powerPS
# 
# Here I have seen that most of the data is between 50 to 300.

# In[ ]:


print ((str((float(len(autodf.powerPS[autodf.powerPS > 300])) / len(autodf.powerPS)) * 100) + " %"))
print (len(autodf.powerPS[autodf.powerPS > 300]))
#print len(autodf.powerPS)
print ("Number of zeros = " + str(len(autodf.powerPS[autodf.powerPS == 0])))

print( "\nSize of the dataset - " + str(len(autodf)))
autodf = autodf[(autodf.powerPS <= 300) & (autodf.powerPS > 50)]
print( "\nSize of the dataset - " + str(len(autodf)))


# In[ ]:


#autodf.powerPS.hist(bins=100)
sns.distplot(autodf.powerPS)


# In[ ]:


sns.boxplot(autodf.powerPS)


# ### 11. model

# In[ ]:


autodf.model = autodf.model.astype('category')


# In[ ]:


sns.factorplot('model',data=autodf,kind='count',size=10)
#toomuch data. Let's keep it anyway. We can try to implement machine learning algorithms (I'm a noob)


# ### 12.kilometer

# In[ ]:


print (autodf.kilometer.count())
print (max(autodf.kilometer))
print (min(autodf.kilometer))
sns.boxplot(autodf.kilometer)


# ### 13.monthOfRegistration

# In[ ]:


autodf['monthOfRegistration'] = autodf.monthOfRegistration.astype('category')


# In[ ]:


sns.factorplot('monthOfRegistration',data=autodf,kind='count')


# In[ ]:


#sns.boxplot(autodf.monthOfRegistration)


# ### 14.fuelType

# In[ ]:


sns.factorplot('fuelType',data=autodf,kind='count')


# ### 15. brand

# In[ ]:


p = sns.factorplot('brand',kind='count',data=autodf,size=10)
p.set_xticklabels(rotation=90)


# ### 16. notRepairedDamage

# In[ ]:


x = pd.DataFrame(autodf.notRepairedDamage.value_counts())
print ("Percentage of cars not repaired = " + str(round( x.ix['ja'] * 100 /x.ix['nein'],2)) + " %")


# In[ ]:


sns.factorplot('notRepairedDamage',data=autodf,kind='count')


# ### 17. dateCreated

# In[ ]:


#


# ### 19. postalCode

# In[ ]:


#autodf.postalCode.value_counts().plot(kind='bar)


# ### 20. lastSeen

# In[ ]:


#Letitbe


# A dataframe for correlations

# In[ ]:


#


# In[ ]:


#

