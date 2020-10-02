#!/usr/bin/env python
# coding: utf-8

# In[2]:


#importing the required libraries
import seaborn as sns # for plotting
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt#for plotting
import os #file handling in os
print(os.listdir("../input")) #list the current data location


# In[12]:


#loading dataset using pandas
data = pd.read_csv('../input/RegionalInterestByConditionOverTime.csv')


# In[23]:


#looking for random 5 data samoles
data.sample(5)


# > **Report**
# 
# The above dataset is about the google search made by the various people of different districts in US.
# It contains search from 2004 to 2017. It is a 210 *  128 dimension dataset. Here 210 = no. of rows in the dataset, 128 = no. of columns/attributes in the dataset.
# It is a multivariate analytical dataset. The attributes columns consists of various diseases like cancer, cardiovascular , stroke, depression, rehab etc search made by the people from 2004 to 2017.
# It also contains the geoCode values of the various districts of the US country.

# In[5]:


#checking for the size of the dataset
data.shape


# > **Looking for the missing Values**
# 
# Firstly we check for the missing values in  the dataset.
# If there are any missing values , we need to proceed to the data preprocessing steps.
# 

# In[6]:


#check missing values
data.columns[data.isnull().any()]


# Since it contains no missing values so we proceed further for Analysis of the dataset.

# > **Separate numeric data for visualization of the searches in the yearly-basis**
# 
# We separate the numeric data of the dataset and rename as data1 so that the increment/decrement of searches made by the people  can be found on a yealy basis.

# In[15]:


#separate variables into new data frames
data1 = data.select_dtypes(include=[np.number])


# In[20]:


sns.jointplot('2004+cancer', '2017+cancer', data=data1)


# In[22]:


sns.jointplot('2016+cancer', '2017+cancer', data=data1)


# ** As you can see, the second of these distributions is significantly closer to linear than the first one is! **

# > **Plotting all the diseases in a year wise manner** 

# In[16]:


#2004-2017
#cancer cardiovascular stroke depression rehab vaccine diarrhea obesity diabetes
yearWiseMean = {}
for col in data1.columns:
    if '+' in col:
        year = col.split('+')[0]
        disease = col.split('+')[-1]
        if not disease in yearWiseMean:
            yearWiseMean[disease] = {}
        if not year in yearWiseMean[disease]:
            yearWiseMean[disease][year] = np.mean(list(data1[col]))

plt.figure(figsize=(18, 6))
ax = plt.subplot(111)
plt.title("Year wise google medical search", fontsize=20)

ax.set_xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13])
ax.set_xticklabels(list(yearWiseMean['cancer'].keys()))
lh = {}
for disease in yearWiseMean:
    lh[disease] = plt.plot(yearWiseMean[disease].values())
plt.legend(lh, loc='best')


# >** We can make some of the inferences from the graph above , they are described at the end.**
# 
# Again we need to see the various searches made by the people of US in a district-wise by plotting in a heatmap.
# In this heatmap we can also see the highest search made various district of US.

# In[17]:


healthSearchData=data


# In[18]:


statesData = pd.DataFrame(healthSearchData.iloc[:,0])
#healthSearchData = healthSearchData.drop(['dma'],axis=1)

meanDict = {}
yearList = []
illnessList = []
for col in healthSearchData.columns:
    if '+' in col:
        yearList.append(col.split('+')[0])
        illnessList.append(col.split('+')[-1])
        
for index, row in healthSearchData.iterrows():
    for illness in illnessList:
        searchCountList = []
        for year in yearList:
            searchCountList.append(row[year+ '+' +illness])
        if not illness in meanDict:
            meanDict[illness] = []
        meanDict[illness].append(np.mean(searchCountList))
yearWiseMeanDf = pd.DataFrame.from_dict(meanDict, orient='columns', dtype=None)
heatMapData = statesData.join(yearWiseMeanDf)
heatMapData.set_index('dma', inplace=True, drop=True)

import seaborn as sns
plt.figure(figsize=(10, 25))
plt.title("State wise illness search", fontsize=16)
ax = plt.subplot(111)
ax.spines["top"].set_visible(False)    
ax.spines["bottom"].set_visible(False)    
ax.spines["right"].set_visible(False)    
ax.spines["left"].set_visible(False)
ax.get_xaxis().tick_bottom()    
ax.get_yaxis().tick_left()
ax = sns.heatmap(heatMapData)


# ****Conclusion****
# 
# Questions:
# 
# 
# 1) From the above analysis , determine which disease  has more and which has least search and also give if there any significance changes made?
# 
# 
# Ans: From the year wise google medical search ,we came to conclusion that Cancer is the most searched illness and cardiovascular is the least searched illness.
# Surprisingly, in 2017, diabetes is the highest searched illness.
# 
# 2)From the above analysis,  the curve of cancer disease yealy are shown in joint-plot .Point out few points from there.
# 
# Ans: a)From the first jointplot we plot the cancer illness i.e(2004+cancer,2017+cancer). Here we see that the search made by the people gradually increases in 2017 of cancer illness.
# b)From the second jointplot we plot the cancer illness i.e(2016+cancer,2017+cancer). Here we see that the search made by the people is almost same in both the year 2016 and 2017 of the cancer illness.
