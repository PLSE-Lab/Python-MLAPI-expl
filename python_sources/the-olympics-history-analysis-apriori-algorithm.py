#!/usr/bin/env python
# coding: utf-8

# ## Using Aprioi to analyse the olympics history
# 

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from mlxtend.preprocessing import TransactionEncoder
print(os.listdir('../input/'))


# In[ ]:


data = pd.read_csv("../input/athlete_events.csv")
#data.info()
print(data.shape)
data.head(1)


# # Data preprocessing

# In[ ]:


print(data.shape)
print(data.columns)
print(data.isna().sum())


# ## The data has 9474 null values in Age , hence will be substituting with the mode

# In[ ]:


AgeMode = int(data['Age'].mode()[0])
print('Age Mode', AgeMode)
data['Age'] = data['Age'].fillna(AgeMode)
data['Age'].isna().sum()


# ## 22% of the Height and 23% of the Weight column is having NaN values
# ## The NaN values are substitued with mode

# In[ ]:


HeightMode = data['Height'].mode()[0] ## 0 -> the column wise mode 
print('Height Mode',HeightMode)
data['Height'] = data['Height'].fillna(HeightMode)
data['Height'].isna().sum()


# In[ ]:


WeightMode = data['Weight'].mode()[0]
print('Weight Mode',WeightMode)
data['Weight'] = data['Weight'].fillna(WeightMode)
data['Weight'].isna().sum()


# ## The Medal column has NaN values if no medal is won , we Substitute the NaN values with a string 'None'

# In[ ]:


data['Medal'] = data['Medal'].fillna('None')
data['Medal'].isna().sum()


# ## After Removing the NaN values

# In[ ]:


data.isna().sum()


# ## We want to find the frequent itemsets for the following columns
# ###    ['Sex','Age', 'Height','Weight','Team','Medal']

# In[ ]:



# TO group a range of values together
def convertDataToRange(data,col,span):
    column = []
    for index,rows in data.iterrows():
        i = data.loc[index,col]
        temp = int(i/span)*span
        val = '(',col,':',str(temp),'-',str(temp+span),')'
        
        column.append("".join(val))
        #print(i,temp,val)
    return column       


# In[ ]:


def makeSelectiveSportDataset(fromData,sport,colList):
    toData = fromData[fromData['Sport'] == sport]
    toData =toData[colList]
    #to.head(5)
    med = ['Gold','Silver','Bronze']
    toData = toData[toData.Medal.isin(med)]
    return(toData)


# In[ ]:


def convertDataForApriori(data):
    data['Height']  = convertDataToRange(data,'Height',10)

    data['Weight']  = convertDataToRange(data,'Weight',10)

    data['Age']  = convertDataToRange(data,'Age',10)
    return data


# In[ ]:


def applyApriori(dataset,support):
    rec = dataset.values.tolist()
    # Finding Frequent Item Sets

    te = TransactionEncoder()
    te_ary = te.fit(rec).transform(rec)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    
    # Applying Aprioi Algo
    from mlxtend.frequent_patterns import apriori
    
    freq_Itemsets =  apriori(df,min_support = support, use_colnames = True)
    freq_Itemsets['length'] = freq_Itemsets['itemsets'].apply(lambda x:len(x))
    return(freq_Itemsets[freq_Itemsets['length'] > 1])


# # Finding Frequent ItemSets on different Sports

# # Atheletics

# In[ ]:


Athletics = makeSelectiveSportDataset(data,'Athletics',['Sex','Age', 'Height','Weight','Team','Medal'])

Athletics= convertDataForApriori(Athletics)

freq_Itemsets_Athletics = applyApriori(Athletics,0.4)
print("Athletics Data")
print(freq_Itemsets_Athletics)


# ### Result:
#      1. Age group between 20-30 is common.
#      2. Height and Weight does not matter.
#      3. Men have a higher chance of winning

# # Archery

# In[ ]:


Archery = makeSelectiveSportDataset(data,'Archery',['Sex','Age', 'Height','Weight','Team','Medal'])
Archery= convertDataForApriori(Archery)
freq_Itemsets_Archery = applyApriori(Archery,0.4)
print("Archery Data")

print(freq_Itemsets_Archery)


# ### Result:
#      1. Age distribution is even.
#      2. Height and Weight ((Height:180-190), (Weight:70-80)) is found to be common .
# 

# # Basket Ball

# In[ ]:


BasketBall = makeSelectiveSportDataset(data,'Basketball',['Sex','Age', 'Height','Weight','Team','Medal'])

BasketBall= convertDataForApriori(BasketBall)

freq_Itemsets_BasketBall =  applyApriori(BasketBall,0.4)
print("Basket Ball Data")

freq_Itemsets_BasketBall


# ### Result:
#      1. Age group between 20-30 is common.
#      2. Height and Weight does not matter.
# 

# # Boxing

# In[ ]:


Boxing = makeSelectiveSportDataset(data,'Boxing',['Sex','Age', 'Height','Weight','Team','Medal'])
Boxing= convertDataForApriori(Boxing)

freq_Itemsets_Boxing = applyApriori(Boxing,0.4)
print("Boxing Data")
freq_Itemsets_Boxing


# ### Result:
#      1. Age group between 20-30 is common.
#      2. Height ranging between ((Height:180-190)) is found to be common .
# 

# # Cycling

# In[ ]:


Cycling = makeSelectiveSportDataset(data,'Cycling',['Sex','Age', 'Height','Weight','Team','Medal'])

Cycling= convertDataForApriori(Cycling)
Cycling.head()

freq_Itemsets_Cycling = applyApriori(Cycling,0.4)
print("Cycling Data")

print(freq_Itemsets_Cycling)


# ### Result:
#      1. Age group between 20-30 is common.
#      2. Age-Height-Weight ranging between (Age:20-30),(Height:180-190), (Weight:70-80)) is found to be doing well .
#     

# # Diving 

# In[ ]:


Diving = makeSelectiveSportDataset(data,'Diving',['Sex','Age', 'Height','Weight','Team','Medal'])
Diving= convertDataForApriori(Diving)
Diving.head()

freq_Itemsets_Diving = applyApriori(Diving,0.4)
print("Diving Data")

freq_Itemsets_Diving


# ### Result:
#      1. Age group between 20-30 is common.
#      2. Height and Weight does not matter.
# 

# # Football

# In[ ]:


Football = makeSelectiveSportDataset(data,'Football',['Sex','Age', 'Height','Weight','Team','Medal'])
Football= convertDataForApriori(Football)
Football.head()

freq_Itemsets_Football = applyApriori(Football,0.4)
print("Footbal Data")
print(freq_Itemsets_Football)


# ### Result:
#      1. Age group between 20-30 has a high support, hence greater chance of winning a medal.
#      2. Age-Height-Weight ranging between (Age:20-30),(Height:180-190), (Weight:70-80)) is found to be doing well .
#     

# # Gymnastics

# In[ ]:


Gymnastics = makeSelectiveSportDataset(data,'Gymnastics',['Sex','Age', 'Height','Weight','Team','Medal'])

Gymnastics= convertDataForApriori(Gymnastics)

freq_Itemsets_Gymnastics = applyApriori(Gymnastics,0.4)
print("Gymnastics Data")

print(freq_Itemsets_Gymnastics)


# ### Result:
#      1. Age group between 20-30 is common.
#      2. Height and Weight ((Height:180-190), (Weight:70-80)) is found to be common .
# 

# # Handball

# In[ ]:


Handball = makeSelectiveSportDataset(data,'Handball',['Sex','Age', 'Height','Weight','Team','Medal'])
Handball= convertDataForApriori(Handball)

freq_Itemsets_Handball = applyApriori(Handball,0.4)
print("HandBall Data")

print(freq_Itemsets_Handball)


# ### Result:
#      1. Age group between 20-30 is common.
#      2. Height and Weight does not matter.
# 

# # Judo

# In[ ]:


Judo = makeSelectiveSportDataset(data,'Judo',['Sex','Age', 'Height','Weight','Team','Medal'])
Judo= convertDataForApriori(Judo)

freq_Itemsets_Judo = applyApriori(Judo,0.4)
print("Judo Data")
print(freq_Itemsets_Judo)


# ### Result:
#      1. Age group between 20-30 is common and have won many bronze medals
#      2. Height and Weight does not matter.
# 

# # Speed Skating

# In[ ]:


SpeedSkt = makeSelectiveSportDataset(data,'Speed Skating',['Sex','Age', 'Height','Weight','Team','Medal'])
SpeedSkt= convertDataForApriori(SpeedSkt)
SpeedSkt.head()

freq_Itemsets_SpeedSkt = applyApriori(SpeedSkt,0.4)
print("Speed Skating Data")

print(freq_Itemsets_SpeedSkt)


# ### Result:
#      1. Age group between 20-30 is common.
#      2. Height ranging between ((Height:180-190)) is common	
# 

# # $$ $$
# 
# # Conclusion
# 
# ## I have applied the Apriori algorithm on a few selective sports  to  find out what are common traits (Winning DNA) among the medal winnners.
# ### The sports list:
#     1. Athletics
#     2. Archery
#     3. Basket ball
#     4. Boxing
#     5. Cycling
#     6. Diving
#     7. Football
#     8. Gymnastics
#     9. Handball
#     10. Judo
#     11. Speed Skating
#     
#    ### In general, Age ranging between (20-30),the height ranging between (180-190)cms and the weight ranging between (70-80)kg was found common among most of these sports. The respective results have been written below the code.

# In[ ]:





# In[ ]:


'''
med = ['Gold','Silver','Bronze']
medalC1 = data[data.Medal.isin(med)]
medalC = medalC1.groupby(['Team']).count()
medalC.reset_index(drop = False,inplace = True)
medalC = medalC[['Team','Medal']]

#medalC = medalC.reset_index(drop=False)
medalCount = medalC[['Team','Medal']]
medalCount.sort_values('Team',inplace = True)
medalCount.head()
'''

