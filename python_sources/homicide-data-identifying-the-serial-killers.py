#!/usr/bin/env python
# coding: utf-8

# An attempt to identify similar incidents in homicide dataset using 

# In[ ]:



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns #visualization
sns.set_style('whitegrid')


hdata = pd.read_csv('../input/database.csv')
np.unique(hdata['Crime Type'])


# We aren't interested in Manslaughter by Negligence. Unlikely to be serial killer activity.

# In[ ]:


hdata.head(n=1)


# In[ ]:



data=hdata[(hdata['Crime Type']!='Manslaughter by Negligence')]
np.unique(hdata['Relationship'])    


# To avoid suspicion, serial killers are more likely to go after people they aren't related to.Most murders committed by family members are likely to be over some disputes or financial/personel reasons.  I'm filtering out such records for this model.

# In[ ]:


hdata=hdata[(hdata['Relationship']=='Stranger')|(hdata['Relationship']=='Unknown')|(hdata['Relationship']=='Acquaintance')|(hdata['Relationship']=='Neighbor')]


# I've decided to focus on victim specific data for creating this model because most serial killers target victims based on some criteria like victim age, ethnicity etc. 
# 
# Serial killers probably also have a Modus Operandi. So I've retained the 'Weapon' column. 
# 
# Geographic proximity is another aspect. Though Serial killers don't really care about state boundaries, I thought this might give us more insights 

# In[ ]:


hdata.drop(['Agency Code','Crime Type','Record ID', 'Year', 'Month', 'Crime Solved', 'City','Agency Name','Agency Type', 'Perpetrator Sex', 'Perpetrator Age', 'Perpetrator Race','Perpetrator Ethnicity','Record Source', 'Perpetrator Count','Incident','Agency Name','Victim Count'], axis=1,inplace=True)

#columns we'll use for the model
hdata.columns


# In[ ]:


hdata = hdata[(hdata['Victim Race']!='Unknown')&(hdata['Victim Ethnicity']!='Unknown')&(hdata['Victim Sex']!='Unknown')]
hdata = hdata[(hdata['Weapon']!='Unknown')]              
np.unique(hdata['Victim Age'])


# In[ ]:


hdata = hdata[hdata['Victim Age']!=998]
sns.kdeplot(hdata['Victim Age'])


# Strange spike around 99 year olds. More likely to be a data entry error or practice and less likely there is a serial killer out there  targeting 99 year olds
# 

# In[ ]:


hdata = hdata[hdata['Victim Age']!=99]


# Converting the Victim Age column into category data to fit our cosine similarity model. Clearly there is a category cut-off around age 18, another around age 60. Other category boundaries aren't quite so obvious. So I've gone with following categories
# 

# In[ ]:


def f(x):
    if x > 60:
        x = 'S'
    elif x <18:
        x = 'C'
    elif (x>=18) & (x<40):
        x = 'A'
    else: x = 'M'
    return x

hdata['Victim Age'] = hdata['Victim Age'].apply(f)
hdata = pd.get_dummies(hdata)


# I'm using the Cosine similarity. But since all the vectors have the same magnitude in our model, I'm simply using the dot products
#  

# In[ ]:


#Due to Kaggle memory limitations, I'm experimenting with a much smaller dataset. This will be helpful when we try to create a heatmap. I often ran into memory issues at that step
hdata_head = hdata.head(n=500)
SimilarityMatrix = hdata_head.dot(hdata_head.transpose())

#We do not care about diagonal elements which simply calculates a records similarity with itself
np.fill_diagonal(SimilarityMatrix.values, 0)

#heatmap for Similarity Matrix
sns.heatmap(SimilarityMatrix,xticklabels=False,yticklabels=False)


# Above graph shows a heat map for the similarity Matrix we calculated earlier. Each cell shows the similarity rating between two records, as indexed by their respective row and column numbers. The diagonal elements show similarity of a record with itself. I've set it zero to avoid confustion.
# 
#  Clearly, there are clusters of records with higher similarity to each other. This model only uses 7 columns from the input data. With more Victim specific data, the model can be enhanced. 
# 
# We can now look at one of the records individually (As an example, Record with index 52 is shown in the graph below) to see how similar it is to other incidents. We are mainly interested in the fraction of records that have the highest similar rating of 7. 
# 

# In[ ]:


sns.countplot(SimilarityMatrix.loc[52,:])


# Disclaimer: This is just an experimental model and shouldn't be used as evidence for law enforcement.
