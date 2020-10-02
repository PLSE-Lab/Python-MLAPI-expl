#!/usr/bin/env python
# coding: utf-8

# * [1.DataOverview](#1.DataOverview)
# * [2.Visualization](#2.Visualization)
#     * [2.1&nbsp;Visualization&nbsp;for&nbsp;species](#21)
#     * [2.2&nbsp;Visualization&nbsp;for&nbsp;street](#22)
#         * [2.2.1&nbsp;Visualization&nbsp;for&nbsp;East_14th_st](#221)
# * [3.Decision_tree](#3.Decision_tree)

# In[ ]:


import numpy as np 
import pandas as pd 
import geopandas as gpd
import folium 
import matplotlib.pyplot as plt
import seaborn as sns
import os 
from folium import plugins
import pylab as pl


# In[ ]:


os.listdir("../input/oakland-street-trees/")


# # 1.DataOverview

# This data size is 38613,and the variable include 'OBJECTID','WELLWIDTH','WELLLENGTH',and so on.
# 
# There are 5 missing data in this data.We drop out the 5 data at first.
# 
# According to the different visualization,I will do other data pre-processing.

# In[ ]:


data=pd.read_csv("../input/oakland-street-trees/oakland-street-trees.csv")
data.head()


# In[ ]:


len(data)


# In[ ]:


data.isnull().sum()


# In[ ]:


data=data.dropna(subset=['SPECIES'])


# # 2.Visualization

# <h3 id="21">2.1&nbsp;Visualization&nbsp;for&nbsp;species</h3>

# There are 196 species of tree in this data. If I visualize for all data,it will be complicated.
# 
# So,most of my visualizations are for Top 10.

# In[ ]:


len(data.SPECIES.unique())


# In[ ]:


fig,ax=plt.subplots(1,2,figsize=(15,8))
clr = ("blue", "forestgreen", "gold", "red", "purple",'cadetblue','hotpink','orange','darksalmon','brown')
data.SPECIES.value_counts().sort_values(ascending=False)[:10].sort_values().plot(kind='barh',color=clr,ax=ax[0])
ax[0].set_title("Top 10 species by counts",size=20)
ax[0].set_xlabel('counts',size=18)


count=data['SPECIES'].value_counts()
groups=list(data['SPECIES'].value_counts().index)[:10]
counts=list(count[:10])
counts.append(count.agg(sum)-count[:10].agg('sum'))
groups.append('Other')
type_dict=pd.DataFrame({"group":groups,"counts":counts})
clr1=('brown','darksalmon','orange','hotpink','cadetblue','purple','red','gold','forestgreen','blue','plum')
qx = type_dict.plot(kind='pie', y='counts', labels=groups,colors=clr1,autopct='%1.1f%%', pctdistance=0.9, radius=1.2,ax=ax[1])
plt.legend(loc=0, bbox_to_anchor=(1.15,0.4)) 
plt.subplots_adjust(wspace =0.5, hspace =0)
plt.ylabel('')


# In[ ]:


Top10_species_counts=data[data['SPECIES'].isin(list(data.SPECIES.value_counts()[:10].index[:10]))]
Top10_species_counts.groupby(['SPECIES','LOWWELL'])['OBJECTID'].agg('count').unstack('LOWWELL').plot.barh(figsize=(15,8))
plt.ylabel('')
plt.xlabel('counts',size=18)
plt.title('Lowwell for Top 10 species',size=20)
plt.tick_params(labelsize=16)


# In[ ]:


fig,ax=plt.subplots(figsize=(25,12))
sns.boxplot(x="SPECIES", y="WELLWIDTH", data=Top10_species_counts)
ax.set_title("Boxplot of WELLWIDTH for Top 10 species",size=20)


# In[ ]:


data['Location 1'][0]


# In[ ]:


data['Long']=data['Location 1'].apply(lambda x: x.split(',')[0].split(':')[1].strip().strip("'"))
data['Lat']=data['Location 1'].apply(lambda x: len(x.split(':')))
data=data.drop(index=data[data.Lat<8].index)
data['Lat']=data['Location 1'].apply(lambda x: x.split(',')[5].split(':')[1].strip('}').strip().strip("'"))
data=data.drop(index=data[data.Lat.isin(['{"address"'])].index)
data.Lat=data.Lat.astype(float)
data.Long=data.Long.astype(float)
data.head()


# <h3 id="22">2.2&nbsp;Visualization&nbsp;for&nbsp;street</h3>

# In[ ]:


data['street']=data.STNAME.apply(lambda x:x.split(',')[0].split(':')[1].split("'")[1])
data=data.drop(index=data[~data.street.isin(['{"address"'])].index)
data['street']=data.STNAME.apply(lambda x:x.split(',')[0].split(':')[2])
data=data.drop(index=data[data.street==' "None"'].index)
data.street.value_counts()


# In[ ]:


fig,ax=plt.subplots(figsize=(12,8))
clr = ("blue", "forestgreen", "gold", "red", "purple",'cadetblue','hotpink','orange','darksalmon','brown')
data.street.value_counts().sort_values(ascending=False)[:10].sort_values().plot(kind='barh',color=clr,ax=ax)
ax.set_title("Top 10 street by trees",size=20)
ax.set_xlabel('trees',size=18)


# In[ ]:


fig,ax=plt.subplots(figsize=(15,8))
Top10_street_counts=data[data['street'].isin(list(data.street.value_counts()[:10].index[:10]))]
sns.boxplot(x="street", y="WELLWIDTH", data=Top10_street_counts)
ax.set_title("Boxplot of WELLWIDTH for Top 10 street",size=20)


# In[ ]:


data[data.street==' "M L KING JR WY"'][data.WELLWIDTH>30]


# Remove the outlier

# In[ ]:


fig,ax=plt.subplots(figsize=(15,8))
data=data[data.WELLWIDTH<30]
Top10_street_counts=data[data['street'].isin(list(data.street.value_counts()[:10].index[:10]))]
sns.boxplot(x="street", y="WELLWIDTH", data=Top10_street_counts)
ax.set_title("Boxplot of WELLWIDTH for Top 10 street(Remove the outlier)",size=20)


# <h3 id="221">2.2.1&nbsp;Visualization&nbsp;for&nbsp;East_14th_st</h3>

# In[ ]:


East_14th=data[data.street==' "EAST 14TH ST"']
East_14th.head()


# In[ ]:


fig,ax=plt.subplots(1,2,figsize=(16,8))
East_14th.SPECIES.value_counts()[:10].sort_values().plot(kind='barh',color=clr,ax=ax[0])
ax[0].set_title("Top 10 Species by counts on East_14th_st",size=20)
ax[0].set_xlabel("counts",size=18)

East_14th.groupby(['WELLWIDTH'])['OBJECTID'].agg('count').plot.bar(ax=ax[1])
pl.xticks(rotation=360)
ax[1].set_title("Distribution of WELLWIDTH on East_14th_st",size=20)
ax[1].set_ylabel("counts",size=18)
ax[1].set_xlabel('WELLWIDTH',size=18)


# # 3.Decision_tree

# In[ ]:


data.head()


# Decision tree for Top 3 species by counts.Remove the data which WELLWIDTH is greater than 30.
# 
# Remove the data which WELLLENGTH is greater than 20.Remove the data which PAREAWIDTH is greater than 20
# 
# 

# In[ ]:


data_tree=pd.DataFrame({'WELLWIDTH':Top10_species_counts.WELLWIDTH,'WELLLENGTH':Top10_species_counts.WELLLENGTH,'PAREAWIDTH':Top10_species_counts.PAREAWIDTH,'SPECIES':Top10_species_counts.SPECIES})
data_tree=data_tree[data_tree.SPECIES.isin(['Platanus acerifolia','Liquidambar styraciflua','Pyrus calleryana cvs'])]
data_tree=data_tree[data_tree.WELLWIDTH<30]
fig,ax=plt.subplots(3,1,figsize=(25,20))
sns.boxplot(x="SPECIES", y="WELLWIDTH", data=data_tree,ax=ax[0])
ax[0].set_title("Boxplot of WELLWIDTH for data_tree",size=20)

data_tree=data_tree[data_tree.WELLLENGTH<20]
sns.boxplot(x="SPECIES", y="WELLLENGTH", data=data_tree,ax=ax[1])
ax[1].set_title("Boxplot of WELLLENGTH for data_tree",size=20)

data_tree=data_tree[data_tree.PAREAWIDTH<20]
sns.boxplot(x="SPECIES", y="PAREAWIDTH", data=data_tree,ax=ax[2])
ax[2].set_title("Boxplot of PAREAWIDTH for data_tree",size=20)


# In[ ]:


from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
data_tree['SPECIES_new'] = labelencoder.fit_transform(data_tree['SPECIES'])
data_tree.head()


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
x_train,x_test,y_train,y_test=train_test_split(data_tree[['WELLWIDTH','WELLLENGTH','PAREAWIDTH']],data_tree[['SPECIES_new']],test_size=0.2,random_state=0)
tree=DecisionTreeClassifier(criterion='entropy',max_depth=4,random_state=0)
tree=tree.fit(x_train,y_train)
predict=tree.predict(x_test)
y=y_test['SPECIES_new']
from sklearn import metrics
print("acc_train:",metrics.accuracy_score(y_train,tree.predict(x_train)))
print("acc_test:",metrics.accuracy_score(y,predict))

