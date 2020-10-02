#!/usr/bin/env python
# coding: utf-8

# * [1.DataOverview](#1.DataOverview)
# * [2.Visualization](#2.Visualization)
#     * [2.1 Visualization for correlation](#21)
#     * [2.2 Visualization for MSSubClass](#22)
#     * [2.3 Visualization for MSZoning](#23)
#     * [2.4 Visualization for Neighborhood](#24)
#     * [2.5 Visualization for OverallQual](#25)
#     * [2.6 Visualization for OverallCond](#26)
# * [3.Regression_tree](#3.Regression_tree)
#     

# In[ ]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import os
import seaborn as sns
import math


# In[ ]:


os.listdir("../input/house-prices-advanced-regression-techniques")


# # 1.DataOverview

# This data size is 1460,and the variable have 81 types,including ID,MSSubClass,and so on.
# 
# In this kernel,I focus on the MSSubClass,MSZonong,Neighborhood,OverallQual,and OverallCond.
#     

# In[ ]:


train=pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
test=pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
train


# From the bar chart,we can know taht the number of missing value in each variable.If you wanna use 
# 
# these variables to create Regression Model,you have to watch out them.

# In[ ]:


plt.figure(figsize=(8,8))
train.isnull().sum().sort_values(ascending=False)[:19].sort_values().plot.barh(color='plum')
plt.title('counts of missing value in the train data',size=20)
plt.xlabel('counts')


# # 2.Visualization

# <h3 id="21">2.1 Visualization for correlation</h3>

# In[ ]:


plt.figure(figsize=(15,15))
corr = train.corr()
sns.heatmap(corr, xticklabels=corr.columns,yticklabels=corr.columns)
plt.title("correlation plot",size=28)


# <h3 id="22">2.2 Visualization for MSSubClass</h3>

# In[ ]:


fig,ax=plt.subplots(1,2,figsize=(15,8))
clr = ("blue", "forestgreen", "gold", "red", "purple",'cadetblue','hotpink','orange','darksalmon','brown')
train.MSSubClass.value_counts().sort_values(ascending=False)[:10].sort_values().plot(kind='barh',color=clr,ax=ax[0])
ax[0].set_title("Top 10 MSSubClass by counts",size=20)
ax[0].set_xlabel('counts',size=18)


count=train.MSSubClass.value_counts()
groups=list(train.MSSubClass.value_counts().index)[:10]
counts=list(count[:10])
counts.append(count.agg(sum)-count[:10].agg('sum'))
groups.append('Other')
type_dict=pd.DataFrame({"group":groups,"counts":counts})
clr1=('brown','darksalmon','orange','hotpink','cadetblue','purple','red','gold','forestgreen','blue','plum')
qx = type_dict.plot(kind='pie', y='counts', labels=groups,colors=clr1,autopct='%1.1f%%', pctdistance=0.9, radius=1.2,ax=ax[1])
plt.legend(loc=0, bbox_to_anchor=(1.15,0.8)) 
plt.subplots_adjust(wspace =0.5, hspace =0)
plt.ylabel('')


# I remove the data which saleprice is greater than 400000 to balance the distribution of price for MSSubClass.

# In[ ]:


fig,ax=plt.subplots(2,1,figsize=(15,15))
sns.boxplot(x="MSSubClass", y="SalePrice", data=train,ax=ax[0])
ax[0].set_title("Boxplot of Price for MSSubClass",size=20)

train=train[train.SalePrice<=400000]
sns.boxplot(x="MSSubClass", y="SalePrice", data=train,ax=ax[1])
ax[1].set_title("Boxplot of Price for MSSubClass(price<=400000)",size=20)


# <h3 id="23">2.3 Visualization for MSZoning</h3>

# In[ ]:


fig,ax=plt.subplots(1,2,figsize=(15,8))
clr = ("blue", "forestgreen", "gold", "red", "purple",'cadetblue','hotpink','orange','darksalmon','brown')
train.MSZoning.value_counts().sort_values(ascending=False).plot(kind='bar',color=clr,ax=ax[0])
ax[0].set_title("bar chart for MSZoning",size=20)
ax[0].set_xlabel('counts',size=18)
ax[0].tick_params(axis='x',rotation=360)

sns.boxplot(x="MSZoning", y="SalePrice", data=train,ax=ax[1])
ax[1].set_title("Boxplot of Price for MSZoning",size=20)


# <h3 id="24">2.4 Visualization for Neighborhood</h3>

# In[ ]:


fig,ax=plt.subplots(1,2,figsize=(15,8))
clr = ("blue", "forestgreen", "gold", "red", "purple",'cadetblue','hotpink','orange','darksalmon','brown')
train.Neighborhood.value_counts().sort_values(ascending=False)[:10].sort_values().plot(kind='barh',color=clr,ax=ax[0])
ax[0].set_title("Top 10 Neighborhood by counts",size=20)
ax[0].set_xlabel('counts',size=18)


count=train.Neighborhood.value_counts()
groups=list(train.Neighborhood.value_counts().index)[:10]
counts=list(count[:10])
counts.append(count.agg(sum)-count[:10].agg('sum'))
groups.append('Other')
type_dict=pd.DataFrame({"group":groups,"counts":counts})
clr1=('brown','darksalmon','orange','hotpink','cadetblue','purple','red','gold','forestgreen','blue','plum')
qx = type_dict.plot(kind='pie', y='counts', labels=groups,colors=clr1,autopct='%1.1f%%', pctdistance=0.9, radius=1.2,ax=ax[1])
plt.legend(loc=0, bbox_to_anchor=(1.15,0.8)) 
plt.subplots_adjust(wspace =0.5, hspace =0)
plt.ylabel('')


# In[ ]:


fig,ax=plt.subplots(figsize=(25,15))
sns.boxplot(x="Neighborhood", y="SalePrice", data=train,ax=ax)
ax.set_title("Boxplot of Price for Neighborhood",size=20)


# <h3 id="25">2.5 Visualization for OverallQual</h3>

# In[ ]:


fig,ax=plt.subplots(1,2,figsize=(15,8))
clr = ("blue", "forestgreen", "gold", "red", "purple",'cadetblue','hotpink','orange','darksalmon','brown')
train.groupby(['OverallQual'])['Id'].agg('count').plot(kind='bar',color=clr,ax=ax[0])
ax[0].set_title("Top 10 OverallQual by counts",size=20)
ax[0].set_xlabel('counts',size=18)


count=train.groupby(['OverallQual'])['Id'].agg('count')
groups=list(train.groupby(['OverallQual'])['Id'].agg('count').index)
counts=list(count)
type_dict=pd.DataFrame({"group":groups,"counts":counts})
clr1=("blue", "forestgreen", "gold", "red", "purple",'cadetblue','hotpink','orange','darksalmon','brown')
qx = type_dict.plot(kind='pie', y='counts', labels=groups,colors=clr1,autopct='%1.1f%%', pctdistance=0.9, radius=1.2,ax=ax[1])
plt.legend(loc=0, bbox_to_anchor=(1.20,0.8)) 
plt.subplots_adjust(wspace =0.5, hspace =0)
plt.ylabel('')


# In[ ]:


fig,ax=plt.subplots(figsize=(25,15))
sns.boxplot(x="OverallQual", y="SalePrice", data=train,ax=ax)
ax.set_title("Boxplot of Price for OverallQual",size=20)


# <h3 id="26">2.6 Visualization for OverallCond</h3>

# In[ ]:


fig,ax=plt.subplots(1,2,figsize=(15,8))
clr = ("blue", "forestgreen", "gold", "red", "purple",'cadetblue','hotpink','orange','darksalmon','brown')
train.groupby(['OverallCond'])['Id'].agg('count').plot(kind='bar',color=clr,ax=ax[0])
ax[0].set_title("Top 10 OverallCond by counts",size=20)
ax[0].set_xlabel('counts',size=18)


count=train.groupby(['OverallCond'])['Id'].agg('count')
groups=list(train.groupby(['OverallCond'])['Id'].agg('count').index)
counts=list(count)
type_dict=pd.DataFrame({"group":groups,"counts":counts})
clr1=("blue", "forestgreen", "gold", "red", "purple",'cadetblue','hotpink','orange','darksalmon','brown')
qx = type_dict.plot(kind='pie', y='counts', labels=groups,colors=clr1,autopct='%1.1f%%', pctdistance=0.9, radius=1.2,ax=ax[1])
plt.legend(loc=0, bbox_to_anchor=(1.20,0.8)) 
plt.subplots_adjust(wspace =0.5, hspace =0)
plt.ylabel('')


# In[ ]:


fig,ax=plt.subplots(figsize=(25,15))
sns.boxplot(x="OverallCond", y="SalePrice", data=train,ax=ax)
ax.set_title("Boxplot of Price for OverallCond",size=20)


# # 3.Regression_tree

# In[ ]:


data_tree=train[['MSSubClass','MSZoning','Neighborhood','OverallQual','OverallCond','SalePrice']]
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
data_tree['MSZoning_new'] = labelencoder.fit_transform(data_tree['MSZoning'])
data_tree['Neighborhood_new'] = labelencoder.fit_transform(data_tree['Neighborhood'])
data_tree.head()


# The mean absolute error (MAE) is 0.1447 for the regression tree.which is not bad.

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
x_train,x_test,y_train,y_test=train_test_split(data_tree[['MSSubClass','MSZoning_new','Neighborhood_new','OverallQual','OverallCond']],data_tree[['SalePrice']],test_size=0.1,random_state=300)
tree=DecisionTreeRegressor(criterion='mse',max_depth=4,random_state=0)
tree=tree.fit(x_train,y_train)
y=y_test['SalePrice']
predict=tree.predict(x_test)
print(np.mean(abs(np.multiply(np.array(y_test.T-predict),np.array(1/y_test)))))


# In[ ]:


data_tree_for_test=test[['Id','MSSubClass','MSZoning','Neighborhood','OverallQual','OverallCond']]
data_tree_for_test.isnull().sum()


# In[ ]:


data_tree_for_test[data_tree_for_test.MSZoning.isnull()==True]


# In[ ]:


data_tree_for_test.MSZoning.value_counts()


# We replace the NaN in MSZoning with 'RL' because 'RL' is the most count in MSZoning.

# In[ ]:


data_tree_for_test.MSZoning[[455,756,790,1444]]='RL'


# In[ ]:


from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
data_tree_for_test['MSZoning_new'] = labelencoder.fit_transform(data_tree_for_test['MSZoning'])
data_tree_for_test['Neighborhood_new'] = labelencoder.fit_transform(data_tree_for_test['Neighborhood'])
data_tree_for_test.head()


# In[ ]:


predict_test=tree.predict(data_tree_for_test[['MSSubClass','MSZoning_new','Neighborhood_new','OverallQual','OverallCond']])
submit=pd.DataFrame({'Id':data_tree_for_test.Id,'SalePrice':predict_test})
submit.head()


# In[ ]:


submit.to_csv('submission.csv',index=False)

