#!/usr/bin/env python
# coding: utf-8

# <a id="0"></a> 
# # INTRODUCTION
# In this program , I will use seaborn library to analyze Black Friday data .  Graphics I will draw are as below :
# <br>
# 1. [Bar Plot](#1)
# 2. [Point Plot](#2)
# 3. [Joint Plot](#3)
# 4. [Pie Chart](#4)
# 5. [Lm Plot](#5)
# 6. [Kde Plot](#6)
# 7. [Violin Plot](#7)
# 8. [Heatmap](#8)
# 9. [Box Plot](#9)
# 10. [Swarm Plot](#10)
# 11. [Pair Plot](#11)
# 12. [Count Plot](#12)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
get_ipython().run_line_magic('matplotlib', 'inline')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


# read BlackFriday.csv file data from input directory and create dataframe named data
df = pd.read_csv("../input/BlackFriday.csv")


# In[ ]:


# check information about columns data 
df.info()


# In[ ]:


# Let's see first 10 data of dataframe to have knowledge data itself.
df.head(10)


# In[ ]:


df["Product_Category_1"].fillna(0, inplace=True)
df["Product_Category_2"].fillna(0, inplace=True)
df["Product_Category_3"].fillna(0, inplace=True)

df.columns


# <a id="1"></a> 
# # 1.Bar Plot

# In[ ]:


# Bar Plot
# Most common 50 Product_id sold in Black Friday
                   
prd_count = Counter(df.Product_ID)         
most_common_prd= prd_count.most_common(50)  

x,y = zip(*most_common_prd)
x,y = list(x),list(y)
# 
plt.figure(figsize=(15,8))
ax= sns.barplot(x=x, y=y,palette = sns.cubehelix_palette(len(x)))
plt.xlabel('Id of Product')
plt.xticks(rotation=90)
plt.ylabel('Frequency')
plt.title('Most Common 15 Product Sold in Black Friday')


# In[ ]:


# Bar Plot
# For each Occupation, percentage of number of Product categories purchased 

df_Product = pd.merge(df[df.Product_Category_1 != 0 ].groupby(['Occupation']).count().reset_index('Occupation')[['Occupation','Product_Category_1']], 
                      df[df.Product_Category_2 != 0 ].groupby(['Occupation']).count().reset_index('Occupation')[['Occupation','Product_Category_2']],    
                       how='outer', left_index=True, on='Occupation')

df_Product = pd.merge(df_Product, 
                      df[df.Product_Category_3 != 0 ].groupby(['Occupation']).count().reset_index('Occupation')[['Occupation','Product_Category_3']],
                       how='outer', left_index=True, on='Occupation')

occ_list = df.sort_values(by=['Occupation'])['Occupation'].unique()

share_prd_cat1 = []
share_prd_cat2 = []
share_prd_cat3 = []

for i in occ_list:
    x = df_Product[df_Product['Occupation']==i]
    total = sum(x.Product_Category_1) + sum(x.Product_Category_2) + sum(x.Product_Category_3)
    share_prd_cat1.append(np.round(100 * (sum(x.Product_Category_1) / total), decimals=2, out=None))
    share_prd_cat2.append(np.round(100 * (sum(x.Product_Category_2) / total), decimals=2, out=None))
    share_prd_cat3.append(np.round(100 * (sum(x.Product_Category_3) / total), decimals=2, out=None))
    
# visualization
f,ax = plt.subplots(figsize = (10,8))

sns.barplot(x=share_prd_cat1,y=occ_list,color='red',alpha = 0.5,label='Product Category 1',orient='h' )
sns.barplot(x=share_prd_cat2,y=occ_list,color='blue',alpha = 0.6,label='Product Category 2',orient='h' )
sns.barplot(x=share_prd_cat3,y=occ_list,color='cyan',alpha = 0.7,label='Product Category 3',orient='h' )

ax.legend(loc='lower right',frameon = True)     # legendlarin gorunurlugu
ax.set(xlabel='Percentage of Products', ylabel='States',title = "Percentage of # Products Purchased")
plt.xticks(rotation=90)



# In[ ]:


# Percentahge of 1-5 type of Product category1 item for each Occupation
df_Product =df[df.Product_Category_1 != 0 ].groupby(['Occupation','Product_Category_1']).count().reset_index('Occupation').reset_index('Product_Category_1')[['Occupation','Product_Category_1', 'Product_ID']]
df_Product.rename(columns={'Product_ID': 'Count'}, inplace = True)

occ_list = df_Product.sort_values(by=['Occupation'])['Occupation'].unique()
prd_list = df_Product.sort_values(by=['Product_Category_1'])['Product_Category_1'].unique()
prd_list = prd_list[0:5]

pool_of_names = []
for i in prd_list:
    globals()['Category{0}'.format(i)] =[]  
    if i == 1 :
        globals()['Color{0}'.format(i)] ="cyan"
    elif i == 2 :
        globals()['Color{0}'.format(i)] ="blue"
    elif i == 3 :
        globals()['Color{0}'.format(i)] ="red"
    elif i == 4 :
        globals()['Color{0}'.format(i)] ="yellow"
    elif i == 5 :
        globals()['Color{0}'.format(i)] ="fuchsia"
        
   
for i in occ_list:
    x = df_Product[df_Product['Occupation']==i]
    total = sum(x.Count)
    for j in prd_list:
         percentage = np.round(100 * (sum(x[x.Product_Category_1 ==j].Count) / total), decimals=2, out=None)
         globals()['Category{0}'.format(j)].append(percentage)
         if percentage < 25:
             globals()['Alpha{0}'.format(j)] = 0.8
         else:
             globals()['Alpha{0}'.format(j)] = 0.2
         
# visualization
f,ax = plt.subplots(figsize = (9,10))

for j in prd_list:
    sns.barplot(x= globals()['Category{0}'.format(j)],
                y=occ_list,
                color=globals()['Color{0}'.format(j)],
                alpha = globals()['Alpha{0}'.format(j)],

                label='Product Category1 {0}'.format(j),
                orient='h' )


ax.legend(loc='lower right',frameon = True)     # legendlarin gorunurlugu
ax.set(xlabel='Percentage of Products', ylabel='States',title = "Percentage of number of ProductCategory1 type Purchased")
plt.xticks(rotation=90)


# **[Go To Top](#0)**

# <a id="2"></a> 
# # 2.Point Plot

# In[ ]:


#%%Point Plot
# high school graduation rate vs Poverty rate of each state

df_Product = pd.merge(df[df.Product_Category_1 != 0 ].groupby(['Occupation']).count().reset_index('Occupation')[['Occupation','Product_Category_1']], 
                      df[df.Product_Category_2 != 0 ].groupby(['Occupation']).count().reset_index('Occupation')[['Occupation','Product_Category_2']],    
                       how='outer', left_index=True, on='Occupation')

df_Product = pd.merge(df_Product, 
                      df[df.Product_Category_3 != 0 ].groupby(['Occupation']).count().reset_index('Occupation')[['Occupation','Product_Category_3']],
                       how='outer', left_index=True, on='Occupation')
occ_list = df_Product.sort_values(by=['Occupation'])['Occupation'].unique()

# visualize
f,ax1 = plt.subplots(figsize =(10,5))
    
sns.pointplot(x='Occupation',y='Product_Category_1',data=df_Product, color='red',alpha=0.6)
sns.pointplot(x='Occupation',y='Product_Category_2',data=df_Product ,color='cyan',alpha=0.6)
sns.pointplot(x='Occupation',y='Product_Category_3',data=df_Product ,color='limegreen',alpha=0.6)

plt.text(5,0.50,'Product category1',color='red',fontsize = 12,style = 'italic')
plt.text(10,15,'Product category2',color='cyan',fontsize = 12,style = 'italic')
plt.text(15,20,'Product category3',color='limegreen',fontsize = 12,style = 'italic')

plt.xlabel('Occupation',fontsize = 14,color='black')
plt.ylabel('Product Category Purchased Count',fontsize = 14,color='black')
plt.title('Product Category Purchased Count Vs Occupation',fontsize = 16,color='red')
plt.grid()


# **[Go To Top](#0)**

# <a id="3"></a> 
# # 3.Joint Plot

# In[ ]:


# Visualization of count(Product_Category_1) vs  count(Product_Category_2) of each state with different style of seaborn code
# joint kernel density
# pearsonr= if it is 1, there is positive correlation and if it is, -1 there is negative correlation.
# If it is zero, there is no correlation between variables
# Show the joint distribution using kernel density estimation 
df_Product = pd.merge(df[df.Product_Category_1 != 0 ].groupby(['Occupation']).count().reset_index('Occupation')[['Occupation','Product_Category_1']], 
                      df[df.Product_Category_2 != 0 ].groupby(['Occupation']).count().reset_index('Occupation')[['Occupation','Product_Category_2']],    
                       how='outer', left_index=True, on='Occupation')

g = sns.jointplot(df_Product.Product_Category_1, df_Product.Product_Category_2, kind="kde", size=7)
plt.show()


# **[Go To Top](#0)**

# <a id="4"></a> 
# # 4.Pie Plot

# In[ ]:


# Product_Category_1 rates in black friday data 
labels = df.Product_Category_1.value_counts().index
sizes = df.Product_Category_1.value_counts().values

colors = ['grey','blue','red','yellow','green','brown','grey','blue','red','yellow','green','brown','grey','blue','red','yellow','green','brown']
explode = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

# visual
plt.figure(figsize = (7,7))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%')
plt.title('Product_Category_1 rates in black friday data',color = 'blue',fontsize = 15)


# **[Go To Top](#0)**

# <a id="5"></a> 
# # 5.Lm Plot

# In[ ]:


# Visualization of count(Product_Category_1 purchased bye men ) vs  count(Product_Category_1 purchased bye women ) of each state with different style of seaborn code
# lmplot 
# Show the results of a linear regression within each dataset

df_Gender = df.groupby(['Gender','Product_Category_1']).count().reset_index('Gender').reset_index('Product_Category_1')

# Filter female and male 
df_Female = df_Gender[df_Gender['Gender'] =='F'] 
df_Male = df_Gender[df_Gender['Gender'] =='M'] 

# get only 3 columns 
df_Female = df_Female[['Product_Category_1', 'Product_ID']]
df_Male = df_Male[['Product_Category_1', 'Product_ID']]

# Rename column Product_ID as Count
df_Female.rename(columns={'Product_ID': 'CountFemale'}, inplace = True)
df_Male.rename(columns={'Product_ID': 'CountMale'}, inplace = True)

df_Female.sort_values(by=['Product_Category_1'], inplace = True)
df_Male.sort_values(by=['Product_Category_1'], inplace = True)

df_Product = pd.merge(df_Female, 
                      df_Male,    
                      how='outer', left_index=True, on='Product_Category_1')
sns.lmplot(x="CountMale", y="CountFemale", data=df_Product)
plt.show()


# **[Go To Top](#0)**

# <a id="6"></a> 
# # 6.Kde Plot

# In[ ]:


#%%  kde plot
sns.kdeplot(df_Product.CountMale, df_Product.CountFemale, shade=True, cut=3)
plt.show()


# **[Go To Top](#0)**

# <a id="7"></a> 
# # 7.Violin Plot

# In[ ]:


# distribution of Product_Category_1, Product_Category_2, Product_Category_3 types 
# Show each distribution with both violins and points
# Use cubehelix to get a custom sequential palette
df_Product = df[( df.Product_Category_2 != 0 ) & (df.Product_Category_3 != 0)][['Product_Category_1','Product_Category_2','Product_Category_3']]

pal = sns.cubehelix_palette(2, rot=-.5, dark=.3)
sns.violinplot(data=df_Product, palette=pal, inner="points")
plt.show()


# **[Go To Top](#0)**

# <a id="8"></a> 
# # 8.Heatmap

# In[ ]:


#correlation map
# Visualization of black friday data for Product_Category_1, Product_Category_2, Product_Category_3 types
df_Product = df[( df.Product_Category_2 != 0 ) & (df.Product_Category_3 != 0)][['Product_Category_1','Product_Category_2','Product_Category_3']]

f,ax = plt.subplots(figsize=(5, 5))
sns.heatmap(df_Product.corr(), annot=True, linewidths=0.5,linecolor="red", fmt= '.1f',ax=ax)
plt.show()


# **[Go To Top](#0)**

# <a id="9"></a> 
# # 9.Box Plot

# In[ ]:


# purchase for Age and Gender
df_Product = df[(df.Age != '0-17') & (df.Age !='18-2')][['Purchase','Age','Gender']].sort_values(by=['Age'])

# Plot the orbital period with horizontal boxes
sns.boxplot(x="Age", y="Purchase", hue="Gender", data=df_Product, palette="PRGn")
plt.show()


# **[Go To Top](#0)**

# <a id="10"></a> 
# # 10.Swarm Plot

# In[ ]:


# purchase for Age and Gender for 1000 record
df_Product = df.loc[:1000,['Purchase','Age','Gender']][(df.Age != '0-17') & (df.Age !='18-2')].sort_values(by=['Age'])


sns.swarmplot(x="Age", y="Purchase",hue="Gender", data=df_Product)
plt.show()


# **[Go To Top](#0)**

# <a id="11"></a> 
# # 11.Pair Plot

# In[ ]:


#%% pair plot
#Count of Product_Category_1 purchased by male and Female
df_Gender = df.groupby(['Gender','Product_Category_1']).count().reset_index('Gender').reset_index('Product_Category_1')

# Filter female and male 
df_Female = df_Gender[df_Gender['Gender'] =='F'].sort_values(by=['Product_Category_1'])
df_Male = df_Gender[df_Gender['Gender'] =='M'].sort_values(by=['Product_Category_1']) 

# Rename column Product_ID as Count
df_Female.rename(columns={'Product_ID': 'CountFemale'}, inplace = True)
df_Male.rename(columns={'Product_ID': 'CountMale'}, inplace = True)

df_Product = pd.merge(df_Female, 
                      df_Male,    
                      how='outer', left_index=True, on='Product_Category_1')[['CountFemale','CountMale']]

sns.pairplot(df_Product)
plt.show()


# **[Go To Top](#0)**

# <a id="12"></a> 
# # 12.Count Plot

# In[ ]:


sns.countplot(df.Occupation)
#sns.countplot(kill.manner_of_death)
plt.title("Occupation",color = 'blue',fontsize=15)
plt.show()


# **[Go To Top](#0)**

# 
