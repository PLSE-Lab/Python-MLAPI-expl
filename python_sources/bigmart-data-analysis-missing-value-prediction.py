#!/usr/bin/env python
# coding: utf-8

# # BigMart Sales Data
# ##### By Chinthaka Liyana Arachchi (2020-06-01)

# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


df_train=pd.read_csv('/kaggle/input/big-mart-sales-data-train-data-set/Train.csv')
df_train.head()


# ## Dataset Summary
# ### Understand the features and checking the dataset for missing values
# ##### Total Number of Rows = 8523, Total Number of Columns = 12
# ##### Missing values in Item_Weight (7060 non-null   float64) & Outlet_Size (6113 non-null   object)

# In[ ]:


df_train.info()
df_train.isnull().sum()
sns.heatmap(df_train.isnull(), yticklabels=False)


# ### Understand and get statistical values for <b>Item_Weight</b> wrt Item_Type - Remove Null Values

# In[ ]:


weight_grp=df_train.groupby(['Item_Type','Outlet_Location_Type'])
weight_grp=weight_grp['Item_Weight'].agg(['count','min','max','mean','median']).reset_index()
weight_grp.head()


# #### Replace null values with mean values by considering the product type. (To Item_Weight column)

# In[ ]:


def add_weight(cols):
    weight=cols[0]
    itype=cols[1]
    location=cols[2]
    
    if pd.isnull(weight):
        for items in weight_grp.itertuples():
            if (items[1]==itype) and (items[2]==location):
                return items[6]
    else:
        return weight


# In[ ]:


df_train['Item_Weight']=df_train[['Item_Weight','Item_Type','Outlet_Location_Type']].apply(add_weight, axis=1)
df_train['Item_Weight'].head(10)


# ### Removed null values in Item_Weight Column
# #### Summary as follows

# In[ ]:


df_train.info()
df_train.isnull().sum()
sns.heatmap(df_train.isnull(), yticklabels=False)


# ### Understand and get statistical values for Outlet_Size wrt Outlet_Location_Type & Outlet_Type - Remove Null Values 

# In[ ]:


df_train.head()


# In[ ]:


g=sns.catplot(x='Outlet_Type', y='Item_Outlet_Sales', hue='Outlet_Size', data=df_train, kind='swarm')
plt.tight_layout()
g.set_xticklabels(rotation=45)


# ##### FIndings : We can replace Supermarket Type 2 & Type 3 as Medium and Grocery Stores as Small. But there's issue with Supermarket Type 1 (Above Line)
# ##### With Following plot, we can say that if It's supermarket type 1 & Tier 2 => Small, supermarket type 1 & Tier 3 => High (But we can't explicitly say that supermarket type 1 & Tier 1 result. It may be Small or Medium
# ###### If Supermarket Type 2 or Supermarket Type 2 => Medium
# ###### If Grocery Store => Small
# ###### If supermarket type 1 & Tier 2 => Small
# ###### If supermarket type 1 & Tier 3 => High

# In[ ]:


g2=sns.catplot(x='Outlet_Location_Type', y='Item_Outlet_Sales', hue='Outlet_Size', data=df_train, kind='violin', col='Outlet_Type')
plt.tight_layout()
g2.set_xticklabels(rotation=45)


# #### Understand the missing values in Outlet_Size variable more

# In[ ]:


osize_grp=df_train.groupby(['Outlet_Type','Outlet_Location_Type','Outlet_Size'])
osize_grp=osize_grp['Outlet_Size'].count()
osize_grp


# In[ ]:


df_train['Outlet_Size'].value_counts()


# In[ ]:


df_train.isnull().sum()


# #### Fill missing Values

# In[ ]:


def add_osize(cols):
    osize=cols[0]
    olocation=cols[1]
    otype=cols[2]
    
    if pd.isnull(osize):
        if (otype=='Supermarket Type2') or (otype=='Supermarket Type3'):
            return 'Medium'
        elif otype=='Grocery Store':
            return 'Small'
        else:
            if olocation=='Tier 2':
                return 'Small'
            elif olocation=='Tier 3':
                return 'High'
            else:
                return 'Small'
    else:
        return osize
        


# In[ ]:


df_train['Outlet_Size']=df_train[['Outlet_Size','Outlet_Location_Type','Outlet_Type']].apply(add_osize, axis=1)
df_train['Outlet_Size'].head(10)


# In[ ]:


sns.heatmap(df_train.isnull(), yticklabels=False)
df_train['Outlet_Size'].value_counts()


# ### Missing values filling completed. 
# ##### All 'Outlet_Size' missing values falls under 'Small' category. Therefore with the above justification, the prediction accuracy was 100%

# In[ ]:


df_train.corr()


# ## The impact on Sales by refering Visibility factor (Visibility, Item Type and Total Sale)

# In[ ]:


visib_grp=df_train.groupby(['Item_Visibility','Item_Type'])
visib_grp=visib_grp[['Item_Outlet_Sales']].mean().reset_index()
visib_grp


# In[ ]:


g3=sns.relplot(hue='Item_Outlet_Sales', y='Item_Visibility', data=visib_grp, x='Item_Type')
plt.tight_layout()
g3.set_xticklabels(rotation=90)


# ##### As correlation, r value between Item_Visibility & Item_Outlet_Sales (-0.128625), there is a negative trend where Item_Visibility is getting closer to 0 (zero), the sales values tends to the maximum. 
# ##### But when it comes to above diagram, there's no significant relationship with product type with reference to the relationship to Item_Visibility & Item_Outlet_Sales variables.

# ## The impact on Sales by refering Visibility factor (Visibility, Item Type and Total Sale, Outlet_Location_Type)

# In[ ]:


visib_grp2=df_train.groupby(['Item_Visibility','Outlet_Location_Type','Outlet_Type'])
visib_grp2=visib_grp2[['Item_Outlet_Sales']].mean().reset_index()
visib_grp2


# In[ ]:


g4=sns.relplot(x='Item_Outlet_Sales', y='Item_Visibility', data=visib_grp2, col='Outlet_Type', hue='Outlet_Location_Type')
plt.tight_layout()
g4.set_xticklabels(rotation=90)


# ##### By refering to above diagram (g4), there is no significant clustering relationship with item visibility with the Store location, store type or the sales.

# In[ ]:


df_train.head()


# ## Understand the sales with respect to Item_Fat_Content

# In[ ]:


fat_grp=df_train.groupby('Item_Fat_Content')
fat_grp=fat_grp[['Item_Outlet_Sales']].mean().reset_index()
fat_grp


# ### Replace LF & low fat with Low Fat and reg with Regular

# In[ ]:


def add_fat(cols):
    fat=cols[0]
    if (fat=='LF') or (fat=='low fat'):
        return 'Low Fat'
    elif fat == 'reg':
        return 'Regular'
    else:
        return fat


# In[ ]:


df_train['Item_Fat_Content']=df_train[['Item_Fat_Content']].apply(add_fat, axis=1)
df_train['Item_Fat_Content'].head(10)


# ### Grouping after data cleaning

# In[ ]:


fat_grpn=df_train.groupby('Item_Fat_Content')
fat_grpn=fat_grpn[['Item_Outlet_Sales']].agg(['mean','count','sum']).reset_index()
fat_grpn


# In[ ]:


fig, (ax1,ax2,ax3) = plt.subplots(nrows=1,ncols=3)
fat_pie1=ax1.pie(fat_grpn[('Item_Outlet_Sales','mean')], labels=fat_grpn['Item_Fat_Content'],autopct='%1.1f%%',textprops={'color':"w"})
fat_pie2=ax2.pie(fat_grpn[('Item_Outlet_Sales','sum')], labels=fat_grpn['Item_Fat_Content'],autopct='%1.1f%%',textprops={'color':"w"})
fat_pie3=ax3.pie(fat_grpn[('Item_Outlet_Sales','count')], labels=fat_grpn['Item_Fat_Content'],autopct='%1.1f%%',textprops={'color':"w"})
ax1.set_title('Mean',bbox={'facecolor':'0.8', 'pad':1})
ax2.set_title('Total Sales',bbox={'facecolor':'0.8', 'pad':1})
ax3.set_title('Item Count',bbox={'facecolor':'0.8', 'pad':1})
plt.show()


# #### From above pie charts we can come to an conclusion that more revenue came and most number of unites sold in Low Fat category but the average unit sale is higher in regular product category

# In[ ]:


fat_grpn=df_train.groupby(['Item_Fat_Content','Item_Type'])
fat_grpn=fat_grpn[['Item_Outlet_Sales']].mean().reset_index()
fat_grpn


# In[ ]:


g5=sns.catplot(x='Item_Type', y='Item_Outlet_Sales', data=fat_grpn, hue='Item_Fat_Content')
plt.tight_layout()
g5.set_xticklabels(rotation=90)


# #### In here we can see that for most products, orange dots are in higher than the blue dots except for "Breakfast", "Snak Foods", "Soft Drinks" and "Starchy Foods"
# #### This represent that the average revenue from all products (except the ones mentioned above) are higher in <b>Regular</b> category

# ##### For further analysis and predictions, this dataset can be used and it will be available on https://www.kaggle.com/brijbhushannanda1979/bigmart-sales-data

# In[ ]:




