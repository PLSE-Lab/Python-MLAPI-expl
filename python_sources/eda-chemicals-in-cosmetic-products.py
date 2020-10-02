#!/usr/bin/env python
# coding: utf-8

# ## Reading data
# 
# First we remove duplicates in this dataset.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_original = pd.read_csv('../input/chemicals-in-cosmetics/chemicals-in-cosmetics.csv')
df = df_original.drop_duplicates()
print('The original database shape:', df_original.shape)
print('Database without duplicates:', df.shape)


# In[ ]:


df.head()


# ## Investigating chemical counts

# In[ ]:


df['ChemicalName'].value_counts().size


# Overall there are 123 different chemicals that are reported.
# 
# Based on the description, each entry in column 'ChemicalCount' is the total number of current chemicals reported for a product. This number does not include chemicals that have been removed from a product.

# In[ ]:


df['ChemicalCount'].describe()


# In average, products contain at least one chemical. Notice there are products with 0 chemicals, and there are products with 9 reported chemicals.
# 
# Let's first investigate products where 'ChemicalCount'=0.

# In[ ]:


df.loc[df.ChemicalCount==0].head()


# The number of chemicals being equal to zero suggests that the chemicals were removed from the product (reported in 'ChemicalDateRemoved'). This can be verified by checking if there are NaN values in this column.

# In[ ]:


# when the result is False, there are no NaN values
df.loc[df.ChemicalCount==0]['ChemicalDateRemoved'].isnull().max()


# In the sequel, we will focus on products that have reported at least one chemical and that are still in use (not discontinued).

# In[ ]:


df_n0 = df.loc[(df.ChemicalCount>0) & (df['DiscontinuedDate'].isna())]


# The maximum number of chemicals that is reported in a product is 9. We can find these products:

# In[ ]:


df_n0.loc[df.ChemicalCount==9]


# It turns out it is only one product, where each chemical is separately reported.
# 
# The following code is used to generate the bar chart showing the number of products per number of chemicals. In counting the number of products, different color, scent and/or flavor of the product are neglected (e.g. 'Professional Eyeshadow Base' can be beige or bright, but it is counted only once with the identification number 'CDPHId'=26).

# In[ ]:


df_n0.loc[df['CDPHId']==26]


# In[ ]:


data = df_n0.groupby(['ChemicalCount']).nunique()['CDPHId']

fig = plt.figure(figsize=(9,7))
ax = plt.subplot(111)
ax.bar(data.index, data.values, log=True, align='center', alpha=0.5, edgecolor='k')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.set_xticks(np.arange(1,10))

for x,y in zip(data.index,data.values):
    plt.annotate(y, (x,y), textcoords="offset points", xytext=(0,4), ha='center') 

ax.set_title('Number of reported products containing chemicals', fontsize=15)
ax.title.set_position([.5, 1.05])
ax.set_xlabel('Number of chemicals', fontsize=12)
ax.set_ylabel('Number of products (log scale)', fontsize=12)

plt.show()


# ## Chemicals in baby products
# 
# Baby products represent one of the primary categories in this dataset. 

# In[ ]:


baby_prod = df_n0.loc[df_n0['PrimaryCategory']=='Baby Products']
baby_prod.head()


# The next code is used to find all chemicals present in baby products (listed and in a graph).

# In[ ]:


baby_prod_chem = baby_prod['ChemicalName'].value_counts()
print(baby_prod_chem)


# The long name 'Retinol/retinyl esters, when in daily dosages ...' will be replaced with 'Retinol'. The long name is stored in 'long_text', and a remark is given below the graph.

# In[ ]:


long_text = baby_prod_chem.index[2]
print('Old chemical name: ', long_text)
print()
baby_prod_chem.rename({baby_prod_chem.index[2]: 'Retinol *'}, inplace=True)
print('New chemical name: ', baby_prod_chem.index[2])


# In[ ]:


fig = plt.figure(figsize=(10, 6))
ax = plt.subplot(111)
ax.barh(baby_prod_chem.index, baby_prod_chem.values, color='red', alpha=0.6)

ax.xaxis.grid(linestyle='--', linewidth=0.5)

for x,y in zip(baby_prod_chem.values,baby_prod_chem.index):
    ax.annotate(x, (x,y), textcoords="offset points", xytext=(4,0), va='center') 

ax.set_title('Chemicals in baby products', fontsize=15)
ax.title.set_position([0.5,1.02])
ax.set_xlabel('Number of baby products', fontsize=12)
ax.set_xticks(np.arange(0,18,5))
plt.text(-0.15,-0.2, "* "+long_text, size=12, transform=ax.transAxes)

plt.show()


# List of all baby product names, containing at least one chemical, sorted by subcategory.

# In[ ]:


reported_baby_prod = baby_prod[['ProductName', 'CompanyName', 'SubCategory']].sort_values('SubCategory')
reported_baby_prod.columns=['Baby product', 'Company', 'Type of product']
reported_baby_prod.style.hide_index()


# In[ ]:




