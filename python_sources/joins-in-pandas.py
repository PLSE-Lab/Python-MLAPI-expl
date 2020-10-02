#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd


# ## 1. Inner Join

# In[ ]:


product=pd.DataFrame({
    'Product_ID':[101,102,103,104,105,106,107],
    'Product_name':['Watch','Bag','Shoes','Smartphone','Books','Oil','Laptop'],
    'Category':['Fashion','Fashion','Fashion','Electronics','Study','Grocery','Electronics'],
    'Price':[299.0,1350.50,2999.0,14999.0,145.0,110.0,79999.0],
    'Seller_City':['Delhi','Mumbai','Chennai','Kolkata','Delhi','Chennai','Bengalore']
})


# In[ ]:


product


# In[ ]:


customer=pd.DataFrame({
    'id':[1,2,3,4,5,6,7,8,9],
    'name':['Olivia','Aditya','Cory','Isabell','Dominic','Tyler','Samuel','Daniel','Jeremy'],
    'age':[20,25,15,10,30,65,35,18,23],
    'Product_ID':[101,0,106,0,103,104,0,0,107],
    'Purchased_Product':['Watch','NA','Oil','NA','Shoes','Smartphone','NA','NA','Laptop'],
    'City':['Mumbai','Delhi','Bangalore','Chennai','Chennai','Delhi','Kolkata','Delhi','Mumbai']
})


# In[ ]:


customer


# In[ ]:


pd.merge(product,customer,on='Product_ID')


# In[ ]:


# if the column names are different
pd.merge(product,customer,left_on='Product_name',right_on='Purchased_Product')


# In[ ]:


## seller and customer both belong to the same city.
pd.merge(product,customer,how='inner',left_on=['Product_ID','Seller_City'],right_on = ['Product_ID','City'])


# ## 2. Full Join

# In[ ]:


pd.merge(product,customer,on='Product_ID',how='outer')


# In[ ]:


pd.merge(product,customer,on='Product_ID',how='outer',indicator=True)


# ## 3. Left Join

# In[ ]:


pd.merge(product,customer,on='Product_ID',how='left')


# ## 4. Right Join

# In[ ]:


pd.merge(product,customer,on='Product_ID',how='right')


# #### Handling Redundancy/Duplicates in Joins

# In[ ]:


# Dummy dataframe with duplicate values

product_dup=pd.DataFrame({'Product_ID':[101,102,103,104,105,106,107,103,107],
                          'Product_name':['Watch','Bag','Shoes','Smartphone','Books','Oil','Laptop','Shoes','Laptop'],
                          'Category':['Fashion','Fashion','Fashion','Electronics','Study','Grocery','Electronics','Fashion','Electronics'],
                          'Price':[299.0,1350.50,2999.0,14999.0,145.0,110.0,79999.0,2999.0,79999.0],
                          'Seller_City':['Delhi','Mumbai','Chennai','Kolkata','Delhi','Chennai','Bengalore','Chennai','Bengalore']})


# In[ ]:


product_dup


# In[ ]:


pd.merge(product_dup,customer,on='Product_ID',how='inner')


# In[ ]:


# Drop duplicates
pd.merge(product_dup.drop_duplicates(),customer,on='Product_ID',how='inner')


# In[ ]:


# "Validate" to keep the duplicates
pd.merge(product_dup,customer,on="Product_ID",how='inner',validate='many_to_many')


# In[ ]:




