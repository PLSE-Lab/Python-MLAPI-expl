#!/usr/bin/env python
# coding: utf-8

# ### Importing Files

# In[ ]:


import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# In[ ]:


df = pd.read_csv("../input/listings_summary.csv")


# In[ ]:


df.shape


# In[ ]:


format(df.duplicated().sum()) # Check wether duplicate entries are their or not. here 0 represent no duplkicate data is their.


# In[ ]:


df.columns


# In[ ]:


columns=['id','space','description','neighborhood_overview','property_type','room_type','accommodates','bathrooms',
         'bedrooms','beds','bed_type','amenities','square_feet','price','security_deposit','cleaning_fee','extra_people',
         'minimum_nights','review_scores_rating','instant_bookable','cancellation_policy']


# In[ ]:


temp_data=df[columns].set_index('id')


# In[ ]:


temp_data.head()


# ### Cleaning Data

# In[ ]:


temp_data.isnull().sum(axis=0)


# In[ ]:


temp_data.security_deposit.fillna('$0.00', inplace=True) 


# In[ ]:


temp_data.cleaning_fee.fillna('$0.00',inplace=True)


# In[ ]:


temp_data[['security_deposit','cleaning_fee']].isnull().sum(axis=0)


# In[ ]:


temp_data.head(3)


# In[ ]:


temp_data.room_type.value_counts()


# In[ ]:


temp_data.dtypes


# In[ ]:


temp_data.price=list(map(lambda x: x.replace(',',''),temp_data.price))


# In[ ]:


temp_data.security_deposit=list(map(lambda x: x.replace(',',''),temp_data.security_deposit))


# In[ ]:


temp_data.cleaning_fee=list(map(lambda x: x.replace(',',''),temp_data.cleaning_fee))


# In[ ]:


temp_data['price'].value_counts()


# In[ ]:


temp_data.price=list(map(lambda x: x.replace('$','0'),temp_data.price))


# In[ ]:


temp_data.security_deposit=list(map(lambda x: x.replace('$','0'),temp_data.security_deposit))


# In[ ]:


temp_data.cleaning_fee=list(map(lambda x: x.replace('$','0'),temp_data.cleaning_fee))


# In[ ]:


temp_data['security_deposit'].value_counts()


# In[ ]:


temp_data['cleaning_fee'].value_counts()


# In[ ]:


temp_data['price'].value_counts()


# In[ ]:


temp_data[['price','security_deposit','cleaning_fee']]=temp_data[['price','security_deposit','cleaning_fee']].astype(float)


# In[ ]:


temp_data['Total_price']=temp_data[['security_deposit','cleaning_fee','price']].sum(axis=1)


# In[ ]:


temp_data.head(2)


# In[ ]:


avg_perRoom=temp_data.groupby(['room_type'])['price'].agg(np.mean)
#data.groupby(['neighbourhood','room_type'])['price'].agg(['mean'])


# In[ ]:


avg_perRoom.plot(kind='bar', figsize=(8,8), fontsize=10,color='green')
plt.title("Average Price per Room Type",fontsize=15)
plt.xlabel("Room Type",fontsize=12)
plt.ylabel("Average price in dollar",fontsize=12)
plt.show()


# ### Above graph shows that Average amount we have to pay per Room Type

# # Room Type, Bed Type and it's count

# In[ ]:


temp_data.bed_type.value_counts()


# In[ ]:


bed_perRoom=temp_data.groupby(['room_type'])['bed_type']


# In[ ]:


bed_perRoom.value_counts()


# # Room Type, Property and its count

# In[ ]:


property_type_perRoom=temp_data.groupby(['room_type'])['property_type']
                 


# In[ ]:


property_type_perRoom.value_counts()


# # Room Type and it's Average Review Score

# In[ ]:


score_perRoom=temp_data.groupby(['room_type'])['review_scores_rating'].agg(np.mean)


# In[ ]:


score_perRoom.plot(kind='pie',autopct="%1.1f%%",startangle=90,figsize=(8,8))


# ### Above Fig shows that Private and Entire home/apt has same average score

# In[ ]:


# Cancellation Policy per Room Type


# In[ ]:


temp_data.cancellation_policy.value_counts()


# In[ ]:


cancellation_perRoom=temp_data.groupby(['room_type'])['cancellation_policy']


# In[ ]:


cancellation_perRoom.value_counts()


# # Correlation Matrix

# In[ ]:


f,ax = plt.subplots(figsize=(9,9))
sns.heatmap(temp_data.corr(),annot=True,linewidth=.5,fmt ='.2f',ax = ax)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




