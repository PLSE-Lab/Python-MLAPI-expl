#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import scipy.stats as stats


# In[ ]:


df=pd.read_excel("/kaggle/input/superstore/US Superstore data.xls")


# In[ ]:


df.head()


# In[ ]:


df.tail()


# In[ ]:


df.info()


# In[ ]:


df.describe()


# In[ ]:


df.columns


# In[ ]:


pd.set_option('display.max_columns', None)
df.head()


# In[ ]:


df.isnull().sum().sum()


# In[ ]:


df["Segment"].value_counts().plot(kind="bar")


# In[ ]:





# In[ ]:


data=df[['Sales','Quantity','Discount','Profit']]
sns.heatmap(data.corr(),annot=True)


# **Here we can observe that there is a positive correlation between Profit and sales
# i.e when sales increases profit las increases.**

# In[ ]:


df['Category'].value_counts()


# In[ ]:


df['Category'].value_counts().plot(kind="bar")


# **Here it can be observed that the sale of office supplies is way higher than that of the other two categories**

# In[ ]:


pd.crosstab(df['Segment'],df['Category']).plot(kind="bar",stacked=True)


# **The graph shows the differnt category under each segment**
# 

# In[ ]:


sns.distplot(df["Sales"])


# In[ ]:


df.columns


# **It can be observed that the data here is higly right skewed**

# In[ ]:



sns.scatterplot("Sales",'Profit',data=df)


# **Here it can be seen that it is not necessary that with the increase in sale profit increases.**

# In[ ]:


axes,fig=plt.subplots(0,1,figsize=(18,5))
sns.scatterplot("Discount",'Profit',data=df)


# **Here it can be observed that when the discount is till 0.3, there is a profit.**
# 
# **But if the discount increases beyond 0.3 there is a loss happening**

# In[ ]:


axes,fig=plt.subplots(0,1,figsize=(18,5))
sns.scatterplot('Sales','Discount',data=df)


# **AFTER A POINT WHEN DISCOUNT IS INCREASING THE TOTAL SALES AMOUNT IS DECREASING**

# In[ ]:


df['Sub-Category'].value_counts().plot(kind="bar")


# **THE SUB-CATEGORY IS ARRANGED ON THE BASIS OF MOST SELLING PRODUCTS**

# In[ ]:


pd.crosstab(df["Region"],df["Category"],df["Profit"],aggfunc='sum').plot(kind="bar",stacked=True)


# **There is more profit from the East an west regions**

# In[ ]:


pd.crosstab(index=df["Category"],columns=df["Segment"],values=df["Profit"],aggfunc="sum").plot(kind="bar",stacked=True)


# **Although office supplies is the most selling category but the profit is highest from the technology sector
# Under Technology the purchasing ie the profit has comE  more  from the  Consumers segment**

# In[ ]:


pd.crosstab(index=df["Category"],columns=df["Ship Mode"],values=df["Profit"],aggfunc="sum").plot(kind="bar",stacked=True)


# In[ ]:


sns.lmplot(x="Profit",y="Sales",data=df,fit_reg=False,col="Category")
plt.show()


# **The profit is very low almost 0 in the Furniture sector
# also the profit is high in the Technology sector**

# In[ ]:


sns.lmplot(x="Profit",y="Sales",data=df,fit_reg=False,col="Ship Mode")


# **The profit is very high when the ship mode is Standard class**
# 
# **NO or very less profit when the ship mode is same day**

# In[ ]:


axes,fig=plt.subplots(0,1,figsize=(18,5))
sns.barplot("Sub-Category","Profit",data=df)


# **HERE IT CAN BE OBSERVED THE PROFIT OR THE LOSSES WITH RESPECT TO THE EACH SUB CATEGORY**
# **TABLE,BOOKCASES,FASTENERS ARE BASICALLY IN LOSS**
# **PROFIT IS HIGHEST FROM THE COPIERS SUB-CATEGORY**

# In[ ]:


sns.scatterplot("Quantity","Profit",data=df)


# In[ ]:





# ## THE FINAL INSIGHTS

# **1. When the discount is till 0.3, there is a profit,But if the discount increases beyond 0.3 there is a loss happening**
# 
# **2.Although office supplies is the most selling category but the profit is highest from the technology sector Under which the  the profit has come more from the Consumers segment**
# 
# **3.Although Copiers is the least selling sub-category but has given the most profit out of all the sub-categories.**
# 
# **4.There is a huge loss from the furniture section**
# 
# **5.The profit is high when the ship mode is "Standard Class" and the Profit is negligible when the ship Mode is "Same day"**
# 
# **6.The profit is more from the east and west region of the country.**
# 

# ##  SUGGESTIONS TO THE BUSSINESS

# **1.THE DISCOUNT SHOULD NOT BE INCREASED BEYOND 0.3.**
# 
# **2.THE FURNITURE CATEGORY IS CAUSING A LOT OF LOSS, SO THE COMPANY CAN STOP SELLING FURNITURES OR SHOULD INCREASE THE PRICE OF THE FURNITURE CATEGORY OR CAN TRY TO REDUCE THE OVERALL COST OF THE PRODUCT**
# 
# **3.IF THE SHIPPING MODE IS "SAME DAY", THE SHIPPING CHARGES SHOULD BE INCREASED.**
# 
# **4.THE COMPANY SHOULD FOCUS MORE ON THE TECHNOLOGY SECTION**

# In[ ]:




