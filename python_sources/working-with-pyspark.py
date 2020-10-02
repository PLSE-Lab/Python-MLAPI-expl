#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install pyspark')


# In[ ]:


from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import *
spark = SparkSession.builder.appName('data_processing').getOrCreate()
from pyspark.sql.functions import udf


# In[ ]:


df = spark.read.csv("../input/sales/Sales Records.csv",header=True,inferSchema=True)


# In[ ]:


df.show(5)


# In[ ]:


df.dtypes


# In[ ]:


df = df.withColumnRenamed('Item Type','Item_Type').withColumnRenamed('Sales Channel','Sales_Channel').withColumnRenamed('Order Priority','Order_Priority')
df = df.withColumnRenamed('Order Date','Order_Date').withColumnRenamed('Order ID','Order_ID').withColumnRenamed('Ship Date','Ship_Date')
df = df.withColumnRenamed('Units Sold','Units_Sold').withColumnRenamed('Unit Price','Unit_Price').withColumnRenamed('Unit Cost','Unit_Cost')
df = df.withColumnRenamed('Total Revenue','Total_Revenue').withColumnRenamed('Total Cost','Total_Cost').withColumnRenamed('Total Profit','Total_Profit')


# ### Select Statements

# In[ ]:


df.select(['Item_Type']).distinct().show()


# In[ ]:


df.select(['Region','Country']).distinct().show()


# ### Filter Statements

# In[ ]:


df[(df.Order_Priority == 'H') 
   & (df.Country == 'United States of America') ].show(5,False)


# In[ ]:


df.filter(df.Sales_Channel == 'Online').filter(df.Region == 'North America').show(5,False)


# In[ ]:


df.filter(df.Sales_Channel == 'Offline').filter(df.Region == 'North America').show(5,False)


# ### Where Statements

# In[ ]:


df.where((df.Units_Sold > 5000) | (df.Total_Revenue >= 100000)).show(5,False)


# In[ ]:


df.where((df.Order_Priority == 'H') & (df.Total_Profit >= 1000000)).show(5,False)


# ### Groupby/Aggregation

# In[ ]:


df.groupby('Region').agg(F.sum('Total_Profit').alias('Region Profits')).show(5,False)


# In[ ]:


df.groupby('Item_Type').agg(F.sum('Total_Profit').alias('Item Profits')).show(5,False)


# In[ ]:


df.groupby('Country').agg(F.sum('Total_Profit').alias('Sum Total Profits')).show(5,False)


# In[ ]:


df.groupby('Sales_Channel').agg(F.sum('Total_Profit').alias('Channel Profits')).show(5,False)


# In[ ]:


df.groupby('Region').agg(F.mean('Total_Revenue').alias('Sum Total Revenue')).show(5,False)


# In[ ]:


df.groupby('Country').agg(F.mean('Total_Revenue').alias('Average Revenue')).show(5,False)


# In[ ]:


df.groupby('Region').agg(F.mean('Total_Cost').alias('Item Cost')).show(5,False)


# ### Collect Set/List

# In[ ]:


df.groupby('Region').agg(F.collect_set('Item_Type')).show(5)


# In[ ]:


df.groupby("Region").agg(F.collect_list("Units_Sold")).show()


# ### (UDF) User Defined Function

# In[ ]:


def Prof(Total_Profit):
    if Total_Profit >= 937196.46:
        return 'Above Average Profit'
    else: 
        return 'Below Average Profit'


# In[ ]:


Prof_udf=udf(Prof,StringType())
df=df.withColumn('Prof',Prof_udf(df['Total_Profit']))


# In[ ]:


df.select('Total_Profit','Prof').show(5,False)


# In[ ]:


df[df.Total_Profit >= 950000].show(5)


# ### Join

# In[ ]:


df1 = spark.read.csv('../input/working13/ManagerInformation.csv',header=True,inferSchema=True)


# In[ ]:


df1.show()


# In[ ]:


join_df = df.join(df1,on='Region')


# In[ ]:


join_df.groupby('Manager_Name').agg(F.sum('Total_Profit').alias('Manager Profits')).show(5,False)


# ### Pivot

# In[ ]:


df.groupby('Region').pivot('Item_Type').sum('Total_Revenue').fillna(0).show(5,False)


# ### Window Function

# In[ ]:


from pyspark.sql.window import Window
from pyspark.sql.functions import col,row_number


# In[ ]:


WinF = Window.orderBy(df['Total_Profit'].desc())
df = df.withColumn('rank',row_number().over(WinF).alias('rank'))


# In[ ]:


df.show(5)

