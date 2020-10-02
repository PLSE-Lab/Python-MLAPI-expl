#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
import pandas
from pyspark.sql import SparkSession
from pyspark.sql import Window
from pyspark.sql.functions import lit, col, sum, isnull, isnan, mean, max, desc, when, unix_timestamp, to_date, regexp_extract
from pyspark.sql.types import DoubleType, LongType


# In[ ]:


sp = SparkSession.builder.appName("test").getOrCreate()


# In[ ]:


df = sp.read.csv('../input/Google Play Store Apps/googleplaystore.csv', inferSchema=True, header=True)


# In[ ]:


df.limit(5).toPandas()
# df.limit(5).toPandas().head()


# In[ ]:


df.groupBy('Category').count().collect()


# In[ ]:


df.groupBy('Price').count().collect()


# In[ ]:


df.printSchema()


# In[ ]:


df = df.withColumn("SizeS", regexp_extract(col('Size'), '([0-9\.]+)[Mk]', 1))
df = df.withColumn("PriceS", when(col('Price') != '0', regexp_extract(col('Price'), '\$([0-9\.]*)', 1)).otherwise(col('Price')))


# In[ ]:


df = df.withColumn("Ratingf", df["Rating"].cast(DoubleType()))
df = df.withColumn("Pricef", df["PriceS"].cast(DoubleType()))
df = df.withColumn("Reviewsf", df["Reviews"].cast(LongType()))
df = df.withColumn("Sizef", df["SizeS"].cast(DoubleType()))
df = df.withColumn("Last Updatedf", to_date(unix_timestamp(col('Last Updated'), 'MMMM dd, yyyy').cast("timestamp")))
df.limit(5).collect()


# In[ ]:


numeric_cols  = [cols[0] for cols in df.dtypes if cols[1] in ['int', 'long', 'double', 'float', 'decimal.Decimal']]
numeric_cols


# In[ ]:


df = df.withColumn("Ratingf", when(col("Ratingf") == float("NaN"), None).otherwise(col("Ratingf")))


# In[ ]:


df.printSchema()


# In[ ]:


(df.count(), len(df.columns))


# In[ ]:


df_null = df.select(*(sum(isnull(col(column)).cast("int")).alias(column) for column in df.columns))
df_null = df_null.withColumn("summary", lit("null"))

df_nan = df.select(*(sum(isnan(col(column)).cast("int")).alias(column) for column in numeric_cols))
df_nan = df_nan.withColumn("summary", lit("nan"))

df_zero = df.select(*(sum((col(column) == 0).cast("int")).alias(column) for column in numeric_cols))
df_zero = df_zero.withColumn("summary", lit("zero"))

df_neg = df.select(*(sum((col(column) < 0).cast("int")).alias(column) for column in numeric_cols))
df_neg = df_neg.withColumn("summary", lit("negative"))

df_inf_p = df.select(*(sum((col(column) == float("Inf")).cast("int")).alias(column) for column in numeric_cols))
df_inf_p = df_inf_p.withColumn("summary", lit("infinity_p"))

df_inf_n = df.select(*(sum((col(column) == -float("Inf")).cast("int")).alias(column) for column in numeric_cols))
df_inf_n = df_inf_n.withColumn("summary", lit("infinity_n"))


# In[ ]:


df_missing = df_null
df_missing_numeric = df_zero.union(df_nan)
df_missing_numeric = df_missing_numeric.union(df_inf_p)
df_missing_numeric = df_missing_numeric.union(df_inf_n)
df_missing_numeric = df_missing_numeric.union(df_neg)


# In[ ]:


df_missing.collect()


# In[ ]:


df_missing_numeric.collect()


# ### Data Fill/Imputation
# * Null: `Sizef`, `Ratingf`, `Pricef`
# 
# #### Strategy
# 
# 1) Sizef: fill as the average of the `Category`
# 
# 2) Ratingf: fill as average of the `Category`
# 
# 3) Pricef: fill 0 if `Type` is 'Free' else fill average of the category

# In[ ]:


window_spec = Window.partitionBy(df["Category"]).rangeBetween(-sys.maxsize, sys.maxsize)
print(window_spec)


# In[ ]:


size_mean = mean(df["Sizef"]).over(window_spec)
print(size_mean)
rating_mean = mean(df["Ratingf"]).over(window_spec)
print(rating_mean)
price_mean = mean(df["Pricef"]).over(window_spec)
print(price_mean)


# In[ ]:


dfi = df
dfi = dfi.withColumn('size_mean', size_mean)
dfi = dfi.withColumn('rating_mean', rating_mean)
dfi = dfi.withColumn('price_mean', price_mean)


# In[ ]:


print(dfi.count(), len(dfi.columns))
dfi.printSchema()


# In[ ]:


dfi = dfi.withColumn('size_imputed', when(isnull(col('Sizef')), col('size_mean')).otherwise(col('Sizef')))
dfi = dfi.withColumn('rating_imputed', when(isnull(col('Ratingf')), col('rating_mean')).otherwise(col('Ratingf')))
dfi = dfi.withColumn('price_imputed', when(isnull(col('Pricef')) & (col('Type') != 'Free'), col('price_mean')).when(isnull(col('Pricef')) & (col('Type') == 'Free'), 0).otherwise(col('Pricef')))                     


# In[ ]:


dfi.limit(5).collect()


# In[ ]:


dfi = dfi.dropna(how='all', subset=["price_imputed", "size_imputed"])


# In[ ]:


dfi.repartition(1).write.csv('google_submission_imputed.csv', header=True)


# In[ ]:


check_col = ['rating_imputed', 'size_imputed', 'price_imputed']
dfi_null = dfi.select(*(sum(isnull(col(column)).cast("int")).alias(column) for column in check_col))
dfi_null = dfi_null.withColumn("summary", lit("null"))


# In[ ]:


dfi_null.collect()

