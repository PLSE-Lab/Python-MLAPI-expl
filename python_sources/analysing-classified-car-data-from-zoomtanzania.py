#!/usr/bin/env python
# coding: utf-8

# ### Analysing Classified Car data from ZoomTanzania
# 
# So, we are going to work on a dataset of classifie cars listed on ZoomTanzania.
# 
# The original data was originally scraped from zoomtanzania.com. A sample of all possible listings is obtained that is ~3930 data points which might decrease during [cleanup](https://towardsdatascience.com/the-art-of-cleaning-your-data-b713dbd49726) but hopefully will not affect our analysis.
# 
# **NOTE: This is my first kaggle (I Learning)**

# In[ ]:


import pandas as pd
import numpy as np

autos = pd.read_csv(r"../input/ZoomTanzania Used Cars.csv", index_col=0)


# In[ ]:


autos.isnull().sum()


# In[ ]:


autos = autos[autos.price_currency == 'TSh']


# In[ ]:


autos.head()


# In[ ]:


autos.price_value = [ v if type(v) is not str else float(v.replace(',', '')) for v in autos.price_value]
autos.year = [ int(v) for v in autos.year]


# In[ ]:


print(autos["price_currency"].unique())
print(autos["posted_weekday"].unique())
print(autos["four_wheel_drive"].unique())
print(autos["delivery_offered"].unique())
print(autos["price_negotiable"].unique())
print(autos["transmission"].unique())
print(autos["location"].unique())

autos.drop(["id", "url", "ad_id", "fetched", "description", "price_currency", 
            "mileage", "import_duty_paid", "current_location"], axis=1, inplace=True)


# Based on the information we got from the data within each column, we decide to delete the following ones:
# 
# - id, url, fetched, ad_id, import_duty_paid, current_location are not relevant in our analysis
# - description not sure what/how to analyse this
# - mileage because more than 20% of the data is null
# - price_currency has only one value

# In[ ]:


autos.info()


# The df.infos() method gives us information about the number of observations, each column data type and the memory used by the df. We can have more details about numerical values with the df.describe() method.

# In[ ]:


autos.describe()


# We can reduce memory allocation by asigning optimal datatypes to values/columns.

# In[ ]:


autos.price_value = autos.price_value.astype(np.uint32)
autos.year = autos.year.astype(np.uint8)
autos.posted_day = autos.posted_day.astype(np.uint8)
autos.page = autos.page.astype(np.uint8)


# In[ ]:


autos.info()


# We then reduced the memory usage :)

# In[ ]:


print(autos.price_value.value_counts().sort_index().head())
print(len(autos.price_value.unique()))
c_prices = autos.price_value.copy()
s_prices = c_prices.sort_values(ascending=False)
s_prices.index = autos.index
print(s_prices.head())


# There is exactly 288 different prices from  0 to 580000000. We can deduce that some of these values are incorrect and might bias our analysis.

# In[ ]:





# ## Conclusion

# In[ ]:




