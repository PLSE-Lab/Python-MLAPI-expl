#!/usr/bin/env python
# coding: utf-8

# # General
# id_31 is caracter data, but anyone does not seem to try correctly use it.
# 
# Someone uses categorycal encoder, but this is not always compelete way.
# 
# If we classify it, we might get useful data.
# 
# I also make another kerenel about id_31.(https://www.kaggle.com/yasagure/fraud-makers-are-earnest-people-about-browser)
# 
# At that time, I focused on the version of the browser and at this time I focused on the kinds of browsers.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pylab as plt
get_ipython().run_line_magic('matplotlib', 'inline')
pd.set_option('display.max_rows', 500)
pd.get_option("display.max_columns",500)


# In[ ]:


folder_path = '../input/'
train_identity = pd.read_csv(f'{folder_path}train_identity.csv')
train_transaction = pd.read_csv(f'{folder_path}train_transaction.csv')


df = pd.merge(train_transaction, train_identity, on='TransactionID', how='left')


# In[ ]:


df.head()


# ## about id_31

# In[ ]:


pd.DataFrame(df.groupby("id_31").count().TransactionID)


# There are a lot of categories in id_31.
# 
# I try to classify them.

# In[ ]:


pd.DataFrame(df.groupby("id_31").count().TransactionID).plot()


# There are big difference between browsers about the total number of them.
# Therefore, I check on the browsers which were used for a lot of transaction

# In[ ]:


df["chrome"]=df['id_31'].str.contains('chrome')*1
##here, I cange bool data to numeric data by mutipling 1( * 1)


# In[ ]:


df["chrome"].head()


# In[ ]:


df["samsung_browser"] = df['id_31'].str.contains('samsung')*1


# In[ ]:


df["safari"] = df['id_31'].str.contains('safari')*1


# In[ ]:


df["opera"] = df['id_31'].str.contains('opera')*1


# In[ ]:


df["ie"] = df['id_31'].str.contains('ie')*1


# In[ ]:


df["google_browser"] = df['id_31'].str.contains('google')*1


# In[ ]:


df["firefox"] = df['id_31'].str.contains('firefox')*1


# In[ ]:


df["edge"] = df['id_31'].str.contains('edge')*1


# when it comes to Android browser, I will use different way because chrome has "chrome for android"

# In[ ]:


df["android_browser"] = df['id_31'].str.contains('android browser')*1
df["android_browser"] = df['id_31'].str.contains('android webview')*1
df["android_browser"] = df['id_31'].str.contains('Generic/Android')*1
df["android_browser"] = df['id_31'].str.contains('Generic/Android 7.0')*1


# # Let's check it!

# In[ ]:


df.groupby("chrome").mean().isFraud


# chrome browser user is more possibly fraud maker than people who do not use it.
# 
# This may be because chrome has a lot of extensions.

# In[ ]:


df.groupby("safari").mean().isFraud


# A lot of safari user are mac user.
# 
# mac user may be less possibly a fraud maker.

# In[ ]:


pd.get_option("display.max_columns",500)
pd.options.display.max_columns = None


# In[ ]:


df.head()


# In[ ]:


df.groupby("DeviceType").mean().isFraud


# In[ ]:


df.groupby("edge").mean().isFraud


# Using these colomns should be useful.

# # problem
# 

# In[ ]:


df.isFraud.mean()


# Supposing from the fraud mean rate of all data, transaction which has id_31 is more doubtful.
# 
# However, I do not know why.
# 
# If you have idea about it, please comment on this kernel.
# 
# ## 8/12
# Gavin Cao gave us a good information.
# If you care about it, please look at the comment of this kernel.

# In[ ]:


df.groupby("isFraud").count().TransactionID


# In[ ]:


df.shape


# In[ ]:


pd.DataFrame(df["id_31"]).info( all,null_counts=True)


# # Conclusion

# The browser's classified columns tell us a lot of information.
# 
# However, there is something which is uncertained.
# 
# We should discuss and exmaine a lot about browsers.
