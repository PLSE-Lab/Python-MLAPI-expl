#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import seaborn as sns
sns.set(style="whitegrid", color_codes=True)


from wordcloud import WordCloud, STOPWORDS

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np # linear algebra
import matplotlib 
import matplotlib.pyplot as plt
import sklearn
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt 
plt.rcParams["figure.figsize"] = [16, 12]
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
filenames = check_output(["ls", "../input"]).decode("utf8").strip()
# helpful character encoding module
import chardet

# set seed for reproducibility
np.random.seed(0)


# In[10]:


df = pd.read_csv('../input/Health_AnimalBites.csv')


# In[3]:


df.dtypes


# In[19]:


df.head()


# In[18]:


df.head_sent_date = pd.to_datetime(df.head_sent_date, infer_datetime_format = True, errors = 'coerce')


# In[17]:


df.release_date = pd.to_datetime(df.release_date, infer_datetime_format = True, errors = 'coerce')


# In[16]:


df.quarantine_date = pd.to_datetime(df.quarantine_date, infer_datetime_format = True, errors = 'coerce')


# In[15]:


df.vaccination_date = pd.to_datetime(df.vaccination_date, infer_datetime_format = True, errors = 'coerce')


# In[13]:


df.bite_date = pd.to_datetime(df.bite_date, infer_datetime_format = True, errors = 'coerce')


# In[ ]:





# In[20]:


df.describe(include = 'O').transpose()


# In[21]:


df.describe(exclude = 'O').transpose()


# In[29]:


df.SpeciesIDDesc.value_counts().plot.bar()


# In[28]:


df.GenderIDDesc.value_counts().plot.bar()


# Want to know if being a female vs male <-> cat bite vs dog bite

# In[24]:


df.columns


# In[27]:


df = df[df.SpeciesIDDesc.isin(['DOG','CAT']) & df.GenderIDDesc.isin(['MALE','FEMALE'])]


# In[30]:


# with help from https://www.kaggle.com/omarayman/chi-square-test-in-python

cont = pd.crosstab(df["SpeciesIDDesc"],df["GenderIDDesc"])
    


# In[31]:


cont


# In[32]:


import scipy
scipy.stats.chi2_contingency(cont)


# For both males and females, dog bites are more common. For male the difference is more pronounced. We could say men tend to be biten by dogs rather than cats, compared to women. 
# 2.1133120633757655e-11 is small number, so we can reject null hypothesis saying that gender does not play a role. Conclusion: gender does play a role. 
