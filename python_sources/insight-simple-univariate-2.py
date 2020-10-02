#!/usr/bin/env python
# coding: utf-8

# In[ ]:


## import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
import seaborn as sns
#Ignore annoying warning from sklearn and seaborn
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn

#other libraiaries
import os
import copy
from collections import defaultdict
from collections import Counter
from sklearn import metrics
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
import re
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


train = pd.read_csv('/kaggle/input/training_v2.csv');train.head()


# In[ ]:


train.icu_type.value_counts().to_frame()


# In[ ]:


h = train[(train['gender'] == 'M') & (train['icu_type'] == 'Med-Surg ICU')]
h.sample(10)


# In[ ]:


plt.figure(figsize=[10,8])
sns.countplot(x='icu_type', hue='hospital_death', data=train)


# ### From the plot above, we can infer that majority of the hospital deaths occur in the med-surg ICU. Therefore, we create a new feature that assigns 1 to med-surg ICU and assigns 0 to others.

# In[ ]:


plt.figure(figsize=[10,8])
sns.countplot(x='icu_type', hue='gender', data=train)


# Based on my hypothesis it was recorded that male died mostly in medical_surg_ICU till date; define a new function to isolate patient in 'medical_surg_ICU and are male' to a binary of 1 and those 'that are medical_surg_ICU and are female'

# In[ ]:


train.readmission_status.value_counts()


# ### drop readmission status: it is a redundancy
