#!/usr/bin/env python
# coding: utf-8

# I recently came across this package called 'Featuretools' for performing Feature Engineering in an automated way. I thought I' d give it a try to see if it brings in any interesting new features to this competition. 
# 
# However, after waiting for so long for the process to complete on a single table, I searched it a little further. It seems that you can manually set the chunk size that the method reads (somewhat like batch in Neural Networks I guess) to improve running times. I was able to only partially improve time taken to complete the process and yet miles away than the respective pandas transformations. 
# 
# I display here my results for the Buro data set, taking under consideration only the first 50,000 rows and performing only a 'mean' aggregate on the data. I have counted elapsed time for a variety of chunk sizes and plotted their times. 
# 
# I wonder if I am missing something or have done something wrong. Elsewise, these times are forbidding of using the tool on the entire data set. 

# In[1]:


#Import libraries and data
import pandas as pd
import numpy as np
import featuretools as ft
import time
import matplotlib.pyplot as plt
import seaborn as sns

buro = pd.read_csv('../input/bureau.csv')


# In[2]:


#Only read of data set
buro = buro.iloc[:50000,:]

buro = buro.reset_index()


# In[3]:


#Create featuretool entities
es = ft.EntitySet(id="buro")

es = es.entity_from_dataframe(entity_id="buro",
                              dataframe=buro,
                              index="index",
                              #time_index="transaction_time",
                              #variable_types={"SK_ID_CURR": ft.variable_types.Categorical},
                              # "EDUCATION": ft.variable_types.Categorical,
                              # "MARRIAGE": ft.variable_types.Categorical,
                              #  }
                              )

es = es.normalize_entity(base_entity_id="buro",
                         new_entity_id="SK_ID_CURRENT",
                         index="SK_ID_CURR",
                         #additional_variables=["DAYS_CREDIT"]
                         )


# In[6]:


#Run 'Deep Feature Synthesis' and record times
chunk_size = []
time_sec=[]

for c in range(250,5250,250):
    start = time.time()

    chunk = c

    print('Creating new features...')
    feature_matrix, feature_defs = ft.dfs(entityset=es,
                                          target_entity="SK_ID_CURRENT",
                                          agg_primitives = ["mean"],
                                          max_depth=1,
                                          chunk_size=chunk)

    stop = time.time()

    #Print elapsed time
    time_elapsed = stop - start
    #print('Chunk size =', chunk_size, ', Time = ', time_elapsed)
    chunk_size.append(chunk)
    time_sec.append(time_elapsed)


# In[10]:


#Plot results
results = pd.DataFrame({'Chunk Size': chunk_size, 'Time in seconds': time_sec})

plt.figure()
sns.barplot('Chunk Size','Time in seconds', data = results)
plt.xticks(rotation=90)
plt.show()


# In[ ]:




