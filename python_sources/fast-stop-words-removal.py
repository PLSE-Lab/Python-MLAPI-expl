#!/usr/bin/env python
# coding: utf-8

# # Fast Stop Words Removal

# ## Introduction
# In this notebook, we will explore six methods of removing stop words and compare their runtime.
# 
# 1. Using list
# 2. Using cached list 
# 3. Using cached numpy array
# 4. Using cached pandas series
# 5. Using regular expression 
# 6. Using cashed set (best method)

# ## Import Libraries 

# In[ ]:


from nltk.corpus import stopwords
import re
import numpy as np
import pandas as pd 
import timeit


# In[ ]:


time = pd.Series()


# ## 1. List

# In[ ]:


def testList():
    text = 'hello bye the the hi'
    text = ' '.join([word for word in text.split() if word not in stopwords.words("english")])
    
start = timeit.default_timer()
for i in range(10000):
    testList()
stop = timeit.default_timer()
print('Time: ', stop - start)  
time['list'] = stop - start


# ## 2. Cached list

# In[ ]:


cache = stopwords.words("english")
def testCachedList():
    text = 'hello bye the the hi'
    text = ' '.join([word for word in text.split() if word not in cache])
    
start = timeit.default_timer()
for i in range(10000):
    testCachedList()
stop = timeit.default_timer()
print('Time: ', stop - start)  
time['cashed list'] = stop - start


# ## 3. Numpy array

# In[ ]:


cache = np.array(stopwords.words("english"))
def testNumpy():
    text = 'hello bye the the hi'
    text = ' '.join([word for word in text.split() if word not in cache])
    
start = timeit.default_timer()
for i in range(10000):
    testNumpy()
stop = timeit.default_timer()
print('Time: ', stop - start)  
time['numpy array'] = stop - start


# ## 4.Pandas series

# In[ ]:


cache = pd.Series(stopwords.words("english"))
def testPandas():
    text = 'hello bye the the hi'
    text = ' '.join([word for word in text.split() if word not in cache])
    
start = timeit.default_timer()
for i in range(10000):
    testPandas()
stop = timeit.default_timer()
print('Time: ', stop - start) 
time['pandas series'] = stop - start


# ## 5.Regular Expression

# In[ ]:


def testRE():
    text = 'hello bye the the hi'
    pattern = re.compile(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*')
    text = pattern.sub('', text)
    
start = timeit.default_timer()
for i in range(10000):
    testRE()
stop = timeit.default_timer()
print('Time: ', stop - start) 
time['re'] = stop - start 


# ## 6. Cashed Set

# In[ ]:


cache = set(stopwords.words("english"))
def testCachedSet():
    text = 'hello bye the the hi'
    text = ' '.join([word for word in text.split() if word not in cache])
    
start = timeit.default_timer()
for i in range(10000):
    testCachedSet()
stop = timeit.default_timer()
print('Time: ', stop - start)  
time['set'] = stop - start 


# ## Graph runtime

# In[ ]:


import seaborn as sns
sns.set(style="whitegrid")
ax = sns.barplot(x=time.index, y=time.values)
var = ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")


# In[ ]:


print(time)


# In[ ]:


7.954735/0.021772


# ## Conclusion
# You will speed up your program by 350 times just by using a cashed set!
