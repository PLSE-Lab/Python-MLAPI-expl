#!/usr/bin/env python
# coding: utf-8

# In[ ]:


file="../input/wp_2gram.txt"
import pandas as pd
import gc


# In[ ]:


f=open(file, 'r', encoding='latin-1')
#f.seek(0)
data = f.read()
f.close()
del f
gc.collect()


# In[ ]:


no_word=len(data)
print("no of word in file :",no_word)
x=data.split("\n")
del data
gc.collect()


# In[ ]:


length=len(x)
print("no of list",length)


# In[ ]:


x_new=x[50000000:]
del x
gc.collect()


# In[ ]:


dist = dict()
for line in x_new:
    y= line.split("\t")
    try:
        dist[' '.join(y[1:])] = int(y[0])
    except:
        print("ignore")


# In[ ]:


del x_new
gc.collect()


# In[ ]:


distpd_5= pd.DataFrame(list(dist.items()), columns=['bigram', 'frequency'])


# In[ ]:


del dist
gc.collect()


# In[ ]:


distpd_5.to_csv("2_gram_9.csv")


# In[ ]:




