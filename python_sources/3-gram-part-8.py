#!/usr/bin/env python
# coding: utf-8

# In[ ]:


file="../input/wp_3gram.txt"
import pandas as pd
import gc


# In[ ]:


f=open(file, 'r', encoding='latin-1')
f.seek(7000000000)
data = f.read(1000000000)
f.close()
del f
gc.collect()
no_word=len(data)
print("no of word in file :",no_word)


# In[ ]:



x=data.split("\n")
del data
gc.collect()


# In[ ]:


length=len(x)
print("no of list",length)


# In[ ]:


x_new=x
del x
gc.collect()


# In[ ]:


gram_value=3
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


distpd_1= pd.DataFrame(list(dist.items()), columns=['bigram', 'frequency'])


# In[ ]:


del dist
gc.collect()


# In[ ]:


distpd_1.to_csv("3_gram_1.csv")


# In[ ]:




