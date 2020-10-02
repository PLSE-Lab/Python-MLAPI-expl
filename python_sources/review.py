#!/usr/bin/env python
# coding: utf-8

# In[ ]:


f=open(file="../input/movies.txt",mode='r',encoding="latin-1")


# In[ ]:


f.seek(0)
data=f.read(4000000000)


# In[ ]:


import gc
len(data)
del f
gc.collect()


# In[ ]:


new_data=data.split("\n")
del data
gc.collect()


# In[ ]:


sumary="review/text: "
sum_len=len(sumary)
f_new=open("review.txt",'w')
for line in new_data:
        if line.startswith(sumary):
            str=line[sum_len:]
            f_new.write(str+"\n")            
f_new.close()


# In[ ]:


del new_data
gc.collect()


# In[ ]:




