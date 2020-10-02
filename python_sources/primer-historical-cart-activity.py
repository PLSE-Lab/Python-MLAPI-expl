#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from pandas import read_csv
data = read_csv("../input/SalesKaggle3.csv")


# In[ ]:


data.head()


# In[ ]:


features = data.columns


# In[ ]:


def get_dtypes(data,features):
    output = {}
    for f in features:
        dtype = str(data[f].dtype)
        if dtype not in output.keys(): output[dtype] = [f]
        else: output[dtype] += [f]
    return output

def show_uniques(data,features):
    for f in features:
        if len(data[f].unique()) < 30:
            print("%s: %s" % (f,data[f].unique()))
        else:
            print("%s: count(%s)" % (f,len(data[f].unique())))

def show_all_uniques(data,features):
    dtypes = get_dtypes(data,features)
    for key in dtypes.keys():
        print(key + "\n")
        show_uniques(data,dtypes[key])
        print()


# In[ ]:


show_all_uniques(data,features)

