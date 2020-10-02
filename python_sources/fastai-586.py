#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# show install info
#import fastai.utils.collect_env
#fastai.utils.collect_env.show_install() # 1.0.39


# In[ ]:


###### https://github.com/wkentaro/gdown
get_ipython().system('pip install gdown')
import gdown
url1 = 'https://drive.google.com/uc?id=1XoKn9-i7nfuqg3vAs96pQL3MtXhdY9hJ'
output1 = 'data.csv'
url2 = 'https://drive.google.com/uc?id=1-AB5GOy8isjh574gurgxxq3A7j1Bayex'
output2 = 'labels.csv'
gdown.download(url1, output1, quiet=False) 
gdown.download(url2, output2, quiet=False)
# https://drive.google.com/file/d/1XoKn9-i7nfuqg3vAs96pQL3MtXhdY9hJ/view  X
# https://drive.google.com/file/d/1-AB5GOy8isjh574gurgxxq3A7j1Bayex/view Y


# In[ ]:


#ls -la


# In[ ]:


from fastai.tabular import *


# In[ ]:


df_data = pd.read_csv('data.csv',index_col=0)
df_data.head()


# In[ ]:


df_labels = pd.read_csv('labels.csv',index_col=None, header=None, names=['A', 'B'])
df_labels.head()


# In[ ]:


df_data['label'] = df_labels['B'].values
df_data.head()


# In[ ]:


df_data.tail()


# In[ ]:


df_data.info()


# In[ ]:


def is_cat(df,c, thresh):
    if len(df[c].unique()) / len(df) < thresh:
        return True
    return False
        


# In[ ]:


cats = [c for c in df_data.columns if is_cat(df_data,c,0.1) and not c in ['label']]
len(cats)


# In[ ]:


cont_names = [c for c in df_data.columns if not (c in cats) and not c in ['label']]
len(cont_names)


# In[ ]:


dep_var = 'label'
cat_names = cats
procs = [FillMissing, Categorify, Normalize]
path = Path('.')
path


# In[ ]:


data = (TabularList.from_df(df_data, path=path, cat_names=cat_names, cont_names=cont_names, procs=procs)
                           .split_by_idx(list(range(1000,1200)))
                           .label_from_df(cols=dep_var)
                           #.add_test(test)
                           .databunch())


# In[ ]:


data.show_batch(rows=5, ds_type=DatasetType.Valid) 


# In[ ]:


learn = tabular_learner(data, layers=[200,100], metrics=accuracy)


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.fit(2, 8e-4)

