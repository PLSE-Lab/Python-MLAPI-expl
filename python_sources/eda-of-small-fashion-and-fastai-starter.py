#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')
import os
from IPython.display import Image
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
from fastai import *
from fastai.vision import *


# In[ ]:


get_ipython().system('pwd')


# In[ ]:


path = "/kaggle/input/fashion_small/fashion_small"
base='/kaggle/input/fashion_small/fashion_small/resized_images/'
print(os.listdir(path))


# In[ ]:


df = pd.read_csv("/kaggle/input/fashion_small/fashion_small/styles.csv", error_bad_lines=False) ;
df.dropna(inplace=True)


# In[ ]:


l=[]
for i in df['id']:
    if not os.path.exists('/kaggle/input/fashion_small/fashion_small/resized_images/'+str(i) +".jpg"):
        l.append(i)
        df.drop(df[df.id == i].index, inplace=True)


# In[ ]:


df.head()


# In[ ]:


df.nunique()


# In[ ]:


cat_id='subCategory'


# In[ ]:


dff=df.groupby(cat_id)
#gdf=list(dff)


# In[ ]:


nl=dff.count().sort_values(by='id')['id'].reset_index()
nl.columns=[cat_id,'count']
nl


# In[ ]:


nll=nl[nl['count']>=5][cat_id]


# In[ ]:


df=df[df[cat_id].isin(nll)]


# In[ ]:


labels = df[cat_id]
from collections import Counter, defaultdict
counts = defaultdict(int)
for l in labels:
     counts[l] += 1

counts_df = pd.DataFrame.from_dict(counts, orient='index')
counts_df.columns = ['count']
counts_df.sort_values('count', ascending=False, inplace=True)

fig, ax = plt.subplots()
ax = sns.barplot(x=counts_df.index, y=counts_df['count'], ax=ax)
fig.set_size_inches(20,10)
ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=-90);


# In[ ]:


def grid_plot(base,title,image_ids,sufix=''):
    try:
        len(image_ids)
        fn=image_ids
    except:
        fn=list(image_ids)
    columns = 5
    plt.figure(figsize=(20,4*(len(fn)//columns+1)))
    
    for i, image in enumerate(fn):
        plt.subplot(len(fn) / columns + 1, columns, i + 1,title=title+" : "+str(image)).axis('off')
        plt.imshow(mpimg.imread(base+str(image)+sufix))


# In[ ]:


dff=df.groupby(cat_id)


# In[ ]:


for i in dff:
    grid_plot(base,i[0],i[1]['id'].sample(5),'.jpg')


# In[ ]:


bs=128


# In[ ]:


get_ipython().system('nvidia-smi')


# In[ ]:


src=(ImageList.from_df(df, path=path, folder='resized_images', suffix='.jpg', cols=0)
                .split_by_rand_pct(0.2)
                .label_from_df(cols=3)
                .transform(get_transforms(), size=224)
                .databunch(bs=bs,num_workers=0)).normalize(imagenet_stats)


# In[ ]:


src.show_batch()


# In[ ]:


learn = create_cnn(
    src,
    models.resnet34,
    path='.',    
    metrics=accuracy, 
    ps=0.5
)


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot(skip_end=5)


# In[ ]:


learn.fit_one_cycle(3, 1e-2)


# In[ ]:


learn.save('freeze_1')


# In[ ]:


learn.recorder.plot_losses()


# In[ ]:


learn.unfreeze()
learn.fit_one_cycle(5, max_lr=slice(1e-4,1e-3))


# In[ ]:


learn.recorder.plot_losses()


# In[ ]:


learn.save('unfreeze')


# In[ ]:


output, target = learn.get_preds(ds_type=DatasetType.Valid)
fn=src.valid_ds.items


# In[ ]:


cls=src.classes


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()

len(src.valid_ds)==len(losses)==len(idxs)


# In[ ]:


interp.plot_top_losses(16, figsize=(15,11),heatmap=False)


# In[ ]:


interp.plot_top_losses(16, figsize=(15,11),heatmap=True)


# In[ ]:


interp.plot_confusion_matrix(figsize=(15,15), dpi=60)


# In[ ]:


interp.most_confused(min_val=2)

