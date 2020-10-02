#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')
import os
import pandas as pd
from fastai import *
from fastai.vision import *


# In[ ]:


get_ipython().system('pwd')


# In[ ]:


path = "/kaggle/input/fashion_small/fashion_small"
print(os.listdir(path))


# In[ ]:


df = pd.read_csv("/kaggle/input/fashion_small/fashion_small/styles.csv", error_bad_lines=False);


# In[ ]:


df.head()


# In[ ]:


l=[]
for i in df['id']:
    if not os.path.exists('/kaggle/input/fashion_small/fashion_small/resized_images/'+str(i) +".jpg"):
        l.append(i)
        df.drop(df[df.id == i].index, inplace=True)


# In[ ]:


bs=64


# In[ ]:


get_ipython().system('nvidia-smi')


# In[ ]:


src=(ImageList.from_df(df, path=path, folder='resized_images', suffix='.jpg', cols=0)
                .split_by_rand_pct(0.2)
                .label_from_df( cols=3)
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


learn.fit_one_cycle(5, 1e-2)


# In[ ]:


learn.save('freeze_1')


# In[ ]:


learn.recorder.plot_losses()


# In[ ]:


learn.unfreeze()
learn.fit_one_cycle(6, max_lr=slice(1e-4,1e-3))


# In[ ]:


learn.recorder.plot_losses()


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()

len(src.valid_ds)==len(losses)==len(idxs)


# In[ ]:


len(src.classes)


# In[ ]:


doc(interp.plot_top_losses)


# In[ ]:


interp.plot_top_losses(9, figsize=(15,11),heatmap=False)


# In[ ]:


interp.plot_confusion_matrix(figsize=(12,12), dpi=60)


# In[ ]:


interp.most_confused(min_val=2)


# In[ ]:


learn.save('/kaggle/working/unfreeze')


# In[ ]:




