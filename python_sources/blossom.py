#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from fastai.vision import *
from fastai.metrics import error_rate


# In[ ]:


bs = 64

get_ipython().system("ls '../input/flower_data/flower_data/'")


# In[ ]:


path = Path('../input/flower_data/flower_data/')


# In[ ]:


path.ls()


# In[ ]:


img_size = 224
data = ImageDataBunch.from_folder(path, ds_tfms=get_transforms(), 
                                  valid='valid', size=img_size, bs = bs) .normalize(imagenet_stats)


# In[ ]:


import json

with open('../input/cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)


# In[ ]:


class_names = data.classes


# In[ ]:


for i in range(0,len(class_names)):
    class_names[i] = cat_to_name.get(class_names[i])
class_names[20]


# In[ ]:


data.classes


# In[ ]:


data.show_batch(rows=3, figsize=(8,7))


# In[ ]:


learn = cnn_learner(data, models.resnet34, metrics=accuracy, model_dir='/kaggle/working')


# In[ ]:


learn.model


# In[ ]:


learn.fit_one_cycle(5)


# In[ ]:


learn.save('stage-1')


# In[ ]:


## Results

interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)


# In[ ]:


interp.plot_top_losses(9, figsize=(15,11))


# In[ ]:


interp.plot_confusion_matrix(figsize=(12,12), dpi=60)


# In[ ]:


interp.most_confused(min_val=2)


# In[ ]:


learn.unfreeze()


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(5)


# 

# In[ ]:


learn.save('stage-2');


# In[ ]:


learn.load('stage-2');


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)


# In[ ]:


interp.most_confused(min_val=2)


# In[ ]:


interp.plot_confusion_matrix(figsize=(12,12),cmap='viridis', dpi=60)


# In[ ]:


learn.recorder.plot_losses()


# In[ ]:


learn.export('/kaggle/working/export.pkl')


# In[ ]:


newpath = '../input/hackathon-blossom-flower-classification/test set/'

test = ImageDataBunch.from_folder(newpath, ds_tfms=get_transforms(), 
                                  valid='valid', size=img_size, bs = bs) .normalize(imagenet_stats)
len(test)


# In[ ]:


learn = load_learner('/kaggle/working', test=test)

preds, _ = learn.get_preds(ds_type=DatasetType.Test)


# In[ ]:


thresh = 0.2
labelled_preds = [' '.join([learn.data.classes[i] for i,p in enumerate(pred) if p > thresh]) for pred in preds]


# In[ ]:


labelled_preds[:5]


# In[ ]:


fnames = [f.name[:-4] for f in learn.data.test_ds.items]


# In[ ]:


df = pd.DataFrame({'image_name':fnames, 'tags':labelled_preds}, columns=['image_name', 'tags'])
df.head(10)

