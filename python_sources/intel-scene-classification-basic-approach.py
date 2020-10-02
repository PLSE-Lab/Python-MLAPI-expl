#!/usr/bin/env python
# coding: utf-8

# In[ ]:


path = "../input/scene_classification/scene_classification/train/"


# In[ ]:


from fastai import *
from fastai.vision import *


# In[ ]:


bs = 256


# In[ ]:


df = pd.read_csv('../input/scene_classification/scene_classification/train.csv')
df.head()


# In[ ]:


tfms = get_transforms(flip_vert=False,max_zoom=1.0,max_warp=0)
data = (ImageList.from_csv(path, csv_name = '../train.csv') 
        .split_by_rand_pct()              
        .label_from_df()            
        .add_test_folder(test_folder = '../test')              
        .transform(tfms, size=256)
        .databunch(num_workers=0))


# In[ ]:


data.show_batch(rows=3, figsize=(8,10))


# In[ ]:


print(data.classes)


# In[ ]:


learn = create_cnn(data, models.resnet34, metrics=[error_rate, accuracy], model_dir="/tmp/model/")


# In[ ]:


learn.fit_one_cycle(4)


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)


# In[ ]:


interp.plot_top_losses(9, figsize=(15,11))


# In[ ]:


interp.plot_confusion_matrix(figsize=(7,7), dpi=60)


# In[ ]:


learn.save('/kaggle/working/stage-1')


# In[ ]:


learn.unfreeze()


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(1, max_lr=slice(1e-6, 1e-4))


# In[ ]:


learn.save('/kaggle/working/stage-2')


# In[ ]:


# preds,_ = learn.get_preds(ds_type=DatasetType.Test)


# In[ ]:


# labelled_preds = []
# for pred in preds:
#     labelled_preds.append(int(np.argmax(pred)))
    
# # labelled_preds[0:10]
# len(labelled_preds)


# In[ ]:


# import os
# filenames = os.listdir('../input/scene_classification/scene_classification/test/')


# In[ ]:


# len(filenames) == len(labelled_preds)


# In[ ]:


# submission = pd.DataFrame(
#     {'image_name': filenames,
#      'label': labelled_preds,
#     })


# In[ ]:


# submission.to_csv('first_submission.csv')


# In[ ]:


# download the notebook before committing

