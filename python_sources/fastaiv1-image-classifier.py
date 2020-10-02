#!/usr/bin/env python
# coding: utf-8

# This is a simple kernel you can use as a base experiment for your first submission, implemented with the fastAI v1 library, by [fast.ai](https://www.fast.ai/).  
# It is based on the [first lesson](https://nbviewer.jupyter.org/github/fastai/course-v3/blob/master/nbs/dl1/lesson1-pets.ipynb) of the course-v3, which is explained in [this](https://course.fast.ai/videos/?lesson=1) video.  
# If you are interested in other edited FastAI ipynb, you can find another one here:
# * [fastaiv1 Collaborative Filtering](https://www.kaggle.com/gianfa/fastaiv1-collaborative-filtering/)

# In[ ]:


# %reload_ext autoreload
# %autoreload 2
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Main imports

# In[ ]:


import torch
import cv2
import matplotlib.pyplot as plt
import PIL
print(PIL.PILLOW_VERSION)


# In[ ]:


from fastai.vision import *
from fastai.metrics import error_rate
import fastai
fastai.__version__


# In[ ]:


import os
import pandas as pd


# ## Data Retrieving

# In[ ]:


os.listdir('../input/')


# In[ ]:


train_folder = '../input/train/train/'
test_folder = '../input/test1/test1/'
print(os.listdir(train_folder)[:10])
print(os.listdir(test_folder)[:10])


# ## Data Loading
# It is very fast to do with fastAI

# Take the image names and split them by regex

# In[ ]:


np.random.seed(2)
fnames = get_image_files(train_folder)
print(fnames[:5])
pat = re.compile(r'(cat|dog)\.\d+\.jpg') # we specify a regex for finding cat or dog images


# In[ ]:


sz = 64
bs = 64
data = ImageDataBunch.from_name_re(
                                train_folder,
                                fnames,
                                pat,
                                ds_tfms=get_transforms(),
                                size=sz, bs=bs,
                                valid_pct = 0.25,
                                num_workers = 0, # for code safety on kaggle
).normalize(imagenet_stats)
data


# In[ ]:


data.classes


# In[ ]:


data.show_batch(rows=4, figsize=(7,6))


# ## Modeling and Training

# In[ ]:


learn = cnn_learner(
    data,
    models.resnet34,
    metrics=error_rate,
    model_dir="/tmp/model/"
)


# In[ ]:


learn.data


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(1, max_lr=slice(2e-3))


# You can see that after a single cycle it's able to reach around 0.03 of valid_loss, great job fasta.ai!

# In[ ]:


learn.unfreeze()
learn.fit_one_cycle(4)


# ## Results

# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)
losses,idxs = interp.top_losses()
print(len(data.valid_ds)==len(losses)==len(idxs))


# In[ ]:


interp.plot_top_losses(9, figsize=(15,11))


# In[ ]:


interp.plot_confusion_matrix(figsize=(5,5), dpi=100)


# In[ ]:


mc = interp.most_confused(min_val=2)
mcc = [x[0] for x in mc[:5]]
mcc


# In[ ]:


train = pd.DataFrame(os.listdir(train_folder))


# In[ ]:


a = ['0']
train.sample(n=10, random_state=1)


# ## Prepare submission

# In[ ]:





# In[ ]:


item1_path = data.items[100]
print(item1_path)
item1 = data.open( item1_path )
item1


# In[ ]:


pred_class, pred_idx, outputs = learn.predict(item1)
probs = torch.nn.functional.softmax(np.log(outputs), dim=0)
print(pred_class)
print(probs)


# Now the **Test Set**

# In[ ]:


data_test = ImageList.from_folder(test_folder).split_none().label_empty()
data_test


# In[ ]:


dst = data_test.train.x[:20]
dst


# Now add _dst_ to the learner dataBunch

# In[ ]:


learn.data


# In[ ]:


data.add_test(items=dst)
data


# In[ ]:


learn.data = data
learn.data


# In[ ]:


pred_probs, pred_class = learn.get_preds(ds_type=DatasetType.Test)


# In[ ]:


print(pred_probs)
print(pred_class)
print((pred_probs.numpy()[:,0]>0.5)+0)


# In[ ]:


df = pd.DataFrame(os.listdir(test_folder))
print(len(df))
df.head()


# In[ ]:


img_idx = 8
print(pred_class[img_idx])
data_test.train.x[img_idx]
plt.imshow(plt.imread(test_folder+df[0].iloc[img_idx]))
# test_folder+df[0]


# In[ ]:





# In[ ]:



submission_data = [ids, pred_class]

df = pd.DataFrame(submission_data).T
df.columns = ['id','label']
df.head()


# In[ ]:


df.to_csv('./kaggle_catsdogs.csv', index=False)
print( os.path.exists('./kaggle_catsdogs.csv') )


# In[ ]:


get_ipython().system('rm -rf dogscats')


# In[ ]:




