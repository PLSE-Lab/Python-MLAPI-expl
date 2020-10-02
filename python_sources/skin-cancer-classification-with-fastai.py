#!/usr/bin/env python
# coding: utf-8

# # Skin cancer classification with fastai library
# 
# Check out the awesome [fastai](https://docs.fast.ai) libary and the great [online courses](https://course.fast.ai/). 
# 
# The obtained accuracy with fastai is >90% using transfer learning with a pre-trained resnet50 and just a few lines of code!

# In[ ]:


#the data is deployed twice...
get_ipython().system('ls /kaggle/input/skin-cancer-malignant-vs-benign/')


# In[ ]:


from fastai.vision import *
import torchvision


# In[ ]:


path = Path('/kaggle/input/skin-cancer-malignant-vs-benign/data')
classes = ['malignant','benign']
ImageList.from_folder(path)


# Using the data_block api to create a databunch. The images in the test folder become the validation set.

# In[ ]:


tfms = get_transforms(do_flip=True,flip_vert=True)
data = (ImageList.from_folder(path) #Where to find the data? -> in path and its subfolders
        .split_by_folder('train','test')              #How to split in train/valid? -> use the folders
        .label_from_folder()            #How to label? -> depending on the folder of the filenames
        .transform(tfms, size=224)       #Data augmentation? -> use tfms with a size of 64
        .databunch(bs=32))


# In[ ]:


data


# In[ ]:


data.classes


# In[ ]:


data.show_batch(rows=3, figsize=(7,8))


# In[ ]:


data.classes, data.c, len(data.train_ds), len(data.valid_ds)


# ## Train model with transfer learning

# In[ ]:


# model loading implemented thanks to https://www.kaggle.com/faizu07/kannada-mnist-with-fastai
get_ipython().system('mkdir -p /tmp/.cache/torch/checkpoints')
get_ipython().system('cp /kaggle/input/fastai-pretrained-models/resnet50-19c8e357.pth /tmp/.cache/torch/checkpoints/resnet50-19c8e357.pth')
learn = cnn_learner(data, models.resnet50, metrics=[error_rate, accuracy], model_dir = Path('../kaggle/working'),path = Path("."),pretrained=True)


# For pretrained==True -> check out the trainable layers, currently only  the batchnorm parameters and the last layer weights can be trained!

# In[ ]:


learn.summary() 


# In[ ]:


learn.fit_one_cycle(4)


# In[ ]:


learn.save('stage-1')


# In[ ]:


learn.load('stage-1')
learn.unfreeze()


# In[ ]:


learn.lr_find(start_lr=1e-9, end_lr=1e-1)


# In[ ]:


learn.recorder.plot(suggestion=True)


# In[ ]:


learn.fit_one_cycle(6, max_lr=slice(1e-04,1e-05)) 


# In[ ]:


learn.save('stage-2')
#learn.export('skin_classifier.pkl')


# ## Interpretation

# In[ ]:


learn.load('stage-2');interp = ClassificationInterpretation.from_learner(learn)


# In[ ]:


interp.plot_confusion_matrix()


# In[ ]:


interp.plot_top_losses(12)


# In[ ]:




