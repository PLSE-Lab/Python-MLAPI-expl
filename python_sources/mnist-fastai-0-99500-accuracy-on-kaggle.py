#!/usr/bin/env python
# coding: utf-8

# #  MNIST with FastAI 

# > Thanks to this notebook "MNIST Resnet_bigger+Mish_in_shortcut+transforms" that was used partially arrange the dataset
# 

# # 1. Imports  and file check

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# this import is to avoid warnings

# In[ ]:


import warnings # remove Warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")


# the one below is to reload the latest packages

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# FastAI imports

# In[ ]:


from fastai.vision import *
from fastai.metrics import *


# Standard imports for ML

# In[ ]:


import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')


# # 2. Quick exam of the dataset provided

# In[ ]:


path = '/kaggle/input/digit-recognizer/'


# This is the path were you find the datasets in Kaggle

# In[ ]:


df_train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
df_train.head()


# quick look of train and test dataset, the only difference is the label column on the train dataset.

# In[ ]:


df_test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
df_test.head()


# # 3. Formatting the data for ML

# Class created in the ["MNIST Resnet_bigger+Mish_in_shortcut+transforms"](https://www.kaggle.com/rincewind007/mnist-resnet-bigger-mish-in-shortcut-transforms/comments) to format the data for FastAI models

# In[ ]:


class CustomImageList(ImageList):
    def open(self, fn):
        if(fn.size == 785):
            fn = fn[1:]
        img = fn.reshape(28,28)
        img = np.stack((img,)*1, axis=-1)
        return Image(pil2tensor(img, dtype=np.float32))
    
    @classmethod
    def from_csv_custom(cls, path:PathOrStr, csv_name:str, imgIdx:int=1, header:str='infer', **kwargs)->'ItemList': 
        df = pd.read_csv(Path(path)/csv_name, header=header)
        res = super().from_df(df, path=path, cols=0, **kwargs)
        
        res.items = df.iloc[:,imgIdx:].apply(lambda x: x.values / 255.0, axis=1).values
        
        return res
    
    @classmethod
    def from_csv_custom_test(cls, path:PathOrStr, csv_name:str, imgIdx:int=1, header:str='infer', **kwargs)->'ItemList': 
        df = pd.read_csv(Path(path)/csv_name, header=header)
        res = super().from_df(df, path=path, cols=0, **kwargs)
        
        res.items = df.iloc[:,imgIdx:].apply(lambda x: x.values / 255.0, axis=1).values
        print(res)
        return res
    
    
    
    @classmethod
    def from_df_custom(cls, path:PathOrStr, df:DataFrame, imgIdx:int=1, header:str='', **kwargs)->'ItemList': 
        res = super().from_df(df, path=path, cols=0, **kwargs)
        
        res.items = df.iloc[:,imgIdx:].apply(lambda x: x.values / 255.0, axis=1).values
        
        return res


# In[ ]:


test = CustomImageList.from_csv_custom_test(path=path, csv_name='test.csv', imgIdx=0)


# The test dataset contains 28000 items 1,28,28

# In[ ]:


data = (CustomImageList.from_csv_custom(path=path, csv_name='train.csv', imgIdx=1)
                .split_by_rand_pct(.02)
                .label_from_df(cols='label') #cols='label'
                .add_test(test, label=0)
                .transform(get_transforms(do_flip=False,max_rotate=15,max_warp=0.4))
                .databunch(bs=128, num_workers=0)
                .normalize(imagenet_stats))


# We created our data for the training

# In[ ]:


data.train_ds


# the train dataset contains 41160 images

# In[ ]:


data.valid_ds


# The validation dataset contains the 2% of the training data, in total 840 images

# We now created the databunch of 128 images to use in our model and we quickly visually examine the data

# In[ ]:


data.show_batch(rows=5, figsize=(7,7))


# # 4. Create the model

# We create our learner now using the data that we just formatted and chosing the model.
# Here FastAi is very flexible, I personally experimented with:
# * resnet18 
# * resnet34
# * resnet50
# * resnet101
# * resnet152
# 
# You can find more info about the models [here](https://docs.fast.ai/vision.models.html#Computer-Vision-models-zoo)
# 
# I encourage you to try and change this values in the model selection and analyze the results, you will learn a lot

# In order to keep it simple I started with the one who should give the best results

# In[ ]:


model_selected = models.resnet152


# create the learner

# In[ ]:


learn = cnn_learner(data, model_selected, 
                    metrics=error_rate,callback_fns=ShowGraph,model_dir='/kaggle/working').mixup()


# classic FastAI [learning rate finder](https://docs.fast.ai/basic_train.html#Model-fitting-methods)
# this is the link to the [scientific paper](https://arxiv.org/abs/1506.01186) if you want to go deeper

# In[ ]:


learn.lr_find()


# plot of the learning rate with [suggested best lr](https://docs.fast.ai/callbacks.lr_finder.html#Suggested-LR) to adopt

# In[ ]:


learn.recorder.plot(suggestion=True)


# set the lr suggested as lr for training

# In[ ]:


min_grad_lr = learn.recorder.min_grad_lr
min_grad_lr


# # 5. Fit the model

# Fit the model with the [one cycle policy](https://docs.fast.ai/callbacks.one_cycle.html#Training-with-the-1cycle-policy)
# 
# here is the scientific [paper link](https://arxiv.org/abs/1803.09820) if you want to go deeper

# I suggest to experiment with the epoch number, for me 30 epochs works pretty well

# In[ ]:


learn.fit_one_cycle(30,min_grad_lr)


# remember to save your job ... of course I learned the hard way

# In[ ]:


learn.save('r152-fit1')


# ### Fit with the full dataset

# Now we are going to train with the full dataset removing the split, so we are going to use all the 42000 

# In[ ]:


data = (CustomImageList.from_csv_custom(path=path, csv_name='train.csv', imgIdx=1)
                #.split_by_rand_pct(.02) --- this split has been removed
                .split_none()# this line replaces the previous split
                .label_from_df(cols='label') #cols='label'
                .add_test(test, label=0)
                .transform(get_transforms(do_flip=False,max_rotate=15,max_warp=0.4))
                .databunch(bs=128, num_workers=0)
                .normalize(imagenet_stats))


# we replace the data in our learner with the new data

# In[ ]:


data.train_ds


# Now the data contains all the 42000 images of the training set

# we replace the data inside the model to retrain with those extra 840 samples

# In[ ]:


learn.data = data


# Fit for extra 10 epochs with the full dataset

# In[ ]:


learn.fit_one_cycle(10,min_grad_lr)


# and save again

# In[ ]:


learn.save('r152-fit2')


# ### Unfreeze

# [Unfreeze the whole model](https://docs.fast.ai/basic_train.html#Learner.unfreeze) in order to train all the layers

# In[ ]:


learn.unfreeze()


# Follow what we did before

# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot(suggestion=True)


# In[ ]:


min_grad_lr = learn.recorder.min_grad_lr
min_grad_lr


# Fit for an extra 10 epochs.

# In[ ]:


learn.fit_one_cycle(10,min_grad_lr)


# # 6. Predictions

# > In order to have a correct interpretation of out data we should compare it towards our validation dataset, BUT we also trained the validation dataset this time in order to achieve the maximum score on Kaggle, so we are forced to look at our model performance through something that we trained.
# > > For this reason we recreate the dataset with the command line below.

# In[ ]:


data = (CustomImageList.from_csv_custom(path=path, csv_name='train.csv', imgIdx=1)
                .split_by_rand_pct(.02)
                .label_from_df(cols='label') #cols='label'
                .add_test(test, label=0)
                .transform(get_transforms(do_flip=False,max_rotate=15,max_warp=0.4))
                .databunch(bs=128, num_workers=0)
                .normalize(imagenet_stats))
learn.data = data


# We create the predictions and the output file

# In[ ]:


# get the predictions
predictions, *_ = learn.get_preds(DatasetType.Test)
labels = np.argmax(predictions, 1)
# output to a file
submission_df = pd.DataFrame({'ImageId': list(range(1,len(labels)+1)), 'label': labels})
submission_df.to_csv(f'submission.csv', index=False)


# quick look at the submission format

# In[ ]:


submission_df


# Create the [classification interpretation](https://docs.fast.ai/vision.learner.html#ClassificationInterpretation) from FastAI

# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)


# and explore the [confusion matrix](https://docs.fast.ai/train.html#ClassificationInterpretation.plot_confusion_matrix)

# In[ ]:


interp.plot_confusion_matrix()


# for vision the [plot top losses](https://docs.fast.ai/vision.learner.html#_cl_int_plot_top_losses) is anyway my favourite

# In[ ]:


interp.plot_top_losses(9,cmap='gray')


# > This resnet18 scored 0.99014, wich is not bad.
# 
# > The best score was achieved with resnet152 with a score of 0.99600.

# If you were so patient to read up to here I hope you found some valuable information ...

# ... and if you liked it 

# # ***... please upvote ... Thank you!!! :)))***

# In[ ]:




