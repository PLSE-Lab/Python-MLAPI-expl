#!/usr/bin/env python
# coding: utf-8

# Versions:
# 
# * v3 - First running version of the code. However, runtime is about 2 hours which will be too long for submission.
# * v4 - Reduced number of epochs to train
# * v5 - turn off internet connection for submission.

# # Pretraining For APTOS Blindness Detection
# 
# In this kernel, I use my dataset of cropped and resized images from the previous [retinopathy detection competition](https://www.kaggle.com/c/diabetic-retinopathy-detection) to train a ResNet50 model. I then take that model and train on this competition dataset. I am not sure if this will work (as the image distributions are different) but we will see!

# In[ ]:


import os
files = os.listdir('../input/diabetic-retinopathy-resized/resized_train_cropped/resized_train_cropped')
print('Number of files: ',len(files)) 


# In[ ]:


# Making pretrained weights work without needing to find the default filename
if not os.path.exists('/tmp/.cache/torch/checkpoints/'):
        os.makedirs('/tmp/.cache/torch/checkpoints/')
get_ipython().system("cp '../input/resnet50/resnet50.pth' '/tmp/.cache/torch/checkpoints/resnet50-19c8e357.pth'")


# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from fastai import *
from fastai.vision import *
import pandas as pd
import matplotlib.pyplot as plt


# In[ ]:


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

SEED = 999
seed_everything(SEED)


# In[ ]:


print('Make sure cuda is installed:', torch.cuda.is_available())
print('Make sure cudnn is enabled:', torch.backends.cudnn.enabled)


# # Reading data
# Here I am going to open the dataset with pandas and check distribution of labels.

# In[ ]:


base_image_dir = os.path.join('..', 'input/diabetic-retinopathy-resized')
df = pd.read_csv(os.path.join(base_image_dir, 'trainLabels_cropped.csv'))
df['path'] = df['image'].map(lambda x: os.path.join(base_image_dir,'resized_train_cropped/resized_train_cropped','{}.jpeg'.format(x)))
df = df.drop(columns=['image'])
df = df.sample(frac=1).reset_index(drop=True) #shuffle dataframe
df.head(10)


# The dataset is highly imbalanced, with many samples with no disease:

# In[ ]:


df['level'].hist(figsize = (10, 5))


# In[ ]:


bs =16 #smaller batch size is better for training, but may take longer
sz=512


# Here, I load the dataset into the `ImageItemList` class provided by `fastai`. The fastai library also implements various transforms for data augmentation to improve training. While there are some defaults that I leave intact, I add vertical flipping (`do_flip=True`) as this has been commonly used for this particular problem.

# In[ ]:


tfms = get_transforms(do_flip=True,flip_vert=True,max_rotate=360,max_warp=0,max_zoom=1.1,max_lighting=0.1,p_lighting=0.5)
src = (ImageList.from_df(df=df,path='./',cols='path') #get dataset from dataset
        .split_by_rand_pct(0.2) #Splitting the dataset
        .label_from_df(cols='level') #obtain labels from the level column
      )
data= (src.transform(tfms,size=sz,resize_method=ResizeMethod.SQUISH,padding_mode='zeros') #Data augmentation
        .databunch(bs=bs,num_workers=4) #DataBunch
        .normalize(imagenet_stats) #Normalize     
       )


# In[ ]:


data.show_batch(rows=3, figsize=(7,6))


# In[ ]:


print(data.classes)
len(data.classes),data.c


# # Training on previous dataset

# **Training:**
# 
# We use transfer learning, where we retrain the last layers of a pretrained neural network. I use the ResNet50 architecture trained on the ImageNet dataset, which has been commonly used for pre-training applications in computer vision. Fastai makes it quite simple to create a model and train:

# In[ ]:


import torchvision
from fastai.metrics import *
from fastai.callbacks import *
learn = cnn_learner(data, models.resnet50, wd = 1e-5, metrics = [accuracy,KappaScore(weights='quadratic')],callback_fns=[partial(CSVLogger,append=True)])
learn.unfreeze()


# In[ ]:


learn.lr_find()
learn.recorder.plot(suggestion=True)


# In[ ]:


learn.fit_one_cycle(2, max_lr=slice(1e-6,4e-2))


# In[ ]:


learn.recorder.plot_losses()


# In[ ]:


learn.recorder.plot_metrics()


# In[ ]:


learn.save('prev-dataset')


# # Checking results on previous dataset

# We look at our predictions and make a confusion matrix.

# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)


# In[ ]:


interp.plot_confusion_matrix(figsize=(12,12), dpi=60)


# # Loading APTOS data
# 
# Now let's move on to the APTOS competition dataset. Here I am going to open the dataset with pandas, check distribution of labels.
# 

# In[ ]:


base_image_dir = os.path.join('..', 'input/aptos2019-blindness-detection/')
train_dir = os.path.join(base_image_dir,'train_images/')
df = pd.read_csv(os.path.join(base_image_dir, 'train.csv'))
df['path'] = df['id_code'].map(lambda x: os.path.join(train_dir,'{}.png'.format(x)))
df = df.drop(columns=['id_code'])
df = df.sample(frac=1).reset_index(drop=True) #shuffle dataframe
df.head(10)


# Note that the distribution of images are slightly different:

# In[ ]:


df['diagnosis'].hist(figsize = (10, 5))


# In[ ]:


bs = 64 #smaller batch size is better for training, but may take longer
sz=224


# In[ ]:


tfms = get_transforms(do_flip=True,flip_vert=True,max_rotate=360,max_warp=0,max_zoom=1.1,max_lighting=0.1,p_lighting=0.5)
src = (ImageList.from_df(df=df,path='./',cols='path') #get dataset from dataset
        .split_by_rand_pct(0.2) #Splitting the dataset
        .label_from_df(cols='diagnosis') #obtain labels from the level column
      )
data= (src.transform(tfms,size=sz,resize_method=ResizeMethod.SQUISH,padding_mode='zeros') #Data augmentation
        .databunch(bs=bs,num_workers=4) #DataBunch
        .normalize(imagenet_stats) #Normalize     
       )


# # Training
# 
# Time to train on our competition dataset:

# In[ ]:


learn = cnn_learner(data, base_arch=models.resnet50, metrics = [KappaScore(weights='quadratic')],callback_fns=[partial(CSVLogger,append=True)])
learn.load('prev-dataset')


# In[ ]:


learn.lr_find()
learn.recorder.plot(suggestion=True)


# In[ ]:


learn.fit_one_cycle(1,max_lr = 1e-2)


# In[ ]:


learn.recorder.plot_losses()
learn.recorder.plot_metrics()


# In[ ]:


learn.unfreeze()
learn.lr_find()
learn.recorder.plot(suggestion=True)


# In[ ]:


learn.fit_one_cycle(3,max_lr = slice(1e-6,1e-3))


# In[ ]:


learn.recorder.plot_losses()
learn.recorder.plot_metrics()


# For both datasets, we should train longer, but because of time constraints for submitting, I have limited training time. I will create a two-part kernel with longer training later.

# # Create Submission
# 
# Now that we have our model working on the APTOS dataset, we can create a submission and see how this model fares.

# In[ ]:


sample_df = pd.read_csv('../input/aptos2019-blindness-detection/sample_submission.csv')
sample_df.head()


# In[ ]:


learn.data.add_test(ImageList.from_df(sample_df,'../input/aptos2019-blindness-detection',folder='test_images',suffix='.png'))


# In[ ]:


preds,y = learn.get_preds(ds_type=DatasetType.Test)


# In[ ]:


sample_df.diagnosis = preds.argmax(dim=-1).numpy().astype(int)
sample_df.head()


# In[ ]:


sample_df.to_csv('submission.csv',index=False)

