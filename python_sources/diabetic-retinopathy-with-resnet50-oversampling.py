#!/usr/bin/env python
# coding: utf-8

# # Diabetic Retinopathy 
# 
# Diabetic retinopathy (DR), also known as diabetic eye disease, is a medical condition in which damage occurs to the retina due to diabetes mellitus. It is a leading cause of blindness. Diabetic retinopathy affects up to 80 percent of those who have had diabetes for 20 years or more. Diabetic retinopathy often has no early warning signs. **Retinal (fundus) photography with manual interpretation is a widely accepted screening tool for diabetic retinopathy**, with performance that can exceed that of in-person dilated eye examinations. 
# 
# The below figure shows an example of a healthy patient and a patient with diabetic retinopathy as viewed by fundus photography ([source](https://www.biorxiv.org/content/biorxiv/early/2018/06/19/225508.full.pdf)):
# 
# ![image.png](data:image/jpeg;base64,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)
# 
# An automated tool for grading severity of diabetic retinopathy would be very useful for accerelating detection and treatment. Recently, there have been a number of attempts to utilize deep learning to diagnose DR and automatically grade diabetic retinopathy. This includes this [competition](https://kaggle.com/c/diabetic-retinopathy-detection) and [work by Google](https://ai.googleblog.com/2016/11/deep-learning-for-detection-of-diabetic.html). Even one deep-learning based system is [FDA approved](https://www.fda.gov/NewsEvents/Newsroom/PressAnnouncements/ucm604357.htm). 
# 
# Clearly, this dataset and deep learning problem is quite well-characterized. 
# 
# # A look at the data:
# 
# Data description from the competition:
# 
# >You are provided with a large set of high-resolution retina images taken under a variety of imaging conditions. A left and right field is provided for every subject. >Images are labeled with a subject id as well as either left or right (e.g. 1_left.jpeg is the left eye of patient id 1).
# >
# >A clinician has rated the presence of diabetic retinopathy in each image on a scale of 0 to 4, according to the following scale:
# >
# >0 - No DR
# >
# >1 - Mild
# >
# >2 - Moderate
# >
# >3 - Severe
# >
# >4 - Proliferative DR
# >
# >Your task is to create an automated analysis system capable of assigning a score based on this scale.
# 
# ...
# 
# > Like any real-world data set, you will encounter noise in both the images and labels. Images may contain artifacts, be out of focus, underexposed, or overexposed. A major aim of this competition is to develop robust algorithms that can function in the presence of noise and variation.
# 
# **A minor problem!**
# 
# Due to the large image file size, there are **only 1000 files with labels** and one csv file with the labels of all the images in the directory available in the kernel, as demonstrated below. The actual competition had on the order of 35,000 files so this is clearly **a very small subset of the data**. 
# In addition, this is a highly imbalanced dataset. 
# 
# Originally, I had aimed to create a model that would be close to the SOTA, but clearly, with only 1000 images that's not possible.
# 
# **AIM OF KERNEL:** I will utilize transfer learning, oversampling, and progressive resizing on this small, imbalanced dataset.
# Given that many real-world datasets are also small and imbalanced, it will be interesting to see how far these techniques will take us.

# In[ ]:


import os
files = os.listdir('../input')
print('trainLabels.csv' in files) #Is the labels csv in the directory?
print(len(files)) #There should be 1000 images + 1 csv file = 1001 files


# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


get_ipython().system('pip install fastai==1.0.42')


# In[ ]:


from fastai import *
from fastai.vision import *
import pandas as pd
import matplotlib.pyplot as plt


# In[ ]:


print('Make sure cuda is installed:', torch.cuda.is_available())
print('Make sure cudnn is enabled:', torch.backends.cudnn.enabled)


# # Reading data
# Here I am going to open the dataset with pandas, check distribution of labels, and oversample to reduce imbalance.

# In[ ]:


base_image_dir = os.path.join('..', 'input')
df = pd.read_csv(os.path.join(base_image_dir, 'trainLabels.csv'))
df['path'] = df['image'].map(lambda x: os.path.join(base_image_dir,'{}.jpeg'.format(x)))
df['exists'] = df['path'].map(os.path.exists) #Most of the files do not exist because this is a sample of the original dataset
df = df[df['exists']]
df = df.drop(columns=['image','exists'])
df = df.sample(frac=1).reset_index(drop=True) #shuffle dataframe
df.head(10)


# The dataset is highly imbalanced, with many samples for level 0, and very little for the rest of the levels. 

# In[ ]:


df['level'].hist(figsize = (10, 5))


# In[ ]:


df.pivot_table(index='level', aggfunc=len)


# This is a function to oversample the dataset (so some of the images of levels 1-4 are present multiple times in the dataset):

# In[ ]:


def balance_data(class_size,df):
    train_df = df.groupby(['level']).apply(lambda x: x.sample(class_size, replace = True)).reset_index(drop = True)
    train_df = train_df.sample(frac=1).reset_index(drop=True)
    print('New Data Size:', train_df.shape[0], 'Old Size:', df.shape[0])
    train_df['level'].hist(figsize = (10, 5))
    return train_df


# In[ ]:


from sklearn.model_selection import train_test_split
train_df, val_df = train_test_split(df,test_size=0.2) # Here we will perform an 80%/20% split of the dataset, with stratification to keep similar distribution in validation set


# In[ ]:


train_df['level'].hist(figsize = (10, 5))
len(val_df)


# In[ ]:


train_df.pivot_table(index='level', aggfunc=len)


# In[ ]:


train_df = balance_data(train_df.pivot_table(index='level', aggfunc=len).max().max(),train_df) # I will oversample such that all classes have the same number of images as the maximum
train_df['level'].hist(figsize = (10, 5))


# In[ ]:


df = pd.concat([train_df,val_df]) #beginning of this dataframe is the oversampled training set, end is the validation set
len(df)


# In[ ]:


from PIL import Image

im = Image.open(train_df['path'][1])
width, height = im.size
print(width,height) 


# The images are actually quite big. We will resize to a much smaller size and try to use progressive resizing to our advantage when dealing with such a small dataset.

# In[ ]:


bs = 16 #smaller batch size is better for training, but may take longer
sz=224


# Here, I load the dataset into the `ImageItemList` class provided by `fastai`. The fastai library also implements various transforms for data augmentation to improve training. While there are some defaults that I leave intact, I add vertical flipping (`do_flip=True`) and 360 deg. `max_rotate=360` as those have been commonly used for this particular problem.
# 
# Typically, one would use the `ImageDataBunch` class to load the dataset much easier, but since I needed to do some custom tasks for splitting and oversampling, I have to use this customized creation of the DataBunch.

# In[ ]:


tfms = get_transforms(do_flip=True,flip_vert=True,max_rotate=360,max_warp=0,max_zoom=1.1,max_lighting=0.1,p_lighting=0.5)
src = (ImageItemList.from_df(df=df,path='./',cols='path') #get dataset from dataset
        .split_by_idx(range(len(train_df)-1,len(df))) #Splitting the dataset
        .label_from_df(cols='level') #obtain labels from the level column
      )
data= (src.transform(tfms,size=sz) #Data augmentation
        .databunch(bs=bs,num_workers=0) #DataBunch
        .normalize(imagenet_stats) #Normalize
       )


# In[ ]:


data.show_batch(rows=3, figsize=(7,6))


# We can see that because of the center cropping and data augmentation, some of the images are cut off or have weird artifacts. This is probably because the fundus image was not 100% centered in the original data. For now, I will ignore these problems.

# In[ ]:


print(data.classes)
len(data.classes),data.c


# # Training (Transfer learning) 

# The Kaggle competition used the Cohen's quadratically weighted kappa so I have that here to compare. This is a better metric when dealing with imbalanced datasets like this one, and for measuring inter-rater agreement for categorical classification (the raters being the human-labeled dataset and the neural network predictions). Here is an implementation based on the scikit-learn's implementation, but converted to a pytorch tensor, as that is what fastai uses.

# In[ ]:


from sklearn.metrics import cohen_kappa_score
def quadratic_kappa(y_hat, y):
    return torch.tensor(cohen_kappa_score(torch.argmax(y_hat,1), y, weights='quadratic'),device='cuda:0')


# **Training:**
# 
# We use transfer learning, where we retrain the last layers of a pretrained neural network. I use the ResNet50 architecture trained on the ImageNet dataset, which has been commonly used for pre-training applications in computer vision. Fastai makes it quite simple to create a model and train:

# In[ ]:


import torchvision
learn = create_cnn(data, models.resnet50, metrics = [accuracy,quadratic_kappa])


# We use the learning-rate finder developed by Dr. Leslie Smith and implemented by the fastai team in their library:

# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# Here we can see that the loss decreases fastest around `lr=2e-3` so that is what we will use to train: 

# In[ ]:


learn.fit_one_cycle(4,max_lr = 2e-3)


# In[ ]:


learn.recorder.plot_losses()


# In[ ]:


learn.save('stage-1-224')


# The previous model only trained the last model. We can unfreeze the rest of the model, and train the rest of the model using discriminative learning rates. The first layers aren't changed as much, with lower learning rates, while the last layers are changed more, with higher learning rates. We use the learning rate finder again, and use a range of learning rates for different layers in the neural network.

# In[ ]:


learn.unfreeze()


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(4, max_lr=slice(1e-6,1e-4))


# In[ ]:


learn.recorder.plot_losses()


# In[ ]:


learn.save('stage-2-224')


# # Progressive resizing

# Progressive resizing is a technique developed by Jeremy Howard as part of the fast.ai class and for the [DAWNBench challenge](https://www.fast.ai/2018/08/10/fastai-diu-imagenet/). The idea is that we train with smaller images at the beginning, and retrain with larger images, which will have more information to learn from. This could be very helpful for dealing with small datasets, and I decided to give this a try. I only resize once due to kernel time limitations, but theoretically, we could continue to resize and see if that improves accuracy, especially since the retinal images are so big in size.

# In[ ]:


data = (src.transform(tfms,size=sz*2) #Data augmentation
        .databunch(bs=bs,num_workers=0) #DataBunch
        .normalize(imagenet_stats) #Normalize
       )


# In[ ]:


learn.data = data
data.train_ds[0][0].shape


# In[ ]:


learn.freeze()


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(4,max_lr=1e-3)


# In[ ]:


learn.recorder.plot_losses()


# In[ ]:


learn.save('stage-1-448')


# In[ ]:


learn.unfreeze()


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(4,max_lr=slice(1e-6,1e-4))


# In[ ]:


learn.recorder.plot_losses()


# In[ ]:


learn.save('stage-2-448')


# # Checking results

# We look at our predictions and make a confusion matrix.

# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)


# In[ ]:


interp.plot_top_losses(9, figsize=(15,11))


# In[ ]:


interp.plot_confusion_matrix(figsize=(12,12), dpi=60)


# Quite impressively, with only 1000 images, oversampling, and transfer learning, we acheive decent agreement with the correct labels! 

# # Future work:
# There are other methods for dealing with imbalanced datasets. The most common alternative is to use a **weighted loss function**, with the class weights dependent on the distribution of the dataset.
# 
# Some possible future experiments include:
# 1. Training the dataset without the healthy (0) class, then retrain with the healthy class.
# 2. Possibly using [mixup](https://forums.fast.ai/t/mixup-data-augmentation/22764) as a form of data augmentation. It would be interesting to see how well it would perform with medical data.
# 3. Utilize [rectangular images](https://www.fast.ai/2018/08/10/fastai-diu-imagenet/) instead of cropping. Most of the data has an aspect ratio of ~1.5, and by default, the `fastai` library center-crops squares to pass into the network. Note for this dataset, it may not provide significant advantage as the circular retinal images are already padded, and cropping them removes much of the padding. 
# 
# # Acknowledgements
# Thanks to the `fastai` library and [fast.ai course](https://course.fast.ai/). The techniques used here were based mainly on lesson 1 and lesson 3 of the course.