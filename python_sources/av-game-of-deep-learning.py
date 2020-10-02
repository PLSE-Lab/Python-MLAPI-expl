#!/usr/bin/env python
# coding: utf-8

# In this competition, i will be using Kaggle platform for modelling (GPU and internet enabled). The libraries i will be using are fastai datablock api which is built on Pytorch. please visit [here](https://www.fast.ai//) for all the course materials.
# Thanks to `Jeremy Howard` and `Rachel Thomas`.
# 

# ### Load libraries and read the data

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# from sklearn.metrics import f1_score
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import seaborn as sns
import matplotlib.pyplot as plt
import os
import cv2
import glob
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# - lets import the fastai libraries 

# In[ ]:


from pathlib import Path
from fastai import *
from fastai.vision import *
import torch
from fastai.callbacks.hooks import *


# In[ ]:


## set the data folder
data_folder = Path("../input")


# In[ ]:


data_path = "../input/train/images/"
path = os.path.join(data_path , "*jpg")


# In[ ]:


files = glob.glob(path)
data=[]
for file in files:
    image = cv2.imread(file)
    data.append(image)


# In[ ]:


## read the csv data files
train_df = pd.read_csv('../input/train/train.csv')
test_df = pd.read_csv('../input/test_ApKoW4T.csv')
submit = pd.read_csv('../input/sample_submission_ns2btKE.csv')


# In[ ]:


train_df.shape, test_df.shape


# In[ ]:


train_df.head()


# In[ ]:


test_df.head()


# In[ ]:


# submit.head()


# In[ ]:


train_df.groupby('category').count()


# In[ ]:


sns.countplot(x='category' , data=train_df)


# - we have 5 categories as mentioned in the problem

# In[ ]:


train_images = data[:6252]
test_images= data[6252:]


# In[ ]:


## mapping the ship categories  
category = {'Cargo': 1, 
'Military': 2, 
'Carrier': 3, 
'Cruise': 4, 
'Tankers': 5}


# - we will plot pictures from all the classes to look at those cool ships

# In[ ]:


def plot_class(cat):
    
    fetch = train_df.loc[train_df['category']== category[cat]][:3]
    fig = plt.figure(figsize=(20,15))
    
    for i , index in enumerate(fetch.index ,1):
        plt.subplot(1,3 ,i)
        plt.imshow(train_images[index])
        plt.xlabel(cat + " (Index:" +str(index)+")" )
    plt.show()


# In[ ]:


plot_class('Cargo')


# In[ ]:


plot_class('Military')


# In[ ]:


plot_class('Carrier')


# In[ ]:


plot_class('Tankers')


# In[ ]:


plot_class('Cruise')


# These pictures seems to be taken from side and not from top (like picture taken from a satellite) except may be few of them which look like taken from above (although not from high above). We also noticed that all the pictures are of different sizes. we have to make sure they are of same sizes before modelling.
# 
# From deep learning context, we do not have very large number of images per category. So we will heavily depend on data augmentation, otherwise, it will easily cause overfitting.

# ### Modelling Approach

# - i am using fastai datablock api to create our databunch and train model using cnn

# In[ ]:


# doc(src.transform)


# Let's define the transformations to be done to the images.
# - random flipping of the images `do_flip=True`. Tried with `False` as well.
# - switch off vertical flipping (it's default behaviour). This option is useful when pictures are taken from high above.
# - let's rotate the pictures a bit randomly `max_rotate=10` (it's already default ).
# - zoom in a higher bit `max_zoom` (as we are dealing with ships picture that are small compared to overall image).
# - `max_warp` is set to zero as it seems to perform better in this case.
# - apply lighting and probability of affine function .
# For details on transformation visit [here][1]
# 
# [1]: https://docs.fast.ai/vision.transform.html

# In[ ]:


##transformations to be done to images
tfms = get_transforms(do_flip=False,flip_vert=False ,max_rotate=10.0, max_zoom=1.22, max_lighting=0.22, max_warp=0.0, p_affine=0.75,
                      p_lighting=0.75)
#, xtra_tfms=zoom_crop(scale=(0.9,1.8), do_rand=True, p=0.8))

## create databunch of test set to be passed
test_img = ImageList.from_df(test_df, path=data_folder/'train', folder='images')


# In[ ]:


np.random.seed(145)
## create source of train image databunch
src = (ImageList.from_df(train_df, path=data_folder/'train', folder='images')
       .split_by_rand_pct(0.2)
       #.split_none()
       .label_from_df()
       .add_test(test_img))


# Let's create our databunch. I will be using `size = 299` for modelling purpose, however let's try even higher size picture to improve accuracy further. But beware that we have to adjust batchsize accordingly to run out of memory. in case of 299 size `bs=32` is used while for 484 or even 599, a smaller batchsize should be used. 
# 
# The reflection padding mode seems to work better in this case (`padding_mode='reflection'`).      
# we will use Squishing resize method.      
# Finally, we normalize the parameters using imagenet_stats 

# In[ ]:


data = (src.transform(tfms, size=299,padding_mode='reflection',resize_method=ResizeMethod.SQUISH)
        .databunch(path='.', bs=32, device= torch.device('cuda:0')).normalize(imagenet_stats))

# data = (src.transform(tfms, size=484,padding_mode='reflection',resize_method=ResizeMethod.SQUISH)
#         .databunch(path='.', bs=16, device= torch.device('cuda:0')).normalize(imagenet_stats))


# In[ ]:


## lets see the few images from our databunch
data.show_batch(rows=3, figsize=(12,12))


# In[ ]:


print(data.classes)


# In[ ]:


# doc(cnn_learner)


# Now we will create cnn learner. 

# In[ ]:


#lets create learner. tried with resnet152, densenet201, resnet101
learn = cnn_learner(data=data, base_arch=models.resnet101, metrics=[FBeta(beta=1, average='macro'), accuracy],
                    callback_fns=ShowGraph)

# learn = cnn_learner(data=data, base_arch=models.densenet161, metrics=[FBeta(beta=1, average='macro'), accuracy],
#                     callback_fns=ShowGraph).mixup()


# `mixup()` on the above learner refer to a pretty aberrant feature which seems very unfamailiar to us (human) however this technique works better for computers. Details of the paper can be found [here](https://arxiv.org/abs/1710.09412).  The theory behind is like this:
# - we do not train dirctely on the raw data images instead the model is trained on mixes of images i.e. we add 2 or more images to combine a single picture by this: `new_image = t * image1 + (1-t) * image2` (not necessary to take only 2 images). By using this same technique, targets are changed as well `new_target = t * target1 + (1-t) * target2` ; where t is a float between 0 and 1.
# - Let's take an example: we are training on cat/dog dataset and we mix 2 images (one of each kind) and we finally get a single image of like this. It not clear to us what image is this; may be 70% dog and 30% cat.
# ![image](https://docs.fast.ai/imgs/mixup.png)
# - One  thing to note that, the mixup model may perform better than the regular one but when you compare the traing/validation losses, the mixup model has loss far greater than the regular one (although accuracy seems better in mixup model). This is because the mixup model predictions are less confident about the target. i.e. when we do prediction through our normal model, model seems pretty confident about the target (one target probability prediction will be close to 1 and others will be close to 0). When we predict through mixup model, the probabilities values of the targets will more likely to be close to one another).
# For details visit [here](https://docs.fast.ai/callbacks.mixup.html)

# In[ ]:


#learn.opt_func = optim.Adam
#learn.crit = FocalLoss()
# learn_gen = None
# gc.collect()
# torch.cuda.empty_cache()
learn.summary()


# Learning rate finder plots lr vs loss relationship for a Learner. The idea is to reduce the amount of guesswork on picking a good starting learning rate.
# If you pass `suggestion=True` in `learn.recorder.plot`, you will see the point where the gardient is the steepest with a
# red dot on the graph. We can use that point as a first guess for an LR. Details can be found [here](https://docs.fast.ai/basic_train.html#lr_find)

# In[ ]:


#lets find the correct learning rate to be used from lr finder
learn.lr_find()
learn.recorder.plot(suggestion=True)


# In[ ]:


#lets start with steepset slope point. adding wd (weight decay) not to overfit as we are running 15 epochs 
lr = 3e-03
#learn.fit_one_cycle(10, slice(lr))
learn.fit_one_cycle(15, slice(lr), wd=0.2)


# Now Unfreeze entire model.This Sets every layer group to trainable (i.e. `requires_grad=True`).

# In[ ]:


#lets plot the lr finder record
learn.unfreeze()
learn.lr_find()
learn.recorder.plot(suggestion=True)


# In[ ]:


# train for  more cycles after unfreezing
learn.fit_one_cycle(10,slice(1e-05,lr/8),wd=0.15)
#learn.fit_one_cycle(10, slice(5e-06, lr/8))


# lets freeze the all layers except last 3  as these are initial layers for finding recurring pattern/ shapes/corners etc. (not exactly helpful in finding ships). so its better not to change stats of those layers

# In[ ]:


learn.freeze_to(-3)


# In[ ]:


## finding the LR
learn.lr_find()
learn.recorder.plot(suggestion=True)


# - Train for few  more cycles (we will be setting two LRs in below trainings: first one to train the initial layers and second to
# train last layers ).    
# - As initial layers' stats are imagenet stats which are helpful in finding patterns (discussed above) not 
# the exact ships, so we will be training those layers with very low learning rates (to not to greatly change those initial layer
# parameters)

# In[ ]:


learn.fit_one_cycle(6, slice(1e-06, lr/10),wd=0.1)


# In[ ]:


## freezing initial all layers except last 2 layers
learn.freeze_to(-2)


# In[ ]:


## training for few cylcles more
learn.fit_one_cycle(6, slice(5e-07, lr/20),wd=0.1)


# In[ ]:


learn.freeze_to(-1)


# In[ ]:


## training even more
learn.fit_one_cycle(5, slice(1e-07, lr/30),wd=0.05)


# In[ ]:


learn.fit_one_cycle(6, slice(1e-07, lr/100))


# ### Interpretation

# In[ ]:


#lets see the most mis-classified images (on validation set)
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_top_losses(9, figsize=(7,6))


# In[ ]:


learn.recorder.plot_losses()


# In[ ]:


interp.plot_confusion_matrix(figsize=(6,6), dpi=60) ## on validation set


# It seems like our model is finding difficult to distinguish between ship 1 (cargo) and 5 (tanker)

# In[ ]:


interp.most_confused(min_val=4) ## on validation set


# In[ ]:


idx=1
x,y = data.valid_ds[idx]
x.show()


# In[ ]:


k = tensor([
    [0.  ,-5/3,1],
    [-5/3,-5/3,1],
    [1.  ,1   ,1],
]).expand(1,3,3,3)/6


# In[ ]:


t = data.valid_ds[1][0].data; t.shape


# In[ ]:


edge = F.conv2d(t[None], k)


# In[ ]:


show_image(edge[0], figsize=(5,5));


# In[ ]:


m = learn.model.eval();
xb,_ = data.one_item(x)
xb_im = Image(data.denorm(xb)[0])
xb = xb.cuda()


# In[ ]:


def hooked_backward(cat=y):
    with hook_output(m[0]) as hook_a: 
        with hook_output(m[0], grad=True) as hook_g:
            preds = m(xb)
            preds[0,int(cat)].backward()
    return hook_a,hook_g


# In[ ]:


hook_a,hook_g = hooked_backward()
acts  = hook_a.stored[0].cpu()
acts.shape


# In[ ]:


avg_acts = acts.mean(0)
avg_acts.shape
torch.Size([11, 11])


# In[ ]:


def show_heatmap(hm):
    _,ax = plt.subplots()
    xb_im.show(ax)
    ax.imshow(hm, alpha=0.6, extent=(0,352,352,0),
              interpolation='bilinear', cmap='magma');


# In[ ]:


show_heatmap(avg_acts)


# Predicted using TTA as it could improve accuracy further (Test time augmentation).
# Applies Test Time Augmentation to our learner on the dataset.    
# Here We take the average of our regular predictions (with a weight beta) with the average of predictions obtained through augmented 
# versions of the training set (with a weight 1-beta). Details can be found [here](https://docs.fast.ai/basic_train.html#Test-time-augmentation)

# In[ ]:


##learn.TTA improves score further. lets see for the validation set
pred_val,y = learn.TTA(ds_type=DatasetType.Valid)
from sklearn.metrics import f1_score, accuracy_score
valid_preds = [np.argmax(pred_val[i])+1 for i in range(len(pred_val))]
valid_preds = np.array(valid_preds)
y = np.array(y+1)
accuracy_score(valid_preds,y),f1_score(valid_preds,y, average='micro')


# Now we have achieved more than 98% accuracy.

# ### Prediction

# - Before predicting on the test set, i generally (sometimes) remove the validation set and try all these steps above on whole train set for modelling.  This is to produce  model not on 80% data but 100% data and predict the testset
# - This can done by changing in one line in data creation i.e. change `.split_by_rand_pct(0.2)` to `.split_none()`.

# In[ ]:


preds,_ = learn.TTA(ds_type=DatasetType.Test)
#preds,_ = learn.get_preds(ds_type = DatasetType.Test)
labelled_preds = [np.argmax(preds[i])+1 for i in range(len(preds))]

labelled_preds = np.array(labelled_preds)


# In[ ]:


#create submission file
df = pd.DataFrame({'image':test_df['image'], 'category':labelled_preds}, columns=['image', 'category'])
df.to_csv('submission.csv', index=False)


# In[ ]:


## function to create download link
from IPython.display import HTML
def create_download_link(title = "Download CSV file", filename = "data.csv"):  
    html = '<a href={filename}>{title}</a>'
    html = html.format(title=title,filename=filename)
    return HTML(html)


# In[ ]:


create_download_link(filename = 'submission.csv')


# In[ ]:


df.category.unique()


# In[ ]:


df.head()


# ### Final submission

# - At this point, i have submission files of around 22 which are created with different augmentation techniques, with/without mixup, different pretrained model (used resnet101, resnet152, densenet161, densenet169), different image sizes (tried with size 224, 299, 484, 599). All the prediction are based on TTA. 
# - The final submission was created based on voting technique by all the submission predicted categories.
# - I could have tried with predicting the probabilities of the image category of all these submissions and then the final submission could have been based on average of those probabilities values. But I am guessing there might have been one challenge: as mentioned above in the notebook, the mixup model predctions are less confident about the target and hence due to this the average probabilities could have been very different leading to different results (may be). 
# 
# For e.g. lets take cat/dog image: without mixup, the model predicts cat (prob: 0.94) and dog (0.06) while the mixup model predicts cat (prob:0.56) and dog (0.44) and the average of these 2 models is cat( 0.75) and dog(0.25). Well this might seems fine, however let's imagine we have more target categories to predict, and in this scenario the mixup model probabilities could alter the average probabilities. 

# In[ ]:




