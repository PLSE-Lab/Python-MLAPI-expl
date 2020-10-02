#!/usr/bin/env python
# coding: utf-8

# <center><h1>104 flower Image Classification</h1></center>
# <center><img src='https://hgtvhome.sndimg.com/content/dam/images/hgtv/fullset/2015/11/10/0/CI_Costa-Farms-Ballad-aster.jpg.rend.hgtvcom.966.644.suffix/1447169929799.jpeg' width=300 height=200>
# <img src='https://www.top13.net/wp-content/uploads/2014/11/28-small-flowers-1024x682.jpg' width=300 height=200>
# <img src='https://www.top13.net/wp-content/uploads/2014/11/17-small-flowers.jpg' width=300 height=200></center>
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 

# 
# # Using a pretrained ResNet50 model
# 
# ## with metrics = f1_score 
# >average = macro 
# 
# 
# * because there is class imbalance in the data set
# 

# ### If it was helpful please upvote 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
from fastai.vision import *
from fastai import *
import matplotlib.pyplot as plt
import seaborn as sns
from functools import partial
from tqdm.notebook import tqdm
import gc
from pylab import imread,subplot,imshow,show
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv("../input/flower-goggle-tpu-classification/flowers_idx.csv")
label = pd.read_csv("../input/flower-goggle-tpu-classification/flowers_label.csv")


# In[ ]:


df.head()


# In[ ]:


label.head()


# In[ ]:


path = Path('/kaggle/input/flower-goggle-tpu-classification/flower_tpu/flower_tpu/flowers_google/')
path1 = Path('/kaggle/input/flower-goggle-tpu-classification/flower_tpu/flower_tpu/test/test/')


# In[ ]:


group = label.flower_class.values


# In[ ]:


flower=df.groupby(['flower_cls'])


# In[ ]:


flow_dict = defaultdict(list)
for i in range(len(group)):
    b = group[i]
    for j in range(5):
        a = flower.get_group(b).iloc[j][0]
        k = path/str('flowers_google/'+str(a)+str('.jpeg'))
        flow_dict[b].append(str(k))


# <center><h1> A glimpse of each category of flower.</h1></center>

# In[ ]:


for i in range(len(group)):
    
    b = group[i]
    m =flow_dict[b]
    
    plt.figure(1,figsize=(20,6))
    
    print(b)
    plt.subplot(151)
    plt.axis('Off')
    plt.imshow(imread(m[0]))
    
    plt.subplot(152)
    plt.axis('Off')
    plt.imshow(imread(m[1]))
    
    plt.subplot(153)
    plt.axis('Off')
    plt.imshow(imread(m[2]))
    
    plt.subplot(154)
    plt.axis('Off')
    plt.imshow(imread(m[3]))
    
    plt.subplot(155)
    plt.axis('Off')
    plt.imshow(imread(m[4]))
    
    plt.show()


# # Forming a databunch

# In[ ]:


tfms = get_transforms(do_flip=True,max_rotate=0.1,max_lighting=0.15)


# In[ ]:


test = (ImageList.from_folder(path1,extensions='.jpeg'))


# In[ ]:


data = (ImageList.from_df(df,path,folder='flowers_google',suffix='.jpeg',cols='id')
                .split_by_rand_pct(0.15)
                .label_from_df(cols='flower_cls')
                .transform(tfms)
                .add_test(test)
                .databunch(bs=128)
                .normalize(imagenet_stats))


# In[ ]:


data.show_batch(rows=4)


# ### Total classes, length of train, validation and test set

# In[ ]:


len(data.classes),len(data.train_ds),len(data.valid_ds),len(data.test_ds)


# # Using a pretrained ResNet50 model
# 
# ## with metrics = f1_score 
# >average = macro 
# 
# 
# * because there is class imbalance in the data set

# In[ ]:


fb = FBeta()
fb.average='macro'


# In[ ]:


arch = models.resnet50


# In[ ]:


#!mkdir -p /tmp/.cache/torch/checkpoints/
#!cp /kaggle/input/resnet50/resnet50.pth  /root/.cache/torch/checkpoints/resnet50-19c8e357.pth


# # Using mixed precision training
# 
# * Mixed precision training utilizes half-precision to speed up training, achieving the same accuracy in some cases as single-precision training using the same hyper-parameters. 
# * Memory requirements are also reduced, allowing larger models and minibatches

# In[ ]:


try:
    learn = cnn_learner(data, arch, metrics = [fb],model_dir='/kaggle/working').to_fp16()
except:
    get_ipython().system('mkdir -p /tmp/.cache/torch/checkpoints/')
    get_ipython().system('cp /kaggle/input/resnet50/resnet50.pth  /root/.cache/torch/checkpoints/resnet50-19c8e357.pth')
    
    learn = cnn_learner(data, arch, metrics = [fb],model_dir='/kaggle/working').to_fp16()


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


gc.collect()


# # Model Summary

# In[ ]:


learn.summary()


# In[ ]:


lr = 1e-2


# In[ ]:


gc.collect()


# In[ ]:


learn.fit_one_cycle(6,lr,moms=(0.9,0.8))


# # Classification Interpretation

# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)


# In[ ]:


interp.plot_top_losses(12,figsize=(20,8))


# # Most Confused

# In[ ]:


interp.most_confused(min_val=3)


# In[ ]:


learn.save('model1')


# In[ ]:


learn.export('/kaggle/working/flower.pkl')


# In[ ]:


img = open_image('/kaggle/input/flower-goggle-tpu-classification/flower_tpu/flower_tpu/test/test/d9cb87ad0.jpeg')
print(learn.predict(img)[0])
img


# In[ ]:


img = open_image('/kaggle/input/flower-goggle-tpu-classification/flower_tpu/flower_tpu/test/test/bcb18c6e4.jpeg')
print(learn.predict(img)[0])
img


# In[ ]:


img = open_image('/kaggle/input/flower-goggle-tpu-classification/flower_tpu/flower_tpu/test/test/d15a4d94c.jpeg')
print(learn.predict(img)[0])
img


# <center><h1> If it was helpful please upvote </h1></center>

# In[ ]:


samp = pd.read_csv('/kaggle/input/flower-classification-with-tpus/sample_submission.csv')


# In[ ]:


samp.head()


# # Prediction and Submission

# In[ ]:


n = samp.shape[0]
path2 = '/kaggle/input/flower-goggle-tpu-classification/flower_tpu/flower_tpu/test/test/'


# In[ ]:


for i in range(n):
  idc = samp.iloc[i][0]
  k = path2 + idc + '.jpeg'
  k = open_image(k)
  ans = learn.predict(k)[0]
  samp.loc[[i],1:] = str(ans)

print("Done Prediction saved ---> ")
  


# In[ ]:


samp.head(10)


# * ## Time to replace flower label with ids

# In[ ]:


lab = {}
for i in range(label.shape[0]):
  sha = label.iloc[i]
  lab[sha[1]]=int(sha[0])


# In[ ]:


samp.label.replace(lab,inplace=True)


# In[ ]:


samp.head(20)


# In[ ]:


samp.to_csv('submission.csv',index=False)


# <center><h1> If it was helpful please upvote </h1></center>

# In[ ]:




