#!/usr/bin/env python
# coding: utf-8

# ## FastAI Vision CNN learner on tabular data

# Even though the digit recognizer dataset is organized as a table, it is indeed pictures of handwritten digits. We usually treat it as any other tabular classification problem, but it may be worth transforming it into a set of pictures and applying fastai.vision library on it. Fastai Vision is pretty good, let's see how it holds up against Tabular in its home turf.
# 
# We are going to literally turn rows into pictures. To do that we need to make our own Imagelist class to change how the data loads into our databunch. 

# In[ ]:


import numpy as np
import pandas as pd
from fastai.vision import *
from fastai.tabular import *
import matplotlib.pyplot as plt
import re


# In[ ]:


path=Path('/kaggle/working/')


# We are adding the index number for now for easier manipulation. We'll be removing it later.

# In[ ]:


train=pd.read_csv('/kaggle/input/train.csv')
train['index'] = train.index
train.head()


# ## Creating Custom ImageList
# 
# We need an ImageList to chuck into the cnn. So we should make a sub-class of the ImageList and change how it loads the data.

# In[ ]:


class PixelList(ImageList):
    def open(self,index):
        regex = re.compile(r'\d+')
        fn = re.findall(regex,index)
        df_fn = self.inner_df[self.inner_df.index.values == int(fn[0])]
        img_pixel = df_fn.drop(labels=['label','index'],axis=1).values
        img_pixel = img_pixel.reshape(28,28)
        img_pixel = np.stack((img_pixel,)*3,axis=-1)
        return vision.Image(pil2tensor(img_pixel,np.float32).div_(255))


# **Step By Step :**
# 
# 1. Declaring our own PixelList class, subclass of ImageList so that it inherits everything.
# 2. While subclassing an ItemList we can modify three methods and "open" is one of them. We are going to override this method because it is called when ImageList tries to open an image.
#         For more details check [here](https://docs.fast.ai/tutorial.itemlist.html#Creating-a-custom-ItemList-subclass).
# 3. Defining a regular expression to find integers and compile it.
# 4. Finding and save the rows matching index number.
# 5. Picking a dataframe of only the index we have already chosen to read as an image.
# 6. Droping index and label, they are not parts of the picture and taking an array of values.
# 7. Reshaping 784x1 to 28x28 array.
# 8. Blowing the image up 3 times.
# 9. Creating an Image from the array using vision.image library.
#         For more details check [here](https://docs.fast.ai/vision.image.html#pil2tensor).

# Now we call it just like our vanilla ImageList. We have to pass cols='index' because PixelList "open" expects that argument.

# In[ ]:


src = (PixelList.from_df(train,'.',cols='index')
      .split_by_rand_pct(0.1)
      .label_from_df(cols='label'))


# Getting transforms is a bit tricky. These are not ordinary images, so flipping, zooming and cropping will yield weird instances. Also padding needs to be zeros, we can not expect reflection on the paddings.

# In[ ]:


tfms=get_transforms(rand_pad(padding=5,size=28,mode='zeros'))


# Finally we get them as size 128 and normalize them. We're not using imagenet_stats for the same reason.

# In[ ]:


data = (src.transform(tfms,size=128)
       .databunch(num_workers=5,bs=48)
       .normalize())


# Let's look at the data now.

# In[ ]:


data.show_batch(rows=3,figsize=(10,7))


# ## Training Resnet
# 
# This part is pretty self-explanatory. We get a Resnet101, train for a bit and see the results.

# In[ ]:


learner=cnn_learner(data,models.resnet101,metrics=[FBeta(),accuracy,error_rate])


# In[ ]:


learner.lr_find()
learner.recorder.plot(suggestion=True)


# In[ ]:


lr=9e-3


# In[ ]:


learner.fit_one_cycle(5,slice(lr))


# In[ ]:


learner.save('Resnet1')


# In[ ]:


learner.lr_find()
learner.recorder.plot(suggestion=True)


# Let's unfreeze all layers except first two and train some more.

# In[ ]:


learner.freeze_to(2)


# In[ ]:


learner.fit_one_cycle(4,slice(5e-6,lr/50))


# In[ ]:


learner.save('Resnetfinal')


# ## Training Tabular Learner
# 
# We just create a databunch, get a tabular learner and train it.
# Tabular learner will consider this a classfication problem by default. All 784 columns are going to become features, so we have to be careful not to overfit.

# In[ ]:


train=pd.read_csv('/kaggle/input/train.csv')
data=TabularDataBunch.from_df(path,train,dep_var='label',valid_idx=range(4000,6000))


# In[ ]:


tablearner=tabular_learner(data,layers=[200,100],ps=[0.001,0.01],emb_drop=0.004,metrics=accuracy)


# In[ ]:


tablearner.lr_find()
tablearner.recorder.plot(suggestion=True)


# In[ ]:


lr=2.2e-2


# In[ ]:


tablearner.fit_one_cycle(10,slice(lr),wd=0.1)


# In[ ]:


tablearner.lr_find()
tablearner.recorder.plot(suggestion=True)


# In[ ]:


tablearner.fit_one_cycle(4,4e-6,wd=0.1)


# ## Verdict [spoilers: Resnet wins]
# 
# After some trial and error, I've found tabular learner to work best under the params I've chosen above. With no input pre-processing, Tabulars best accuracy is around 97~98%.
# Where Resnet gives <99% results every time. It seems resnet has asserted its dominance again.

# ## What About DenseNet?
# 
# I have experimented a bit with densenet as well. But as it takes an obscene amount of time to train, I've decided not to include it this round.
# DenseNet201 gave >98%, but not quite like resnet; not doing justice to the time it took to train.
# 

# ## Submit!
# 
# So we have our winner.
# 

# In[ ]:


df_test = pd.read_csv('../input/test.csv')
df_test['label'] = 0
df_test['index'] = df_test.index
df_test.head()


# In[ ]:


learner.load('Resnetfinal')


# In[ ]:


learner.data.add_test(PixelList.from_df(df_test,path='.',cols='index'))


# In[ ]:


pred_test = learner.get_preds(ds_type=DatasetType.Test)
test_result = torch.argmax(pred_test[0],dim=1)
result = test_result.numpy()


# In[ ]:


final = pd.Series(result,name='Label')
submission = pd.concat([pd.Series(range(1,28001),name='ImageId'),final],axis=1)
submission.to_csv('submit.csv',index=False)

