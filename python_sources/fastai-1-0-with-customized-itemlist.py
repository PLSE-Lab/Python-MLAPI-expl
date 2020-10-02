#!/usr/bin/env python
# coding: utf-8

# # **FastAI 1.0 with classic MNIST dataset **

# Kaggle provides MNIST dataset as csv file, that means standard fastai API won't work in this case, the three factory methods(for vision)  .from_folder, from_csv, from_df all asking a filename (fn) with label pair to get your databunch for training. 
# 
# One solution we see in class is that you can write your own dense neural nets, but if you want to use CNN, or fastai Resnet layer functions, you will have to preprocess your data. Either use fastai provided called Lambda layer or preprocess your data as numpy array that has dimension (m, c, h, w) 
# 
# What if we want to use fastai API with transfer learning?  
# 
# In the following section, I would like to share how to customize a simple ItemList with datablock API so we can go back to fastai routine. 

# # Loading related library

# In[ ]:


# !conda install -c fastai fastai --yes #using latest 1.0.48 as 1.0.46 learner will have read-only issue


# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from fastai import *
from fastai.vision import *
import numpy as np
import pandas as pd
import re


# # Pre-Porcessing

# The train.csv file contains **Label** column with pixel values. If we think it as **filename, label** pair, all we need is a filename for each of our data
# 
# Let's create a filename for each of the training data, I used index of each example as filename. The idea here is besides label, we need to prepare each data with a name
# 
# For example, training set item 0, we name it 0. filename = 0, label =1

# In[ ]:


df_train = pd.read_csv('../input/train.csv')
df_train['fn'] = df_train.index
df_train.head()


# # Customized ItemList for pixel values

# Since we already load the data to dataframe, we can use from_df() API to load the data. 
# However, simply calling ImageItemList.from_df() won't work, we need to find a way to overide the default behavior : Instead of loading img files, it needs to load dataframe values, and make it RGB channels. 
# 
# For example, from 784 to (28,28) to (3,28,28)

# Custom ItemList provides a great way to override the standard behavior. You can check fastai doc tutorials for more [Docs](https://docs.fast.ai/tutorial.itemlist.html) 
# 
# Most importantly, since our custom ItemList will be similar to the ImageItemList, we only need to tell the library how to read our pixel value data.
# 
# It turns out that we only need to change the open() method, which get() will call to get data. 
# 
# So fastai vision is following this path
# 
# **from_csv() / from_folder() / from_df() --> eventually calls get() ---> eventually calls open()**
# 
# This is why in the custom ItemList tutorial get() is most important function to override. 
# 
# 
# 

# FInally,
# from_df() will pass **path/filename** to get image file, we need to properly open the img when fn is passed. 
# 
# 1. get fn
# 2. according to the fn, gets the pixel value from dataframe, this is internally saved in the self.xtra **(for 1.0.4x version, self.xtra renamed to self.inner_df)**
# 3. reshape the img, stack the gray channel make it RBG
# 4. fastai provides a API called **pil2tensor() ** which takes npdarry and return pytorch tensor
# 5. return vision.Image() as Image class takes pixel

# In[ ]:


class PixelImageItemList(ImageList):
    def open(self,fn):
        regex = re.compile(r'\d+')
        fn = re.findall(regex,fn)
        df = self.inner_df[self.inner_df.fn.values == int(fn[0])]
        df_fn = df[df.fn.values == int(fn[0])]
        img_pixel = df_fn.drop(labels=['label','fn'],axis=1).values
        img_pixel = img_pixel.reshape(28,28)
        img_pixel = np.stack((img_pixel,)*3,axis=-1)
        return vision.Image(pil2tensor(img_pixel,np.float32).div_(255))


# # FastAI datablock API

# We are ready to go, now we can use standard fastai datablock API to create databunch
# 
# 1. get data, since we are calling from_df, we will pass dataframe with col that tags our data. The path passed is './fn' 
# 2. random split, 80-20 train - valid split
# 3. get labels from df, where we pass col = 'label'.
# 4. add transform (optional), we are only zero pad and random zoom in this case. calling get_transform() with flipping / lighting wont do much good for the 28*28 grey imgs(even though is RGB now)
# 5. create databunch, and normalize data using pretrained model stats 
# 

# In[ ]:


src = (PixelImageItemList.from_df(df_train,'.',cols='fn')
      .split_by_rand_pct()
      .label_from_df(cols='label'))


# In[ ]:


data = (src.transform(tfms=(rand_pad(padding=5,size=28,mode='zeros'),[]))
       .databunch(num_workers=2,bs=128)
       .normalize(imagenet_stats))


# Lets take a look of our data, notice that the grey img now is turned to RGB(3 channels) with size (3,28,28). 

# In[ ]:


data.show_batch(rows=3,figsize=(10,7))


# The label and data is correct, and we can see the data is randomly pad and zoomed 
# 
# we can further test the shape of our data to make sure it is correct

# In[ ]:


print(data.train_ds[0][1]) #label
data.train_ds[0][0] #img


# In[ ]:


data.train_ds[0][0].data,data.train_ds[0][0].data.shape,data.train_ds[0][0].data.max()


# Start training using standard fastai 

# In[ ]:


learn = cnn_learner(data,models.resnet50,metrics=accuracy,model_dir='/kaggle/model')


# In[ ]:


learn.lr_find(end_lr=100)


# In[ ]:


learn.recorder.plot()


# In[ ]:


lr = 1e-2


# In[ ]:


learn.fit_one_cycle(5,slice(lr))


# Unfreeze the model and train a little bit more

# In[ ]:


learn.unfreeze()
learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(8,slice(2e-5,lr/5))


# **Updated: Using resnet 50 gives a boost of performance to 0.994 LB score**
# 
# 99.2 is not bad, consider we are only training with resnet 34 and train loss is still higher than valid loss, which means there is still room for you to train longer.
# 
# Also, we can simply load a resnet 50 with little bit more data argumentation, such as rotate image -10 to 10 degrees. (Which can be easily done by fastai framework)
# 
# Let's take a look of some errors. 

# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)
interp.plot_top_losses(9,figsize=(6,6))


# From the top 9 losses, we can try the following
# 
# 1. remove some 'mislabeled' data? 
# 2. rotate the trainting set a bit should help 
# 3. we can train longer
# 
# So the model still got bit space to improve. 

# In[ ]:


learn.show_results()


# # fin

# Pack the test set as what we did for the training set.

# In[ ]:


df_test = pd.read_csv('../input/test.csv')
df_test['label'] = 0
df_test['fn'] = df_test.index
df_test.head()


# It turns out that you dont need to override add_test() to work. 
# 
# The default library takes ItemList to add as test set
# 
# We can simply create another PixelImageItemList and add it to the model.

# In[ ]:


learn.data.add_test(PixelImageItemList.from_df(df_test,path='.',cols='fn'))


# In[ ]:


pred_test = learn.get_preds(ds_type=DatasetType.Test)


# In[ ]:


test_result = torch.argmax(pred_test[0],dim=1)
result = test_result.numpy()


# Or you can take advantage of the fastai TTA (test time argumentation, which act like ensemble way of predicting. averages regular prediction and test time argumentated predication)
# 
# However, in this model, we didn't apply transfroms to vaildation set, therefore we can skip this part and submit our predication base on no TTA

# In[ ]:


# preds = learn.TTA(ds_type=DatasetType.Test)
# pred = torch.argmax(preds[0],dim=1)


# In[ ]:


final = pd.Series(result,name='Label')
submission = pd.concat([pd.Series(range(1,28001),name='ImageId'),final],axis=1)
submission.to_csv('fastai-res34-0.992.csv',index=False)


# In[ ]:


submission.head()


# In[ ]:




