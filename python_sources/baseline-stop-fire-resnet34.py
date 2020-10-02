#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('ls ../input/fire-test/challenge1/')


# In[ ]:


from fastai.vision import *
from fastai import *


# Reading the path of files

# In[ ]:


path = Path('../input/fire-test/challenge1')


# In[ ]:


smoke_path = path/'smoke'
no_smoke_path = path/'no_smoke'


# In[ ]:


smoke_filenames = get_files(smoke_path)
no_smoke_filenames = get_files(no_smoke_path)
len(smoke_filenames),len(no_smoke_filenames)


# Therefore, the dataset is balanced, about 50% smoke and 50% non smoke images. Which is not what happend in the real world

# Let's take a look of the first image in both folders

# In[ ]:


smoke_img = open_image(smoke_filenames[0])
smoke_img.size


# In[ ]:


smoke_img


# In[ ]:


no_smoke_img = open_image(no_smoke_filenames[0])
no_smoke_img.size


# In[ ]:


no_smoke_img


# Building piplines
# 
# Argumentation:
# 1. flip left and right
# 2. rotate 10 degree
# 3. lighting .2
# 4. zoom 10%
# 5. warp angle 20%
# 
# Resize img size to (128,128)

# In[ ]:


tfms = get_transforms()


# In[ ]:


data = (ImageList
        .from_folder(path,include=['smoke','no_smoke'])
        .split_by_rand_pct()
        .label_from_folder()
        .transform(tfms,size=(128,128))
        .databunch(bs=64)
        .normalize(imagenet_stats)
)


# In[ ]:


data.show_batch(rows=3,figsize=(12,10))


# Using imagenet pretrained model resnet 34 with customized head (see head arch in the following cell)
# 
# 1. stacked [maxpool and avgpool]
# 2. few drop out, batchnorm, linear layers for more calculation

# In[ ]:


# loading imagenet pre-trained model, also using mix precision for training 
learn = cnn_learner(data,models.resnet34,metrics=[accuracy],model_dir='/kaggle/working').to_fp16() 


# In[ ]:


learn.model[1] # customized Head


# In[ ]:


# Finding best learning rate
learn.lr_find()


# In[ ]:


learn.recorder.plot(suggestion=True)


# In[ ]:


lr = 3e-3
learn.fit_one_cycle(3,slice(lr))


# In[ ]:


learn.unfreeze() # Unfreeze the body, fine tune the whole model


# In[ ]:


learn.lr_find()
learn.recorder.plot(suggestion=True)


# In[ ]:


learn.save('resnet34-stage-1-128')


# In[ ]:


learn.fit_one_cycle(5,slice(1e-5,lr/5))


# Progressive resizing to 256

# In[ ]:


data_256 = (ImageList
        .from_folder(path,include=['smoke','no_smoke'])
        .split_by_rand_pct()
        .label_from_folder()
        .transform(tfms,size=(256,256))
        .databunch(bs=64)
        .normalize(imagenet_stats)
)


# In[ ]:


learn.save('res34-stage-2-128')


# In[ ]:


learn.data = data_256
learn.freeze_to(-1)


# In[ ]:


learn.to_fp16(); # using mix precision 


# In[ ]:


learn.lr_find()
learn.recorder.plot(suggestion=True)


# In[ ]:


lr = 1e-3
learn.fit_one_cycle(3,slice(lr))


# In[ ]:


learn.unfreeze()


# In[ ]:


learn.save('res34-stage-1-256')


# In[ ]:


learn.lr_find()
learn.recorder.plot(suggestion=True)


# In[ ]:


learn.fit_one_cycle(5,slice(1e-5,lr/5))


# In[ ]:


learn.save('res34-stage-2-256')


# In[ ]:


y_p, y, loss = learn.get_preds(with_loss=True)


# In[ ]:


y_p.shape,y.shape


# In[ ]:


pred = y_p.argmax(dim=-1).float()


# In[ ]:


pred.shape


# In[ ]:


from sklearn.metrics import f1_score


# In[ ]:


f1_score(y,pred)


# In[ ]:


learn.show_results()


# In[ ]:


interp = ClassificationInterpretation(learn, y_p, y, loss)


# In[ ]:


interp.plot_confusion_matrix()


# In[ ]:


interp.plot_top_losses(9,figsize=(12,12))


# # Fin
# 
# 1. If we are fitting the model with the balanced dataset, it doesn't reflect the real world case (where you have maybe 99% of time no smoke, only 1% of time that has smoke)
# 2. As you can see from the top losses, the model is not doing a very good job prediction the early fires. When the smokes are in the initial phase,the model is not able to capture the smoke.

# # Next
# 
# 1. To build the early fire detector, we will need to remove the easy examples (the late fire with lots of smokes in the dataset)
# 2. Use segmentation / bounding boxes 
# 3. To have more data

# In[ ]:




