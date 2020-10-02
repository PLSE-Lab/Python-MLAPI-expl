#!/usr/bin/env python
# coding: utf-8

# # Creat data-set from Google Image

# In[ ]:


from fastai.vision import *


# In[ ]:


get_ipython().system('ls ../input')


# In[ ]:


folder = 'house'
file = 'house.txt'


# In[ ]:


path = Path('data/total')
dest = path/folder
dest.mkdir(parents=True, exist_ok=True)


# In[ ]:


path.ls()


# In[ ]:


download_images(Path('../input')/file, dest, max_pics=200)


# In[ ]:


folder = 'nohouse'
file = 'nohouse.txt'


# In[ ]:


path = Path('data/total')
dest = path/folder
dest.mkdir(parents=True, exist_ok=True)
path.ls()


# In[ ]:


classes = ['house','nohouse']


# In[ ]:


download_images(Path('../input')/file, dest, max_pics=200)


# In[ ]:


for c in classes:
    print(c)
    verify_images(path/c, delete=True, max_size=500)


# In[ ]:


np.random.seed(42)
data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.2,
        ds_tfms=get_transforms(), size=224, num_workers=2).normalize(imagenet_stats)


# In[ ]:


data.show_batch()


# In[ ]:


data.classes, data.c, len(data.train_ds), len(data.valid_ds)


# # Finish setting up data
# 
# The finall data loaded has 2 classes 
# 
# 'house' means the the img has house in it
# 
# 'nohouse' means the img only has side walk
# 
# The train has 286 pics, the validation has 71 pics
# 
# Notice: I didn't check the balance of the dataset, so it is not evenly distributed, but for demonstration purpose, it should be sufficient

# In[ ]:


learn = cnn_learner(data,models.resnet34,metrics=accuracy)


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(5,1e-2)


# In[ ]:


learn.show_results()


# In[54]:


learn.export()


# In[56]:


path.ls()


# now the model is export.pkl file, I will explain how to use them to classify images, you can also upload the model to a web server, so users can upload images to see if it is house or not

# In[58]:


# when you do prediction, you want to disable GPU, since not most server has GPU ready for prediction
defaults.device = torch.device('cpu')


# In[63]:


get_ipython().system("ls {path/'house'} | head -3")


# In[64]:


get_ipython().system("ls {path/'nohouse'} | head -3")


# In[67]:


house_img = open_image(path/'house/00000002.jpg')
house_img


# In[68]:


nohouse_img = open_image(path/'nohouse/00000001.jpg')
nohouse_img


# In[69]:


learn = load_learner(path)


# In[70]:


learn.predict(house_img)


# In[71]:


learn.predict(nohouse_img)


# # Fin

# As shown above, user can input any image with any size, the model will predict giving a set of result
# 
# 1. Category, a house or not a house
# 2. tensor(0), tensor(1) indicates the one-hot encode result
# 3. Confidence with result, with 1.0 means 100% sure, 0.0 means 0% sure

# In[ ]:




