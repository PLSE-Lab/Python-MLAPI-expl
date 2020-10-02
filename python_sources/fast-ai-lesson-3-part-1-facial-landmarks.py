#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from fastai.vision import *


# # Loading and Preparing the data

# In[ ]:


import requests, zipfile, io
zip_file_url = "https://download.pytorch.org/tutorial/faces.zip"
r = requests.get(zip_file_url)
z = zipfile.ZipFile(io.BytesIO(r.content))
z.extractall()


# In[ ]:


import os
os.listdir()


# In[ ]:


path = "faces"


# In[ ]:


images_and_coordinates = pd.read_csv("faces/face_landmarks.csv")


# In[ ]:


images_and_coordinates.head()


# In[ ]:


images_and_coordinates.describe()


# In[ ]:


def getPointCoordinates(filename):
    singleDFEntry = images_and_coordinates[images_and_coordinates.image_name == filename[len(path +'/'):]]
    onlyDFCoordinates = singleDFEntry.drop(columns='image_name')
    numpyMatrix = onlyDFCoordinates.values.reshape(-1,2)
    numpyMatrix[:, 0], numpyMatrix[:, 1] = numpyMatrix[:, 1], numpyMatrix[:, 0].copy()
    ##print(numpyMatrix.shape)
    return torch.from_numpy(numpyMatrix).float()


# In[ ]:


fileName = path + '/' + images_and_coordinates.image_name[65]
samplePoints = getPointCoordinates(fileName)
img = open_image(fileName)
samplePoints = ImagePoints(FlowField(img.size, samplePoints), scale=True)
img.show(y=samplePoints)


# In[ ]:


src = (PointsItemList.from_csv(path, 'face_landmarks.csv')
        .split_by_rand_pct(0.2)
        .label_from_func(getPointCoordinates))


# In[ ]:


src


# In[ ]:


data = (src.transform(tfm_y=True, size=(160,160))
        .databunch(bs=16).normalize(imagenet_stats))   


# In[ ]:


data.show_batch(3, figsize=(9,6))


# # Creating and Training the Model

# In[ ]:


learn = cnn_learner(data, models.resnet34)


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


lr = 1e-2
learn.fit_one_cycle(5, slice(lr))


# In[ ]:


learn.show_results()

