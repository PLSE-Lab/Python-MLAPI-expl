#!/usr/bin/env python
# coding: utf-8

# # Recommendation System based Embedding Approach 
# 
# ## 2020-03-07 Ahn Sangho
# 
# ## Thanks to 
# 
# - [KCDC](http://www.cdc.go.kr/index.es?sid=a2)
# - [datartist
# and 9 collaborators
# ](https://www.kaggle.com/kimjihoo/coronavirusdataset/data) at Kaggle
# - [Jeremy Howard](https://www.kaggle.com/jhoward) fastai
# 
# The `COVID-19` is now a global problem. 
# 
# Among many countries, South Korea is doing its best to protect virus with KCDC, various medical staff and many unknown people.
# 
# This dataset is also part of this effort. KCDC is producing detailed and accurate information, and datartist at el. are refining and sharing it.
# 
# This analysis suggests a way to embed place and patient information from route of patient. Specifically, I show fastai's collaborative filtering module to embedding this information. 
# 
# I sincerely hope that `COVID-19` will be well prevented and finished globally as soon as possible.
# 
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from pathlib import Path

root_dir = Path("/kaggle/")
base_dir = root_dir / "input"

data_path = base_dir / "coronavirusdataset"
print(data_path)


# # 1. Getting the Data
# 
# 
# The first step is to look at the data roughly, as you probably already know.
# 
# ## 1.1. Load and Head

# In[ ]:


import pandas as pd 

patient_df = pd.read_csv(data_path / "patient.csv")
print(patient_df.shape)
patient_df.head().T


# In[ ]:


route_df = pd.read_csv(data_path / "route.csv")
print(route_df.shape)
route_df.head().T


# ## 1.2. Wrangling for Rec Input
# 
# To Embedding patient and place information, some wraggling process needed. 
# 
# In recommendation system, rating is common. However, this dataset does not have rating information. 
# 
# Therefore, I've used a trick to represent visit information as 1. As a view of matrix factorization, I expect this to work.
# 
# ![MF](https://miro.medium.com/max/1689/1*Zhm1NMlmVywn0G18w3exog.png)
# Image from [this website](https://medium.com/@connectwithghosh/simple-matrix-factorization-example-on-the-movielens-dataset-using-pyspark-9b7e3f567536)

# In[ ]:


import numpy as np 

patient_id = route_df["id"]
place_id = route_df["city"] + "_"+ route_df["visit"]
one_hot = pd.Series(np.ones(patient_id.shape))

route_data = pd.DataFrame({
    "patient_id": patient_id,
    "place_id": place_id,
    "one": one_hot
})


# In[ ]:


route_data.groupby("place_id").count()


# # 2. Training Collab
# 
# Now, We can make `learner` and train model using fastai.
# 
# 
# The following steps are all common analysis process of fastai.
# 
# ## 2.1. Create DataBunch

# In[ ]:


from fastai.collab import *


# In[ ]:


patient, place, visit = "patient_id", "place_id", "one"

data = CollabDataBunch.from_df(route_data, seed=42, valid_pct=0, item_name=place) 


# In[ ]:


data.show_batch()


# ## 2.2. Create Learner and Find LR

# In[ ]:


y_range = [0, 1.5]


# In[ ]:


learn = collab_learner(data, n_factors=40, y_range=y_range, wd=1e-1)


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot(skip_end=15, suggestion=True)


# ## 2.3. Train Learner

# In[ ]:


learn.fit_one_cycle(10, 4e-02)


# In[ ]:


learn.recorder.plot_losses()


# # 3. Interpretation
# 
# After Training, we finally get embedding vectors for place and patient information. 
# 
# In both cases, a PCA-based dimension reduction was applied to the weights. 
# 
# The results are plotting on a two-dimensional plane.

# In[ ]:


learn.model


# ## 3.1. Place Embedding

# In[ ]:


place_array = learn.data.train_ds[0][0].classes["place_id"]

place_weight = learn.model.i_weight.weight.to("cpu")

place_pca =  place_weight.pca(2)

fac0, fac1 = place_pca.t()
fac0


# In[ ]:


import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

mpl.rcParams['axes.unicode_minus'] = False

# idxs = np.random.choice(len(top_itemes), 300, replace=False)

X = fac0.detach().numpy()
Y = fac1.detach().numpy()

plt.figure(figsize=(30,30))
plt.scatter(X, Y)
for i, x, y in zip(place_array, X, Y):
    plt.text(x,y,i, fontsize=15)
plt.show()


# 

# ## 3.2. Patient Embedding

# In[ ]:


patient_array = learn.data.train_ds[0][0].classes["patient_id"]

patient_weight = learn.model.u_weight.weight

patient_pca =  patient_weight.pca(2)

fac0, fac1 = patient_pca.t()
fac0


# In[ ]:


import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

mpl.rcParams['axes.unicode_minus'] = False

# idxs = np.random.choice(len(top_itemes), 300, replace=False)

X = fac0.detach().numpy()
Y = fac1.detach().numpy()

plt.figure(figsize=(30,30))
plt.scatter(X, Y)
for i, x, y in zip(patient_array, X, Y):
    plt.text(x,y,i, fontsize=15)
plt.show()


# 
# ![tree](http://t1.daumcdn.net/brunch/service/user/30eI/image/aFIiPM9FzmE3mRkE0VR_GAsS5wU.jpg)
# 
# `COVID-19` has been reported to spread patient 6 to patients 10 and 11, as well as in embedding vectors. 
# 
# I think this information will be better represented with more data.

# # 4. Future Work
# 
# The current dataset is still being added.
# 
# Because of the characteristic of embedding, the more data you get, the better the results, when this dataset is updated, the result will also be updated.
# 
# In addition, since it does not simply end in embedding, I will use this as input and apply it to our prediction and classification tasks.

# In[ ]:




