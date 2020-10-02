#!/usr/bin/env python
# coding: utf-8

# # Import modules

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
from fastai.vision import *
import matplotlib.pyplot as plt
import seaborn as sns
import os
import shutil


# # Define paths

# In[ ]:


path = Path('../input/plant-pathology-2020-fgvc7/images')


# In[ ]:


path.ls()[0:3]


# Train and test files are all mixed in the **images** folder. We are going to create 2 folders called Train and Test and then move the files there. Just to organize things a bit...

# In[ ]:


outpath = Path('/kaggle/working')
outpath.ls()


# In[ ]:


ii = open_image(path/'Train_1767.jpg')
ii.show()


# Great. Now let's take a look into the data

# # EDA

# In[ ]:


train = pd.read_csv('../input/plant-pathology-2020-fgvc7/train.csv')
test = pd.read_csv('../input/plant-pathology-2020-fgvc7/test.csv')


# In[ ]:


classes=['healthy', 'multiple_diseases', 'rust', 'scab']


# In[ ]:


train.head(5)


# In[ ]:


# https://www.kaggle.com/otzhora/fastai-simple-efficientnet-ensemble-solution
def get_tag(row):
    if row.healthy:
        return "healthy"
    if row.multiple_diseases:
        return "multiple_diseases"
    if row.rust:
        return "rust"
    if row.scab:
        return "scab"
def transform_data(train_labels):
    train_labels.image_id = [image_id+'.jpg' for image_id in train_labels.image_id]
    train_labels['tag'] = [get_tag(train_labels.iloc[idx]) for idx in train_labels.index]
    train_labels = train_labels.drop(columns=['healthy', 'multiple_diseases', 'rust', 'scab'])
    return train_labels


# In[ ]:


train_tag = transform_data(train)


# In[ ]:


train_tag


# Alright. So our data is a set of images and each of these can be assigned to 4 labels, namely healthy, multiple_diseases, rust and scab. 
# 
# I'm not a botanist, so I will just look some quick references explaining what these diseases are.

# ## Visualizing diseases

# ### Cedar apple rust

# Cedar apple rust (*Gymnosporangium juniperi-virginianae*) is a fungal disease that requires juniper plants to complete its complicated two year life-cycle. Spores overwinter as a reddish-brown gall on young twigs of various juniper species. In early spring, during wet weather, these galls swell and bright orange masses of spores are blown by the wind where they infect susceptible apple and crab-apple trees. The spores that develop on these trees will only infect junipers the following year. From year to year, the disease must pass from junipers to apples to junipers again; it cannot spread between apple trees.
# 
# On apple and crab-apple trees, look for pale yellow pinhead sized spots on the upper surface of the leaves shortly after bloom. These gradually enlarge to bright orange-yellow spots which make the disease easy to identify. Orange spots may develop on the fruit as well. Heavily infected leaves may drop prematurely.
# 
# 
# <img src="https://upload.wikimedia.org/wikipedia/commons/2/27/Cedar_apple_rust_cycle_for_Wikipedia.jpg" width="500px">
# Figure 1. Life cycle of the cedar apple rust (Source: https://en.wikipedia.org/wiki/Gymnosporangium_juniperi-virginianae)
# 
# <img src="http://www.missouribotanicalgarden.org/Portals/0/Gardening/Gardening%20Help/images/Pests/CedarApple_Rust168.jpg" width="500px">
# 
# Figure 2. Apple leafs infected with *G. juniperi-virginianae* (Source: http://www.missouribotanicalgarden.org/gardens-gardening/your-garden/help-for-the-home-gardener/advice-tips-resources/pests-and-problems/diseases/rusts/cedar-apple-rust.aspx)
# 
# <img src="https://www.fs.fed.us/wildflowers/plant-of-the-week/images/cedarapplerust/Gymnosparangium_juniperi-virginianae_23A.jpg" width="500px">
# 
# Figure 3. Yellow spots on apple leaves (Source: https://www.fs.fed.us/wildflowers/plant-of-the-week/gymnosporangium_juniperi-virginianae.shtml)
# 
# 
# Source: https://www.planetnatural.com/pest-problem-solver/plant-disease/cedar-apple-rust/

# Questions:
# 
# - Are there different steps of the life cycle represented on the dataset (e.g. horns, brownish spots and yellowish spots)?

# In[ ]:


rust = train_tag[train_tag['tag']=='rust']


# In[ ]:


rust.head()


# In[ ]:


rust_leaves = list(rust['image_id']);len(rust_leaves)


# In[ ]:


_,axs = plt.subplots(1,3,figsize=(22,22))
open_image(path/'Train_3.jpg').show(ax=axs[0],title='1')
open_image(path/'Train_10.jpg').show(ax=axs[1],title='2')
open_image(path/'Train_15.jpg').show(ax=axs[2],title='3')


# Apparently we do have different stages of the life cycle. For instance, image 3 shows an orange spot with yellow borders, while image 1 shows a mostly yellow spot with a darker center.

# Question:
# 
# - Can the different stages influence our predictions?

# Now let's do the same steps for the other classes

# ### Apple scab

# In[ ]:


scab = train_tag[train_tag['tag']=='scab']
scab.head(10)


# In[ ]:


scab_leaves = list(scab['image_id']);len(scab_leaves)


# A serious disease of apples and ornamental crabapples, apple scab (*Venturia inaequalis*) attacks both leaves and fruit. The fungal disease forms **pale yellow or olive-green spots on the upper surface of leaves**. **Dark, velvety spots may appear on the lower surface**. Severely infected leaves become twisted and puckered and may drop early in the summer.
# 
# Symptoms on fruit are similar to those found on leaves. Scabby spots are sunken and tan and may have velvety spores in the center. As these spots mature, they become larger and turn brown and corky. Infected fruit becomes distorted and may crack allowing entry of secondary organisms. Severely affected fruit may drop, especially when young.
# 
# Apple scab overwinters primarily in fallen leaves and in the soil. Disease development is favored by wet, cool weather that generally occurs in spring and early summer. Fungal spores are carried by wind, rain or splashing water from the ground to flowers, leaves or fruit. During damp or rainy periods, newly opening apple leaves are extremely susceptible to infection. The longer the leaves remain wet, the more severe the infection will be. Apple scab spreads rapidly between 55-75 degrees F.
# 
# <img src="https://www.backyardnature.net/n/09/090802sc.jpg" width="500px">
# 
# Figure 4. Apple scab (Source: https://www.backyardnature.net/n/09/090802sc.jpg)
# 
# Source: https://www.planetnatural.com/pest-problem-solver/plant-disease/apple-scab/

# In[ ]:


sc = open_image(path/'Train_0.jpg')
sc


# ### Multiple diseases

# In[ ]:


multi_d = train_tag[train_tag['tag']=='multiple_diseases']
multi_d.head(10)


# In[ ]:


multi_leaves =  list(multi_d['image_id'])


# This time I'm not going to classifying each of the diseases on the leaves (because I'm not an expert in plants diseases). We are just going to take a look at some samples.

# In[ ]:


mt1 = open_image(path/'Train_122.jpg')
mt2 = open_image(path/'Train_113.jpg')
mt3 = open_image(path/'Train_95.jpg')
_,axs = plt.subplots(1,3,figsize=(22,22))
mt1.show(ax=axs[0],title='1')
mt2.show(ax=axs[1],title='2')
mt3.show(ax=axs[2],title='3')


# ### Healthy

# In[ ]:


ht = train_tag[train_tag['tag']=='healthy']
ht.head(10)


# In[ ]:


healthy_leaves = list(ht['image_id'])
healthy_leaves[0:3]


# In[ ]:


ht1 = open_image(path/'Train_2.jpg')
ht2 = open_image(path/'Train_4.jpg')
ht3 = open_image(path/'Train_5.jpg')
_,axs = plt.subplots(1,3,figsize=(22,22))
ht1.show(ax=axs[0],title='1')
ht2.show(ax=axs[1],title='2')
ht3.show(ax=axs[2],title='3')


# ## Class distribution

# We have 4 classes in this dataset:
# 
# * Healthy
# * Multiple diseases
# * Scab
# * Rust
# 
# What is the distribution of these classes? Are there more samples of one class than the others?

# In[ ]:


print(len(scab),len(rust),len(multi_d),len(ht))


# In[ ]:


dd = train_tag.groupby('tag')['image_id'].count().reset_index()


# In[ ]:


dd


# In[ ]:


sns.barplot(x='tag', y='image_id', data=dd)


# There's only a small number of leaves with multiple diseases. The dataset is dominated by scab, rust and healthy plants.

# In[ ]:


# https://www.kaggle.com/lextoumbourou/plant-pathology-2020-eda-training-fastai2
_, axes = plt.subplots(ncols=4, nrows=1, constrained_layout=True, figsize=(10, 3))
for ax, column in zip(axes, classes):
    train[column].value_counts().plot.bar(title=column,ax=ax)

plt.show()


# There is an imbalance, which is most pronouced for multiple_diseases.

# # Prepare data for training

# # Train model

# ## Data splitting

# We are going to split our dataset using a stratificantion strategy because our data is imbalanced

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


trainsplit,validsplit = train_test_split(train_tag,test_size=0.30,random_state=42,stratify=train_tag['tag'])


# In[ ]:


trainsplit.shape,validsplit.shape


# In[ ]:


trainsplitdd = trainsplit.groupby('tag')['image_id'].count().reset_index()


# In[ ]:


trainsplitdd


# In[ ]:


validplitdd = validsplit.groupby('tag')['image_id'].count().reset_index()


# In[ ]:


validplitdd


# In[ ]:


sns.barplot(x='tag', y='image_id', data=trainsplitdd)


# In[ ]:


valid_idx = validsplit.index
train_idx = trainsplit.index


# ## Create databunch

# In[ ]:


tfms = get_transforms()


# In[ ]:


np.random.seed(42)
src = ImageList.from_df(path=path,df=train_tag).split_by_idx(valid_idx).label_from_df('tag')


# In[ ]:


src


# In[ ]:


data_512 = src.transform(tfms,size=512).databunch(bs=8).normalize(imagenet_stats)
#data_1020 = src.transform(tfms,size=1020).databunch(bs=8).normalize(imagenet_stats)


# In[ ]:


data_512,data_512.show_batch(5,figsize=(7,7)),data_512.classes,data_512.c


# ## Select architeture

# In[ ]:


learn = cnn_learner(data_512,models.densenet169,wd=1e-4,metrics=accuracy,model_dir=outpath)


# ## Fit

# In[ ]:


learn.lr_find(num_it=300)


# In[ ]:


learn.recorder.plot(suggestion=True)


# In[ ]:


lr = 2e-3
learn.fit_one_cycle(8,lr)


# In[ ]:


learn.recorder.plot_losses()


# In[ ]:


learn.save('stage1_512')


# In[ ]:


(outpath).ls()


# In[ ]:


learn.load('stage1_512')


# In[ ]:


learn.unfreeze()


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(8,slice(1e-6,1e-5/5))


# In[ ]:


learn.save('stage2_512')


# ## Going bigger 

# In[ ]:


#data_1020 = src.transform(tfms,size=1020).databunch(bs=8).normalize(imagenet_stats)

#learn.data = data_1020
#data_1020.train_ds[0][0].shape


# In[ ]:


#learn.freeze()


# In[ ]:


#learn.lr_find()


# In[ ]:


#learn.recorder.plot()


# In[ ]:


#learn.fit_one_cycle(5, slice(lr))


# In[ ]:


#learn.save('stage1_1020')


# In[ ]:




#learn.unfreeze()


# In[ ]:


#learn.fit_one_cycle(5, slice(1e-5, lr/5))


# In[ ]:


#learn.save('stage2_1020')


# # Interpretation and predictions

# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)


# In[ ]:


interp.plot_confusion_matrix()


# Multiple diseases was underrepresented on the validation set. We will try a stratified split next time.

# In[ ]:


interp.plot_top_losses(k=16,figsize=(20,20),heatmap=False)


# We can see some leaves with very faint spots, indicating rust or scab, being classified as healthy. Maybe changing the lightining can solve this.
# 
# As expected, multiple_diseases are being misclassified as other diseases due to low representation on the validation set.

# In[ ]:


learn.load('stage2_512')


# In[ ]:


learn.export(outpath/'export.pkl')


# In[ ]:


outpath.ls()


# In[ ]:


test_images = ImageList.from_folder(path)
test_images.filter_by_func(lambda x: x.name.startswith("Test"))


# In[ ]:


test_df = pd.read_csv('../input/plant-pathology-2020-fgvc7/test.csv')


# In[ ]:


test_df.image_id = [image_id+'.jpg' for image_id in test_df.image_id]


# In[ ]:


test_df.head()


# In[ ]:


learn = load_learner(outpath)


# In[ ]:


all_test_preds = []
from tqdm import tqdm_notebook as tqdm
for item in tqdm(test_images.items):
    name = item.name[:-4]
    img = open_image(item)
    preds = learn.predict(img)[2]
    all_test_preds.append(preds)
   # test_df.loc[name]['healthy'] = preds[0]
   # test_df.loc[name]['multiple_diseases'] = preds[1]
   # test_df.loc[name]['rust'] = preds[2]
  #  test_df.loc[name]['scab'] = preds[3]


# In[ ]:


aa = [f.numpy() for f in all_test_preds]


# In[ ]:


bb = np.stack(aa,axis=0)
len(bb)


# In[ ]:


bb


# In[ ]:


test_df_output = pd.concat([test_df, pd.DataFrame(bb, columns=classes)], axis=1).round(6)


# In[ ]:


test_df_output.head()


# In[ ]:


test_df_output['image_id'] = test_df['image_id'].str.strip('.jpg')


# In[ ]:


test_df_output.to_csv('/kaggle/working/submission3.csv',index=False)


# In[ ]:


data.save(outpath/'data.pkl')


# In[ ]:


outpath.ls()


# In[ ]:




