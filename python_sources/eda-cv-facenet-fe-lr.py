#!/usr/bin/env python
# coding: utf-8

# # Log
# - Version 5:
#     - Use VGGFace to replace FaceNet: 2048 features instead of 128 features.
#     - Do not use very similar pictures of the same person, in order to save RAM. Use FaceNet to pick similar pictures.

# # Introduction
# 
# > This is my first kernel shared :)
# 
# - The main idea of this kernel is to share some observations on the dataset and some recommandations on the cross-validation folders. 
# - The Pipeline is: firstly use FaceNet to have features extracted, then use these features to train a traditional machine learning model, and use this model to predict on test set.
# 
# The FaceNet idea was inspired by [Khoi Nguyen](https://www.kaggle.com/suicaokhoailang) and his [kernel](https://www.kaggle.com/suicaokhoailang/facenet-baseline-in-keras-0-749-lb). 
# 
# I am new to Deep Learning, so firstly I only used deep net as feature extractor, then use traditional way to train the model. Here we can use other deep net, like VGGFace, to replace FaceNet as feature extractor. After feature extraction, we can test on very different traditional machine learning models.

# # Load useful libraries

# In[ ]:


import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from imageio import imread
from skimage.transform import resize
from keras.models import load_model
from tqdm._tqdm_notebook import tqdm_notebook
from sklearn.model_selection import GroupKFold

from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


# In[ ]:


import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)
warnings.filterwarnings('ignore') #this one works good!


# # Use nearest neighbor to remove similar pictures

# In[ ]:


train_df = pd.read_csv("../input/recognizing-faces-in-the-wild/train_relationships.csv")
train_df.head()


# In[ ]:


# Find all the train images

def findAllTrain(train_folder):
    train_li=[]
    for fam in os.listdir(train_folder):
        for pers in os.listdir(os.path.join(train_folder,fam)):
            for pic in os.listdir(os.path.join(train_folder,fam,pers)):
                train_li.append(os.path.join(fam,pers,pic))
    
    return train_li

train_fd = '../input/recognizing-faces-in-the-wild/train'

train_file_li=findAllTrain(train_fd)

print('There are {} images in the train dataset.'.
      format(len(train_file_li)))

#Create a dict to store all the train images
train_file_dict=dict(zip(train_file_li,range(len(train_file_li))))

# Create a DataFrame to store all the train images
train_file_df = pd.DataFrame()
train_file_df['image_fp']=train_file_li
train_file_df.sample(5)


# In[ ]:


get_ipython().system('pip install git+https://github.com/rcmalli/keras-vggface.git')


# In[ ]:


from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from keras.preprocessing import image


# In[ ]:


# Convolution Features
vgg_features = VGGFace(include_top=False, input_shape=(224, 224, 3), pooling='avg', model='resnet50') # pooling: None, avg or max

# After this point you can use your model to predict.


# In[ ]:



def load_images(filepaths,target_size=(224, 224)):
    
    aligned_images = []
    for filepath in filepaths:
        img = image.load_img(filepath, target_size=target_size)
        x_ = image.img_to_array(img)
        aligned_images.append(x_)
            
    return preprocess_input(np.array(aligned_images),version=2)

def calc_embs(filepaths,batch_size=512,target_size=(224, 224)):
    pd = []
    for start in tqdm_notebook(range(0, len(filepaths), batch_size)):
        aligned_images = load_images(filepaths[start:start+batch_size],target_size=target_size)
        pd.append(vgg_features.predict_on_batch(aligned_images))
    embs = np.concatenate(pd)

    return embs


# In[ ]:


feature_size=2048


# In[ ]:


# Calculate embs for train images

train_embs = calc_embs([os.path.join("../input/recognizing-faces-in-the-wild/train", f) for f in train_file_df['image_fp']])
train_file_df=pd.concat([train_file_df, pd.DataFrame(train_embs,columns=['fe'+str(i) for i in range(feature_size)])],axis=1)
train_file_df.head()


# In[ ]:


# Get family ID for each image
train_file_df['fam_person']=" "
train_file_df['fam_person']=train_file_df['image_fp'].apply(lambda x: x[:x.find('P')-1])
train_file_df.head()


# In[ ]:


from sklearn.neighbors import NearestNeighbors

radius=60
neigh = NearestNeighbors(radius=radius)

for person in train_file_df['fam_person'].unique():
    if person=='F0601/MID1':
        print(person)
        print(len(train_file_df.loc[train_file_df['fam_person']==person,:]))
        person_df = train_file_df.loc[train_file_df['fam_person']==person,:]
        neigh.fit(person_df.iloc[:,1:2049])
#         rng = neigh.radius_neighbors([person_df.iloc[0,1:2049]])
#         print(len(rng[0][0])-1)
        print(person_df.apply(lambda x: len(neigh.radius_neighbors([x[1:2049]])[0][0])-1,axis=1))
        print(person_df.apply(lambda x: neigh.radius_neighbors([x[1:2049]]),axis=1))


# In[ ]:


show2pic('../input/recognizing-faces-in-the-wild/train/',
         train_file_df.loc[2427,'image_fp'] + '-' + train_file_df.loc[2437,'image_fp'])


# In[ ]:


show2pic('../input/recognizing-faces-in-the-wild/train/',
         train_file_df.loc[3313,'image_fp'] + '-' + train_file_df.loc[3319,'image_fp'])


# In[ ]:


# use this function to show some image pairs.
def show2pic(fd,paire):
    plt.figure(figsize=(7,10))
    plt.subplot(121)
    plt.imshow(imread(os.path.join(fd,paire.split('-')[0])))
    plt.axis('off')
    plt.title(paire.split('-')[0])
    plt.subplot(122)
    plt.imshow(imread(os.path.join(fd,paire.split('-')[1])))
    plt.axis('off')
    plt.title(paire.split('-')[1])


# In[ ]:


show2pic('../input/recognizing-faces-in-the-wild/train/',
         train_file_df.loc[3330,'image_fp'] + '-' + train_file_df.loc[3336,'image_fp'])


# # Get image pairs
# 
# Find all the possible image pairs for training.

# In[ ]:


train_df = pd.read_csv("../input/recognizing-faces-in-the-wild/train_relationships.csv")
train_df.head()


# In[ ]:


# Find all the train images

def findAllTrain(train_folder):
    train_li=[]
    for fam in os.listdir(train_folder):
        for pers in os.listdir(os.path.join(train_folder,fam)):
            for pic in os.listdir(os.path.join(train_folder,fam,pers)):
                train_li.append(os.path.join(fam,pers,pic))
    
    return train_li

train_fd = '../input/recognizing-faces-in-the-wild/train'

train_file_li=findAllTrain(train_fd)

print('There are {} images in the train dataset.'.
      format(len(train_file_li)))

#Create a dict to store all the train images
train_file_dict=dict(zip(train_file_li,range(len(train_file_li))))

# Create a DataFrame to store all the train images
train_file_df = pd.DataFrame()
train_file_df['image_fp']=train_file_li
train_file_df.sample(5)


# In[ ]:


# Find all the image pairs with kinship

train_fd = '../input/recognizing-faces-in-the-wild/train'

index_p1_li=[]
index_p2_li=[]

for idx, row in tqdm_notebook(train_df.iterrows(), total=len(train_df)):
    if os.path.isdir(os.path.join(train_fd,row['p1'])) and os.path.isdir(os.path.join(train_fd,row['p2'])): # some folders do not exist !!
        for p1_pic in os.listdir(os.path.join(train_fd,row['p1'])):
            for p2_pic in os.listdir(os.path.join(train_fd,row['p2'])):
                index_f1=train_file_dict[os.path.join(row['p1'].split('/')[0],row['p1'].split('/')[1],p1_pic)]
                index_f2=train_file_dict[os.path.join(row['p2'].split('/')[0],row['p2'].split('/')[1],p2_pic)]
                if index_f1<index_f2: # force the image pairs to have the same order of persons
                    index_p1_li.append(index_f1)
                    index_p2_li.append(index_f2)
                else:
                    index_p1_li.append(index_f2)
                    index_p2_li.append(index_f1)
                    
train_pairs_kinship=pd.DataFrame()
train_pairs_kinship['p1']=index_p1_li
train_pairs_kinship['p2']=index_p2_li

index_p1_li=[]
index_p2_li=[]

print('Total image pairs with kinship: {}'.format(len(train_pairs_kinship)))
train_pairs_kinship.sample(5)


# **How about the image pairs from the same person? Should they be used as positive samples (with kinship)? I think YES.**
# 
# Because basicly we are training a model to identify the similarity of two images, the same person's images can bring us more postive samples. 

# In[ ]:


# make image pairs of the same person
# for example: for this person "F0002\MID1", there are 10 images in the folder, so it can make 10*9/2=45 pairs.

def make_pair_same_person(source,pre_path):
    res_p1_li = []
    res_p2_li = []
    for p1 in range(len(source)):
        for p2 in range(p1+1,len(source)):
            index_f1=train_file_dict[os.path.join(pre_path,source[p1])]
            index_f2=train_file_dict[os.path.join(pre_path,source[p2])]
            if index_f1<index_f2: # force the image pairs to have the same order of persons
                res_p1_li.append(index_f1)
                res_p2_li.append(index_f2)
            else:
                res_p1_li.append(index_f2)
                res_p2_li.append(index_f1)
            
    return (res_p1_li,res_p2_li)

index_p1_li = []
index_p2_li = []
for fam in os.listdir(train_fd):
    for pers in os.listdir(os.path.join(train_fd,fam)):
        res_temp = make_pair_same_person([pic for pic in os.listdir(os.path.join(train_fd,fam,pers))],os.path.join(fam,pers))
        index_p1_li.extend(res_temp[0])
        index_p2_li.extend(res_temp[1])

train_pairs_same=pd.DataFrame()
train_pairs_same['p1']=index_p1_li
train_pairs_same['p2']=index_p2_li

index_p1_li = []
index_p2_li = []

print('Total image pairs of same person: {}'.format(len(train_pairs_same)))

train_pairs_same.sample(5)


# In[ ]:


fig, ax = plt.subplots(figsize=(12,2))
y2show=[len(train_pairs_kinship),len(train_pairs_same)]
plt.barh(range(2),y2show,0.35)
plt.title('Image pair number')
plt.yticks(range(2), ('With kinship', 'From same person'),)
plt.box(on=None)
plt.xticks([], [])
for i, v in enumerate(y2show):
    ax.text(v+1000, i-0.05, str(v), color='blue', fontweight='bold')


# In[ ]:


train_pairs_kinship=pd.concat([train_pairs_kinship,train_pairs_same],ignore_index=True) # Combine them together
train_pairs_same=None # to free RAM
print('Total POSITIVE image pairs: {}'.format(len(train_pairs_kinship)))


# In[ ]:


# Get all the possible image pairs

index_p1_li = []
index_p2_li = []

for p1 in tqdm_notebook(range(len(train_file_li))):    
    for p2 in range(p1+1,len(train_file_li)):
        index_p1_li.append(p1)
        index_p2_li.append(p2)

train_pairs_all=pd.DataFrame()
train_pairs_all['p1']=index_p1_li
index_p1_li = []
train_pairs_all['p2']=index_p2_li
index_p2_li = []

print('Total image pairs: {}'.format(len(train_pairs_all)))

train_pairs_all.sample(5)


# In[ ]:


# Add a col "is_related": 1 if POS, 0 if NEG

kin_index=np.arange(len(train_pairs_all))[train_pairs_all.merge(train_pairs_kinship, on=['p1','p2'],how='left', indicator=True)['_merge']=='both']
train_pairs_all['is_related']=0
train_pairs_all.loc[kin_index,'is_related']=1
kin_index=None # to free RAM
train_pairs_kinship=None # to free RAM


# In[ ]:


fig, ax = plt.subplots(figsize=(12,2))
y2show=[train_pairs_all.query('is_related == 0').shape[0],train_pairs_all.query('is_related == 1').shape[0]]
plt.barh(range(2),y2show,0.35)
plt.title('Image pair number')
plt.yticks(range(2), ('No Kinship (NEG)','With kinship (POS)'))
plt.box(on=None)
plt.xticks([], [])
for i, v in enumerate(y2show):
    ax.text(v+1000, i-0.05, str(v), color='blue', fontweight='bold')


# In[ ]:


print("The number of negative samples is {:.0f} times of positive samples!".
      format(train_pairs_all.query('is_related == 0').shape[0]/train_pairs_all.query('is_related == 1').shape[0]))


# # Create folders for cross-validation
# 
# It is very important to have good cross-validation folders, in order to:
# - Avoid data leakage
# - Optimize model parameters
# - Close the gap between your validation set and LB score
# 
# The main idea is: **The same family does NOT appear in two different folds!** 
# 
# This is the same as **GroupKFold** in Scikit-Learn. However, we can not use GroupKFold directly in this case. If we define one "family" as one "group", then it will be difficult to define the family ID for negative samples (image pair with NO kinship), because the 2 persons in negative image pairs can be from 2 different families.
# 
# So we have to create our own group number for NEG samples. Firstly, use GroupKFold to seperate POS samples into N folders (use family ID as group). Then we can get a family list for each folder. And this family list can be used to get NEG samples for each folders. For example, the 2 persons in a NEG sample are from family-1 and family-2, and both families are in folder-A's family list, then this NEG sample can be assigned to folder-A.
# 
# However, this method has a problem on this dataset!! Let's see below:

# In[ ]:


# Get family ID for each image

train_file_df['fam']=-1
train_file_df['fam']=train_file_df['image_fp'].apply(lambda x: int(x[1:5]))
train_file_df.reset_index(inplace=True)
train_file_df.head()


# In[ ]:


print('There are {} families in the train set.'.format(len(train_file_df.fam.unique())))


# In[ ]:


# Get family ID for each POSimage pair (use p1 only)
train_pairs_kinship = train_pairs_all.query('is_related == 1')
train_pairs_kinship=train_pairs_kinship.merge(train_file_df[['index','fam']], left_on='p1',right_on='index',how='left').drop(columns=['index'])
train_pairs_kinship.sample(5)


# In[ ]:


fig, ax = plt.subplots(figsize=(12,5))
sns.countplot(x='fam',data=train_pairs_kinship,
              order=train_pairs_kinship.fam.value_counts().iloc[:20].index)
plt.title('Top 20 families (image pair with kinship | POS samples)')
ax.text(12, 30000, 'Average POS samples per family is {:.0f}'.format(len(train_pairs_kinship)/len(train_pairs_kinship.fam.unique())),fontsize=12)
for i, v in enumerate(train_pairs_kinship['fam'].value_counts()[:20]):
    ax.text(i-0.4, v+500, str(v),color='gray')
plt.box(on=None)
plt.yticks([]);


# In[ ]:


print('Family 601 contains {:.0f}% of image pair of all the POS samples!'.format(train_pairs_kinship['fam'].value_counts().tolist()[0]/len(train_pairs_kinship)*100))
print('Family 9 contains {:.0f}% of image pair of all the POS samples.'.format(train_pairs_kinship['fam'].value_counts().tolist()[1]/len(train_pairs_kinship)*100))


# In[ ]:


sns.distplot(train_pairs_kinship.fam.value_counts());


# As shown above, **the family 601 represents 35% of POS samples**.
# 
# What happens? If you open the family folder, you'll find that it's **British Royal Family** ! Of course!
# 
# This kernel [EDA with Plotly-Smart, Cute and Pretty People](https://www.kaggle.com/gowrishankarin/eda-with-plotly-smart-cute-and-pretty-people) by [Gowri Shankar](https://www.kaggle.com/gowrishankarin) shows great visualizations on this.
# 
# Why does it cause a problem to create our CV folders? 
# 
# We want each folder to have equivalent number of samples. If we cut our samples into 3 or more folders, and we don't want the same family appears in two different folds (to avoid data leakage), so the British Royal Family will take one whole folder. It will bias the cross validation score. 
# 
# So, we may reduce the POS sample number per family to a certain limit, like 3000. If the number is above the limit, only use 3000 random samples from tha family. 

# In[ ]:


limit_number = 300 # Only use 300 because of lack of RAM for 2048 features

index_li = train_pairs_kinship['fam'].value_counts()[lambda x:x<=limit_number].index
train_fam_lim_df = train_pairs_kinship[train_pairs_kinship['fam'].isin(index_li)]

for i in train_pairs_kinship['fam'].value_counts()[lambda x:x>limit_number].index:
    df_temp = train_pairs_kinship.query('fam == {}'.format(i)).sample(limit_number,replace=False,random_state=2019)
    train_fam_lim_df = pd.concat([train_fam_lim_df, df_temp])
    
train_fam_lim_df=train_fam_lim_df.reset_index() # Reset index for GroupKFold method

print('Number of POS samples in the selected dataset: {}'.format(len(train_fam_lim_df)))


# In[ ]:


sns.distplot(train_fam_lim_df.fam.value_counts());


# In[ ]:


gkf = GroupKFold(n_splits=6) # Group 6 as test set, Group0-5 as CV folders.

train_fam=train_fam_lim_df['fam']

fam_group=np.ones(max(train_fam_lim_df['fam'])+1)*(-1)
fam_group=fam_group.astype(int)

for idx,( _, test_index) in enumerate(gkf.split(X=train_fam,groups=train_fam)):
    print("Group {}: {}".format(idx,np.unique(train_fam[test_index])))
    fam_group[np.unique(train_fam[test_index])]=idx
    print('-'*85)


# 6 groups have been created, now it's time to add NEG samples to each group.
# 
# Is it the best to have the equal number of POS and NEG samples? Or shall we have more portion for NEG samples, like 2:1? I will test other portions later.

# In[ ]:


# Get group ID for each image

train_file_df['group']=train_file_df['fam'].apply(
    lambda x: fam_group[x])

train_file_df.sample(5)


# In[ ]:


# Get group ID for each image pair

tqdm_notebook.pandas()
group_li=train_file_df['group'].tolist()

train_pairs_all['group1']=train_pairs_all['p1'].progress_apply(lambda x: group_li[x])
train_pairs_all['group2']=train_pairs_all['p2'].progress_apply(lambda x: group_li[x])
tmp_li = (train_pairs_all['group1']==train_pairs_all['group2'])*(train_pairs_all['group1']+1)-1
train_pairs_all.drop(columns=['group1','group2'],inplace=True)
train_pairs_all['group']=tmp_li
tmp_li=None # to free RAM
train_pairs_all.sample(5)


# In[ ]:


fig, ax = plt.subplots(figsize=(12,5))
sns.countplot(y='group',data=train_pairs_all,orient='v')
plt.title('Number of image pair in each group')
ax.text(30000000,1, '"-1" means no group is assigned.',fontsize=16)
ax.text(25000000,2,
        '{:.0f}% of image pairs have no group assigned.'.format(train_pairs_all.query('group == -1').shape[0]/len(train_pairs_all)*100),
        fontsize=16)
plt.box(on=None)
plt.xticks([]);


# In[ ]:


# Drop group==-1
train_pairs_all = train_pairs_all[train_pairs_all['group']!=-1]
# Shuffle
train_pairs_all = train_pairs_all.sample(frac=1,random_state=2019)


# In[ ]:


train_dataset_df=train_fam_lim_df
train_dataset_df['group']=train_dataset_df['fam'].apply(lambda x: fam_group[x])
train_dataset_df.drop(columns=['index','fam'],inplace=True)
train_dataset_df.head()


# In[ ]:


group_num = train_fam_lim_df.groupby('group')['is_related'].count().tolist()
portion=1 # get equal number of NEG / POS
df_temp = pd.concat(
    [t.head(int(group_num[g]*portion)) for g, t in train_pairs_all.query('is_related == 0').groupby('group', sort=False, as_index=False)],
    ignore_index=True)

train_dataset_df=pd.concat([train_dataset_df,df_temp],ignore_index=True)
train_dataset_df.shape


# In[ ]:


# to free RAM
train_pairs_all=None
train_pairs_kinship=None


# In[ ]:


fig, ax = plt.subplots(figsize=(12,5))
sns.countplot(y='group',data=train_dataset_df,orient='v',hue='is_related')
plt.title('Number of image pair in each group')
plt.box(on=None)
plt.xticks([]);


# In[ ]:


train_dataset_df.head()


# # Use VGGFace to calculate 2048 features

# Thanks : [VGGFace Baseline 197X197](https://www.kaggle.com/hsinwenchang/vggface-baseline-197x197) for the example of using VGGFace.

# In[ ]:


get_ipython().system('pip install git+https://github.com/rcmalli/keras-vggface.git')


# In[ ]:


from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from keras.preprocessing import image


# In[ ]:


# Convolution Features
vgg_features = VGGFace(include_top=False, input_shape=(224, 224, 3), pooling='avg', model='resnet50') # pooling: None, avg or max

# After this point you can use your model to predict.


# In[ ]:


def load_images(filepaths,target_size=(224, 224)):
    
    aligned_images = []
    for filepath in filepaths:
        img = image.load_img(filepath, target_size=target_size)
        x_ = image.img_to_array(img)
        aligned_images.append(x_)
            
    return preprocess_input(np.array(aligned_images),version=2)


# In[ ]:


def calc_embs(filepaths,batch_size=512,target_size=(224, 224)):
    pd = []
    for start in tqdm_notebook(range(0, len(filepaths), batch_size)):
        aligned_images = load_images(filepaths[start:start+batch_size],target_size=target_size)
        pd.append(vgg_features.predict_on_batch(aligned_images))
    embs = np.concatenate(pd)

    return embs


# In[ ]:


# feature_size=128
feature_size=2048


# In[ ]:


# Calculate embs for train images

train_embs = calc_embs([os.path.join("../input/recognizing-faces-in-the-wild/train", f) for f in train_file_df['image_fp']])
train_file_df=pd.concat([train_file_df, pd.DataFrame(train_embs,columns=['fe'+str(i) for i in range(feature_size)])],axis=1)
train_file_df.head()


# In[ ]:


train_embs = None # to free RAM


# In[ ]:


# Use absolute distance as final features

p1_df = train_dataset_df.merge(train_file_df, left_on='p1',right_on='index',how='left').iloc[:,8:]
p2_df = train_dataset_df.merge(train_file_df, left_on='p2',right_on='index',how='left').iloc[:,8:]

train_dataset_df = pd.concat([train_dataset_df, abs(p1_df-p2_df)],axis=1)
p1_df=None
p2_df=None
train_dataset_df.head()


# In[ ]:


train_dataset_df.shape


# # Train model

# In[ ]:


# shuffle the dataset
train_dataset_df=train_dataset_df.sample(frac=1,random_state=2019).reset_index(drop=True)
train_dataset_df.head()


# In[ ]:


# X=train_dataset_df.iloc[:,4:]
# y=train_dataset_df.iloc[:,2]
# X.shape, y.shape


# In[ ]:


# X=None
# y=None


# In[ ]:


# X_train=X[train_dataset_df['group']!=5]
# X_test=X[train_dataset_df['group']==5]
# y_train=train_dataset_df['is_related'][train_dataset_df['group']!=5]
# y_test=train_dataset_df['is_related'][train_dataset_df['group']==5]

# y_train_group=train_dataset_df['group'][train_dataset_df['group']!=5]

# X_train.shape,X_test.shape,y_train.shape,y_test.shape,y_train_group.shape


# In[ ]:


# X_train=train_dataset_df.query('group != 5').iloc[:,4:]
# X_test=train_dataset_df.query('group == 5').iloc[:,4:]
# y_train=train_dataset_df.query('group != 5')['is_related']
# y_test=train_dataset_df.query('group == 5')['is_related']

# y_train_group=train_dataset_df.query('group != 5')['group']


# In[ ]:


# X_train=None
# X_test=None
# y_train=None
# y_test=None
# y_train_group=None


# In[ ]:


# group kfolder
group_kfold = GroupKFold(n_splits=5)


# In[ ]:


# # this is a check of GroupKFold result

# for train_index, test_index in group_kfold.split(X_train, y_train, y_train_group):
#     #print("TRAIN:", train_index, "TEST:", test_index)
#     print(np.unique(y_train_group.as_matrix()[train_index]))
#     print(np.unique(y_train_group.as_matrix()[test_index]))
#     print('-'*20)


# In[ ]:


# # this is a check of GroupKFold result

# for train_index, test_index in group_kfold.split(train_dataset_df.query('group != 5').iloc[:,4:], 
#                                                  train_dataset_df.query('group != 5')['is_related'], 
#                                                  train_dataset_df.query('group != 5')['group']):
#     #print("TRAIN:", train_index, "TEST:", test_index)
#     print(np.unique(train_dataset_df.query('group != 5')['group'].as_matrix()[train_index]))
#     print(np.unique(train_dataset_df.query('group != 5')['group'].as_matrix()[test_index]))
#     print('-'*20)


# Only LogisticRegression model is tested. You can use other more advanced model, like lightgbm, to train on the same dataset. And use GridSearchCV to tweat super parameters.

# In[ ]:


model=LogisticRegression(random_state=2019)
res=cross_validate(model,train_dataset_df.query('group != 5').iloc[:,4:],
                   train_dataset_df.query('group != 5')['is_related'],
                   cv=group_kfold,n_jobs=1,
                   groups=train_dataset_df.query('group != 5')['group'],
                   scoring=('accuracy', 'roc_auc'))
print("Mean ROC_AUC score: {:.4f} (std: {:.4f})".format(res['test_roc_auc'].mean(),res['test_roc_auc'].std()))


# In[ ]:


from lightgbm import LGBMClassifier

model = LGBMClassifier(random_state=2019,
                       n_jobs=-1,
                      n_estimators=1000,
                      num_leaves=40,
                      max_depth=12)


# In[ ]:


# model = LGBMClassifier(random_state=2019,n_jobs=1)
# res=cross_validate(model,train_dataset_df.query('group != 5').iloc[:,4:],
#                    train_dataset_df.query('group != 5')['is_related'],
#                    cv=group_kfold,n_jobs=1,
#                    groups=train_dataset_df.query('group != 5')['group'],
#                    scoring=('accuracy', 'roc_auc'))
# print("Mean ROC_AUC score: {:.4f} (std: {:.4f})".format(res['test_roc_auc'].mean(),res['test_roc_auc'].std()))


# In[ ]:


# test on Test Set
model.fit(train_dataset_df.query('group != 5').iloc[:,4:],train_dataset_df.query('group != 5')['is_related'])
print("ROC_AUC socre on test set: {:.3f}".format(roc_auc_score(train_dataset_df.query('group == 5')['is_related'],
                                                               model.predict_proba(train_dataset_df.query('group == 5').iloc[:,4:])[:,1])))


# # Predict and Export result

# In[ ]:


# Calculate embs for test images
test_images = os.listdir("../input/recognizing-faces-in-the-wild/test/")
test_embs = calc_embs([os.path.join("../input/recognizing-faces-in-the-wild/test/", f) for f in test_images])


# In[ ]:


img2idx = dict()
for idx, img in enumerate(test_images):
    img2idx[img] = idx


# In[ ]:


test_df = pd.read_csv("../input/recognizing-faces-in-the-wild/sample_submission.csv")
test_df.head()


# In[ ]:


test_np = []
for idx, row in tqdm_notebook(test_df.iterrows(), total=len(test_df)):
    imgs = [test_embs[img2idx[img]] for img in row.img_pair.split("-")]
    test_np.append(abs(imgs[0]-imgs[1]))
test_np = np.array(test_np)


# In[ ]:


# Predict
model.fit(X,y)
probs = model.predict_proba(test_np)[:,1]

sub_df = pd.read_csv("../input/recognizing-faces-in-the-wild/sample_submission.csv")
sub_df.is_related = probs


# In[ ]:


sub_df.hist();


# In[ ]:


# use this function to show some image pairs.
def show2pic(fd,paire):
    plt.figure(figsize=(7,10))
    plt.subplot(121)
    plt.imshow(imread(os.path.join(fd,paire.split('-')[0])))
    plt.axis('off')
    plt.title(paire.split('-')[0])
    plt.subplot(122)
    plt.imshow(imread(os.path.join(fd,paire.split('-')[1])))
    plt.axis('off')
    plt.title(paire.split('-')[1])


# In[ ]:


sub_df.sort_values('is_related',ascending=False).head(10)


# In[ ]:


# here is an example of the top 5th result.

show2pic('../input/recognizing-faces-in-the-wild/test/',sub_df.loc[4636,'img_pair'])


# In[ ]:


# export result to csv file
sub_df.to_csv("submission.csv", index=False)


# In[ ]:





# In[ ]:





# In[ ]:


import sys

# These are the usual ipython objects, including this one you are creating
ipython_vars = ['In', 'Out', 'exit', 'quit', 'get_ipython', 'ipython_vars']

# Get a sorted list of the objects and their sizes
sorted([(x, sys.getsizeof(globals().get(x))/1e6) for x in dir() if not x.startswith('_') and x not in sys.modules and x not in ipython_vars], key=lambda x: x[1], reverse=True)

