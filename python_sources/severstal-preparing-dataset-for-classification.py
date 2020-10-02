#!/usr/bin/env python
# coding: utf-8

# In[ ]:


DATASET_DIR = '../input/severstal-steel-defect-detection/'
TEST_SIZE = 0.1
RANDOM_STATE = 123


# # Import modules

# In[ ]:


import pandas as pd
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from shutil import copyfile
from sklearn.model_selection import train_test_split


# In[ ]:


df = pd.read_csv(os.path.join(DATASET_DIR, 'train.csv'))


# In[ ]:


df['Image'] = df['ImageId_ClassId'].map(lambda x: x.split('_')[0])
df['HavingDefection'] = df['EncodedPixels'].map(lambda x: 0 if x is np.nan else 1)

image_col = np.array(df['Image'])
image_files = image_col[::4]
y = np.array(df['HavingDefection']).reshape(-1, 4)


# In[ ]:


df.head()


# In[ ]:


num_img_class_1 = np.sum(y[:, 0])
num_img_class_2 = np.sum(y[:, 1])
num_img_class_3 = np.sum(y[:, 2])
num_img_class_4 = np.sum(y[:, 3])
print('Class 1:', num_img_class_1)
print('Class 2:', num_img_class_2)
print('Class 3:', num_img_class_3)
print('Class 4:', num_img_class_4)


# # Split dataset into training and validation sets

# In[ ]:


X_train, X_val, y_train, y_val = train_test_split(image_files, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)


# In[ ]:


df.head()


# In[ ]:


print(X_train.shape)
print(y_train.shape)
print(X_val.shape)
print(y_val.shape)


# # Visualize some images and corresponding labels

# In[ ]:


train_pairs = np.array(list(zip(X_train, y_train)))
samples = train_pairs[np.random.choice(train_pairs.shape[0], 10, replace=False), :]

fig, axes = plt.subplots(5, 2, figsize=(30, 20))
for i in range(10):
    sample = samples[i]
    img_path = os.path.join(DATASET_DIR, 'train_images', sample[0])
    img = cv2.imread(img_path)

    axes[i//2][i%2].imshow(img/255)
    axes[i//2][i%2].set_title('{} - ({})'.format(sample[0], ', '.join(sample[1].astype(np.str))))
plt.show()


# In[ ]:


val_pairs = np.array(list(zip(X_val, y_val)))
samples = val_pairs[np.random.choice(val_pairs.shape[0], 10, replace=False), :]

fig, axes = plt.subplots(5, 2, figsize=(30, 20))
for i in range(10):
    sample = samples[i]
    img_path = os.path.join(DATASET_DIR, 'train_images', sample[0])
    img = cv2.imread(img_path)

    axes[i//2][i%2].imshow(img/255)
    axes[i//2][i%2].set_title('{} - ({})'.format(sample[0], ', '.join(sample[1].astype(np.str))))
plt.show()


# # Copy images into right folders

# In[ ]:


get_ipython().system('mkdir train_images')
get_ipython().system('mkdir val_images')


# In[ ]:


for image_file in X_train:
    src = os.path.join(DATASET_DIR, 'train_images', image_file)
    dst = os.path.join('./train_images', image_file)
    copyfile(src, dst)

for image_file in X_val:
    src = os.path.join(DATASET_DIR, 'train_images', image_file)
    dst = os.path.join('./val_images', image_file)
    copyfile(src, dst)


# # Zip training and validation sets

# In[ ]:


get_ipython().system('apt install zip')


# In[ ]:


get_ipython().system('zip -r -m -1 -q train_images.zip ./train_images')
get_ipython().system('zip -r -m -1 -q val_images.zip ./val_images')


# In[ ]:


# y_train = list(map(lambda x: ' '.join(x.astype(np.str)), y_train))
# y_val = list(map(lambda x: ' '.join(x.astype(np.str)), y_val))
y_train = [' '.join(y.astype(np.str)) for y in y_train]
y_val = [' '.join(y.astype(np.str)) for y in y_val]


# In[ ]:


print(len(y_train))
print(len(y_val))


# # Save labels

# In[ ]:


train_set = {
    'ImageId': X_train,
    'Label': y_train
}

val_set = {
    'ImageId': X_val,
    'Label': y_val
}

train_df = pd.DataFrame(train_set)
val_df = pd.DataFrame(val_set)

train_df.to_csv('./train.csv', index=False)
val_df.to_csv('./val.csv', index=False)


# In[ ]:


train_df.head()


# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


train_df['blabels'] = train_df['Label'].map(lambda x: int(x.split()[0])+2*int(x.split()[1])+3*int(x.split()[2])+4*int(x.split()[3]))


# In[ ]:


train_df.head()


# In[ ]:


print(train_df["blabels"].nunique()); classes = list(set(train_df["blabels"])); classes


# In[ ]:


train_df.to_csv('train_df.csv')


# In[ ]:





# In[ ]:


for i in classes:
    print("Number of items in class {} is {}".format(i,len(train_df[train_df["blabels"] == i])))


# In[ ]:


from fastai import *
from fastai.vision import *

data = ImageDataBunch.from_csv('../input/severstal-steel-defect-detection/', folder = 'severstal-steel-defect-detection', csv_labels = "train.csv",
                               test = 'test_images',suffix=".zip", size = 36, ds_tfms = get_transforms())
data.path = pathlib.Path('.')
data.normalize(imagenet_stats)

learn = create_cnn(data,resnet50,pretrained = True,metrics = accuracy)
learn.fit_one_cycle(5)

learn.unfreeze()
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(3,max_lr = slice(1e-6,3e-4))

interp = ClassificationInterpretation.from_learner(learn)
interp.plot_top_losses(9)
interp.plot_confusion_matrix()
preds,y = learn.TTA()
acc = accuracy(preds, y)
print('The validation accuracy is {} %.'.format(acc * 100))

def generateSubmission(learner):
    submissions = pd.read_csv('../input/sample_submission.csv')
    id_list = list(submissions.id)
    preds,y = learner.TTA(ds_type=DatasetType.Test)
    pred_list = list(preds[:,1])
    pred_dict = dict((key, value.item()) for (key, value) in zip(learner.data.test_ds.items,pred_list))
    pred_ordered = [pred_dict[Path('../input/test/' + id + '.zip')] for id in id_list]
    submissions = pd.DataFrame({'id':id_list,'label':pred_ordered})
    submissions.to_csv("submission_{}.csv".format(pred_score),index = False)

generateSubmission(learn)


# In[ ]:


tfms = get_transforms(do_flip = True,flip_vert = True,max_zoom = 1.1)


# In[ ]:




