#!/usr/bin/env python
# coding: utf-8

# # EDA CSV

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 
import gc
import os
import PIL

from scipy import stats
from multiprocessing import Pool
from PIL import ImageOps, ImageFilter
from tqdm import tqdm
pd.set_option("max_columns",300)
pd.set_option("max_rows",1103)
from wordcloud import WordCloud

tqdm.pandas()


# In[ ]:


df_train = pd.read_csv('../input/imet-2019-fgvc6/train.csv')
train_path = '../input/imet-2019-fgvc6/train/'
label_df = pd.read_csv('../input/imet-2019-fgvc6/labels.csv')


# In[ ]:


label_names = label_df['attribute_name'].values

num_labels = np.zeros((df_train.shape[0],))
train_labels = np.zeros((df_train.shape[0], len(label_names)))

for row_index, row in enumerate(df_train['attribute_ids']):
    num_labels[row_index] = len(row.split())    
    for label in row.split():
        train_labels[row_index, int(label)] = 1


# In[ ]:


culture, tag, unknown = 0, 0, 0

for l in label_names:
    if l[:3] == 'cul':
        culture += 1
    elif l[:3] == 'tag':
        tag += 1
    else:
        unknown += 1
        
print(f'Culture : {culture}')
print(f'Tag     : {tag}')
print(f'Unknown : {unknown}')
print(f'Total   : {culture + tag + unknown}')


# In[ ]:


label_df['is_culture'] = label_df['attribute_name'].apply(lambda x: 1 if 'culture' in x else 0)
attribute_count = label_df['is_culture'].value_counts()

ax = sns.barplot(['Tag', 'Culture'], attribute_count.values, alpha=0.8)
for p in ax.patches:
    ax.annotate(f'{p.get_height()}\n{p.get_height() * 100 / label_df.shape[0]:.2f}%',
                (p.get_x() + p.get_width()/2., p.get_height()), 
                ha='center', 
                va='center', 
                fontsize=12, 
                color='black',
                xytext=(2,-20), 
                textcoords='offset points')
plt.title('Culture/Tag')
plt.xlabel('attribute type')
plt.ylabel('Frequency')


# In[ ]:


label_sum = np.sum(train_labels, axis=0)

culture_sequence = label_sum[:398].argsort()[::-1]
tag_sequence = label_sum[398:].argsort()[::-1]

culture_labels = [label_names[x][9:] for x in culture_sequence]
culture_counts = [label_sum[x] for x in culture_sequence]

tag_labels = [label_names[x + 398][5:] for x in tag_sequence]
tag_counts = [label_sum[x + 398] for x in tag_sequence]


# In[ ]:


for i in range(len(culture_labels)):
    print(culture_labels[i],':',culture_counts[i])


# In[ ]:


df = pd.DataFrame({'Culture_label': culture_labels,'Culture_count': culture_counts})
df.to_csv('cutr_labe.csv',index=True)


# In[ ]:


for i in range(len(tag_labels)):
    print(tag_labels[i],':',tag_counts[i])


# In[ ]:


df = pd.DataFrame({'Tags_label': tag_labels,'Tags_count': tag_counts})
df.to_csv('tags_label.csv',index=True)


# In[ ]:


plt.figure(figsize=(20,15))

plt.subplot(1,2,1)
ax1 = sns.barplot(y=culture_labels[:20], x=culture_counts[:20], orient="h")
plt.title('Label Counts by Culture (Top 20)',fontsize=15)
plt.xlim((0, max(culture_counts)*1.15))
plt.yticks(fontsize=15)

for p in ax1.patches:
    ax1.annotate(f'{int(p.get_width())}\n{p.get_width() * 100 / df_train.shape[0]:.2f}%',
                (p.get_width(), p.get_y() + p.get_height() / 2.), 
                ha='left', 
                va='center', 
                fontsize=12, 
                color='black',
                xytext=(7,0), 
                textcoords='offset points')

plt.subplot(1,2,2)    
ax2 = sns.barplot(y=tag_labels[:20], x=tag_counts[:20], orient="h")
plt.title('Label Counts by Tag (Top 20)',fontsize=15)
plt.xlim((0, max(tag_counts)*1.15))
plt.yticks(fontsize=15)

for p in ax2.patches:
    ax2.annotate(f'{int(p.get_width())}\n{p.get_width() * 100 / df_train.shape[0]:.2f}%',
                (p.get_width(), p.get_y() + p.get_height() / 2.), 
                ha='left', 
                va='center', 
                fontsize=12, 
                color='black',
                xytext=(7,0), 
                textcoords='offset points')

plt.tight_layout()
plt.show()


# In[ ]:


plt.figure(figsize=(20,8))

ax = sns.countplot(num_labels)
plt.xlabel('Number of Labels')
plt.title('Number of Labels per Image', fontsize=20)

for p in ax.patches:
    ax.annotate(f'{p.get_height() * 100 / df_train.shape[0]:.3f}%',
            (p.get_x() + p.get_width() / 2., p.get_height()), 
            ha='center', 
            va='center', 
            fontsize=11, 
            color='black',
            xytext=(0,7), 
            textcoords='offset points')


# In[ ]:


train_attr_ohot = np.zeros((len(df_train), len(label_df)), dtype=int)

for idx, attr_arr in enumerate(df_train.attribute_ids.str.split(" ").apply(lambda l: list(map(int, l))).values):
    train_attr_ohot[idx, attr_arr] = 1


# In[ ]:


names_arr = label_df.attribute_name.values
df_train["attribute_names"] = [", ".join(names_arr[arr == 1]) for arr in train_attr_ohot]


# In[ ]:


df_train["attr_num"] = train_attr_ohot.sum(axis=1)
df_train["culture_attr_num"] = train_attr_ohot[:, :398].sum(axis=1)
df_train["tag_attr_num"] = train_attr_ohot[:, 398:].sum(axis=1)


# In[ ]:


df_train.head()


# In[ ]:


#for i in range(len(df_train["attribute_names"])):
#    print(df_train["attribute_names"][i],':',df_train["culture_attr_num"][i])


# In[ ]:


fig = plt.figure(figsize=(15, 10))
fig.subplots_adjust(hspace=0.4)
ax2 = fig.add_subplot(3,1,2,)
sns.countplot(df_train.culture_attr_num, ax=ax2)
ax2.set_title("number of 'culture' attributes each art has")
for p in ax2.patches:
    ax2.annotate(f'{p.get_height() * 100 / df_train.shape[0]:.3f}%',
            (p.get_x() + p.get_width() / 2., p.get_height()), 
            ha='center', 
            va='center', 
            fontsize=11, 
            color='black',
            xytext=(0,7), 
            textcoords='offset points')
ax3 = fig.add_subplot(3,1,3,)
ax3.set_title("number of 'tag' attributes each art has")
sns.countplot(df_train.tag_attr_num, ax=ax3)
for p in ax3.patches:
    ax3.annotate(f'{p.get_height() * 100 / df_train.shape[0]:.3f}%',
            (p.get_x() + p.get_width() / 2., p.get_height()), 
            ha='center', 
            va='center', 
            fontsize=11, 
            color='black',
            xytext=(0,7), 
            textcoords='offset points')


# # EDA by Visualize Images

# Example of images with tags

# In[ ]:


from cv2 import cv2
i = 1
df_train["attribute_ids"]=df_train["attribute_ids"].apply(lambda x:x.split(" "))
df_train["id"]=df_train["id"].apply(lambda x:x+".png")
plt.figure(figsize=[30,30])
for img_name in os.listdir("../input/imet-2019-fgvc6/train/")[5:10]:   
    img = cv2.imread("../input/imet-2019-fgvc6/train/%s" % img_name)[...,[2, 1, 0]]
    plt.subplot(5, 1, i)
    plt.imshow(img)
    ids = df_train[df_train["id"] == img_name]["attribute_ids"]
    print(ids)
    title_val = []
    for tag_id in ids.values[0]:
        att_name = label_df[label_df['attribute_id'].astype(str) == tag_id]['attribute_name'].values[0]
        title_val.append(att_name)
    plt.title(title_val)
    i += 1
    
plt.show()


# # Check duplicated images

# In[ ]:


import os
import numpy as np
from PIL import Image
import pandas as pd
import hashlib


# In[ ]:


def check_md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


# In[ ]:


file_names = []
path_root = '../input/imet-2019-fgvc6/train/'
for filename in os.listdir(path_root)[0:20]:
    file_names.append(check_md5(path_root+filename))
print(len(file_names))


# In[ ]:


unit = np.unique(file_names,return_counts=True)
count = 0
for i in range(len(unit[1])):
    if unit[1][i]>1:
        count += 1
        print('Duplicated Images')
if count == 0:
    print('NOT Duplicated Images')


# In[ ]:





# # Padding image

# In[ ]:


def padding_image(path_img):
    image_old = Image.open(path_img)
    width, height = image_old.size

    if width > height:
        distance_max = width
        array = np.zeros([distance_max, distance_max, 3], dtype=np.uint8)
        array.fill(0)
        image_new = Image.fromarray(array)

        xmin = 0
        ymin = int((distance_max / 2) - (height / 2))
        xmax = distance_max
        ymax = int((distance_max / 2) + (height / 2))

        image_new.paste(image_old, (xmin, ymin, xmax, ymax))
        return image_new

    elif width < height:
        distance_max = height
        array = np.zeros([distance_max, distance_max, 3], dtype=np.uint8)
        array.fill(0)
        image_new = Image.fromarray(array)

        xmin = int((distance_max / 2) - (width / 2))
        ymin = 0
        xmax = int((distance_max / 2) + (width / 2))
        ymax = distance_max

        image_new.paste(image_old, (xmin, ymin, xmax, ymax))
        return image_new

    else:
        return image_old


# In[ ]:


import random
random_filenames = random.choices(os.listdir(path_root), k=5)
for filename in random_filenames:
    plt.imshow(np.array(Image.open(path_root+filename)))
    plt.figure()
    plt.imshow(padding_image(path_root+filename))
    plt.figure()


# In[ ]:





# In[ ]:


import fastai
from fastai.vision import *
fastai.__version__


# # Setup

# In[ ]:


BATCH  = 126
SIZE   = 250
path = Path('../input/imet-2019-fgvc6/') # iMet data path


# In[ ]:


get_ipython().system('ls ../input/resnet50/')


# In[ ]:


# Making pretrained weights work without needing to find the default filename
from torch.utils import model_zoo
Path('models').mkdir(exist_ok=True)
get_ipython().system("cp '../input/resnet50/resnet50.pth' 'models/'")
def load_url(*args, **kwargs):
    model_dir = Path('models')
    filename  = 'resnet50.pth'
    if not (model_dir/filename).is_file(): raise FileNotFoundError
    return torch.load(model_dir/filename)
model_zoo.load_url = load_url


# # Data

# In[ ]:


# Load train dataframe
train_df = pd.read_csv(path/'train.csv')
train_df.head()


# In[ ]:


# Load labels dataframe
labels_df = pd.read_csv(path/'labels.csv')
labels_df.head()


# In[ ]:


# Load sample submission
test_df = pd.read_csv(path/'sample_submission.csv')
test_df.head()


# # Create data object using datablock API

# In[ ]:


tfms = get_transforms(do_flip=True, flip_vert=False, max_rotate=0.10, max_zoom=1.5, max_warp=0.2, max_lighting=0.2,
                     xtra_tfms=[(symmetric_warp(magnitude=(-0,0), p=0)),])


# In[ ]:


train, test = [ImageList.from_df(df, path=path, cols='id', folder=folder, suffix='.png') 
               for df, folder in zip([train_df, test_df], ['train', 'test'])]
data = (train.split_by_rand_pct(0.05, seed=42)
        .label_from_df(cols='attribute_ids', label_delim=' ')
        .add_test(test)
        .transform(tfms, size=SIZE, resize_method=ResizeMethod.PAD, padding_mode='border',)
        .databunch(path=Path('.'), bs=BATCH).normalize(imagenet_stats))


# In[ ]:


data


# # Create learner with pretrenet model and FocalLoss
# For problems with high class imbalance Focal Loss is usually a better choice than the usual Cross Entropy Loss.

# In[ ]:


# Source: https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/78109
class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma

    def forward(self, logit, target):
        target = target.float()
        max_val = (-logit).clamp(min=0)
        loss = logit - logit * target + max_val +                ((-max_val).exp() + (-logit - max_val).exp()).log()

        invprobs = F.logsigmoid(-logit * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        if len(loss.size())==2:
            loss = loss.sum(dim=1)
        return loss.mean()


# In[ ]:


learn = cnn_learner(data, base_arch=models.resnet50, loss_func=FocalLoss(), metrics=fbeta)


# In[ ]:


ls ../input


# In[ ]:


get_ipython().system("cp '../input/models2/stage-1.pth' 'models/'")


# In[ ]:


def find_best_fixed_threshold(preds, targs, do_plot=True):
    score = []
    thrs = np.arange(0, 0.5, 0.01)
    for thr in progress_bar(thrs):
        score.append(fbeta(valid_preds[0],valid_preds[1], thresh=thr))
    score = np.array(score)
    pm = score.argmax()
    best_thr, best_score = thrs[pm], score[pm].item()
    print(f'thr={best_thr:.3f}', f'F2={best_score:.3f}')
    if do_plot:
        plt.plot(thrs, score)
        plt.vlines(x=best_thr, ymin=score.min(), ymax=score.max())
        plt.text(best_thr+0.03, best_score-0.01, f'$F_{2}=${best_score:.3f}', fontsize=14);
        plt.show()
    return best_thr

i2c = np.array([[i, c] for c, i in learn.data.train_ds.y.c2i.items()]).astype(int) # indices to class number correspondence

def join_preds(preds, thr):
    return [' '.join(i2c[np.where(t==1)[0],1].astype(str)) for t in (preds[0].sigmoid()>thr).long()]


# In[ ]:


learn.load('stage-1')


# In[ ]:


# Validation predictions
valid_preds = learn.get_preds(DatasetType.Valid)
best_thr = find_best_fixed_threshold(*valid_preds)


# # Train the model

# In[ ]:


# Find a good learning rate
learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.unfreeze()
learn.fit_one_cycle(16, slice(1e-4,1e-3))
learn.freeze()
learn.save('stage-2', return_path=True)


# In[ ]:


# Validation predictions
valid_preds = learn.get_preds(DatasetType.Valid)
best_thr = find_best_fixed_threshold(*valid_preds)


# In[ ]:


learn.recorder.plot()
learn.recorder.plot_losses()
learn.recorder.plot_metrics()


# In[ ]:


learn.export()


# # Get predictions

# In[ ]:





# In[ ]:





# In[ ]:


# Test predictions
#test_preds = learn.get_preds(DatasetType.Test)
#test_df.attribute_ids = join_preds(test_preds, best_thr)
#test_df.head()


# In[ ]:


#test_df.to_csv('submission.csv', index=False)


# ## TTA

# In[ ]:


# Validation predictions with TTA
#valid_preds = learn.TTA(ds_type=DatasetType.Valid)
#best_thr = find_best_fixed_threshold(*valid_preds)


# In[ ]:


# Test predictions with TTA
test_preds = learn.TTA(ds_type=DatasetType.Test)
test_df.attribute_ids = join_preds(test_preds, best_thr)
test_df.head()


# In[ ]:


test_df.to_csv('submission.csv', index=False) 


# In[ ]:


# Find a good learning rate
learn.lr_find()
learn.recorder.plot()

