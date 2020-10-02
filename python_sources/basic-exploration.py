#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from fastai import *
from fastai.vision import *
import seaborn as sns
from PIL import Image
from matplotlib import pyplot as plt


# In[ ]:


path = Path('../input/aptos2019-blindness-detection/')


# In[ ]:


path.ls()


# In[ ]:


df = pd.read_csv(path/'train.csv')
df.head()


# In[ ]:


df_test = pd.read_csv(path/'test.csv')
df_test.head()


# In[ ]:


print(len(df))
print(len(df_test))


# In[ ]:


print(df.isna().sum()) 
print('-' * 20)
print(df_test.isna().sum())


# In[ ]:


df.diagnosis.value_counts()


# In[ ]:


# plot the value counts as histogram
b = sns.countplot(df['diagnosis'])
b.axes.set_title('Distribution of diagnosis', fontsize = 30)
b.set_xlabel('Diagnosis', fontsize = 20)
b.set_ylabel('Count', fontsize = 20)
plt.show()


# In[ ]:


im = Image.open("../input/aptos2019-blindness-detection/train_images/08b6e3240858.png")


# In[ ]:


print(im.format, im.size, im.mode)


# In[ ]:


im = Image.open("../input/aptos2019-blindness-detection/train_images/0ca0aee4d57e.png")


# In[ ]:


print(im.format, im.size, im.mode)


# In[ ]:


# plot the various sizes of images
def get_image_sizes(folder):
    image_list = (path/folder).ls()
    heights = []
    widths = []
    ids = []

    for image in image_list:
        im = Image.open(image)
        height, width = im.size
        heights.append(height)
        widths.append(width)
        ids.append(str(image)[-16:-4])
        
    return pd.DataFrame({'id_code': ids,
                         'height': heights,
                         'width': widths})


# In[ ]:


size_df = get_image_sizes('train_images')
size_df.head()


# In[ ]:


size_df_test = get_image_sizes('test_images')
size_df_test.head()


# In[ ]:


plt.hist(size_df['height'])


# In[ ]:


plt.hist(size_df_test['height'])


# In[ ]:


plt.hist(size_df['width'])


# In[ ]:


plt.hist(size_df_test['width'])


# In[ ]:


# plot the images from 0 and 4 to see the difference
df_0 = df[df['diagnosis'] == 0]
df_0.head()


# In[ ]:


df_4 = df[df['diagnosis'] == 4]
df_4.head()


# In[ ]:


data = (ImageList.from_df(df_0,path,folder='train_images',suffix='.png')
        .split_by_rand_pct(0.1, seed=42)
        .label_from_df()
        .transform(get_transforms(),size=128)
        .databunch()).normalize(imagenet_stats)


# In[ ]:


# add figsize argument
data.show_batch(rows=3)


# In[ ]:


data = (ImageList.from_df(df_4,path,folder='train_images',suffix='.png')
        .split_by_rand_pct(0.1, seed=42)
        .label_from_df()
        .transform(get_transforms(),size=128)
        .databunch()).normalize(imagenet_stats)


# In[ ]:


data.show_batch(rows=3)


# ## Without augmentation

# In[ ]:


data = (ImageList.from_df(df_4,path,folder='train_images',suffix='.png')
        .split_by_rand_pct(0.1, seed=42)
        .label_from_df()
        .transform([],size=128)
        .databunch()).normalize(imagenet_stats)


# In[ ]:


data.show_batch(rows=3, )


# In[ ]:


# in the next notebook work with various augmentations

