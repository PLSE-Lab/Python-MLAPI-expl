#!/usr/bin/env python
# coding: utf-8

# # This code is made for if you download the metadata in your PC
# ### I hope that it will be helpful for beginner :)
# ---
# 

# # Install and import package

# In[ ]:


get_ipython().system('pip install requests wget wordcloud')


# In[ ]:


import os
import os.path as pth

import wget
from multiprocessing import Pool

import pandas as pd
from collections import Counter
from wordcloud import WordCloud
from matplotlib import pyplot  as plt


# # Download data for visualization
# ### This part based on web crawling.
# ### If you wonder more detail, Please to visit my another kernel ([Download metadata and segmentation images](https://www.kaggle.com/bbchip/download-metadata-and-segmentation-images))

# In[ ]:


def get_metadata(url, base_path='./'):
    filename = url.split('/')[-1]
    full_filename = pth.join(base_path, filename)
    if pth.exists(full_filename):
        return full_filename, 1
    wget.download(url, out=base_path)
    ### If you can't use wget, you can use below blocked code
    ### But It's much slower than wget...
    # data = requests.get(url).data
    # with open(full_filename, 'wb') as f:
    #     f.write(data)
    return full_filename, 0


# In[ ]:


data_base_path = 'metadata'
metadata_path = pth.join(data_base_path, 'Metadata')
box_path = pth.join(data_base_path, 'Boxes')


# In[ ]:


metadata_url = 'https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv'
train_box_url = 'https://storage.googleapis.com/openimages/2018_04/train/train-annotations-bbox.csv'
validation_box_url = 'https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv'
test_box_url = 'https://storage.googleapis.com/openimages/v5/test-annotations-bbox.csv'

url_list = [metadata_url, train_box_url, validation_box_url, test_box_url]
path_list = [metadata_path] + [box_path]*3

os.makedirs(metadata_path, exist_ok=True)
os.makedirs(box_path, exist_ok=True)
pool = Pool(8)
for filename, status in pool.starmap(get_metadata, zip(url_list, path_list)):
    if status == 0:
        print(filename + ' is saved.')
    elif status == 1:
        print(filename + ' is already exist.')
    else:
        print('???')
pool.close()
pool.join()


# # Make the label table. {LabelName:RealName}
# ###  ['Pandas'](http://zeldahagoshipda.com) is useful package for data analysis.
# ### I recommand you to get used to this package for data analysis

# In[ ]:


label_filename = pth.join(metadata_path, 'class-descriptions-boxable.csv')
df = pd.read_csv(label_filename, header=None, index_col=None)
label_dict = dict(df.values)
dict(list(label_dict.items())[:10])


# # Count the number of label
# ### ['Collections'](https://docs.python.org/3/library/collections.html) has a lot of useful function.

# In[ ]:


train_box_filename = pth.join(box_path, 'train-annotations-bbox.csv')
val_box_filename = pth.join(box_path, 'validation-annotations-bbox.csv')
test_box_filename = pth.join(box_path, 'test-annotations-bbox.csv')


# In[ ]:


df = pd.read_csv(train_box_filename)
labels = df['LabelName'].values
train_cnt = Counter(labels)
    
df = pd.read_csv(val_box_filename)
labels = df['LabelName'].values
val_cnt = Counter(labels)

df = pd.read_csv(test_box_filename)
labels = df['LabelName'].values
test_cnt = Counter(labels)


# In[ ]:


train_cnt.most_common(5), val_cnt.most_common(5), test_cnt.most_common(5)


# In[ ]:


train_cnt_dict = {label_dict[k]:v for k, v in train_cnt.items()}
val_cnt_dict = {label_dict[k]:v for k, v in val_cnt.items()}
test_cnt_dict = {label_dict[k]:v for k, v in test_cnt.items()}


# # Make a word cloud
# ### It made by wordcloud module ([And there is a lot of exameple!](https://amueller.github.io/word_cloud/auto_examples/index.html#example-gallery))
# 
# ### You can make your wordcloud (Just tune the "max_word, color, mask image, iamge size")
# ### I recommend you to apply mask first, Because it makes word-cloud more beautiful.

# In[ ]:


wc = WordCloud(max_words=300
                , background_color='white'
#                 , width=1920, height=1080
#                 , mask=mask
#                 , color_func=MakeColor
                )
wc.generate_from_frequencies(train_cnt_dict)
# wc.to_file(pth.join(your_directory, wordcloud_filename))

plt.figure()
plt.axis("off")
plt.title('Train')
plt.imshow(wc, interpolation='bilinear')


# In[ ]:


wc = WordCloud(max_words=300
                , background_color='white'
                )
wc.generate_from_frequencies(val_cnt_dict)

plt.figure()
plt.axis("off")
plt.title('Validation')
plt.imshow(wc, interpolation='bilinear')


# In[ ]:


wc = WordCloud(max_words=300
                , background_color='white'
                )
wc.generate_from_frequencies(val_cnt_dict)

plt.figure()
plt.axis("off")
plt.title('Test')
plt.imshow(wc, interpolation='bilinear')

