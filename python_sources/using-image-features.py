#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from zipfile import ZipFile
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from nltk.stem.snowball import SnowballStemmer
import os
print(os.listdir("../input"))
from matplotlib import pyplot as plt
import seaborn as sns

# Any results you write to the current directory are saved as output.


# In[3]:


train_zip  = ZipFile('../input/avito-demand-prediction/train_jpg.zip')


# In[4]:


filenames = train_zip.namelist()[1:]
len(filenames)


# In[5]:


train_zip.close()
del train_zip


# In[6]:


train_zip  = ZipFile('../input/avito-demand-prediction/test_jpg.zip')
filenames = train_zip.namelist()[1:]
print(len(filenames))
train_zip.close()
del train_zip


# # image base features

# In[7]:


train_image_info = pd.read_csv('../input/extract-image-features-train-files/train_img_feat.csv',index_col=0)
train_image_info.shape


# In[8]:


train_image_size = pd.read_csv('../input/extract-image-features-train-files/train_img_files_info.csv',index_col=0)
train_image_size.shape


# In[9]:


get_ipython().run_cell_magic('time', '', "image_features = train_image_info.merge(train_image_size,on='image',how='left')")


# In[10]:


image_features.head()


# In[10]:


image_features['dim'] = image_features['dim'].apply(lambda x:eval(str(x)))


# In[11]:


image_features['colors'] = image_features['colors'].apply(lambda x: eval(str(x)))


# In[12]:


image_features['width'] = image_features['dim'].apply(lambda x: x[0])
image_features['height'] = image_features['dim'].apply(lambda x: x[1])
image_features['red_avg'] = image_features['colors'].apply(lambda x: x[0])
image_features['green_avg'] = image_features['colors'].apply(lambda x: x[1])
image_features['blue_avg'] = image_features['colors'].apply(lambda x: x[2])


# In[13]:


image_features.drop(columns=['dim','colors','csize'],inplace=True)


# In[14]:


image_features.head()


# In[15]:


def make_features(image_features):
    image_features['width_height_diff'] = image_features[['width','height']].diff(axis=1)['height']
    image_features['green_blue_diff'] = image_features[['green_avg','blue_avg']].diff(axis=1)['blue_avg']
    image_features['green_red_diff'] = image_features[['green_avg','red_avg']].diff(axis=1)['red_avg']
    image_features['red_blue_diff'] = image_features[['red_avg','blue_avg']].diff(axis=1)['blue_avg']
    image_features['width_height_ratio'] = image_features['width']/image_features['height']
    image_features['total_pixel'] = image_features['width']*image_features['height']
    return image_features


# In[16]:


image_features = make_features(image_features)


# In[17]:


image_features.head()


# In[18]:


def show_corr(df):
    f, ax = plt.subplots(figsize=[10,7])
    sns.heatmap(df.corr(),
                annot=False, fmt=".2f",cbar_kws={'label': 'Correlation Coefficient'},cmap="plasma",ax=ax, linewidths=.5)
    ax.set_title("Dense Features Correlation Matrix")
    plt.savefig('correlation_matrix.png')
show_corr(image_features)


# In[19]:


image_features.columns


# # yuv features

# In[122]:


test_image_yuv = pd.read_csv('../input/script-image-features-test-yuv-multproces/test_jpg_img_feat_saturation.csv',index_col=0)
train_image_yuv = pd.read_csv('../input/script-image-features-train-yuv-multprocess/train_jpg_img_feat_saturation.csv',index_col=0)


# In[123]:


yuv_col = [ 'bright_avg', 'u_avg', 'yuv_v_avg', 'bright_std', 'bright_min','birght_max', 'bright_diff']


# In[124]:


test_image_yuv.columns


# In[137]:


test_image_yuv.rename(columns={'v_avg':'yuv_v_avg'},inplace=True,index=str)
train_image_yuv.rename(columns={'v_avg':'yuv_v_avg'},inplace=True,index=str)


# In[138]:


test_image_yuv.columns,train_image_yuv.columns


# In[32]:


test_image_yuv.head()


# In[54]:


image_features.shape,test_image_yuv.shape,train_image_yuv.shape


# In[34]:


# test_image_yuv= test_image_yuv[using_col]
# train_image_yuv= train_image_yuv[using_col]


# In[55]:


train_image_yuv.head()


# In[56]:


image_features = image_features.merge(train_image_yuv,on='image',how='left')


# In[57]:


image_features.head()


# # hsv features

# In[59]:


train_path_ = '../input/script-image-features-train-hsv-batch%s/train_jpg_img_feat_saturation.csv'
train_ids = [1,15,2,25,3,4]
test_path_ = '../input/script-image-features-test-hsv-batch%s/test_jpg_img_feat_saturation.csv'
test_ids = [1,2]
train_hsv_df = pd.DataFrame()
test_hsv_df = pd.DataFrame()
for i in train_ids:
    p = train_path_%i
    train_hsv_df = train_hsv_df.append(pd.read_csv(p,index_col=0))
    print(p,train_hsv_df.shape)

for i in test_ids:
    p = test_path_%i
    test_hsv_df = test_hsv_df.append(pd.read_csv(p,index_col=0))
    print(p,test_hsv_df.shape)
train_hsv_df.shape,test_hsv_df.shape


# In[139]:


train_hsv_df.rename(columns={'v_avg':'hsv_v_avg'},inplace=True,index=str)
test_hsv_df.rename(columns={'v_avg':'hsv_v_avg'},inplace=True,index=str)


# In[60]:


image_features = image_features.merge(train_hsv_df,on='image',how='left')
image_features.head()


# In[66]:


train_hsv_df.columns


# In[67]:


hsv_col = ['hue_avg', 'sst_avg', 'hsv_v_avg', 'sat_std', 'sat_min', 'sat_max', 'sat_diff']


# In[61]:


show_corr(image_features)


# # colorfull

# In[70]:


train_path_ = '../input/script-image-features-train-color-batch%s/train_jpg_img_feat_saturation.csv'
test_path_ = '../input/script-image-features-test-color-multprocess/test_jpg_img_feat_saturation.csv'
train_ids= [1,2,3,4,5,6]
# train_ids= [1,2,3,5,6]
test_color_df = pd.read_csv(test_path_,index_col=0)
train_color_df = pd.DataFrame()
for i in train_ids:
    p = train_path_%i
    x = pd.read_csv(p,index_col=0)
    print(p,x.shape)
    train_color_df = train_color_df.append(x)
train_color_df.shape,test_color_df.shape


# In[39]:


del train_image_info,train_image_size
import gc
gc.collect()


# In[71]:


image_features = image_features.merge(train_color_df,on='image',how='left')
image_features.head()


# In[73]:


color = ['colorfull']


# # std

# In[30]:


train_std_df = pd.read_csv("../input/script-image-features-train-pil-std-batch1/train_jpg_img_feat_saturation.csv",index_col=0)
test_std_df = pd.read_csv("../input/script-image-features-test-pil-std-batch1/test_jpg_img_feat_saturation.csv",index_col=0)
std_cols = ['r_std', 'g_std', 'b_std', 'r_md', 'g_md', 'b_md']
image_features = image_features.merge(train_std_df,on='image',how='left')


# In[31]:


image_features.head()


# # xception

# In[25]:


train_xception_df = pd.read_csv('../input/xception-train-features-starter-include-top/train_xception.csv',index_col=0)
test_xception_df = pd.read_csv('../input/xception-test-features-starter-include-top/test_xception.csv',index_col=0)


# In[33]:


train_xception_df.columns = ['image','item_label', 'xception_prob', 'xception_var',
       'xception_nonzero']
test_xception_df.columns = ['image','item_label', 'xception_prob', 'xception_var',
       'xception_nonzero']


# In[34]:


train_xception_df.head()


# In[35]:


cols = ['item_label', 'xception_prob', 'xception_var',
       'xception_nonzero']


# In[36]:


image_features = image_features.merge(train_xception_df,on='image',how='left')
image_features.head()


# # nima

# In[28]:


train_nima_df = pd.read_csv('../input/neural-image-assessment-train-features-starter/train_xception.csv',index_col=0)
test_nima_df = pd.read_csv('../input/neural-image-assessment-test-feature-starter/test_xception.csv',index_col=0)


# In[29]:


train_nima_df.shape,test_nima_df.shape


# In[30]:


image_features = image_features.merge(train_nima_df,on='image',how='left')


# In[ ]:


nima_cols = ['mean_nima','std_nima_']


# In[31]:


image_features.head()


# # blurr

# In[3]:


train_blurr_df = pd.read_csv('../input/blurr-features-train-part1/train_jpg_img_feat_blurr.csv',index_col=0)
train_blurr_df=train_blurr_df.append(pd.read_csv('../input/blurr-features-train-part2/train_jpg_img_feat_blurr.csv',index_col=0))
train_blurr_df=train_blurr_df.append(pd.read_csv('../input/fork-of-blurr-features-train-part/train_jpg_img_feat_blurr.csv',index_col=0))

test_blurr_df = pd.read_csv('../input/blurr-image-features-test/test_jpg_img_feat_blurr.csv',index_col=0)
train_blurr_df.shape,test_blurr_df.shape


# In[4]:


train_blurr_df.columns


# In[5]:


image_features = image_features.merge(train_blurr_df,on='image',how='left')
image_features.head()


# In[16]:


show_corr(image_features)


# # train

# In[26]:


train_df = pd.read_csv('../input/avito-demand-prediction/train.csv',usecols=['image','item_id',],index_col=0)


# In[78]:


get_ipython().run_cell_magic('time', '', "train_df = train_df.reset_index().merge(image_features,on='image',how='left').set_index('item_id')")


# In[79]:


train_df.head()


# In[80]:


image_cols = image_features.columns


# In[81]:


image_cols


# In[82]:


train_image_features=image_features


# In[93]:


image_features.shape


# 

# # test

# In[38]:


test_image_info = pd.read_csv('../input/extract-image-features-test-file/train_img_feat.csv',index_col=0)
test_file_info = pd.read_csv('../input/extract-image-features-test-file/test_img_files_info.csv',index_col=0)
test_imge_features = test_image_info.merge(test_file_info,on='image',how='left')


# In[39]:


def proce_data(image_features):
    image_features['dim'] = image_features['dim'].apply(lambda x:eval(x))
    image_features['colors'] = image_features['colors'].apply(lambda x: eval(x))
    image_features['width'] = image_features['dim'].apply(lambda x: x[0])
    image_features['height'] = image_features['dim'].apply(lambda x: x[1])
    image_features['red_avg'] = image_features['colors'].apply(lambda x: x[0])
    image_features['green_avg'] = image_features['colors'].apply(lambda x: x[1])
    image_features['blue_avg'] = image_features['colors'].apply(lambda x: x[2])
    return image_features

test_imge_features = proce_data(test_imge_features)
test_imge_features = make_features(test_imge_features)


# In[40]:


test_imge_features.shape


# In[96]:


test_imge_features.head()


# In[87]:


test_imge_features.drop(columns=['dim','colors','csize'],inplace=True)


# In[94]:


test_imge_features = test_imge_features.merge(test_hsv_df,on='image',how='left')
test_imge_features = test_imge_features.merge(test_color_df,on='image',how='left')
test_imge_features.shape


# In[95]:


test_imge_features = test_imge_features.merge(test_image_yuv,on='image',how='left')
test_imge_features.shape


# In[ ]:


test_imge_features = test_imge_features.merge(test_std_df,on='image',how='left')


# In[41]:


test_imge_features = test_imge_features.merge(test_xception_df,on='image',how='left')


# In[ ]:


test_imge_features = test_imge_features.merge(test_nima_df,on='image',how='left')


# In[18]:


test_imge_features = test_imge_features.merge(test_blurr_df,on='image',how='left')


# In[42]:


show_corr(test_imge_features)


# In[100]:


test_df = pd.read_csv('../input/avito-demand-prediction/test.csv',usecols=['image','item_id','image_top_1','description','title'],index_col=0)


# In[101]:


test_df  = test_df.reset_index().merge(test_imge_features,on='image',how='left').set_index('item_id')


# In[102]:


test_df.shape


# In[141]:


test_df[train_image_features.columns].drop(columns=['image']).to_csv('test_image_features.csv.gzip',compression='gzip')
train_df[image_features.columns].drop(columns=['image']).to_csv('train_image_features.csv.gzip',compression='gzip')


# In[83]:


# submission = pd.read_csv('../input/avito-demand-prediction/sample_submission.csv',index_col=0)


# In[ ]:


# submission['deal_probability'] = predict_y
# submission.to_csv('rf_image.csv',index=True)

