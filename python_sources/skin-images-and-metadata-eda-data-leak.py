#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.options.mode.chained_assignment = None
from torch.utils.data import Dataset, DataLoader, Subset
import torch
import torchvision
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pylab import rcParams
import cv2
import gc
import plotly.express as ex
import plotly.graph_objects as go
from plotly.offline import iplot
#cufflinks to link pandas to plotly
import cufflinks as cf
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(style='whitegrid', palette='muted')
rcParams['figure.figsize'] = 14,8
import torchvision.transforms as transforms
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


root_dir = '../input/siim-isic-melanoma-classification/'

train = pd.read_csv(root_dir + 'train.csv')

train.head()


test = pd.read_csv(root_dir + 'test.csv')

test.head()


# In[ ]:


print('Total number of entries in train dataset: ', len(train))
print('Unique patients in train dataset: ',train['patient_id'].nunique())

print('Total number of entries in test dataset: ', len(test))
print('Unique patients in test dataset: ',test['patient_id'].nunique())


# # Meta Data Analysis

# **Age Analysis for both train and test**

# In[ ]:


#Missing values in train
print('Number of missing age values in train data is :',train['age_approx'].isnull().sum())
print('Number of missing age values in test data is :',test['age_approx'].isnull().sum())


# In[ ]:


def show_age_dist(series, bins = None):
    fig = go.Figure()
    hist = go.Histogram(x = series, nbinsx=50)
    fig.add_trace(hist)
    fig.update_layout(title_text = 'Age distribution')
    fig.update_yaxes(title_text='Number of Patients')
    fig.update_xaxes(title_text='Age')
    fig.show()
#     return hist.xbins
    


# In[ ]:


group_age_train = train.groupby('patient_id')['age_approx'].mean()
group_age_test = test.groupby('patient_id')['age_approx'].mean()


# In[ ]:


sns.distplot(group_age_train,bins=50, kde = True).set_title('Age distribution in train dataset')


# In[ ]:


sns.distplot(group_age_test,bins=50, kde = True).set_title('Age distribution in test dataset')


# In[ ]:


train_withAge = train[train['age_approx']>0]


# Lets check its correlation with "Target" variable

# In[ ]:


train_withAge['age_approx'].corr(train_withAge['target'])


# Since the correlation is very low between 'target' and 'age', we can deduce that age is not a significant factor for Melanoma

# ** Gender Distribution **

# In[ ]:


print('Number of records with missing gender information in training set: ', train['sex'].isnull().sum())
print('Number of records with missing gender information in test set: ', test['sex'].isnull().sum())


# In[ ]:


def show_gender_dist(series, title):
    fig = go.Figure()
    df_ = series.value_counts(normalize = True)
    bar_graph = go.Bar(x = df_.index, y = df_.values, width=[0.1,0.1])
    fig.add_trace(bar_graph)
    fig.update_layout(title_text = title, bargap=0.1)
    fig.update_yaxes(title_text='Gender count percentage')
    fig.update_xaxes(title_text='Gender')
    fig.show()
#     return hist.xbins
    


# In[ ]:


show_gender_dist(train['sex'], 'Gender Distribution for Train set')


# In[ ]:


show_gender_dist(test['sex'], 'Gender Distribution for Test set')


# In[ ]:


def show_gender_target_dist(df, title):
    gender_target_df = df.groupby(['sex', 'target'])['benign_malignant'].count().to_frame().reset_index()
    gender_target_df.target = gender_target_df.target.replace({0:'Benign', 1:'Malignant'})
    fig = go.Figure()
    fig.add_trace(go.Bar(x = gender_target_df[gender_target_df['sex']=='female'].target, 
                         y = gender_target_df[gender_target_df['sex']=='female'].benign_malignant, 
                         name='Female',
                         marker_color='indianred',
                         text = gender_target_df[gender_target_df['sex']=='female'].benign_malignant, 
                         textposition = 'auto',
                         width=[0.25,0.25]))

    fig.add_trace(go.Bar(x = gender_target_df[gender_target_df['sex']=='male'].target, 
                         y = gender_target_df[gender_target_df['sex']=='male'].benign_malignant, 
                         name='Male',
                         marker_color='lightsalmon',
                         text = gender_target_df[gender_target_df['sex']=='male'].benign_malignant, 
                         textposition = 'auto',
                         width=[0.25,0.25]))


    fig.update_layout(barmode='group', title=title, bargap=0)
    fig.update_yaxes(title_text='Count')
    fig.update_xaxes(title_text='Benign vs Malignant')
    fig.show()


# In[ ]:


show_gender_target_dist(train, 'Gender-Target distirbution of Train set')


# Although the difference is not huge, but is noticeable enough that male patients have higher malignant cases compared to female patients

# ** Anatomy Site EDA **

# In[ ]:


lesion_loc = train['anatom_site_general_challenge'].value_counts(normalize=True).sort_values(ascending=False)
lesion_loc.iplot(kind='bar', 
                 xTitle='Percentage', 
                 text = lesion_loc.values.tolist(),
                 
                 textposition='outside',
                 title='Distribution of lesion location across train dataset')


# In[ ]:


lesion_loc = test['anatom_site_general_challenge'].value_counts(normalize=True).sort_values(ascending=False)
lesion_loc.iplot(kind='bar', 
                 xTitle='Percentage', 
                 text = lesion_loc.values.tolist(),
                 
                 textposition='outside',
                 title='Distribution of lesion location across test dataset')


# In[ ]:


def show_location_target_dist(df, title):
    df.dropna(subset = ["anatom_site_general_challenge"], inplace=True)
    loc_target_df = df.groupby(['anatom_site_general_challenge', 'target'])['benign_malignant'].count().to_frame().reset_index()
    loc_target_df.target = loc_target_df.target.replace({0:'Benign', 1:'Malignant'})
    locations = df['anatom_site_general_challenge'].unique()
    total_benign = loc_target_df[loc_target_df['target']=='Benign'].benign_malignant.sum()
    total_malignant = loc_target_df[loc_target_df['target']=='Malignant'].benign_malignant.sum()
    fig = go.Figure()
    for l in locations:
#         print(l)
        percent_text = round((loc_target_df[loc_target_df['anatom_site_general_challenge']==l].benign_malignant/[total_benign,total_malignant ])*100, 2)
        percent_text = [str(t)+"%" for t in percent_text]
        fig.add_trace(go.Bar(x = loc_target_df[loc_target_df['anatom_site_general_challenge']==l].target, 
                             y = loc_target_df[loc_target_df['anatom_site_general_challenge']==l].benign_malignant, 
                             name=l,
                             text = percent_text, 
                             textposition = 'outside'
                            ))




    fig.update_layout(barmode='group', title=title, bargap=0.1)
    fig.update_yaxes(title_text='Count')
    fig.update_xaxes(title_text='Benign vs Malignant')
    fig.show()


# In[ ]:


show_location_target_dist(train, 'Location ditribution relative to Target ')


# Let's take a deeper look at location distribution for Malignant lesions

# In[ ]:


def show_location_malignant_dist(df, title):
    df.dropna(subset = ["anatom_site_general_challenge"], inplace=True)
    loc_target_df = df.groupby(['anatom_site_general_challenge', 'target'])['benign_malignant'].count().to_frame().reset_index()
    loc_target_df.target = loc_target_df.target.replace({0:'Benign', 1:'Malignant'})
    loc_target_df = loc_target_df[loc_target_df['target']=='Malignant']
    locations = df['anatom_site_general_challenge'].unique()    
    total_malignant = loc_target_df[loc_target_df['target']=='Malignant'].benign_malignant.sum()
    fig = go.Figure()
    for l in locations:
#         print(l)
        percent_text = round((loc_target_df[loc_target_df['anatom_site_general_challenge']==l].benign_malignant/total_malignant)*100, 2)
        percent_text = [str(t)+"%" for t in percent_text]
        fig.add_trace(go.Bar(x = loc_target_df[loc_target_df['anatom_site_general_challenge']==l].target, 
                             y = loc_target_df[loc_target_df['anatom_site_general_challenge']==l].benign_malignant, 
                             name=l,
                             text = percent_text, 
                             textposition = 'outside'
                            ))




    fig.update_layout(barmode='group', title=title, bargap=0.1)
    fig.update_yaxes(title_text='Count')
    fig.update_xaxes(title_text='Malignant')
    fig.show()


# In[ ]:


show_location_malignant_dist(train, 'Location ditribution relative to Malignant lesions')


# So, Torso, which is the trunk of human body has more chances to have malignant lesions. For that matter, any lesion since even it has higher percentage share for benign lesions as well

# ** Diagnosis EDA **

# In[ ]:


def show_diagnosis_dist(df):
    diagnosis = df['diagnosis'].value_counts().sort_values(ascending=False)
    
    fig = go.Figure()
    fig.add_trace(go.Bar( x = diagnosis.index,
                         y = diagnosis.values,
                         text = diagnosis.values.tolist(),
                         textposition = 'outside'
    
    ))
    fig.update_layout(title='Diagnosis distribution')
    fig.update_yaxes(title_text='Count')
    fig.update_xaxes(title_text='Diagnosis')
    fig.show()


# In[ ]:


show_diagnosis_dist(train)


# So lot of records have unknown diagnosis. However, this information is only available for train dataset

# # EDA on images

# ## Data Leak?
# One of interesting discussions on forum is possible data leak through image resolution and image mean color. Is it really true? Let's check it now

# ** Data Leak through Image resolution: ** Images with high resolution have more pixels, so lets use this intuition and calculate coorelation between image resolution and target. Record with high number of pixels implies that it has high resolution

# In[ ]:


class MelanomaDataset(Dataset):
    def __init__(self, df: pd.DataFrame, im_folder: str, train: bool = True, transforms = None):
        """
        Class initialization
        Args:
            df (pd.DataFrame): DataFrame with data description
            im_folder (str): folder with images
            train (bool): flag of whether a training dataset is being initialized or testing one
            transforms: image transformation method to be applied
            
        """
        self.df = df
        self.transforms = transforms
        self.train = train
        self.im_folder = im_folder
        
    def __getitem__(self, index):
        im_path = os.path.join(self.im_folder, self.df.iloc[index]['image_name'] + '.jpg')
        x = cv2.imread(im_path)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        if self.transforms:
            x = self.transforms(x)
            
        if self.train:
            y = self.df.iloc[index]['target']
            return x, y
        else:
            return x
    
    def __len__(self):
        return len(self.df)


# In[ ]:


melanoma_dataset = MelanomaDataset(train, root_dir+'jpeg/train/', True)

melanoma_dataloader = DataLoader(dataset = melanoma_dataset, batch_size=1, shuffle=False, num_workers=10)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'from tqdm import tqdm\npixels = []\nmean_color = []\ntargets_temp = []\nwidths = []\nheights = []\nfor i, (img, target_) in enumerate(tqdm(melanoma_dataloader)):\n    img = img.squeeze()\n    target_ = target_.squeeze()\n#     print(target_.shape)\n    h,w = img.shape[0], img.shape[1]\n    pixels.append(h*w)\n    widths.append(w)\n    heights.append(h)\n    mean_color.append(np.mean(img.numpy()))\n    targets_temp.append(int(target_.numpy()))\n    del img\n    gc.collect()\n    if(i==1500):\n        break')


# In[ ]:


# plt.scatter(widths, heights, c = targets_temp, cmap = plt.cm.autumn, alpha = 0.5)
fig = go.Figure()
fig.add_trace(go.Scatter(x=widths, y=heights, mode='markers', marker = dict(color=targets_temp)))
fig.update_xaxes(title_text='Widhts')
fig.update_yaxes(title_text='Heights')
fig.show()


# In[ ]:


targets_temp = np.array(targets_temp)
widths = np.array(widths)
heights = np.array(heights)
malignant_indices = np.where(targets_temp==1)


# In[ ]:


benign_indices = np.where(targets_temp==0)

pixels = np.array(pixels)


# In[ ]:


sns.distplot(pixels[malignant_indices]).set_title('Resolution distribution for Malignant Lesions')


# In[ ]:


sns.distplot(pixels[benign_indices]).set_title('Resolution distribution for Benign Lesions')


# In[ ]:


get_ipython().system('pip install chart_studio')


# In[ ]:


import chart_studio.plotly as py
import plotly.figure_factory as ff


# In[ ]:


def density_plot_resolution(indices, title):
    fig = go.Figure()
    fig.add_trace(go.Histogram2dContour(x=widths[indices], y=heights[indices], histfunc='count', colorscale='blues'))
    fig.add_trace(go.Scatter(x=widths[indices], y=heights[indices], mode='markers'))
    fig.add_trace(go.Histogram(
            y = heights[indices],
            xaxis = 'x2',
    
        ))
    fig.add_trace(go.Histogram(
            x = widths[indices],
            yaxis = 'y2',
   
        ))

    fig.update_layout(

        xaxis = dict(
            zeroline = False,
            domain = [0,0.85],
            showgrid = False
        ),
        yaxis = dict(
            zeroline = False,
            domain = [0,0.85],
            showgrid = False
        ),
        xaxis2 = dict(
            zeroline = False,
            domain = [0.85,1],
            showgrid = False
        ),
        yaxis2 = dict(
            zeroline = False,
            domain = [0.85,1],
            showgrid = False
        ),

        bargap = 0,
        hovermode = 'closest',
        showlegend = False,
        title = title
    )
    fig.update_xaxes(title_text='Widhts')
    fig.update_yaxes(title_text='Heights')
    fig.show()


# In[ ]:


density_plot_resolution(benign_indices, 'Density plot of resolution for Benign Lesions')


# In[ ]:


density_plot_resolution(malignant_indices, 'Density plot of resolution for Malignant Lesions')


# In[ ]:


mean_color = np.array(mean_color)
# fig = go.Figure()
fig=(ff.create_distplot([mean_color[benign_indices], mean_color[malignant_indices]], group_labels=['Mean color for Benign', 'Mean color for Malignant']))
fig.show()


# In[ ]:


# Corelation calculation:
pixels_series = pd.Series(pixels)
mean_color_series = pd.Series(mean_color)
targets_temp_series = pd.Series(targets_temp)
print('Correlation between Image resolution and targets is :', pixels_series.corr(targets_temp_series))
print('Correlation between Mean color of images and targets is :', mean_color_series.corr(targets_temp_series))


# Though there is no direct correlation of resolution & mean-color with Target i.e. no linear relationship, the density plots indicate a possible non-linear relationship which can be captured by a neural network model as malignant and benign lesions have different resolution distributions

# In[ ]:


np.save('widths.npy', widths)
np.save('heights.npy', heights)
np.save('targets_temp.npy', targets_temp)
np.save('mean_color.npy', mean_color)
np.save('pixels.npy',pixels)


# # Image Visualization Analysis

# In[ ]:


transform_basic = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256,256)),
    transforms.ToTensor()
    ])

melanoma_dataset = MelanomaDataset(train, root_dir+'jpeg/train/', True, transform_basic)

melanoma_dataloader = DataLoader(dataset = melanoma_dataset, batch_size=10, shuffle=False)

def imshow(img):
    img = img.squeeze()
    plt.imshow(img)


# In[ ]:


dataiter = iter(melanoma_dataloader)

images, targets = dataiter.next()
images = images.permute(0,2,3,1)
images = images.numpy()
targets = targets.numpy()
fig = plt.figure(figsize=(25, 10))
# display 10 images
for idx in np.arange(10):
    ax = fig.add_subplot(2, 10/2, idx+1, xticks=[], yticks=[])
    imshow(images[idx])
    ax.set_title(targets[idx])


# Remember the distribution on diagnosis? Let's analyse that

# In[ ]:


show_diagnosis_dist(train)


# In[ ]:


def display_images_diagnosis(diagnosis, title):
    
    train_temp = train[train['diagnosis']==diagnosis]
    random_indices = np.random.randint(0, len(train_temp), 5)
    
    fig = plt.figure(figsize=(25, 10))
    for idx in np.arange(5):
        img = cv2.imread(root_dir+'jpeg/train/'+train_temp.iloc[random_indices[idx]].image_name+'.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img,(256,256))
        target = train_temp.iloc[random_indices[idx]].target
        ax = fig.add_subplot(2, 5, idx+1, xticks=[], yticks=[])
        imshow(img)
        ax.set_title(target) 
    plt.suptitle(title)


# In[ ]:


display_images_diagnosis('nevus', 'Lesions diagnosed as Nevus')


# In[ ]:


display_images_diagnosis('melanoma', 'Lesions diagnosed as Melanoma')


# In[ ]:


display_images_diagnosis('seborrheic keratosis', 'Lesions diagnosed as Seborrheic Keratosis')


# In[ ]:


display_images_diagnosis('lentigo NOS', 'Lesions diagnosed as Lentigo NOS')


# In[ ]:


display_images_diagnosis('lichenoid keratosis', 'Lesions diagnosed as Lichenoid keratosis')


# From the above images, it can be seen that Nevus lesions have some sort of pinkish shade all over ( No matter how many random indices were picked all nevus images somehow has same shade). Could model easily capture this? The rest of the types i.e. Melanoma, Seborrheic Kertosis, Lentigo NOS, Lichenoid Keratosis etc seem to display similar characteristics at the first glance. To distinguish between them, model has to capture more insighful features like the ones described in ABCDE, 7-point derma checklist etc.

# **To be continued ...**

# In[ ]:




