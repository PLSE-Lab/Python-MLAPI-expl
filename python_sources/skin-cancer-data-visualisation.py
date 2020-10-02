#!/usr/bin/env python
# coding: utf-8

# # **Data Visualisation !**

# The cancer is the most dangerous enemy for humanity, skin kancer (**Melanoma**) a common type of cancer witch we fight!!
# On that kernel we take a look for deferent types of skin birthmarks, that **melenoma** appear as one of it side by side with several skin birthmarks summarized on that seven types:
# 1. **Melanocytic nevi**
# 2. **Melanoma**
# 3. **Benign keratosis-like lesions**
# 4. **Basal cell carcinoma**
# 5. **Actinic keratoses**
# 6. **Vascular lesions**
# 7. **Dermatofibroma**

# Start with importing libraries help us !

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from glob import glob
from PIL import Image
import seaborn as sns


# make the image path dictionary by joining the folder path from base directory `base_skin_dir` and merge the images in jpg format from both the folders `HAM10000_images_part1.zip` and `HAM10000_images_part2.zip`

# In[ ]:


base_skin_dir = os.path.join('..', 'input/skin-cancer-mnist-ham10000')
imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x
                     for x in glob(os.path.join(base_skin_dir, '*', '*.jpg'))}

lesion_type_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign keratosis-like lesions ',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}


# In[ ]:


tile_df = pd.read_csv(os.path.join(base_skin_dir, 'HAM10000_metadata.csv'))
tile_df['path'] = tile_df['image_id'].map(imageid_path_dict.get)
tile_df['cell_type'] = tile_df['dx'].map(lesion_type_dict.get) 
tile_df['cell_type_idx'] = pd.Categorical(tile_df['cell_type']).codes
tile_df.sample(3)


# Showing images samples ...

# In[ ]:


tile_df['image'] = tile_df['path'].map(lambda x: np.asarray(Image.open(x).resize((100,75))))


# In[ ]:


n_samples = 7
fig, m_axs = plt.subplots(7, n_samples, figsize = (4*n_samples, 3*7))
for n_axs, (type_name, type_rows) in zip(m_axs, 
                                         tile_df.sort_values(['cell_type']).groupby('cell_type')):
    n_axs[0].set_title(type_name)
    for c_ax, (_, c_row) in zip(n_axs, type_rows.sample(n_samples, random_state=1234).iterrows()):
        c_ax.imshow(c_row['image'])
        c_ax.axis('off')
fig.savefig('category_samples.png', dpi=300)


# ## Cleaning Data !!
# first thing showing ig there is a `null` data.

# In[ ]:


tile_df.isnull().sum()


# Fill the null values by their mean.

# In[ ]:


tile_df['age'].fillna((tile_df['age'].mean()), inplace=True)


# insure that now we have no empety data.

# In[ ]:


tile_df.isnull().sum()


# ## Exploring Data.
# Exploring different features of the dataset.

# In[ ]:


fig, ax1 = plt.subplots(1, 1, figsize= (10, 5))
tile_df['cell_type'].value_counts().plot(kind='bar', ax=ax1)


# We notice that the *Melanocytic nevi* has the biggest share of data.

# Plotting of Technical Validation field (ground truth) which is dx_type to see the distribution of its 4 categories which are listed below :
# 1. **Histopathology(Histo)**: Histopathologic diagnoses of excised lesions have been performed by specialized dermatopathologists.
# 2. **Confocal**: Reflectance confocal microscopy is an in-vivo imaging technique with a resolution at near-cellular level , and some facial benign with a grey-world assumption of all training-set images in Lab-color space before and after manual histogram changes.
# 3. **Follow-up**: If nevi monitored by digital dermatoscopy did not show any changes during 3 follow-up visits or 1.5 years biologists accepted this as evidence of biologic benignity. Only nevi, but no other benign diagnoses were labeled with this type of ground-truth because dermatologists usually do not monitor dermatofibromas, seborrheic keratoses, or vascular lesions.
# 4. **Consensus**: For typical benign cases without histopathology or followup biologists provide an expert-consensus rating of authors PT and HK. They applied the consensus label only if both authors independently gave the same unequivocal benign diagnosis. Lesions with this type of groundtruth were usually photographed for educational reasons and did not need further follow-up or biopsy for confirmation.

# In[ ]:


tile_df['dx_type'].value_counts().plot(kind='bar')


# The place of the cell on the body.

# In[ ]:


tile_df['localization'].value_counts().plot(kind='bar')


# It seems back, lower extremity, trunk and upper extremity are heavily compromised regions of skin cancer.

# Show the distribution of `age`.

# In[ ]:


tile_df['age'].hist(bins=40)


# That showing that patients age range between 35 and 75.

# Seeing the distribution on gender.

# In[ ]:


tile_df['sex'].value_counts().plot(kind='bar')


# Showing `dx_type` distribution on `age`.

# In[ ]:


sns.boxplot(x='dx_type', y='age', data=tile_df)


# Showing `cell_types` distribution on `age`.

# In[ ]:


plt.figure(figsize=(16,6))
sns.boxplot(x='cell_type', y='age', data=tile_df)


# # Go To Train Model !!
# Now after we took a look on the data, let's build our **CNN** model.
# [click here to go for model kernal ^_^](https://www.kaggle.com/zuhdiabyjayyab/new-skin-cancer)
