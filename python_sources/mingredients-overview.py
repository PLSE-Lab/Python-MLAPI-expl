#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
plt.rcParams["figure.figsize"] = (15, 10)
plt.rcParams["figure.dpi"] = 125
plt.rcParams["font.size"] = 14
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.style.use('ggplot')
sns.set_style("whitegrid", {'axes.grid': False})
plt.rcParams['image.cmap'] = 'gray' # grayscale looks better
from itertools import cycle
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']


# In[ ]:


from pathlib import Path
import numpy as np
import pandas as pd
from skimage.io import imread as imread
from skimage.util import montage as montage2d
from skimage.color import label2rgb
base_dir = Path('..') / 'input'


# # Overview
# Load and process data

# In[ ]:


image_overview_df = pd.read_json(base_dir/ 'image_subset.json')
image_overview_df.sample(3)


# In[ ]:


from itertools import chain
all_ingredients = list(chain.from_iterable(image_overview_df['top_ingredients'].values))
ingredient_list = pd.value_counts(all_ingredients)


# In[ ]:


fig, ax1 = plt.subplots(1, 1, figsize=(20, 5))
ingredient_count = pd.value_counts(all_ingredients)
ingredient_count.plot.bar(ax=ax1)


# In[ ]:


ing_ids = ingredient_count.index.tolist()


# In[ ]:


image_overview_df['ingredients_len'] = image_overview_df['top_ingredients'].map(len)
ing_arr = np.stack(
    image_overview_df.sort_values('ingredients_len')['top_ingredients'].map(lambda x_list: [k in x_list for k in ing_ids]).values,
    0
)
ing_arr.shape


# In[ ]:


fig, ax1 = plt.subplots(1, 1, figsize=(50, 10))
ax1.imshow(ing_arr.T, cmap='viridis')
ax1.set_aspect(20)
ax1.set_xlabel('Image #')
ax1.set_ylabel('Word')
ax1.set_yticks(range(len(ing_ids)))
ax1.set_yticklabels(ing_ids);


# In[ ]:


fig, m_axs = plt.subplots(3, 3, figsize=(20, 20))
for c_ax, (_, c_row) in zip(m_axs.flatten(), 
                            image_overview_df.sample(9).iterrows()):
    c_ax.imshow(imread(base_dir / 'subset' / c_row['image_path']))
    c_ax.set_title('\n'.join(c_row['top_ingredients'][:4]))
    c_ax.axis('off')

