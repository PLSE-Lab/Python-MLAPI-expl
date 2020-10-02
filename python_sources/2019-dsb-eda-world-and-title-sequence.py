#!/usr/bin/env python
# coding: utf-8

# # 2019 DSB EDA : 'World' and 'Title' sequence

# ![Whole World](https://user-images.githubusercontent.com/30274701/68391745-9770aa00-01ab-11ea-87eb-e8fba91917d2.JPG)

# __PBS KIDS Measure Up! App__ have four world(`TREETOPCITY`, `CRYSTALCAVES`, `MAGMAPEAK`, `None`) and __five `Assessment`__. Each world have many titles(courses) and __the APP__ suggest course's procedure.
# 
# In this kernel, find the title's sequence(order).

# ![](https://user-images.githubusercontent.com/30274701/68392259-bfacd880-01ac-11ea-9de5-102f85865019.jpg)

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns
import operator

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


get_ipython().run_cell_magic('time', '', "# Read in the data CSV files\ntrain = pd.read_csv('../input/data-science-bowl-2019/train.csv')\ntrain_labels = pd.read_csv('../input/data-science-bowl-2019/train_labels.csv')\ntest = pd.read_csv('../input/data-science-bowl-2019/test.csv')\nspecs = pd.read_csv('../input/data-science-bowl-2019/specs.csv')\nss = pd.read_csv('../input/data-science-bowl-2019/sample_submission.csv')")


# If we regard `None` world is just intro World, main world is three. In this game, we can find specific `Assessment` is in one world. 
# 
# * Mushroom Sorter -> TREETOPCITY
# * Bird Measurer -> TREETOPCITY
# * Cart Balancer -> CRYSTALCAVES
# * Chest Sorter -> CRYSTALCAVES
# * Cauldron Filler -> MAGMAPEAK
# 
# In test, we have to predict only last `Assesment`, so it's important specific world's procedure.

# In[ ]:


assess_list = ['Bird Measurer', 'Cart Balancer', 'Cauldron Filler', 'Chest Sorter', 'Mushroom Sorter']
world_list = ['MAGMAPEAK', 'TREETOPCITY', 'CRYSTALCAVES']


# In[ ]:


for a in assess_list:
    world = train.loc[train['title'].str.contains(a), 'world'].unique()
    print('{} Assessment in {} World'.format(a, world))


# In the game, each world have several course. My main idea is __most players play the game in order.__
# 
# So I find the path that many users played.

# In[ ]:


print('Find_course functions')
def world_in_label(df, world):
    if world == 'MAGMAPEAK':
        df = df.loc[df['title'].str.contains('Cauldron Filler')]
    elif world == 'TREETOPCITY':
        df = df.loc[(df['title'].str.contains('Bird Measurer')) | (df['title'].str.contains('Mushroom Sorter'))]
    elif world == 'CRYSTALCAVES':
        df = df.loc[(df['title'].str.contains('Cauldron Filler')) | (df['title'].str.contains('Chest Sorter'))]
    
    return df['installation_id'].unique()

def find_course(world):
    label_ids = world_in_label(train_labels, world)
    world_df = train.query('world=="{}"'.format(world))[['installation_id', 'world', 'title']]
    world_df = world_df.loc[world_df['installation_id'].isin(label_ids)]
    ids = world_df['installation_id'].unique()
    num_course = world_df['title'].nunique()
    
    course_d = dict()
    
    print('{} ids'.format(len(ids)))
    for i in ids:
        if i not in label_ids:
            continue
        else:
            id_df = world_df.query('installation_id=="{}"'.format(i))
            if id_df['title'].nunique() == num_course:
                try:
                    course_d[str(id_df['title'].unique())] += 1
                except:
                    course_d[str(id_df['title'].unique())] = 1    
    return course_d


# In[ ]:


w_dict = dict()
for w in world_list:
    print('World : {}'.format(w))
    w_dict[w] = find_course(w)
    print(sorted(w_dict[w].items(), key=operator.itemgetter(1), reverse=True)[:5])
    print('='*100)


# In plot, most users play the APP in particular path.

# In[ ]:


fig, ax = plt.subplots(1,3, figsize=(18,6))
for num, w in enumerate(world_list):
#with sns.axes_style('Set1'):
    plot = pd.DataFrame(sorted(w_dict[w].items(), key=operator.itemgetter(1), reverse=True)[:5]).plot(kind='bar', cmap='summer', ax=ax[num])
    plot.set_title(w, fontsize=20)
    plot.patches[0].set_color('orange')
    plot.set_xticklabels(['Path {}'.format(i+1)  for i in range(5)], rotation=45)
    plot.legend().remove()


# Most user's path is below.

# ### CRYSTALCAVES(15)
# 
# * 'Crystal Caves - Level 1' 
# * 'Chow Time' 
# * 'Balancing Act'
# * 'Chicken Balancer (Activity)' 
# * 'Lifting Heavy Things' 
# * 'Crystal Caves - Level 2' 
# * 'Honey Cake' 
# * 'Happy Camel'
# * 'Cart Balancer (Assessment)' 
# * 'Leaf Leader' 
# * 'Crystal Caves - Level 3'
# * 'Heavy, Heavier, Heaviest' 
# * 'Pan Balance' 
# * 'Egg Dropper (Activity)'
# * 'Chest Sorter (Assessment)'
# 
# 
# ### TREETOPCITY(17)
# 
# * 'Tree Top City - Level 1' 
# * 'Ordering Spheres' 
# * 'All Star Sorting'
# * 'Costume Box'
# * 'Fireworks (Activity)'
# * '12 Monkeys'
# * 'Tree Top City - Level 2' 
# * 'Flower Waterer (Activity)' 
# * 'Pirate's Tale
# * 'Mushroom Sorter (Assessment)'
# * 'Air Show'
# * 'Treasure Map'
# * 'Tree Top City - Level 3'
# * 'Crystals Rule'
# * 'Rulers'
# * 'Bug Measurer (Activity)' 
# * 'Bird Measurer (Assessment)'
# 
# 
# ### MAGMA PEAK(11)
# 
# * 'Magma Peak - Level 1'
# * 'Sandcastle Builder (Activity)' 
# * 'Slop Problem' 
# * 'Scrub-A-Dub' 
# * 'Watering Hole (Activity)'
# * 'Magma Peak - Level 2'
# * 'Dino Drink'
# * 'Bubble Bath' 
# * 'Bottle Filler (Activity)'
# * 'Dino Dive'
# * 'Cauldron Filler (Assessment)'

# Here, __courses in list__ is not match __above picture courses .__
# 
# For example, 'MAGMA PEAK' world have 9 courses in the picture. But train data have 11 unique courses.<br>
# 
# 
# I think `0000 - Level 0` is not in picture because its media type is just `clip`. It seem like intro clip.<br>
# In this case, 'Magma Peak - Level 1' and 'Magma Peak - Level 2' is `clip`.

# In[ ]:


temp = train.query('installation_id=="002db7e3"')
temp = temp.loc[temp['world'] == 'MAGMAPEAK']
print('"Magma Peak - Level 1" type is {}'.format(train.loc[train['title'].str.contains('Magma Peak - Level 1')]['type'].unique()))
display(temp.loc[temp['title'].str.contains('Level 1')])
print('"Magma Peak - Level 2" type is {}'.format(train.loc[train['title'].str.contains('Magma Peak - Level 2')]['type'].unique()))
display(temp.loc[temp['title'].str.contains('Level 2')])


# In[ ]:


#Save dict
pd.DataFrame(w_dict).to_csv('course_dict.csv')


# ## Bonus.
# 
# when extract specific data in Dataframe, `query` is faster than `.loc`.

# In[ ]:


get_ipython().run_cell_magic('time', '', "train.loc[train['installation_id'] == '0006a69f']['world'].unique()")


# In[ ]:


get_ipython().run_cell_magic('time', '', 'train.query(\'installation_id=="0006a69f"\')[\'world\'].unique()')

