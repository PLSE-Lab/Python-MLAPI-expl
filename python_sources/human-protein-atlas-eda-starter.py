#!/usr/bin/env python
# coding: utf-8

# <h1>What to predict </h1>
# 
#  We need to predict  protein organelle localization labels for each sample. There are in total 28 different labels present in the dataset. However, the dataset comprises 27 different cell types of highly different morphology, which affect the protein patterns of the different organelles. All image samples are represented by four filters (stored as individual files), the protein of interest (green) plus three cellular landmarks: nucleus (blue), microtubules (red), endoplasmic reticulum (yellow). The green filter should hence be used to predict the label, and the other filters are used as references.
#  Lets create a dictionary for labels.
#  
#  The organelles are visible in green color channel. Blue color marks the nucleus and microtubules are shown in red
# ![](http://storage.googleapis.com/kaggle-media/competitions/proteins/description_NACC_cropped_opt.png)

# In[ ]:


dicts={
0:  "Nucleoplasm", 
1:  "Nuclear membrane",   
2:  "Nucleoli",   
3:  "Nucleoli fibrillar center" ,  
4:  "Nuclear speckles"   ,
5:  "Nuclear bodies"   ,
6:  "Endoplasmic reticulum",   
7:  "Golgi apparatus"   ,
8:  "Peroxisomes"   ,
9:  "Endosomes"   ,
10:  "Lysosomes"   ,
11:  "Intermediate filaments",   
12:  "Actin filaments"   ,
13:  "Focal adhesion sites",   
14:  "Microtubules"   ,
15:  "Microtubule ends",   
16:  "Cytokinetic bridge",   
17:  "Mitotic spindle"   ,
18:  "Microtubule organizing center" ,  
19:  "Centrosome"   ,
20:  "Lipid droplets",   
21:  "Plasma membrane",   
22:  "Cell junctions"  , 
23:  "Mitochondria"   ,
24:  "Aggresome"   ,
25:  "Cytosol",
26:  "Cytoplasmic bodies",   
27:  "Rods & rings" 
}


# <h2>Let's check train csv</h2>

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))


# In[ ]:


df_train = pd.read_csv('../input/train.csv')
df_train.info()


# In[ ]:


df_train.head()


# In[ ]:


#cheking for nulls
df_train["Target"].isnull().values.sum()


# In[ ]:


from collections import Counter, defaultdict
labels = df_train['Target'].apply(lambda x: x.split(' '))
labels[:5]


# In[ ]:


counts = defaultdict(int)
for l in labels:
    for l2 in l:
        counts[l2] += 1


# In[ ]:


strs=[]
for count in counts.keys(): strs.append(dicts[int(count)])


# In[ ]:


import seaborn as sns
sns.set(style="whitegrid")
df_count = pd.DataFrame({'ptype': strs,
     'Count': list(counts.values())})
df_count.head()


# <h2>Plot plot!</h2>

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
plt.subplots(figsize=(20,15))
ax = sns.barplot(x="ptype", y="Count", data=df_count)
for item in ax.get_xticklabels():
    item.set_rotation(90)


# <h2>Lets check a train image </h2>

# In[ ]:


import cv2
from PIL import Image


# In[ ]:


#opening a green image
img = Image.open('../input/train/00070df0-bbc3-11e8-b2bc-ac1f6b6435d0_green.png')
img


# In[ ]:


print("Image size :",img.size)


# In[ ]:


#opening a neuclus blue image
img = Image.open('../input/train/00070df0-bbc3-11e8-b2bc-ac1f6b6435d0_blue.png')
img


# In[ ]:


#opening a microtubles red image
img = Image.open('../input/train/00070df0-bbc3-11e8-b2bc-ac1f6b6435d0_red.png')
img


# In[ ]:


#opening a endoplasmic reticulum (yellow)
img = Image.open('../input/train/00070df0-bbc3-11e8-b2bc-ac1f6b6435d0_yellow.png')
img
   

