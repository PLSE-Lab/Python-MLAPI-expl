#!/usr/bin/env python
# coding: utf-8

# ## Our goal
# 
# * Understand the dataset

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import itertools as itt
import networkx as nx
from sklearn.preprocessing import MinMaxScaler


# In[ ]:


import sys
for name, module in sorted(sys.modules.items()):
    if name in ['numpy', 'pandas', 'seaborn', 'matplotlib', 'networkx', 'sklearn']:
        if hasattr(module, '__version__'): 
            print(name, module.__version__)


# In[ ]:


train_labels = pd.read_csv("../input/train.csv")
train_labels.head()


# How many samples do we have?

# In[ ]:


train_labels.shape[0]


# ## Helper code

# In[ ]:


label_names = {
    0:  "Nucleoplasm",  
    1:  "Nuclear membrane",   
    2:  "Nucleoli",   
    3:  "Nucleoli fibrillar center",   
    4:  "Nuclear speckles",
    5:  "Nuclear bodies",   
    6:  "Endoplasmic reticulum",   
    7:  "Golgi apparatus",   
    8:  "Peroxisomes",   
    9:  "Endosomes",   
    10:  "Lysosomes",   
    11:  "Intermediate filaments",   
    12:  "Actin filaments",   
    13:  "Focal adhesion sites",   
    14:  "Microtubules",   
    15:  "Microtubule ends",   
    16:  "Cytokinetic bridge",   
    17:  "Mitotic spindle",   
    18:  "Microtubule organizing center",   
    19:  "Centrosome",   
    20:  "Lipid droplets",   
    21:  "Plasma membrane",   
    22:  "Cell junctions",   
    23:  "Mitochondria",   
    24:  "Aggresome",   
    25:  "Cytosol",   
    26:  "Cytoplasmic bodies",   
    27:  "Rods & rings"
}

reverse_train_labels = dict((v,k) for k,v in label_names.items())

def fill_targets(row):
    row.Target = np.array(row.Target.split(" ")).astype(np.int)
    for num in row.Target:
        name = label_names[int(num)]
        row.loc[name] = 1
    return row


# ## Which protein organelle localizations occur most often in images?

# In[ ]:


for key in label_names.keys():
    train_labels[label_names[key]] = 0


# In[ ]:


train_labels = train_labels.apply(fill_targets, axis=1)
train_labels.head()


# In[ ]:


target_counts = train_labels.drop(["Id", "Target"],axis=1).sum(axis=0).sort_values(ascending=False)
plt.figure(figsize=(15,15))
sns.barplot(y=target_counts.index.values, x=target_counts.values, order=target_counts.index)


# In[ ]:


target_counts.tail()


# ### Take-Away
# 
# * We can see that most common protein structures belong to coarse grained cellular components like the plasma membrane, the cytosol and the nucleus. 
# * In contrast small components like the lipid droplets, peroxisomes, endosomes, lysosomes, microtubule ends, rods and rings are very seldom in our train data. For these classes the prediction will be very difficult as we have only a few examples that may not cover all variabilities and as our model probably will be confused during ins learning process by the major classes. Due to this confusion we will make less accurate predictions on the minor classes.
# * Consequently accuracy is not the right score here to measure your performance and validation strategy should be very fine. 

# ## How many targets are most common?

# In[ ]:


train_labels["number_of_targets"] = train_labels.drop(["Id", "Target"],axis=1).sum(axis=1)


# In[ ]:


count_perc = train_labels.groupby("number_of_targets").count()['Id']
count_perc


# In[ ]:


plt.figure(figsize=(5,5))
plt.pie(count_perc,
        labels=["%d targets" % x for x in count_perc.index],
        autopct='%1.1f%%')
plt.ylabel('');


# ### Take-away
# 
# * Most train images only have 1 or two target labels.
# * More than 3 targets are very seldom!

# ## Which targets are correlated?
# 
# Let's see if we find some correlations between our targets. This way we may already see that some proteins often come together.

# In[ ]:


def heatmap(C):
    mask = np.zeros_like(C)
    mask[np.triu_indices_from(mask)] = True
    f, ax = plt.subplots(figsize=(11, 9))
    hm = sns.heatmap(C,
                mask=mask, cmap=sns.diverging_palette(220, 10, as_cmap=True),
                vmax=.3,
                center=0,
                square=True,
                linewidths=1,
                cbar_kws={"shrink": .5})
    hm.set_facecolor('w')
    return hm


# In[ ]:


C = train_labels[train_labels.number_of_targets>1].drop(
    ["Id", "Target", "number_of_targets"],axis=1
).corr()
heatmap(C);


# ### Take-away
# 
# * We can see that many targets only have very slight correlations. 
# * In contrast, endosomes and lysosomes often occur together and sometimes seem to be located at the endoplasmatic reticulum. 
# * In addition we find that the mitotic spindle often comes together with the cytokinetic bridge. This makes sense as both are participants for cellular division. And in this process microtubules and thier ends are active and participate as well. Consequently we find a positive correlation between these targets.

# To spot the correlations, let's select those with absolute value greater than 0.1

# In[ ]:


heatmap(C[abs(C)> 0.1]);


# Now we create a function to get the edges weighted the correlations above 0.1

# In[ ]:


import itertools as itt

def get_correlation_graph(C, threashold = 0.1):
    return [(i, j, {'weight': abs(C.iloc[i, j])})
             for i, j in itt.combinations(range(C.shape[0]), 2)
            if abs(C.iloc[i, j]) >= threashold]

import networkx as nx


# Then, we create the graph:

# In[ ]:


G = nx.Graph()

edges = get_correlation_graph(C)
G.add_edges_from(edges)
graph_pos = nx.spring_layout(G)

plt.figure(figsize=(15,15))
nx.draw(G, graph_pos, alpha=.4)
labels = {i : '\n'.join(C.columns[i].split(' '))
          for i in set([i for (i, j, k) in edges] + [j for (i, j, k) in edges])}
nx.draw_networkx_labels(G, graph_pos, labels, font_size=16);


# ## How are special and seldom targets grouped?

# ### Lysosomes and endosomes
# 
# Let's start with these high correlated features!

# In[ ]:


def find_counts(special_target, labels):
    counts = labels[labels[special_target] == 1].drop(
        ["Id", "Target", "number_of_targets"],axis=1
    ).sum(axis=0)
    counts = counts[counts > 0]
    counts = counts.sort_values()
    return counts


# In[ ]:


lyso_endo_counts = find_counts("Lysosomes", train_labels)

plt.figure(figsize=(10,3))
sns.barplot(x=lyso_endo_counts.index.values, y=lyso_endo_counts.values, palette="Blues");


# ### Rods and rings

# In[ ]:


rod_rings_counts = find_counts("Rods & rings", train_labels)
plt.figure(figsize=(15,3))
sns.barplot(x=rod_rings_counts.index.values, y=rod_rings_counts.values, palette="Greens");


# ### Peroxisomes

# In[ ]:


peroxi_counts = find_counts("Peroxisomes", train_labels)

plt.figure(figsize=(15,3))
sns.barplot(x=peroxi_counts.index.values, y=peroxi_counts.values, palette="Reds");


# ### Microtubule ends

# In[ ]:


tubeends_counts = find_counts("Microtubule ends", train_labels)

plt.figure(figsize=(15,3))
sns.barplot(x=tubeends_counts.index.values, y=tubeends_counts.values, palette="Purples");


# ### Nuclear speckles

# In[ ]:


nuclear_speckles_counts = find_counts("Nuclear speckles", train_labels)

plt.figure(figsize=(15,3))
sns.barplot(x=nuclear_speckles_counts.index.values, y=nuclear_speckles_counts.values, palette="Oranges")
plt.xticks(rotation="70");


# ### Take-away
# 
# * We can see that even with very seldom targets we find some kind of grouping with other targets that reveal where the protein structure seems to be located. 
# * For example, we can see that rods and rings have something to do with the nucleus whereas peroxisomes may be located in the nucleus as well as in the cytosol.
# * Perhaps this patterns might help to build a more robust model!  

# Taking back to the analysis of the correlation made above, let's construct a correlation based on the scaled value counts:

# In[ ]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

def find_counts_normalized(special_target, labels):
    counts = labels[labels[special_target] == 1].drop(
        ["Id", "Target", "number_of_targets"],axis=1
    ).sum(axis=0)
    counts = pd.DataFrame(scaler.fit_transform(counts.astype(float).values.reshape(-1,1)).reshape(-1),
                          index=counts.index, columns=[special_target])
    return counts


# In[ ]:


normed = pd.concat([find_counts_normalized(col, train_labels)
           for col in sorted(train_labels.drop(["Id", "Target", "number_of_targets"], axis=1).columns)],
          axis='columns',
         sort=True)
heatmap(normed.corr());


# Using the same threashold of 0.1.

# In[ ]:


img = normed.corr()
heatmap(img[abs(img) > 0.1]);


# In[ ]:


C = img[abs(img) > 0.1]
G = nx.Graph()

edges = get_correlation_graph(C)
G.add_edges_from(edges)
graph_pos = nx.spring_layout(G)

plt.figure(figsize=(15,15))
nx.draw(G, graph_pos, alpha=.4)
labels = {i : '\n'.join(C.columns[i].split(' '))
          for i in set([i for (i, j, k) in edges] + [j for (i, j, k) in edges])}
nx.draw_networkx_labels(G, graph_pos, labels, font_size=16);


# It looks like we have got a [hair ball problem!](https://image.slidesharecdn.com/bokehdatashader-odscboston2016-160523145441/95/visualizing-a-billion-points-w-bokeh-datashader-16-638.jpg?cb=1464015395)
# 
# The graph made with a 0.15 threashold is more easy to understand and get some insight:

# In[ ]:


C = img

heatmap(C[abs(img) > 0.15])
plt.show()

G = nx.Graph()

edges = get_correlation_graph(C, 0.15)
G.add_edges_from(edges)
graph_pos = nx.spring_layout(G)

plt.figure(figsize=(15,15))
nx.draw(G, graph_pos, alpha=.4)
labels = {i : '\n'.join(C.columns[i].split(' '))
          for i in set([i for (i, j, k) in edges] + [j for (i, j, k) in edges])}
nx.draw_networkx_labels(G, graph_pos, labels, font_size=16)
plt.show();


# Hey! Isn't it obvious that the most common places in the cell will spot Cytosol and Nucleoplasm? What happens if we drop them?

# In[ ]:


D = C.drop(['Cytosol', 'Nucleoplasm'], axis=0).drop(['Cytosol', 'Nucleoplasm'], axis=1)


# In[ ]:


heatmap(D[D > .15])


# In[ ]:


G = nx.Graph()

edges = get_correlation_graph(D, 0.15)
G.add_edges_from(edges)
graph_pos = nx.spring_layout(G)

plt.figure(figsize=(15,15))
nx.draw(G, graph_pos, alpha=.4)
labels = {i : '\n'.join(D.columns[i].split(' '))
          for i in set([i for (i, j, k) in edges] + [j for (i, j, k) in edges])}
nx.draw_networkx_labels(G, graph_pos, labels, font_size=16)
plt.show();


# That it! Look, the patterns are coherent with the simple correlation based graph and with the counting plots.

# In[ ]:


plt.figure(figsize=(15,3))
sns.barplot(x=tubeends_counts.index.values, y=tubeends_counts.values, palette="Purples");

plt.figure(figsize=(15,3))
sns.barplot(x=rod_rings_counts.index.values, y=rod_rings_counts.values, palette="Greens");


# There is something wrong here: 'Rods & Rings' should not be associated with 'Microtubule ends'
# 

# In[ ]:


train_labels[train_labels.number_of_targets == 1].drop(
    ['Id', 'Target', 'number_of_targets'],
    axis='columns'
).sum(axis='rows')


# In[ ]:


train_labels[(train_labels.number_of_targets == 1) & (train_labels['Rods & rings'] == 1) ]


# ## How do the images look like?
# 
# 

# ### Peek into the directory
# 
# Before we start loading images, let's have a look into the train directory to get an impression of what we can find there:

# In[ ]:


from os import listdir

files = listdir("../input/train")
for n in range(10):
    print(files[n])


# Ah, ok, great! It seems that for one image id, there are different color channels present. Looking into the data description of this competition we can find that:
# 
# * Each image is actually splitted into 4 different image files. 
# * These 4 files correspond to 4 different filter:
#     * a **green** filter for the **target protein structure** of interest
#     * **blue** landmark filter for the **nucleus**
#     * **red** landmark filter for **microtubules**
#     * **yellow** landmark filter for the **endoplasmatic reticulum**
# * Each image is of size 512 x 512

# Let's check if the number of files divided by 4 yields the number of target samples:

# In[ ]:


len(files) / 4 == train_labels.shape[0]


# ## How do images of specific targets look like?
# 
# While looking at examples, we can build an batch loader:

# In[ ]:


train_path = "../input/train/"


# In[ ]:


def load_image(basepath, image_id):
    images = np.zeros(shape=(4,512,512))
    images[0,:,:] = plt.imread(basepath + image_id + "_green" + ".png")
    images[1,:,:] = plt.imread(basepath + image_id + "_red" + ".png")
    images[2,:,:] = plt.imread(basepath + image_id + "_blue" + ".png")
    images[3,:,:] = plt.imread(basepath + image_id + "_yellow" + ".png")
    return images

def make_image_row(image, subax, title):
    subax[0].imshow(image[0], cmap="Greens")
    subax[1].imshow(image[1], cmap="Reds")
    subax[1].set_title("stained microtubules")
    subax[2].imshow(image[2], cmap="Blues")
    subax[2].set_title("stained nucleus")
    subax[3].imshow(image[3], cmap="Oranges")
    subax[3].set_title("stained endoplasmatic reticulum")
    subax[0].set_title(title)
    return subax

def make_title(file_id):
    file_targets = train_labels.loc[train_labels.Id==file_id, "Target"].values[0]
    title = " - "
    for n in file_targets:
        title += label_names[n] + " - "
    return title


# In[ ]:


a = load_image('../input/train/', 'e403806e-bbbf-11e8-b2bb-ac1f6b6435d0')
np.shape(a)


# In[ ]:


plt.figure(figsize=(15,15))
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.imshow(a[i], cmap='gray')


# In[ ]:


plt.figure(figsize=(15,15))
plt.imshow(a[0], cmap='gray')


# In[ ]:


a = []
a.append(plt.imread('../input/train/039085dc-bbaa-11e8-b2ba-ac1f6b6435d0_red.png'))
a.append(plt.imread('../input/train/039085dc-bbaa-11e8-b2ba-ac1f6b6435d0_green.png'))
a.append(plt.imread('../input/train/039085dc-bbaa-11e8-b2ba-ac1f6b6435d0_blue.png'))
a.append(plt.imread('../input/train/039085dc-bbaa-11e8-b2ba-ac1f6b6435d0_yellow.png'))


# In[ ]:


plt.figure(figsize=(15,15))
cmaps = [sns.dark_palette("red", as_cmap=True),
         sns.dark_palette("green", as_cmap=True),
         sns.dark_palette("blue", as_cmap=True),
         sns.dark_palette("yellow", as_cmap=True)]
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.imshow(a[i], cmap=cmaps[i])


# In[ ]:


def to_rgba2(img):
    r = np.transpose(np.vectorize(lambda x: (1,0,0,x))(img[0]))
    g = np.transpose(np.vectorize(lambda x: (0,1,0,x))(img[1]))
    b = np.transpose(np.vectorize(lambda x: (0,0,1,x))(img[2]))
    y = np.transpose(np.vectorize(lambda x: (1,1,0,x))(img[3]))
    return np.array([r,g,b,y])


# In[ ]:


get_ipython().run_line_magic('time', 'r = to_rgba2(a)')


# In[ ]:


plt.figure(figsize=(15,15))
for i in range(4):
    plt.imshow(r[i])


# In[ ]:


class TargetGroupIterator:
    
    def __init__(self, target_names, batch_size, basepath):
        self.target_names = target_names
        self.target_list = [reverse_train_labels[key] for key in target_names]
        self.batch_shape = (batch_size, 4, 512, 512)
        self.basepath = basepath
    
    def find_matching_data_entries(self):
        train_labels["check_col"] = train_labels.Target.apply(
            lambda l: self.check_subset(l)
        )
        self.images_identifier = train_labels[train_labels.check_col==1].Id.values
        train_labels.drop("check_col", axis=1, inplace=True)
    
    def check_subset(self, targets):
        return np.where(set(self.target_list).issuperset(set(targets)), 1, 0)
    
    def get_loader(self):
        filenames = []
        idx = 0
        images = np.zeros(self.batch_shape)
        for image_id in self.images_identifier:
            images[idx,:,:,:] = load_image(self.basepath, image_id)
            filenames.append(image_id)
            idx += 1
            if idx == self.batch_shape[0]:
                yield filenames, images
                filenames = []
                images = np.zeros(self.batch_shape)
                idx = 0
        if idx > 0:
            yield filenames, images
            


# Let's try to visualize specific target groups. **In this example we will see images that contain the protein structures lysosomes or endosomes**. Set target values of your choice and the target group iterator will collect all images that are subset of your choice:

# In[ ]:


your_choice = ["Lysosomes", "Endosomes"]
your_batch_size = 3


# In[ ]:


imageloader = TargetGroupIterator(your_choice, your_batch_size, train_path)
imageloader.find_matching_data_entries()
iterator = imageloader.get_loader()


# In[ ]:


file_ids, images = next(iterator)

fig, ax = plt.subplots(len(file_ids),4,figsize=(20,5*len(file_ids)))
if ax.shape == (4,):
    ax = ax.reshape(1,-1)
for n in range(len(file_ids)):
    make_image_row(images[n], ax[n], make_title(file_ids[n]))


# In[ ]:




