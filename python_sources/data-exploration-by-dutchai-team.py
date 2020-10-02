#!/usr/bin/env python
# coding: utf-8

# # Data Exploration
# Dutch AI team
# 
# ## Version
# 
# 25/10/18
# Setting up notebook and team. Doing first data exploration.

# # Introduction
# Starting a new Kaggle competition is always an exciting moment! You will start to explore most often a new domain with a never seen before dataset. Over the coming few week we will not only learn a lot about proteins, but also learn new AI techniques. In the best case, we might actually improve upon existing algorithms. How nice would that be? ;)
# 
# In this notebook we explore the dataset to get a better understanding of the domain and how to handle the data. Machine learning models will we build in a seperate notebook, once we feel confortable with the data.
# 
# 
# ## File descriptions
# A full description of the files can be read here https://www.kaggle.com/c/human-protein-atlas-image-classification/data. The dataset consist of the following files:
# 
# - *train.csv* - filenames and labels for the training set.
# - *sample_submission.csv* - filenames for the test set, and a guide to constructing a working submission.
# - *train.zip* - All images for the training set.
# - *test.zip* - All images for the test set.
# 
# Let's first start with exploring the train.csv file.

# In[ ]:


# load some default libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# define a PATH variables so you easily use this notebook in a different computing environment
PATH = "../input/"

# read csv ad Pandas dataframe
df_train = pd.read_csv(PATH + "train.csv")
df_train.info()


# In[ ]:


# print head of dataframe
df_train.head(5)


# From the full description we learn that the labels are protein organelle localization labels for each sample, defined by
# 
# 0.  Nucleoplasm  
# 1.  Nuclear membrane   
# 2.  Nucleoli   
# 3.  Nucleoli fibrillar center   
# 4.  Nuclear speckles   
# 5.  Nuclear bodies   
# 6.  Endoplasmic reticulum   
# 7.  Golgi apparatus   
# 8.  Peroxisomes   
# 9.  Endosomes   
# 10.  Lysosomes   
# 11.  Intermediate filaments   
# 12.  Actin filaments   
# 13.  Focal adhesion sites   
# 14.  Microtubules   
# 15.  Microtubule ends    
# 16.  Cytokinetic bridge   
# 17.  Mitotic spindle   
# 18.  Microtubule organizing center   
# 19.  Centrosome   
# 20.  Lipid droplets   
# 21.  Plasma membrane   
# 22.  Cell junctions   
# 23.  Mitochondria   
# 24.  Aggresome   
# 25.  Cytosol   
# 26.  Cytoplasmic bodies   
# 27.  Rods & rings
# 
# ## Plotting images
# 
# To get a better intuition for the data, let's plot the images for the first data sample (first row in dataframe).

# In[ ]:


# put labels in list for easy printing
protein_labels = [
    'Nucleoplasm', 
    'Nuclear membrane', 
    'Nucleoli', 
    'Nucleoli fibrillar center', 
    'Nuclear speckles', 
    'Nuclear bodies', 
    'Endoplasmic reticulum', 
    'Golgi apparatus', 
    'Peroxisomes', 
    'Endosomes', 
    'Lysosomes', 
    'Intermediate filaments', 
    'Actin filaments', 
    'Focal adhesion sites', 
    'Microtubules',
    'Microtubule ends', 
    'Cytokinetic bridge', 
    'Mitotic spindle', 
    'Microtubule organizing center',
    'Centrosome', 
    'Lipid droplets',
    'Plasma membrane', 
    'Cell junctions',
    'Mitochondria', 
    'Aggresome', 
    'Cytosol', 
    'Cytoplasmic bodies',
    'Rods & rings']

# define function to print labels
def print_labels(target):
    label_ints = [int(l) for l in target.split()]
    for i in label_ints:
        print("{} - {}".format(i, protein_labels[i]))


# In[ ]:


# each datapoint consists of four images with different color 
# define function to plot those four images
def plot_protein_images(id):
    fig, axs = plt.subplots(1, 4, figsize=(16,4))

    for i, color in enumerate(['red', 'green', 'yellow', 'blue']):
        filename = "train/{}_{}.png".format(id, color)
        im = plt.imread(PATH + filename)
        axs[i].imshow(im, cmap='binary')
        axs[i].set_title(color)


# In[ ]:


plot_protein_images(df_train.Id[0])
print_labels(df_train.Target[0])


# In[ ]:


plot_protein_images(df_train.Id[1])
print_labels(df_train.Target[1])


# From these images we get a better intuition of the data. Sometimes, you should spent some time on making a 'nice' representation of your data. It's not only fun to do, but representing data in different formats might emphasize different aspects of data. In this case, we could try to merge all four images ('channels') into a RGB images (3 channels). For this we define a 4x3 transformation matrix that turns 4 channels into 3 rgb channels.

# In[ ]:


def plot_color_protein_images(id, ax=None, figsize=(10,10)):
    # use ax argument so this function can be using to plot in a grid using axes
    if ax==None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    # read all color images
    all_images = np.empty((512,512,4))
    for i, color in enumerate(['red', 'green', 'yellow', 'blue']):
        all_images[:,:,i] = plt.imread(PATH + "train/{}_{}.png".format(id, color))

    # define transformation matrix
    # note that yellow is made usign red and green
    # but you can tune this color conversion yourself
    T = np.array([[1,0,1,0],[0,1,1,0],[0,0,0,1]])
    
    # convert to rgb
    rgb_image = np.matmul(all_images.reshape(-1, 4), np.transpose(T))
    rgb_image = rgb_image.reshape(all_images.shape[0], all_images.shape[0], 3)
    rgb_image = np.clip(rgb_image, 0, 1)
    
    # plot
    ax.imshow(rgb_image)
    ax.set(xticks=[], yticks=[])


# In[ ]:


plot_color_protein_images(df_train.Id[1])


# Finally, let's print several nice plots of the proteins.

# In[ ]:


# plot color protein images with target as title
n_rows, n_cols = 4, 8
fig, axs = plt.subplots(n_rows, n_cols, figsize=(18, 10))
axs = axs.ravel()
N = n_rows * n_cols
for i in range(N):
    plot_color_protein_images(df_train.Id[i], axs[i])
    axs[i].set_title(df_train.Target[i])


# ## Statistics
# 
# It is important to know how many samples there are per target. Sometimes a dataset is 'imbalanced', e.g. there are many more samples of target A than target B. This could impact the performance of the model and requires special training preparation.

# In[ ]:


# number of samples
n_datapoints = df_train.shape[0]
n_datapoints


# In[ ]:


# count protein targets
count = np.zeros(len(protein_labels))
for target in df_train.Target:
    label_ints = [int(l) for l in target.split()]
    count[label_ints] = count[label_ints] + 1

plt.figure(figsize=(14,6))
plt.bar(range(len(protein_labels)), count)
plt.ylabel('count')
plt.xticks(range(len(protein_labels)), protein_labels, rotation=-90);


# As we can see from this bar diagram, the dataset is imbalanced. This we should take into account when training a model.
# 
# Next, let's see is some targets are correlated.

# In[ ]:


# create array with target encoding
n_labels = len(protein_labels)
a_targets = np.zeros((n_datapoints, n_labels))
for i, target in enumerate(df_train.Target):
    label_ints = [int(l) for l in target.split()]
    a_targets[i, label_ints] = 1 


# In[ ]:


# calculate correlation matrix
C = np.corrcoef(a_targets, rowvar=False)
C.shape


# In[ ]:


plt.figure(figsize=(10,8))
sns.heatmap(C, xticklabels=protein_labels, yticklabels=protein_labels)
plt.title('Correlation matrix')


# In[ ]:




