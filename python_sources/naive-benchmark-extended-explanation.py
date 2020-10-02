#!/usr/bin/env python
# coding: utf-8

# ### This fork extends the explanations and shows the magic behind the initial code.
# 
# Forked from: https://www.kaggle.com/kyleboone/naive-benchmark-galactic-vs-extragalactic
# 
# Initial kernel author: Kyle Boone (https://www.kaggle.com/kyleboone)
# 
# Current kernel author: Ilya Khristoforov (https://www.kaggle.com/darbin)
# 
# # Galactic vs Extragalactic Objects
# 
# The astronomical transients that appear in this challenge can be separated into two distinct groups: ones that are in our Milky Way galaxy (galactic) and ones that are outside of our galaxy (extragalactic). As described in the data note, all of the galactic objects have been assigned a host galaxy photometric redshift of 0. We can use this information to immediately classify every object as either galactic or extragalactic and remove a lot of potential options from the classification. Doing so results in matching the naive benchmark.
# 
# We find that all of the classes are either uniquely galactic or extragalactic except for class 99 which represents the unknown objects that aren't in the training set.

# ## Load the data
# 
# For this notebook, we'll only need the metadata (**training_set_metadata.csv** and **test_set_metadata.csv**).

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[ ]:


# matplotlib setting: create static png graphs instead of interactive ones
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


meta_data = pd.read_csv('../input/training_set_metadata.csv')
test_meta_data = pd.read_csv('../input/test_set_metadata.csv')


# Map the classes to the range 0-14. We manually add in the 99 class that doesn't show up in the training data.

# In[ ]:


# create list of all unique classes and add class 99 (undefiend class) to the list
classes = np.unique(meta_data['target'])
classes_all = np.hstack([classes, [99]])
classes_all


# In[ ]:


# create a dictionary {class : index} to map class number with the index 
# (index will be used for submission columns like 0, 1, 2 ... 14)
target_map = {j:i for i, j in enumerate(classes_all)}
target_map


# In[ ]:


# check the type of the target_map
type(target_map)


# In[ ]:


# create 'target_id' column to map with 'target' classes
# target_id is the index defined in previous step: see dictionary target_map
# this column will be used later as index for the columns in the final submission
target_ids = [target_map[i] for i in meta_data['target']]
meta_data['target_id'] = target_ids
meta_data.head()


# Let's look at which classes show up in galactic vs extragalactic hosts. We can use the hostgal_specz key which is 0 for galactic objects.

# In[ ]:


# hostgal_specz == 0 means that the object is situated in our Milky Way Galaxy, these objects are labeled as 'Galactic'
# other objects exist outside of our Galaxy and thus are labeld as 'Extragalactic'
# hostgal_specz == 0 is the mask which will be used to separate Galactic objects from Extragalactic
# hostgal_specz - is 'host galaxy spectral redshift'. The bigger the redshift - the farrer the object from us.
galactic_cut = meta_data['hostgal_specz'] == 0

# create figure object to hold an image 10x8 (width x height)
plt.figure(figsize=(10, 8))

# create 2 histograms on one image showing the number of objects in each class
# label - is a legend label
# alpha - is a transparency (50%) to see the overlapping between the 2 histograms
# 15 - is the number of bins for hist
# (0, 15) - is the range

# histogram 1 for hostgal_specz == 0 (Galactic objects)
plt.hist(meta_data[galactic_cut]['target_id'], 15, (0, 15), alpha=0.5, label='Galactic')
# histogram 2 for hostgal_specz <> 0 (Extragalactic objects)
plt.hist(meta_data[~galactic_cut]['target_id'], 15, (0, 15), alpha=0.5, label='Extragalactic')

# labels for x axis
plt.xticks(np.arange(15)+0.5, classes_all)

# show y axis in a log format to narrow the distances between the counts for each bar: 
# this simplifies the presentation and helps to determine the overlapings between the two hists
plt.gca().set_yscale("log")

# print axes labels
plt.xlabel('Class')
plt.ylabel('Counts')

# range for x axis (0 - 15)
plt.xlim(0, 15)

# print a legend
plt.legend();

# as we can see classes 6, 16, 53,65 and 92 are all Galactic
# and classes 15, 42, 52, 62,64, 67, 88, 90 and 95 are all Extragalactic
# there is no overlapping between the classes


# There is no overlap at all between the galactic and extragalactic objects in the training set. Class 99 isn't represented in the training set at all. Let's make a classifier that checks if an object is galactic or extragalactic and then assigns a flat probability to each class in that group. We'll include class 99 in both the galactic and extragalactic groups.

# In[ ]:


# Build the flat probability arrays for both the galactic and extragalactic groups

# Extract galactic and Extragalactic classes
galactic_cut = meta_data['hostgal_specz'] == 0
galactic_data = meta_data[galactic_cut]
extragalactic_data = meta_data[~galactic_cut]

galactic_classes = np.unique(galactic_data['target_id'])
extragalactic_classes = np.unique(extragalactic_data['target_id'])

print('Galactic classes:', galactic_classes)
print('Extragalactic classes:', extragalactic_classes)


# In[ ]:


# Add class 99 (id=14) to both groups (Galactic and Extragalactic classes).
galactic_classes = np.append(galactic_classes, 14)
extragalactic_classes = np.append(extragalactic_classes, 14)

print('Galactic classes:', galactic_classes)
print('Extragalactic classes:', extragalactic_classes)


# In[ ]:


# create a 15 zeros array 'galactic_probabilities'
galactic_probabilities = np.zeros(15)
print('Zeros for Galactic probabilities:', galactic_probabilities)

# suppose that probability for the Galactic object to have certain Galactic class is evenly distributed 
# create an array of probabilities for the galactic object to belong to a certain Galactic class
galactic_probabilities[galactic_classes] = 1 / len(galactic_classes)
print('Galactic flat probabilities: ',galactic_probabilities)


# In[ ]:


# create a 15 zeros array 'extragalactic_probabilities'
extragalactic_probabilities = np.zeros(15)
print('Zeros for Extragalactic probabilities:', extragalactic_probabilities)

# suppose that probability for the Extragalactic object to have certain Extragalactic class is evenly distributed 
# create an array of probabilities for the galactic object to belong to a certain Extragalactic class
extragalactic_probabilities[extragalactic_classes] = 1 / len(extragalactic_classes)
print('Extragalactic flat probabilities: ', extragalactic_probabilities)


# Apply this prediction to the data. We simply choose which of the two probability arrays to use based off of whether the object is galactic or extragalactic.

# In[ ]:


# Apply this prediction to a table

# import progress bar package
import tqdm

def do_prediction(table):
    probs = []
    for index, row in tqdm.tqdm(table.iterrows(), total=len(table)):
        
        # we use 'hostgal_photoz' (photometric redshift) here instead of 'hostgal_specz' (spectral redshift)
        # it is the same redshift measure but made faster on a larger area and thus less accurate, but we have this measure for all of the objects 
        
        # if object is in the Milky Way Galaxy
        if row['hostgal_photoz'] == 0:
            prob = galactic_probabilities
            
        # if object is out of the Milky Way Galaxy
        else:
            prob = extragalactic_probabilities
        probs.append(prob)
    
    return np.array(probs)

pred = do_prediction(meta_data)
test_pred = do_prediction(test_meta_data)


# Now write the prediction out and submit it. This notebook gets a score of 2.158 which matches the naive benchmark.

# In[ ]:


test_df = pd.DataFrame(index=test_meta_data['object_id'], data=test_pred, columns=['class_%d' % i for i in classes_all])
test_df.to_csv('./naive_benchmark.csv')

