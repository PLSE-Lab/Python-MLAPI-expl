#!/usr/bin/env python
# coding: utf-8

# # Pneumonia Detection Competition
# ## Data Exploration
# What is pneumonia?
# "Chest X-rays are currently the best available method for diagnosing pneumonia, playing a crucial role in clinical care and epidemiological studies. Pneumonia is responsible for more than 1 million hospitalizations and 50,000 deaths per year in the US alone." - [Link to Stanford ML Group Paper](https://stanfordmlgroup.github.io/projects/chexnet/)
# 
# <img src="https://www.mayoclinic.org/-/media/671275b4a4e64a868f06eb8b18b002fa.jpg" alt="drawing" width="350"/>

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pylab as plt
import pydicom
import os
from os import listdir
from os.path import isfile, join
# print(os.listdir("../input"))


# ## Data Overview
# ### Stage 1 Images - `stage_1_train_images.zip` and `stage_1_test_images.zip`
# - images for the current stage. Filenames are also patient names.
# 
# ### Stage 1 Labels - `stage_1_train_labels.csv` and Stage 1 Sample Submission `stage_1_sample_submission.csv`
# - Which provides the IDs for the test set, as well as a sample of what your submission should look like
# 
# ### Stage 1 Detailed Info - `stage_1_detailed_class_info.csv`
# - contains detailed information about the positive and negative classes in the training set, and may be used to build more nuanced models.

# In[ ]:


# Images Example
train_images_dir = '../input/stage_1_train_images/'
train_images = [f for f in listdir(train_images_dir) if isfile(join(train_images_dir, f))]
test_images_dir = '../input/stage_1_test_images/'
test_images = [f for f in listdir(test_images_dir) if isfile(join(test_images_dir, f))]
print('5 Training images', train_images[:5]) # Print the first 5


# In[ ]:


print('Number of train images:', len(train_images))
print('Number of test images:', len(test_images))


# ## Plot a few training images from `stage_1_train_images.zip`

# In[ ]:


plt.style.use('default')
fig=plt.figure(figsize=(20, 10))
columns = 8; rows = 4
for i in range(1, columns*rows +1):
    ds = pydicom.dcmread(train_images_dir + train_images[i])
    fig.add_subplot(rows, columns, i)
    plt.imshow(ds.pixel_array, cmap=plt.cm.bone)
    fig.add_subplot


# ## Look at labels in `stage_1_train_labels.csv`

# In[ ]:


train_labels = pd.read_csv('../input/stage_1_train_labels.csv')
train_labels.head()


# ## Distribution of Positive Labels

# In[ ]:


# Number of positive targets
print(round((8964 / (8964 + 20025)) * 100, 2), '% of the examples are positive')
pd.DataFrame(train_labels.groupby('Target')['patientId'].count())


# In[ ]:


# Distribution of Target in Training Set
plt.style.use('ggplot')
plot = train_labels.groupby('Target')     .count()['patientId']     .plot(kind='bar', figsize=(10,4), rot=0)


# ## Size of the impacted area
# We can make a new feature called "area" to the train labels data to see what the distribution of areas label look like.

# In[ ]:


plt.style.use('ggplot')
train_labels['area'] = train_labels['width'] * train_labels['height']
plot = train_labels['area'].plot(kind='hist',
                          figsize=(10,4),
                          bins=20,
                          title='Distribution of Area within Image idenfitying a positive target')


# # Plotting Boxes around Images
# Thanks for plotting functions from @peterchang77 `https://www.kaggle.com/peterchang77/exploratory-data-analysis` !!!

# In[ ]:


# Forked from `https://www.kaggle.com/peterchang77/exploratory-data-analysis`
def parse_data(df):
    """
    Method to read a CSV file (Pandas dataframe) and parse the 
    data into the following nested dictionary:

      parsed = {
        
        'patientId-00': {
            'dicom': path/to/dicom/file,
            'label': either 0 or 1 for normal or pnuemonia, 
            'boxes': list of box(es)
        },
        'patientId-01': {
            'dicom': path/to/dicom/file,
            'label': either 0 or 1 for normal or pnuemonia, 
            'boxes': list of box(es)
        }, ...

      }

    """
    # --- Define lambda to extract coords in list [y, x, height, width]
    extract_box = lambda row: [row['y'], row['x'], row['height'], row['width']]

    parsed = {}
    for n, row in df.iterrows():
        # --- Initialize patient entry into parsed 
        pid = row['patientId']
        if pid not in parsed:
            parsed[pid] = {
                'dicom': '../input/stage_1_train_images/%s.dcm' % pid,
                'label': row['Target'],
                'boxes': []}

        # --- Add box if opacity is present
        if parsed[pid]['label'] == 1:
            parsed[pid]['boxes'].append(extract_box(row))

    return parsed

parsed = parse_data(train_labels)

def draw(data):
    """
    Method to draw single patient with bounding box(es) if present 

    """
    # --- Open DICOM file
    d = pydicom.read_file(data['dicom'])
    im = d.pixel_array

    # --- Convert from single-channel grayscale to 3-channel RGB
    im = np.stack([im] * 3, axis=2)

    # --- Add boxes with random color if present
    for box in data['boxes']:
        #rgb = np.floor(np.random.rand(3) * 256).astype('int')
        rgb = [255, 251, 204] # Just use yellow
        im = overlay_box(im=im, box=box, rgb=rgb, stroke=15)

    plt.imshow(im, cmap=plt.cm.gist_gray)
    plt.axis('off')

def overlay_box(im, box, rgb, stroke=2):
    """
    Method to overlay single box on image

    """
    # --- Convert coordinates to integers
    box = [int(b) for b in box]
    
    # --- Extract coordinates
    y1, x1, height, width = box
    y2 = y1 + height
    x2 = x1 + width

    im[y1:y1 + stroke, x1:x2] = rgb
    im[y2:y2 + stroke, x1:x2] = rgb
    im[y1:y2, x1:x1 + stroke] = rgb
    im[y1:y2, x2:x2 + stroke] = rgb

    return im


# In[ ]:


plt.style.use('default')
fig=plt.figure(figsize=(20, 10))
columns = 8; rows = 4
for i in range(1, columns*rows +1):
    fig.add_subplot(rows, columns, i)
    draw(parsed[train_labels['patientId'].unique()[i]])
    fig.add_subplot


# ## A closer look at a Positive and Negative Example

# In[ ]:


fig=plt.figure(figsize=(20, 10))
draw(parsed[train_labels['patientId'].loc[20]])
plt.show()
fig=plt.figure(figsize=(20, 10))
draw(parsed[train_labels['patientId'].loc[10]])
plt.show()


# # EDA of Detailed Class Info

# In[ ]:


detailed_class_info = pd.read_csv('../input/stage_1_detailed_class_info.csv')
detailed_class_info.groupby('class').count()


# In[ ]:


plt.style.use('ggplot')
plot = detailed_class_info.groupby('class').count().plot(kind='bar',
                                                  rot=0,
                                                  title='Count of Class Labels',
                                                  figsize=(10,4))


# In[ ]:


count_labels_per_patient = detailed_class_info.groupby('patientId').count()


# # Images of Each Label Type

# In[ ]:


opacity = detailed_class_info     .loc[detailed_class_info['class'] == 'Lung Opacity']     .reset_index()
not_normal = detailed_class_info     .loc[detailed_class_info['class'] == 'No Lung Opacity / Not Normal']     .reset_index()
normal = detailed_class_info     .loc[detailed_class_info['class'] == 'Normal']     .reset_index()


# ## ** Lung Opacity** Examples

# In[ ]:


plt.style.use('default')
fig=plt.figure(figsize=(20, 10))
columns = 8; rows = 4
for i in range(1, columns*rows +1):
    fig.add_subplot(rows, columns, i)
    draw(parsed[opacity['patientId'].unique()[i]])


# ## ** No Lung Opacity / Not Normal** Examples

# In[ ]:


plt.style.use('default')
fig=plt.figure(figsize=(20, 10))
columns = 8; rows = 4
for i in range(1, columns*rows +1):
    fig.add_subplot(rows, columns, i)
    draw(parsed[not_normal['patientId'].loc[i]])


# ## **Normal** Examples

# In[ ]:


plt.style.use('default')
fig=plt.figure(figsize=(20, 10))
columns = 8; rows = 4
for i in range(1, columns*rows +1):
    fig.add_subplot(rows, columns, i)
    draw(parsed[normal['patientId'].loc[i]])


# # Side By Side Compare of Opacity/Not Normal/Normal

# In[ ]:


fig=plt.figure(figsize=(20, 10))
columns = 3; rows = 1
fig.add_subplot(rows, columns, 1).set_title("Normal", fontsize=30)
draw(parsed[normal['patientId'].unique()[0]])
fig.add_subplot(rows, columns, 2).set_title("Not Normal", fontsize=30)
# ax2.set_title("Not Normal", fontsize=30)
draw(parsed[not_normal['patientId'].unique()[0]])
fig.add_subplot(rows, columns, 3).set_title("Opacity", fontsize=30)
# ax3.set_title("Opacity", fontsize=30)
draw(parsed[opacity['patientId'].unique()[0]])


# In[ ]:


fig=plt.figure(figsize=(20, 10))
columns = 3; rows = 1
fig.add_subplot(rows, columns, 1).set_title("Normal", fontsize=30)
draw(parsed[normal['patientId'].unique()[1]])
fig.add_subplot(rows, columns, 2).set_title("Not Normal", fontsize=30)
# ax2.set_title("Not Normal", fontsize=30)
draw(parsed[not_normal['patientId'].unique()[1]])
fig.add_subplot(rows, columns, 3).set_title("Opacity", fontsize=30)
# ax3.set_title("Opacity", fontsize=30)
draw(parsed[opacity['patientId'].unique()[1]])


# In[ ]:


fig=plt.figure(figsize=(20, 10))
columns = 3; rows = 1
fig.add_subplot(rows, columns, 1).set_title("Normal", fontsize=30)
draw(parsed[normal['patientId'].unique()[2]])
fig.add_subplot(rows, columns, 2).set_title("Not Normal", fontsize=30)
# ax2.set_title("Not Normal", fontsize=30)
draw(parsed[not_normal['patientId'].unique()[2]])
fig.add_subplot(rows, columns, 3).set_title("Opacity", fontsize=30)
# ax3.set_title("Opacity", fontsize=30)
draw(parsed[opacity['patientId'].unique()[2]])


# # Number of Labels Per Patientid
# patients have 0-4 labels, patients can have multiple labels.
# (Only including ones with at least on label)

# In[ ]:


count_labels_per_patient.reset_index().groupby('class').count()


# In[ ]:


# Patients with 4 Labels
count_labels_per_patient.sort_values('class', ascending=False).head()


# In[ ]:


detailed_class_info.loc[detailed_class_info['patientId'] == '7d674c82-5501-4730-92c5-d241fd6911e7']


# # Closer Look of Each Type

# In[ ]:


fig=plt.figure(figsize=(20, 10))
plt.suptitle('"Lung Opacity" Example', fontsize=16)
draw(parsed['7d674c82-5501-4730-92c5-d241fd6911e7'])


# In[ ]:


not_normal = detailed_class_info.loc[detailed_class_info['class'] == 'No Lung Opacity / Not Normal']
not_normal_example = not_normal['patientId']
fig=plt.figure(figsize=(20, 10))
plt.suptitle('"No Lung Opacity / Not Normal" Example', fontsize=16)
draw(parsed['019e035e-2f82-4c66-a198-57422a27925f'])


# In[ ]:


fig=plt.figure(figsize=(20, 10))
plt.suptitle('"Normal" Example', fontsize=16)
draw(parsed['003d8fa0-6bf1-40ed-b54c-ac657f8495c5'])


# # What does the submission look like?

# In[ ]:


pd.read_csv('../input/stage_1_sample_submission.csv').head()


# In[ ]:




