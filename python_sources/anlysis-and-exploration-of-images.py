#!/usr/bin/env python
# coding: utf-8

# # Do you Know About what is Pneumonia?
# 
# Pneumonia is an infection that inflames the air sacs in one or both lungs. The air sacs may fill with fluid or pus (purulent material), causing cough with phlegm or pus, fever, chills, and difficulty breathing. A variety of organisms, including bacteria, viruses and fungi, can cause pneumonia.
# 
# Pneumonia can range in seriousness from mild to life-threatening. It is most serious for infants and young children, people older than age 65, and people with health problems or weakened immune systems.
# 
# <img src="https://lungdiseasenews.com/wp-content/uploads/2015/11/shutterstock_318078170.jpg" alt="Pneumonia" height="500" width="750">
# 
# 
# ## How Is Pneumonia Diagnosed?
# 
# Physical exam: Your doctor will listen to your lungs with a stethoscope. If you have pneumonia, your lungs may make crackling, bubbling, and rumbling sounds when you inhale. You also may be wheezing, and it may be hard to hear sounds of breathing in some areas of your chest.
# 
# ## What is Symptoms of Pneumonia?
# 
# * Chest pain when you breathe or cough
# * Confusion or changes in mental awareness (in adults age 65 and older)
# * Cough, which may produce phlegm
# * Fatigue
# * Fever, sweating and shaking chills
# * Lower than normal body temperature (in adults older than age 65 and people with weak immune systems)
# * Nausea, vomiting or diarrhea
# * Shortness of breath
# 
# # Let Dig out the Data and Find Solutions of  Pneumonia

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import pydicom

import os
print(os.listdir("../input"))
from os import listdir
from os.path import isfile, join

# Any results you write to the current directory are saved as output.


# In[ ]:



### plot packages
import plotly.offline as py
py.init_notebook_mode(connected=True)
from plotly.offline import init_notebook_mode, iplot
import plotly.figure_factory as ff
import matplotlib as plt
import plotly.graph_objs as go
import plotly.tools as tls
get_ipython().run_line_magic('matplotlib', 'inline')
import cufflinks as cf
cf.go_offline()


# In[ ]:


train_images_dir = '../input/stage_1_train_images/'
train_images = [f for f in listdir(train_images_dir) if isfile(join(train_images_dir, f))]
test_images_dir = '../input/stage_1_test_images/'
test_images = [f for f in listdir(test_images_dir) if isfile(join(test_images_dir, f))]
print('5 Training images', train_images[:5]) # Print the first 5


# In[ ]:


train_label = pd.read_csv("../input/stage_1_train_labels.csv")
class_info = pd.read_csv("../input/stage_1_detailed_class_info.csv")


# ## Road Map of Exploration
# * Class Count of stage_1_detailed
# * Target Variable
# * Preprocessing Function 
# * All Case Images of No_Lung_Opacity / Normal /Lung_Opacity
# * Filter the all_class of No_Lung_Opacity / Normal /Lung_Opacity
# * Only Lung_Opacity Image Insights
# * Only Not_Normal_Opacity Image Insights
# * Only Normal_Opacity Image Insights

# In[ ]:


### plot packages
import plotly.offline as py
py.init_notebook_mode(connected=True)
from plotly.offline import init_notebook_mode, iplot
import plotly.figure_factory as ff
import matplotlib as plt
import plotly.graph_objs as go
import plotly.tools as tls
get_ipython().run_line_magic('matplotlib', 'inline')
import cufflinks as cf
cf.go_offline()

trace1 = go.Bar(
            x=class_info['class'].value_counts().index,
            y=class_info['class'].value_counts().values,
        marker=dict(
            color='rgba(222,45,38,0.8)',
        )
    )

data = [trace1]
layout = go.Layout(
    title = 'Class Count of stage_1_detailed'
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='grouped-bar')


# In[ ]:


trace1  = go.Bar(
            x=train_label['Target'].value_counts().index,
            y=train_label['Target'].value_counts().values,
         marker=dict(
                color='rgba(222,99,38,0.8)',
            )
    )

data = [trace1]
layout = go.Layout(
    title = 'Target Variable'
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='basic-bar')


# ### Preprocessing Function 
# 

# In[ ]:


# Forked from `https://www.kaggle.com/peterchang77/exploratory-data-analysis`
def parse_data(df):
    
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

parsed = parse_data(train_label)

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


# ### All Case Images of No_Lung_Opacity / Normal /Lung_Opacity

# In[ ]:


import matplotlib.pylab as plt
plt.style.use('default')
fig=plt.figure(figsize=(25, 25))
columns = 3; rows = 3
for i in range(1, columns*rows +1):
    fig.add_subplot(rows, columns, i)
    draw(parsed[train_label['patientId'].unique()[i]])
    fig.add_subplot


# ### Filter the all_class of No_Lung_Opacity / Normal /Lung_Opacity

# In[ ]:


opacity = class_info     .loc[class_info['class'] == 'Lung Opacity']     .reset_index()
not_normal = class_info     .loc[class_info['class'] == 'No Lung Opacity / Not Normal']     .reset_index()
normal = class_info     .loc[class_info['class'] == 'Normal']     .reset_index()


# ### Only Lung_Opacity Image Insights
# 
# #### Community-acquired pneumonia is the most common type of pneumonia. It occurs outside of hospitals or other health care facilities. It may be caused by:
# 
# * Bacteria - The most common cause of bacterial pneumonia in the U.S. is Streptococcus pneumoniae. This type of pneumonia can occur on its own or after you've had a cold or the flu. It may affect one part (lobe) of the lung, a condition called lobar pneumonia.
# * Bacteria like organisms -  Mycoplasma pneumoniae also can cause pneumonia. It typically produces milder symptoms than do other types of pneumonia. Walking pneumonia is an informal name given to this type of pneumonia, which typically isn't severe enough to require bed rest.
# * Fungi - This type of pneumonia is most common in people with chronic health problems or weakened immune systems, and in people who have inhaled large doses of the organisms. The fungi that cause it can be found in soil or bird droppings and vary depending upon geographic location.
# * Viruses -  Some of the viruses that cause colds and the flu can cause pneumonia. Viruses are the most common cause of pneumonia in children younger than 5 years. Viral pneumonia is usually mild. But in some cases it can become very serious.
# 

# In[ ]:


plt.style.use('default')
fig=plt.figure(figsize=(25, 25))
columns = 3; rows = 3
for i in range(1, columns*rows +1):
    fig.add_subplot(rows, columns, i)
    draw(parsed[opacity['patientId'].unique()[i]])


# ### Only Not_Normal_Opacity Image Insights
# 
# #### Normal factors include:
# 
# * Being hospitalized. You're at greater risk of pneumonia if you're in a hospital intensive care unit, especially if you're on a machine that helps you breathe (a ventilator).
# * Chronic disease. You're more likely to get pneumonia if you have asthma, chronic obstructive pulmonary disease (COPD) or heart disease.
# * Smoking. Smoking damages your body's natural defenses against the bacteria and viruses that cause pneumonia.
# * Weakened or suppressed immune system. People who have HIV/AIDS, who've had an organ transplant, or who receive chemotherapy or long-term steroids are at risk.
# 

# In[ ]:


plt.style.use('default')
fig=plt.figure(figsize=(25, 25))
columns = 3; rows = 3
for i in range(1, columns*rows +1):
    fig.add_subplot(rows, columns, i)
    draw(parsed[not_normal['patientId'].loc[i]])


# ### Only Normal_Opacity Image Insights

# In[ ]:


plt.style.use('default')
fig=plt.figure(figsize=(25, 25))
columns = 3; rows = 3
for i in range(1, columns*rows +1):
    fig.add_subplot(rows, columns, i)
    draw(parsed[normal['patientId'].loc[i]])


# In[ ]:




