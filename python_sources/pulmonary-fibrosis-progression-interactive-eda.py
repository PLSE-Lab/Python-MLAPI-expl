#!/usr/bin/env python
# coding: utf-8

# ### >>>>>>>>>> for using interactive features please fork it and then run <<<<<<<<<<
# 
# ![](https://www.osicild.org/uploads/1/2/2/7/122798879/editor/kaggle-v01-clipped_2.png?1569348761)
# ## If you want to know details about the Pulmonary Fibrosis Progression, check out my other kernel [Pulmonary Fibrosis for Non-Med People](https://www.kaggle.com/redwankarimsony/pulmonary-fibrosis-for-non-med-people) which contains detailed information about the the disease, symptoms, causes and many more. If you like this kernel, please upvote.. 

# In[ ]:


import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


#plotly imports
get_ipython().system('pip install chart_studio')
import plotly.express as px
import chart_studio.plotly as py
import plotly.graph_objs as go
from plotly.offline import iplot
import cufflinks
cufflinks.go_offline()
cufflinks.set_config_file(world_readable=True, theme='pearl')


# ### 1. Load Dataframes

# In[ ]:


train_dir = '../input/osic-pulmonary-fibrosis-progression/train/'
test_dir = '../input/osic-pulmonary-fibrosis-progression/test/'


train_df = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')
test_df = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')
print('Training Dataframe shape: ', train_df.shape)

train_df.head(10)


# ### 2. Train and Test Dataframe Details

# In[ ]:


# Let's have a look at the detailed info about the dataframes
print('Training Dataframe Details: ')
print(train_df.info())

print('\n\nTest Dataframe Details: ')
print(test_df.info())



# ### 3. Number of Patients

# In[ ]:


print('Number of patients in training set:',
      len(os.listdir(train_dir)))
print('Number of patients in test set:',
     len(os.listdir(test_dir)))


# ### 4. Creating Individual Patient Profiles 
# Let's create a dataframe that will contain all the unique patient IDs. 

# In[ ]:


# Creating unique patient lists and their properties. 
patient_ids = os.listdir(train_dir)
patient_ids = sorted(patient_ids)

#Creating new rows
no_of_instances = []
age = []
sex = []
smoking_status = []

for patient_id in patient_ids:
    patient_info = train_df[train_df['Patient'] == patient_id].reset_index()
    no_of_instances.append(len(os.listdir(train_dir + patient_id)))
    age.append(patient_info['Age'][0])
    sex.append(patient_info['Sex'][0])
    smoking_status.append(patient_info['SmokingStatus'][0])

#Creating the dataframe for the patient info    
patient_df = pd.DataFrame(list(zip(patient_ids, no_of_instances, age, sex, smoking_status)), 
                                 columns =['Patient', 'no_of_instances', 'Age', 'Sex', 'SmokingStatus'])
print(patient_df.info())
patient_df.head()


# ### 5. Gender Distribution
# It seems that among the unique patients 78% are Male and rest of them are female. 

# In[ ]:


patient_df['Sex'].value_counts(normalize = True).iplot(kind = 'bar', 
                                                        color = 'blue', 
                                                        yTitle = 'Unique patient count',
                                                        xTitle = 'Gender',
                                                        title = 'Gender Distribution of the unique patients')


# ### 6. Age Distribution of the Patients
# Now let's have a look at the patients age distributions. We observe that the age distribution starts from near 50 years and it tops to 90 years. Therefore, the given data about the pulmonary fibrosis is mainly of the older people.  

# In[ ]:


import scipy

data = patient_df.Age.tolist()
plt.figure(figsize=(18,6))
# Creating the main histogram
_, bins, _ = plt.hist(data, 15, density=1, alpha=0.5)

# Creating the best fitting line with mean and standard deviation
mu, sigma = scipy.stats.norm.fit(data)
best_fit_line = scipy.stats.norm.pdf(bins, mu, sigma)
plt.plot(bins, best_fit_line, color = 'b', linewidth = 3, label = 'fitting curve')
plt.title(f'Age Distribution [ mean = {"{:.2f}".format(mu)}, standard_dev = {"{:.2f}".format(sigma)} ]', fontsize = 18)
plt.xlabel('Age -->')
plt.show()

patient_df['Age'].iplot(kind='hist',bins=25,color='blue',xTitle='Percent distribution',yTitle='Count')


# let's see the average age of patients based on their sex. However, it is observed that the average age of male and female patients are almost same. However, male patients are mostly sick near the age of 70 

# In[ ]:


plt.figure(figsize=(16, 6))
sns.kdeplot(patient_df.loc[patient_df['Sex'] == 'Male', 'Age'], label = 'Male',shade=True)
sns.kdeplot(patient_df.loc[patient_df['Sex'] == 'Female', 'Age'], label = 'Female',shade=True)
plt.xlabel('Age (years)'); plt.ylabel('Density'); plt.title('Distribution of Ages');


# ### 7. Study of SmokingStatus 
# It seems that almost 70% of the patients are ex-smokers (at some point of their lives, they smoked). However, among the patients about 27% of the patients have never smoked in their life. Hilarious thing is that 5% of the patients are still smoking.. LOL

# In[ ]:


patient_df['SmokingStatus'].value_counts(normalize=True).iplot(kind='bar',
                                                      yTitle='Percentage', 
                                                      linecolor='black', 
                                                      opacity=0.8,
                                                      color='blue',
                                                      theme='pearl',
                                                      bargap=0.5,
                                                      title='SmokingStatus Distribution')


# ### 8. Study of Gender vs SmokingStatus

# In[ ]:


patient_df.groupby(['SmokingStatus', 'Sex']).count()['Patient'].unstack().iplot(kind='bar', 
                                                                                yTitle = 'Unique Patient Count',
                                                                                title = 'Gender vs SmokingStatus' )


# ### 9. Age Distribution of the Patients based on Smoking Status

# In[ ]:


plt.figure(figsize=(16, 6))
sns.kdeplot(patient_df.loc[patient_df['SmokingStatus'] == 'Ex-smoker', 'Age'], label = 'Ex-smoker',shade=True)
sns.kdeplot(patient_df.loc[patient_df['SmokingStatus'] == 'Never smoked', 'Age'], label = 'Never smoked',shade=True)
sns.kdeplot(patient_df.loc[patient_df['SmokingStatus'] == 'Currently smokes', 'Age'], label = 'Currently smokes',shade=True)
# Labeling of plot
plt.xlabel('Age (years)'); plt.ylabel('Density'); plt.title('Distribution of Ages');


# ### 10. Interactive Patient Lookup (FVC Decay)
# If you run the following cell, you will get an dropdown menu where you can select any patient ID and it will show you the details present for that patient in the training data frame. It will show you all the information of that patient from the train data frame and also plot the gradual fibrosis progression over time. It plots the decrease of FVC over time.  
# #### Observations:
# * ***Smoking Status doesn't change over time.*** So smoking status for a single patient is always unique. 
# * ***Age of the patient doesn't change over time*** Whatever the age in the beginning of the diagnosis, even after more than 52 weeks patient's age didn't change in the dataframe. 
# 

# In[ ]:


from ipywidgets import interact  #, interactive, IntSlider, ToggleButtons

def patient_lookup(patient_id):
    print(train_df[train_df['Patient'] == patient_id])
    patient_info = train_df[train_df['Patient'] == patient_id].reset_index()
    fig, (ax1,ax2) = plt.subplots(1,2, figsize = (15, 5))
    ax1.plot(patient_info['Weeks'].tolist() , patient_info['FVC'].tolist(), marker = '*', linewidth = 3,color = 'r', markeredgecolor = 'b')
    ax1.set_title('FVC Deterioriation over the Weeks')
    ax1.set_xlabel('Weeks -->')
    ax1.set_ylabel('FVC')
    ax1.grid(True)
    
    ax2.plot(patient_info['Weeks'].tolist() , patient_info['Percent'].tolist(),marker = '*', linewidth = 3,
            color = 'r', markeredgecolor = 'b' )
    ax2.set_title('Percent change over the weeks')
    ax2.set_xlabel('Weeks -->')
    ax2.set_ylabel('Percent(of adult capacity)')
    ax2.grid(True)
    fig.suptitle(f'P_ID: {patient_id}', fontsize = 20) 
    
    
    
interact(patient_lookup, patient_id = patient_ids)


# ### 11. Interactive Patient Lookup (CT Scans)
# Here from the dropdown menu, please select any patient id, and it will show you 16 CT-Scans over time for that particular patient. 

# In[ ]:


import random
import pydicom
def explore_dicoms(patient_id, instance):
    RefDs = pydicom.dcmread('../input/osic-pulmonary-fibrosis-progression/train/' + 
                            patient_id +
                            '/' + 
                            str(instance) + '.dcm')
    plt.figure(figsize=(10, 5))
    plt.imshow(RefDs.pixel_array, cmap='gray');
    plt.title(f'P_ID: {patient_id}\nInstance: {instance}')
    plt.axis('off')


def show_ct_scans(patient_id):
    no_of_instances = int(patient_df[patient_df['Patient'] == patient_id]['no_of_instances'].values[0])
    files = sorted(random.sample(range(1, no_of_instances), 9))
    rows = 3
    cols = 3
    fig = plt.figure(figsize=(12,12))
    for idx in range(1, rows*cols+1):
        fig.add_subplot(rows, cols, idx)
        RefDs = pydicom.dcmread(train_dir + patient_id + '/' + str(files[idx-1]) + '.dcm')
        plt.imshow(RefDs.pixel_array, cmap='gray')
        plt.title(f'Instance: {files[idx-1]}')
        plt.axis(False)
        fig.add_subplot
    fig.suptitle(f'P_ID: {patient_id}') 
    plt.show()


# In[ ]:


# show_ct_scans(patient_ids[0])
interact(show_ct_scans,patient_id = patient_ids)


# ### 12. Sumarizing the Unique Patient Profile

# In[ ]:


import warnings
warnings.filterwarnings('ignore')
import pandas_profiling as pdp
unique_patient_profile  = pdp.ProfileReport(patient_df)


# In[ ]:


unique_patient_profile

