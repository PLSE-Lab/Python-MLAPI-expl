#!/usr/bin/env python
# coding: utf-8

# <h1>EDA with Data Visualization and Augmentation</h1>
# In this **tutorial** we are going to explore the dataset searching for hidden patterns using** statistical** and **manual** methodes.
# We are also going to see how we can perform **Clustering** to cluster images based on the label data we have and also will perform some **data augmentation** to be able to classify the images efficiently.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pydicom
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

import os
print(os.listdir("../input"))


# We will start by exploring the target classes we have in the dataset.

# In[ ]:



patient_classes = pd.read_csv('../input/stage_2_detailed_class_info.csv')
patient_classes.head()


# Before proceeding we need to know the number of target classes and its names.

# In[ ]:


print("Number of classes in the dataset :: %i" %  len(patient_classes["class"].unique()))
print("Classes' names are :: %s" % patient_classes["class"].unique())


# To be able to decide which performance metric we may use, We need to know the number of data examples in eachclass to see if the data is skewed or not. If the data is skewe ( one class has much more data examples than other classes ) then we can't use accuracy because it yeald misleading results but we may use Precision, Recall or F-beta score according to the situation we have.

# In[ ]:


class_count = patient_classes['class'].value_counts()
class_count.plot.bar( ec="orange")
print(class_count)


# It seems that data is not skewed so we will proceed with accuracy as the performance metric for the algorithm.
# Now let's try to see how our train_labels data looks like.

# In[ ]:



train_labels = pd.read_csv('../input/stage_2_train_labels.csv')
print(train_labels.iloc[0])


# From the above example we see that for every label we have a patient ID, and bounding boxes coordinates which are nulls for class 0 and have values for class 1

# Input images are stored in DICOM format which stores description for every image in the dataset. Now let's see what we can benefit from this description.

# In[ ]:



dcm_file = '../input/stage_2_train_images/%s.dcm' % train_labels.patientId.tolist()[0]
dcm_data = pydicom.read_file(dcm_file)
print(dcm_data)  


# Most of the information doesn't seem to be useful except Age and Gender.

# Now let's store all the destinct IDs into a dataframe for future usage.

# In[ ]:


patientIds = train_labels.drop_duplicates('patientId', keep = 'first').patientId.tolist()


# We will now extract the PatientAge and PatientSex fields in the DICOM description for every destince ID in a seperate array for each one and then convert this to a dataframe to be able to use it later.

# In[ ]:


Sex = []
Age = []
for patientId in patientIds:
    dcm_file = '../input/stage_2_train_images/%s.dcm' % patientId
    dcm_data = pydicom.read_file(dcm_file)
    Sex.append(dcm_data.PatientSex)
    Age.append(int(dcm_data.PatientAge))


# In[ ]:


patientInfo = pd.DataFrame({'patientId': patientIds, 'patientSex': Sex, 'patientAge': Age})
patientInfo.dtypes


# Let's now explore the data we collected in the dataframe and search for missing values.

# In[ ]:


patientAge_count = patientInfo['patientAge'].value_counts().sum()
patientSex_count = patientInfo['patientSex'].value_counts().sum()
patient_count = patientInfo['patientId'].value_counts().sum()

print("total number of patientId :: %i" % patient_count )
print("Total number of patients with Non null patientSex :: %i " % patientSex_count )
print("Total number of patients with Non null patientAge :: %i " %  patientAge_count )
print("Number of missing values to be imputed for the first field :: %i " % (patient_count - patientSex_count) )
print("Number of missing values to be imputed for the second field :: %i " % (patient_count - patientAge_count) )


# In[ ]:


patientInfo = patientInfo.set_index('patientId').join(train_labels.set_index('patientId'))[['patientSex', 'patientAge', 'Target']]
patientInfo.reset_index(inplace=True)
patientInfo.head()


# In[ ]:


patientInfo.describe()


# **Exploring Age Feature**

# We will explore the distinct age values in the dataset, Age greater than 90 years and also the frequency for each value using histgrams. Note: the oldest person on earth is 122 years old so any value more than this is probably an outlier.

# In[ ]:



patientInfo['patientAge'].unique()


# In[ ]:



patientInfo['patientAge'].hist()


# In[ ]:



patientInfo[patientInfo['patientAge']>=90]['patientAge'].hist(bins=50)


# In[ ]:


patientInfo[patientInfo['patientAge']>=85]['patientAge'].value_counts()


# # Exploring Images 
# We will explore images from the dataset based on different age groups to see if there is a significant difference in the images to decide if we need further processing.

# In[ ]:



def draw_img(patient_id, title=None):
    dcm_file = '../input/stage_2_train_images/%s.dcm' % patient_id
    dcm_data = pydicom.read_file(dcm_file)
    plt.imshow(dcm_data.pixel_array)
    if title is not None:
        plt.title(title)


# In[ ]:



patients_greater_100 = patientInfo[patientInfo['patientAge']>=100]
patients_less_5 = patientInfo[patientInfo['patientAge']<=5]
patients_mid_age = patientInfo[(patientInfo['patientAge']>=30) & (patientInfo['patientAge']<= 50)]


# In[ ]:



def draw_grid(arr_patients, rows=5, columns=4, titles=None, figsize=(15, 15)):
    fig=plt.figure(figsize=figsize)
    for i in range(1, columns*rows + 1):
        if(i <= len(arr_patients)):
            fig.add_subplot(rows, columns, i)
            if titles is None:
                    draw_img(arr_patients[i - 1])
            else:
                    draw_img(arr_patients[i - 1], title=titles[i - 1])
    plt.show()


# In[ ]:


draw_grid(patients_mid_age['patientId'].tolist())


# In[ ]:


draw_grid(patients_greater_100['patientId'].tolist(), 2, 2)


# In[ ]:


draw_grid(patients_less_5['patientId'].tolist())


# In[ ]:


patientInfo['age_category'] = (patientInfo['patientAge'] // 10) * 10


# In[ ]:


ax = sns.countplot(x="age_category", hue="Target", data=patientInfo)
ax.set_title('Disease per age category')
ax.legend(title='Disease')
ax.legend()


# # Explore Gender
# We will explore the images based on the ggender of the patient.
# First we need to see if the number of patients is the same for males and females.

# In[ ]:


patientInfo['patientSex'].value_counts()


# Now we will try to see is there is a segnificant difference between the radiation images of males and females.

# In[ ]:



draw_grid(patientInfo[patientInfo['patientSex'] == 'M']['patientId'].tolist(), 3, 3, 
          titles=patientInfo[patientInfo['patientSex'] == 'M']['patientAge'].tolist(), figsize=(15, 15))


# In[ ]:



draw_grid(patientInfo[patientInfo['patientSex'] == 'F']['patientId'].tolist(), 3, 3, 
          titles=patientInfo[patientInfo['patientSex'] == 'F']['patientAge'].tolist(), figsize=(15, 15))


# I can see no difference between the images based on the gender of the patient only.

# Now let's try to explore the numbers in each class based on the gender.

# In[ ]:



ax = sns.countplot(x="patientSex", hue="Target", data=patientInfo)
ax.set_title('Disease per gender')
ax.legend(title='Disease')
ax.legend()


# # Age and Gender Exploration
# Now We will explore the age, gender and also the target classes. We will discover the data representation using histograms to see how the data looks like.

# In[ ]:



z = {'F': 1, 'M': 0}
patientInfo['Sex'] = patientInfo['patientSex'].map(z)


# In[ ]:



sns.pairplot(patientInfo[['Sex', 'age_category', 'Target']]);


# It is very important to know the correlation between features and target variable to be able to decide if the feature is important to be used in the model or not. We will use correlation matrix and heat map to see this visually.

# In[ ]:


corr = patientInfo.corr()
corr


# In[ ]:


import seaborn as sns


import matplotlib.pyplot as plt

f, ax = plt.subplots(figsize=(25, 25))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, cmap=cmap, vmax= .3,vmin = - 0.3, center=0,
            square=True, linewidths=.5)


# We also have  to check if any label that is of the +ve class has Nulls in the bounding boxex data. Since only bounding boxes in the 0 class can have nulls since we will not draw anything.

# In[ ]:


print("The number instances for the 0 class:: %i and for the 1 class = %i " % (train_labels.Target.value_counts()[0], train_labels.Target.value_counts()[1]) )


# In[ ]:


train_ones = train_labels[train_labels.Target == 1]
train_ones.head()


# In[ ]:


print("The number of NaN values for bounding box dimentions columns for class 1 data = %s " % train_ones.isna().sum() )


# # Drawing bounding boxes on images
# We will draw the bounding boxes on the images to see how disease looks like in the radiation images .

# In[ ]:



train_labels.Target.value_counts()


# In[ ]:



train_labels.iloc[110]


# In[ ]:




print('all images:', train_labels.shape[0])
print('unique images:', np.unique(train_labels.patientId.tolist()).shape[0])


# In[ ]:



def draw_mult_rects(patientIds, rows=3, cols=3, figsize=(15, 15)):
    
    fig=plt.figure(figsize=figsize)
    
    for i in range(1, len(patientIds)+1):
        fig.add_subplot(rows, cols, i)
        records = train_labels[train_labels.patientId == patientIds[i-1]]
        class_label = patient_classes[patient_classes.patientId == patientIds[i-1]]['class'].tolist()[0]
        dcm_file = '../input/stage_2_train_images/%s.dcm' % patientIds[i-1]
        dcm_data = pydicom.read_file(dcm_file)
        plt.imshow(dcm_data.pixel_array)
        plt.title(class_label)
        for j in range(records.shape[0]):
            record = records.iloc[j]                
            x = record.x
            y = record.y
            width = record.width
            height = record.height
            if x is not None:
                rect = patches.Rectangle((x,y),width,height,linewidth=1,edgecolor='r',facecolor='none')
                plt.gca().add_patch(rect)
    plt.show()


# In[ ]:



draw_mult_rects(train_labels.patientId.unique()[:9])


# # Clustering
# We have noticed a great difference in radiation images based on the Age and gender of the patient, So we will try to cluster images based on the differences in the features for each patient and examine the output histograms and draw an image for each cluster . We will explore the hestograms of each clustering procedure when changing the number of centroids of the clustering algorithm.

# In[ ]:


patientId = patientInfo.patientId
tmp_cluster = patientInfo.drop(['patientId', "Target", 'patientSex', 'age_category'], axis = 1)
tmp_cluster['patientAge'].max()


# In[ ]:


from sklearn.cluster import KMeans

def kmeans(n, tmp_cluster) :
    cluster = KMeans(n_clusters=n, max_iter=300, tol=0.0001, verbose=0, random_state = 0, n_jobs=-1).fit(tmp_cluster)
    return cluster.labels_


# Plotting the histogram of clustered data and also viewing an image from each cluster to see what we can be gathered in each cluster.

# In[ ]:


import matplotlib.pyplot as plt
tmp_cluster_Id = pd.DataFrame()
rows = 0
for i in range(2,10):
    columns = 3
    tmp_cluster["clusters"] = kmeans(i, tmp_cluster)
    n, bins, patches = plt.hist(tmp_cluster["clusters"], facecolor='b')
    tmp_cluster_Id = tmp_cluster.copy()
    tmp_cluster_Id["patientId"] = patientId
    pics = ((tmp_cluster_Id.drop_duplicates('clusters', keep = 'first'))['patientId']).tolist()
    
    if( len( pics ) >= 3) :
        rows = (((len(pics) - 3) / 3) + (len(pics) - 3) % 3) + 1
    else : 
        columns = 2
        rows = 1
    draw_grid(pics, columns = columns, rows = int(rows) )
    plt.show()


# # Data Augmentation
# Here we will perform a very simple data augmentation. We will crop the edges of the images since we didn't need it. When trying this with our model it had significant effect on decreasing the variance.

# In[ ]:


import cv2

def remove_borders(img_data, threshold = 10):
    img_data = img_data[:, np.max(img_data, axis=0) > threshold]
    img_data = img_data[np.max(img_data, axis=1) > threshold]
    img_data = cv2.resize(img_data, (1024, 1024))
    return img_data


# Now we will see how the images look like after removing the boarders. I will show the same images as the above images to be able do see the differences.

# In[ ]:


def read_one_img(patientId):
    dcm_file = '../input/stage_2_train_images/%s.dcm' % patientId
    dcm_data = pydicom.read_file(dcm_file)
    img = dcm_data.pixel_array
    img = remove_borders(img)
    img = cv2.resize(img, (224, 224))
    return img


# In[ ]:


figure=plt.figure(figsize=(15,15))

for (i, j) in enumerate(pics) :
    image = read_one_img(j)
    image_pixels = remove_borders(image)
    figure.add_subplot(3, 3, i+1)
    plt.imshow(image_pixels)
    
plt.show()


# ### Note : The data augmentation part is still in developpment since we can add many other things to it.
