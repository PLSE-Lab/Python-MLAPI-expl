#!/usr/bin/env python
# coding: utf-8

# <div><div class="markdown-converter__text--rendered"><h1><strong>Hemorrhage Types</strong></h1>
# <p>Hemorrhage in the head (intracranial hemorrhage) is a relatively common condition that has many causes ranging from trauma, stroke, aneurysm, vascular malformations, high blood pressure, illicit drugs and blood clotting disorders.  The neurologic consequences also vary extensively depending upon the size, type of hemorrhage and location ranging from headache to death.  The role of the Radiologist is to detect the hemorrhage, characterize the hemorrhage subtype, its size and to determine if the hemorrhage might be jeopardizing critical areas of the brain that might require immediate surgery. </p>
# <p>While all acute (i.e. new) hemorrhages appear dense (i.e. white) on computed tomography (CT), the primary imaging features that help Radiologists determine the subtype of hemorrhage are the location, shape and proximity to other structures (see table).  </p>
# <p>Intraparenchymal hemorrhage is blood that is located completely within the brain itself; intraventricular or subarachnoid hemorrhage is blood that has leaked into the spaces of the brain that normally contain cerebrospinal fluid (the ventricles or subarachnoid cisterns).  Extra-axial hemorrhages are blood that collects in the tissue coverings that surround the brain (e.g. subdural or epidural subtypes). ee figure.) Patients may exhibit more than one type of cerebral hemorrhage, which c may appear on the same image.  While small hemorrhages are less morbid than large hemorrhages typically, even a small hemorrhage can lead to death because it is an indicator of another type of serious abnormality (e.g. cerebral aneurysm). </p>
# <p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F603584%2F56162e47358efd77010336a373beb0d2%2Fsubtypes-of-hemorrhage.png?generation=1568657910458946&amp;alt=media" alt="subtypes of hemorrhage"></p>
# <p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F603584%2Fda30220341a8c77a9925868023698b8f%2FMeninges-en.png?generation=1566848157664036&amp;alt=media" alt="intracranial diagram"></p>
# <p>Image credit: By SVG by Mysid, original by SEER Development Team [1], Jmarchn - Vectorized in Inkscape by Mysid, based on work by SEER Development Team, CC BY-SA 3.0, <a href="https://commons.wikimedia.org/w/index.php?curid=10485059" target="_blank">https://commons.wikimedia.org/w/index.php?curid=10485059</a></p></div></div>
# <br/>
# <br/>
# <div>
#     <h3>References : </h3>
#     <ul>
#         <li>https://www.radiologymasterclass.co.uk/tutorials/ct/ct_acute_brain/ct_brain_details</li>
#         <li>https://radiopaedia.org/articles/windowing-ct</li>
#         <li>https://www.slideshare.net/ganesahyogananthem/ct-numbers-window-width-and-window-level</li>
#     </ul>
# </div>

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import pydicom
import os, glob
import warnings
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
warnings.filterwarnings("ignore")


# In[ ]:


colorcodes = ["#00876c"
,"#4c9c85"
,"#78b19f"
,"#a0c6b9"
,"#c8dbd5"
,"#f1f1f1"
,"#f1cfce"
,"#eeadad"
,"#e88b8d"
,"#df676e"
,"#d43d51"]


colorcodes = ["#003f5c"
,"#2f4b7c"
,"#665191"
,"#a05195"
,"#d45087"
,"#f95d6a"
,"#ff7c43"
,"#ffa600"]


# In[ ]:


print(os.listdir("/kaggle/input/rsna-intracranial-hemorrhage-detection/"))
print(os.listdir("/kaggle/input/rsna-intracranial-hemorrhage-detection/stage_1_test_images")[0])
init_dir_path = "/kaggle/input/rsna-intracranial-hemorrhage-detection/"


# In[ ]:


def window_image(img, window_center,window_width, intercept, slope):
    '''
        img : dicom.pixel array
        window center : Window Center/Level center of the CT numbers
        window width : It is the range of the CT numbers that image contains
        intercept : It helps to specify the linear transformation of CT images
        slope : rescale scope is also used to specify the linear transformation of images
    '''
    img = (img*slope +intercept) #linear transformation
    img_min = window_center - window_width//2 #lower grey level
    img_max = window_center + window_width//2 #upper grey level
    img[img<img_min] = img_min
    img[img>img_max] = img_max
    return img 

def get_first_of_dicom_field_as_int(x):
    #get x[0] as in int is x is a 'pydicom.multival.MultiValue', otherwise get int(x)
    if type(x) == pydicom.multival.MultiValue:
        return int(x[0])
    else:
        return int(x)

def get_windowing(data):
    dicom_fields = [data[('0028','1050')].value, #window center
                    data[('0028','1051')].value, #window width
                    data[('0028','1052')].value, #intercept
                    data[('0028','1053')].value] #slope
    return [get_first_of_dicom_field_as_int(x) for x in dicom_fields]


# In[ ]:


data = pd.read_csv(init_dir_path+"stage_1_train.csv")
smpl = pd.read_csv(init_dir_path+"stage_1_sample_submission.csv")


# In[ ]:


data_filter = pd.DataFrame(data.ID.str.split("_").tolist(), columns=["ID","Number","Class"])
data_filter.ID = data_filter.ID +"_"+data_filter.Number
data_filter["Label"] = data.Label

data_filter = data_filter.drop('Number', 1)


# In[ ]:


data_filter.sample(10)


# In[ ]:


data_dict_true = {}
data_dict_false = {}
classes = list(set(data_filter.Class))
for name in classes:
    data_dict_true[name] = np.array((data_filter.Class == name)&(data_filter.Label==1)).astype(int).sum()
    data_dict_false[name] = np.array((data_filter.Class == name)&(data_filter.Label==0)).astype(int).sum()


# In[ ]:


print("Total Images : ",len(list(set(data_filter.ID))))
print("Total Classes : ",len(classes))
print("Total Labels : ",2)


# In[ ]:


print(data_dict_true)
plt.figure(figsize=(10,7))
plt.bar(list(data_dict_true.keys()),list(data_dict_true.values()), label="Label | 1", color=colorcodes[0])
plt.grid()
plt.title("True Class Dstribution")
plt.ylabel("Count")
plt.xlabel("class")
plt.legend()
_=plt.xticks(rotation=75)


# In[ ]:


print(data_dict_true)
print(data_dict_false)
plt.figure(figsize=(10,7))
plt.bar(list(data_dict_false.keys()),list(data_dict_false.values()),color=colorcodes[6], label="Label | 0")
plt.bar(list(data_dict_true.keys()),list(data_dict_true.values()),color=colorcodes[0],bottom=list(data_dict_false.values()), label="Label | 1")
plt.grid()
plt.title("Class Dstribution")
plt.ylabel("Count")
plt.xlabel("class")
plt.legend()
_=plt.xticks(rotation=75)


# In[ ]:


ds = pydicom.dcmread("/kaggle/input/rsna-intracranial-hemorrhage-detection/stage_1_test_images/"+"ID_0000ca2f6.dcm")


# <div>
#     <h3>Linear Transformation of CT windows (Conversion of <i>Bone to Brain</i> window)</h3>
#     <ul>
#         <li>Window Width(WW) is used to see wide range of CT numbers (or simply window width controls contrast) </li>
#         <li>Window level/center(WL) is center of CT numbers (or simply window level/center controls brightness)  </li>
#         <li>Rescale Intercept and Rescale Slope is used for linear transformation <b>(image*slope + intercept)</b></li>
#         <li>Lower grey level = WL - (WW / 2) </li>
#         <li>Upper grey level = WL + (WW / 2) </li>
#     </ul>
# </div>

# In[ ]:


image = ds.pixel_array
window_center , window_width, intercept, slope = get_windowing(ds)
image_windowed = window_image(image, window_center, window_width, intercept, slope)


# <div>
#     <h3>Brain VS Bone CT Windows</h3>
#     <ul>
#         <li>The roll over images show the 'bone window' images at the same level as the 'brain windows'</li>
#         <li>The brain window images provide limited detail of bone structures</li>
#         <li>The bone window images provide no useful detail of brain structure</li>
#         <li>For understanding <i>Hemorrhage Types</i> we need <b>brain windowing</b> images </li>
#         <li>For understanding of <i>fracture or suture</i> <b>bone windowing</b> can be used</li>
#     </ul>
# </div>

# In[ ]:


plt.figure(figsize=(25,15))
plt.subplot(1,2,1)
plt.title("Bone Windowing")
plt.axis("off")
plt.imshow(ds.pixel_array, cmap=plt.cm.bone)
plt.subplot(1,2,2)
plt.title("Brain Windowing")
plt.axis("off")
_=plt.imshow(image_windowed, cmap=plt.cm.bone)


# <div>
#     <h3>Understanding DICOM metadata</h3>
#     <img class="cp-img panning" src="https://api.media.atlassian.com/file/3ee878d2-cf3c-47a4-b2bc-f285ed41f1e6/image?mode=full-fit&amp;client=4489e027-dff6-46fe-8dbe-66960db0d2a0&amp;token=eyJhbGciOiJIUzI1NiJ9.eyJpc3MiOiI0NDg5ZTAyNy1kZmY2LTQ2ZmUtOGRiZS02Njk2MGRiMGQyYTAiLCJhY2Nlc3MiOnsidXJuOmZpbGVzdG9yZTpmaWxlOjNlZTg3OGQyLWNmM2MtNDdhNC1iMmJjLWYyODVlZDQxZjFlNiI6WyJyZWFkIl19LCJleHAiOjE1NjkxNTEyMDYsIm5iZiI6MTU2OTE0Nzg0Nn0.9sa5c3tVwXldcQE6HYedS1Zc1VMvj1GIHo5UYG3JHzk" alt="iod.jpg" style="width: 604.041px; height: 599px;">
#     <p>Reference : https://dcm4che.atlassian.net/wiki/spaces/d2/pages/1835038/A+Very+Basic+DICOM+Introduction</p>
# </div>

# In[ ]:


ds


# <h3>Sample Classes</h3>

# In[ ]:


plt.figure(figsize=(25,10))
iterIdx=1
for cls in classes:
    idx = data_filter.Class[(data_filter.Class == cls)&(data_filter.Label==1)].index[260]
#     print("Class : ", cls, "  ,Index : ",idx,"  ,ID : ", data_filter.ID[idx])
    
    ds_class_wise = pydicom.dcmread(init_dir_path+"stage_1_train_images/"+data_filter.ID[idx]+".dcm")
    image = ds_class_wise.pixel_array
    window_center , window_width, intercept, slope = get_windowing(ds_class_wise)
    image_windowed = window_image(image, window_center, window_width, intercept, slope)    
    plt.subplot(1,6,iterIdx)
    plt.title(cls)
    plt.axis("off")
    plt.imshow(image_windowed, cmap=plt.cm.bone)
#     plt.show()
    iterIdx+=1


# <H3>Difficult to figure out the images with same distribution</H3>

# In[ ]:


plt.figure(figsize=(25,10))
subdural_img_ids = data_filter[(data_filter.Class=="subdural") & (data_filter.Label==1)]['ID'].iloc[0:5]
imgs_list = {}
plt_idx_d=1
for name in subdural_img_ids:
    ds_class_wise = pydicom.dcmread(init_dir_path+"stage_1_train_images/"+name+".dcm")
    image = ds_class_wise.pixel_array
    window_center , window_width, intercept, slope = get_windowing(ds_class_wise)
    image_windowed = window_image(image, window_center, window_width, intercept, slope)
    plt.title("Subdural Image distributions")
    sns.distplot(image_windowed.flatten(), hist=False, label=name)
    imgs_list[name]=image_windowed

plt.figure(figsize=(25,10))
for name in imgs_list:
    plt.subplot(1,5,plt_idx_d)
    plt.title(name)
    plt.imshow(imgs_list[name], cmap=plt.cm.bone)
    plt.axis('off')
    
    plt_idx_d+=1


# <h3>Some images are going out of the box which having high peek at the end needs to figure out how many of them are following this dribution</h3>

# In[ ]:


plt_idx = 1
plt.figure(figsize=(25,15))

        
for cls in classes:
    img_name = data_filter[(data_filter.Class == cls)&(data_filter.Label==1)]['ID'][100:106]
    plt.subplot(3,2,plt_idx)
    plt.title(cls)
    for name in img_name:
        ds_class_wise = pydicom.dcmread(init_dir_path+"stage_1_train_images/"+name+".dcm")
        image = ds_class_wise.pixel_array
        window_center , window_width, intercept, slope = get_windowing(ds_class_wise)
        image_windowed = window_image(image, window_center, window_width, intercept, slope) 
        minmax = MinMaxScaler(feature_range=(0,1))
        out = minmax.fit_transform(image_windowed)
        sns.distplot(out.flatten(), hist=False, label=name)
    plt_idx+=1


# <H3>Conclusion :</H3>
# <ul>
#     <li>Bone windowing cannot be used in model</li>
#     <li>Brain windowing also maintain some of the feature which Bone is having</li>
#     <li>Some images are not having the expected distribution needs to figure out</li>
#     <li>Deep features have to maintain while making model</li>
# </ul>

# In[ ]:




