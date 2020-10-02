#!/usr/bin/env python
# coding: utf-8

# # Intracranial Hemorrhage Detection
# This challenge is really interesting for me, as I have a background in psychology/neuroscience. I've been working in basic research (brain connectivity, behavior etc.) in healthy participants, with greater goal to understand stroke. So this challenge is really exciting.
# I haven't worked so far with CT, but quite a lot with (f)MRI recordings. I will put together some intuitions, basically just looking at images. 
# 
# I am wrong somewhere, definitely possible - Please comment :)
# 
# ### 11th Version
# I did a little overhaul and tried to put my words into a more concise and understandable manner, as well as putting the code in order.
# 
# # Content
# 1. Looking at the data
# 2. Hounsfield Units (and tranforming DICOMS)
# 3. Windowing (What it means and some code)
# 3. Resampling (Putting the images into same space)
# 4. Segmentation (Extracting the brain)
# 5. Croping (Deleting non-informative rows and columns)
# 6. Pading (Centering the image in the middle, creating images of equal size)
# 7. Normalizing the data
# 9. Conclusions
# 8. Crazy ideas
# 9. Using histograms to classify "any" the presence of any kind of hemorrhage.

# In[ ]:


# Load the required packages:
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Tools
from glob import glob
import os
from tqdm import tqdm_notebook
import pydicom
# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
# Image procesing
from scipy import ndimage
import scipy.misc
from skimage import morphology
from skimage.segmentation import slic
from skimage import measure
from skimage.transform import resize, warp
from skimage import exposure
# Some machine learning as a treat
from sklearn.preprocessing import MaxAbsScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold


# In[ ]:


PATH = '../input/rsna-intracranial-hemorrhage-detection/' # Set up the path.
# Load the stage 1 file
train_csv = pd.read_csv(f'{PATH}stage_1_train.csv')
# Create a path to the train image location:
image_path = os.path.join(PATH, 'stage_1_train_images') + os.sep 
print(image_path)


# In[ ]:


# Check out this kernel: https://www.kaggle.com/currypurin/simple-eda 
# This is a really nice preprocessing of ID and labels :)
train_csv['Image ID'] = train_csv['ID'].apply(lambda x: x.split('_')[1]) 
train_csv['Sub-type'] = train_csv['ID'].apply(lambda x: x.split('_')[2]) 
train_csv = pd.pivot_table(train_csv, index='Image ID', columns='Sub-type')


# In[ ]:


train_csv.head()


# ## 1. Let's have a look at some of the CT images

# In[ ]:


# find hemorrhage images:
hem_img = train_csv.iloc[train_csv['Label']['any'].values == 1].iloc[12:16] # We will look only at a few
plt.figure(figsize=(15,10))
for n, ii in enumerate(hem_img.index):
    plt.subplot(2, 4, n + 1)
    img = pydicom.read_file(image_path + 'ID_' + ii + '.dcm').pixel_array # Read the pixel values
    tmp = hem_img.loc[ii]
    plt.title(tmp.unstack().columns[tmp.unstack().values.ravel() == 1][-1]) # Hacky way to give it a title... 
    plt.imshow(img, cmap='bone')
    plt.subplot(2, 4, n + 5)
    plt.hist(img.ravel())


# ## First Evaluation
# 1. Values in the image have huge ranges. And, very different ranges (i.e. see -3000 in the fourth image). This will provide a challenge later, when we want to normalize the images to a range useful for deep learning, etc.
# 2. In fMRI one would usually do some adjustment of the contrast, just to have a better look at the images (done like in this kernel: https://www.kaggle.com/robinchao/just-visualizing-images). But I learned that values in the CT images, when transformed to Hounsfield units, have an actually meaning.
# 3. We see a lot of other things in the image. Halos could be the head rest, there might be some cushions in there, or even some medical equipment. I think, I might be a good idea to cut these parts out, as there might be some non-medical information in there. 

# #### Hounsfield Units
# Rescaling images to Hounsfield images units, can help us to better understand the histograms of the images. 
# See the wikipedia article:
# https://en.wikipedia.org/wiki/Hounsfield_scale
# 
# Interestingly, there is a table which includes an interpretation for the different values, that might be quite interesting for our analysis. 
# 
# |Substance	|	| HU |
# |-------------------	|---------------------	|------------------	|
# | Subdural hematoma 	| First hours         	| +75 to +100   	|
# |   	|   After 3 days       	|        +65 to +85            	|
# | 	|  After 10-14 days     	|         +35 to +40           	|
# | Other blood       	| Unclotted           	| +13 to +50 	|
# |      	|  Clotted      	|      +50 to +75            	| 
# 
# Another important value to consider is -1024 for air and values greater than 500 for foreign bodies. 

# ## 2. Rescaling to Hounsfield Units
# Fortunately rescaling to Hounsfield units is easy, it's a simple linear transformation and all the important values we need for this are provided in the dicom headers (have a look at this tutorial :https://www.raddq.com/dicom-processing-segmentation-visualization-in-python/). 
# We will just need to multiply the values by the slope (`dicom.RescaleSlope`) and add the intercept (`dicom.RescaleIntercept`).

# In[ ]:


def image_to_hu(image_path, image_id):
    ''' 
    Minimally adapted from https://www.raddq.com/dicom-processing-segmentation-visualization-in-python/
    '''
    dicom = pydicom.read_file(image_path + 'ID_' + image_id + '.dcm')
    image = dicom.pixel_array.astype(np.float64)
         
    # Convert to Hounsfield units (HU)
    intercept = dicom.RescaleIntercept
    slope = dicom.RescaleSlope
    
    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.float64)
        
    image += np.float64(intercept)
    
    image[image < -1024] = -1024 # Setting values smaller than air, to air.
    # Values smaller than -1024, are probably just outside the scanner.
    return image, dicom


# In[ ]:


# find hemorrhage images:
hem_img = train_csv.iloc[train_csv['Label']['any'].values == 1].iloc[12:16] # We will look only at a few
plt.figure(figsize=(15,10))
for n, img_id in enumerate(hem_img.index):
    plt.subplot(2, 4, n + 1)
    img, _ = image_to_hu(image_path, img_id)
    tmp = hem_img.loc[img_id]
    plt.title(tmp.unstack().columns[tmp.unstack().values.ravel() == 1][-1]) # Hacky way to give it a title... 
    plt.imshow(img, cmap='bone')
    plt.subplot(2, 4, n + 5)
    plt.hist(img.ravel())


# ## 3. Windowing
# Although, we can interpret the values in the image now, we cannot really see anything in the image. But now we can look at a certain value range, which we now is interesting. Typically this is done by a process called windowing, where the image is basically clipped to a certain value range. Which is defined as:  
# 
# $WindowLength \pm \frac{WindowWidth}{2}$  
# 
# A range in Hounsfield Units that might be interesting to look at is a width of 80 and a center of 40, described as useful for analyses of brains (source: https://radiopaedia.org/articles/windowing-ct), as well as a width of 130 and a center of 50. 
#   
#     
# Also check out many of the other kernels that give way better explanations, for example this amazing one: https://www.kaggle.com/allunia/rsna-ih-detection-eda-baseline, which I used for the following code.  
# 
# 
# In the long run, testing different windowing values.

# In[ ]:


def image_windowed(image, custom_center=50, custom_width=130, out_side_val=False):
    '''
    Important thing to note in this function: The image migth be changed in place!
    '''
    # see: https://www.kaggle.com/allunia/rsna-ih-detection-eda-baseline
    min_value = custom_center - (custom_width/2)
    max_value = custom_center + (custom_width/2)
    
    # Including another value for values way outside the range, to (hopefully) make segmentation processes easier. 
    out_value_min = custom_center - custom_width
    out_value_max = custom_center + custom_width
    
    if out_side_val:
        image[np.logical_and(image < min_value, image > out_value_min)] = min_value
        image[np.logical_and(image > max_value, image < out_value_max)] = max_value
        image[image < out_value_min] = out_value_min
        image[image > out_value_max] = out_value_max
    
    else:
        image[image < min_value] = min_value
        image[image > max_value] = max_value
    
    return image


# In[ ]:


# find hemorrhage images:
hem_img = train_csv.iloc[train_csv['Label']['any'].values == 1].iloc[12:16] # We will look only at a few
plt.figure(figsize=(15,10))
for n, img_id in enumerate(hem_img.index):
    plt.subplot(2, 4, n + 1)
    img, _ = image_to_hu(image_path, img_id)
    img = image_windowed(img, out_side_val=False)
    tmp = hem_img.loc[img_id]
    plt.title(tmp.unstack().columns[tmp.unstack().values.ravel() == 1][-1]) # Hacky way to give it a title... 
    plt.imshow(img, cmap='bone')
    plt.subplot(2, 4, n + 5)
    plt.hist(img.ravel())


# ## 4. Resampling
# Usually, in neuroimaging we want to make sure that the images are in the same space. The code here https://www.raddq.com/dicom-processing-segmentation-visualization-in-python/ provides a nice code snipped to resample the images. Again this requires some information which is stored in the DICOM files. I adapted the code from the tutorial so that it works for 2D. Normaly these approaches are done in 3D. However, in this challenge we are tasked to classify slices, so using a whole volume of scans might not be possible. The information is stored in `dicom.PixelSpacing`. I decided to go for isotopic pixels of 1 by 1 mm. 
# 
# Note: I am now drawing random images for visualization. I am also windowing the image, so that we can see something in the image.

# In[ ]:


def image_resample(image, dicom_header, new_spacing=[1,1]):
    # Code from https://www.raddq.com/dicom-processing-segmentation-visualization-in-python/
    # Adapted to work for pixels.
    spacing = map(float, dicom_header.PixelSpacing)
    spacing = np.array(list(spacing))
    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)
    
    return image


# In[ ]:


tmp = train_csv.iloc[train_csv['Label']['any'].values == 1].iloc[np.random.randint(
    train_csv.iloc[train_csv['Label']['any'].values == 1].shape[0])].name

hu_img, dicom_header = image_to_hu(image_path, tmp)
resamp_img = image_resample(hu_img, dicom_header)

# Window images, for visualization
hu_img = image_windowed(hu_img)
resamp_img = image_windowed(resamp_img)

plt.figure(figsize=(7.5, 5))
plt.subplot(121)
plt.imshow(hu_img, cmap='bone')
plt.axis('off')
plt.title(f'Orig shape\n{hu_img.shape}')
plt.subplot(122)
plt.imshow(resamp_img, cmap='bone')
plt.title(f'New shape\n{resamp_img.shape}');
plt.axis('off');


# ## 5. Segmentation
# Another pre-processing step, that could help us in the challenge is to extract the brain, and discarding all the other information. I based the segmentation on the code from https://www.raddq.com/dicom-processing-segmentation-visualization-in-python/ . 
# 
# But I learned something by now. In the previous versions I used `slic` from `skimage.segmentation`, however I realized that a simple background seperation using the windowing approach is easier and works more realiably. Further, for background seperation we do not need to use some weird criteria.  
# 
# If you have any suggestions, on how to use `skimage.morphology.dilation` or `skimage.morphology.erosion` more efficiently, I would be glad.

# In[ ]:


def image_background_segmentation(image_path, image_id, WW=40, WL=80, display=False):
    img, dcm_head = image_to_hu(image_path, image_id)
    img = image_resample(img, dcm_head)
    img_out = img.copy()
    # use values outside the window as well, helps with segmentation
    img = image_windowed(img, custom_center=WW, custom_width=WL, out_side_val=True)
    
    # Calculate the outside values by hand (again)
    lB = WW - WL
    uB = WW + WL
    
    # Keep only values inside of the window
    background_seperation = np.logical_and(img > lB, img < uB)
    
    # Get largest connected component:
    # From https://github.com/nilearn/nilearn/blob/master/nilearn/_utils/ndimage.py
    background_seperation = morphology.dilation(background_seperation,  np.ones((5, 5)))
    labels, label_nb = scipy.ndimage.label(background_seperation)
    
    label_count = np.bincount(labels.ravel().astype(np.int))
    # discard the 0 label
    label_count[0] = 0
    mask = labels == label_count.argmax()
    
    # Fill holes in the mask
    mask = morphology.dilation(mask, np.ones((5, 5))) # dilate the mask for less fuzy edges
    mask = scipy.ndimage.morphology.binary_fill_holes(mask)
    mask = morphology.dilation(mask, np.ones((3, 3))) # dilate the mask again

    if display:
        plt.figure(figsize=(15,2.5))
        plt.subplot(141)
        plt.imshow(img, cmap='bone')
        plt.title('Original Images')
        plt.axis('off')

        plt.subplot(142)
        plt.imshow(background_seperation)
        plt.title('Segmentation')
        plt.axis('off')

        plt.subplot(143)
        plt.imshow(mask)
        plt.title('Mask')
        plt.axis('off')

        plt.subplot(144)
        plt.imshow(mask * img, cmap='bone')
        plt.title('Image * Mask')
        plt.suptitle(image_id)
        plt.axis('off')

    return mask * img_out


# In[ ]:


for ii in range(5):
    tmp = train_csv.iloc[train_csv['Label']['any'].values == 1].iloc[np.random.randint(
        train_csv.iloc[train_csv['Label']['any'].values == 1].shape[0])].name
    masked_image = image_background_segmentation(image_path, tmp, display=True)


# ## 6. Cropping
# Cropping the images can now help us to create better or nicer images for a later deep neural network (or what ever)

# In[ ]:


def image_crop(image):
    # Based on this stack overflow post: https://stackoverflow.com/questions/26310873/how-do-i-crop-an-image-on-a-white-background-with-python
    mask = image == 0

    # Find the bounding box of those pixels
    coords = np.array(np.nonzero(~mask))
    top_left = np.min(coords, axis=1)
    bottom_right = np.max(coords, axis=1)

    out = image[top_left[0]:bottom_right[0],
                top_left[1]:bottom_right[1]]
    
    return out


# In[ ]:


plt.figure(figsize=(7.5,5))
for ii in range(3):
    tmp = train_csv.iloc[train_csv['Label']['any'].values == 1].iloc[np.random.randint(
        train_csv.iloc[train_csv['Label']['any'].values == 1].shape[0])].name
    masked_image = image_background_segmentation(image_path, tmp, False)
    masked_image = image_windowed(masked_image)
    cropped_image = image_crop(masked_image)
    plt.subplot(1, 3, ii + 1)
    plt.imshow(cropped_image, cmap='bone')
    plt.title(f'Image Shape:\n{cropped_image.shape}')
    plt.axis('off')


# ## 7. Bring images back to equal spacing
# 
# Pading the images puts the brain in the center and keeps the resampled voxel dimensions. A further thing to test out, might be to resize the images to fill out the whole space. 

# In[ ]:


def image_pad(image, new_height, new_width):
    # based on https://stackoverflow.com/questions/26310873/how-do-i-crop-an-image-on-a-white-background-with-python
    height, width = image.shape

    # make canvas
    im_bg = np.zeros((new_height, new_width))

    # Your work: Compute where it should be
    pad_left = int( (new_width - width) / 2)
    pad_top = int( (new_height - height) / 2)

    im_bg[pad_top:pad_top + height,
          pad_left:pad_left + width] = image

    return im_bg


# In[ ]:


plt.figure(figsize=(7.5, 5))
for ii in range(3):
    tmp = train_csv.iloc[train_csv['Label']['any'].values == 1].iloc[np.random.randint(
        train_csv.iloc[train_csv['Label']['any'].values == 1].shape[0])].name
    masked_image = image_background_segmentation(image_path, tmp, False)
    masked_image = image_windowed(masked_image)
    cropped_image = image_crop(masked_image)
    padded_image = image_pad(cropped_image, 256, 256)
    plt.subplot(1, 3, ii + 1)
    plt.imshow(padded_image, cmap='bone')
    plt.title(f'Image Shape:\n{padded_image.shape}')
    plt.axis('off')


# ## 8. Rescaling (again)
# Finally we want to rescale the images to either be between 0 and 255 or between 0 and 1, so that the networks can work with it, or to use some pretrained networks, that assume certain value ranges. 

# In[ ]:


plt.figure(figsize=(7.5, 5))
for ii in range(3):
    tmp = train_csv.iloc[train_csv['Label']['any'].values == 1].iloc[np.random.randint(
        train_csv.iloc[train_csv['Label']['any'].values == 1].shape[0])].name
    masked_image = image_background_segmentation(image_path, tmp, False)
    masked_image = image_windowed(masked_image)
    cropped_image = image_crop(masked_image)
    padded_image = image_pad(cropped_image, 256, 256)
    padded_image = MaxAbsScaler().fit_transform(padded_image.reshape(-1, 1)).reshape([256, 256])
    plt.subplot(2, 3, ii + 1)
    plt.imshow(padded_image, cmap='bone')
    plt.title(f'Image Shape:\n{padded_image.shape}')
    plt.axis('off')
    plt.subplot(2, 3, ii + 4)
    plt.hist(padded_image.ravel())


# # 9. Conclusion
# We create a preprocessing pipeline in the kernel. Which might be of use to some of you, and we learned a bit about interpreting CT images and what kind of data and maybe what kind of struggles we have to expect. 
# 
# There are some issues I think with the data, which I observed due to the random sampling of images, I am not sure, whether the pipeline introduced them, or they were there before:
# * In some images, the patient's head is way to big and not completely in the field of view
# * Some images have some kind of artifacts - like weird stripes all over, or deletion
# * There might be empty slices in there
# 
# As I haven't done any serious model fitting, what-so-ever with the data, I am not sure whether these weird images have an effect on the general performance. I mean there are quite a lot of images. But it might be worthwhile to be on the look out. 
# 

# # 10. Some ideas
# Just a collection of things, that might be interesting too look at.
# 
# ### Different spaces
# As suggested here https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/discussion/109365#latest-629434 - using different information about the images might be interesting. Who knows? 
# 
# ### Pixel Values
# The Houndfield units say something about the tissue. Histograms of the different values might be used to classify the presence of a hemorrhage.

# ## 11. Classification based on Histograms
# Just as a short run: Classify hemorrhage presence, using histogram values. 

# In[ ]:


hist_bins = np.array([ 0.,  5.,  10., 15., 20., 25., 30., 35., 40., 45., 50., 55., 60.,
       65., 70., 75., 80., 85., 90., 95., 100.])


# In[ ]:


def extract_histogram(image_path, image_id, hist_bins):
    # hu_img, dicom_header = get_pixels_hu(image_path, image_id)
    # windowed_img = set_manual_window(hu_img.copy(), custom_center=40, custom_width=80)
    try:
        masked_image = image_background_segmentation(image_path, image_id, False)
        masked_image = image_windowed(masked_image)
        cropped_image = image_crop(masked_image)

        val, _ = np.histogram(cropped_image.flatten(), bins=hist_bins)
        tmp = val[1:-1] # Remove the first and last bin, as they are probably noisy
        tmp = (tmp - np.mean(tmp)) / np.std(tmp) # z-score
    except:
        tmp=np.zeros(18)
    return tmp 


# In[ ]:


xTrain = np.zeros((6000, 18))
yTrain = np.hstack([np.zeros((3000)), np.ones((3000))])


# In[ ]:


no_hem = train_csv.iloc[train_csv['Label']['any'].values == 0].index.values
hem = train_csv.iloc[train_csv['Label']['any'].values == 1].index.values

for ii, tmp in tqdm_notebook(enumerate(np.random.choice(no_hem, 3000, replace=False))):
    xTrain[ii, :] = extract_histogram(image_path, tmp, hist_bins)

for ii, tmp in tqdm_notebook(enumerate(np.random.choice(hem, 3000, replace=False))):
    xTrain[ii+3000, :] = extract_histogram(image_path, tmp, hist_bins)
    
xTrain[np.isnan(xTrain)] = 0 # somehow, there are some nans in there


# In[ ]:


from scipy.stats import ttest_ind
plt.figure(figsize=(15,10))
for n, vec in enumerate(xTrain.T):
    plt.subplot(3,6, n + 1)
    sns.distplot(vec[yTrain==0])
    sns.distplot(vec[yTrain==1])
    t, p = ttest_ind(vec[yTrain==0], vec[yTrain==1])
    plt.title(f'HU bin {hist_bins[n+1]:2.0f}, t={t:4.2f}')

plt.axis('tight');


# Let's have a look at the different distributions. 
# Looking at the t-values (of course almost everything is significant ~.~), we find the highest difference in distributions around HU 24 - 32, and 44 to 56. These values roughly correspond to older hematomas and blood. Which we would expect, when looking for hemorrhage. Maybe the image histograms can already help us to classify the presence of an hemorrhage?
# 
# ### Table from above
# | Substance | | HU|
# |-------------------	|---------------------	|------------------	|
# | Subdural hematoma 	| First hours         	| +75 to +100   	|
# |   	|   After 3 days       	|        +65 to +85            	|
# | 	|  After 10-14 days     	|         +35 to +40           	|
# | Other blood       	| Unclotted           	| +13 to +50 	|
# |      	|  Clotted      	|      +50 to +75            	| 
# 
# Source: https://en.wikipedia.org/wiki/Hounsfield_scale
# 

# In[ ]:


from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

SKF = StratifiedKFold(5)

logReg_score = cross_val_score(LogisticRegressionCV(cv=5), xTrain, yTrain, cv = SKF)
rfClf_score = cross_val_score(RandomForestClassifier(n_estimators=100), xTrain, yTrain, cv = SKF)

print(f'Logistic Regression Accuracy: {np.mean(logReg_score):4.3f} +/- {np.std(logReg_score):4.3f}')
print(f'Random Forest Accuracy: {np.mean(rfClf_score):4.3f} +/- {np.std(rfClf_score):4.3f}')

