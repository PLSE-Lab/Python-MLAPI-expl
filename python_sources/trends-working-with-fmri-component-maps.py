#!/usr/bin/env python
# coding: utf-8

# ### Making use of the fMRI images in .mat files
# 
# Most of the notebooks and discussion posted on this comp focuses on the table data.  The fMRI maps provided are the spatial weights for each individual subject for the spatially constrained group ICA derived independent components.  The associated time series for each component is what gets correlated with the time series of other components to generate the fnc data table.  So, it might be of interest to use these maps or attempt to derive variables from them.  This notebook provides an example of how one could do this.
# 
# Most of the first few blocks of code are lifted from:
# https://www.kaggle.com/bbradt/loading-and-exploring-spatial-maps
# 
# Thanks to [@bbradt](https://www.kaggle.com/bbradt) for showing how to load and transform mats to nifti images.
# 

# In[ ]:


"""
From: https://www.kaggle.com/bbradt/loading-and-exploring-spatial-maps
"""
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from scipy import stats
from scipy.ndimage.morphology import generate_binary_structure
from scipy.ndimage.measurements import label
import numpy as np # linear algebra
import nilearn as nl
import nilearn.plotting as nlplt
import nibabel as nib
import h5py
import matplotlib.pyplot as plt
from nilearn.masking import apply_mask
from nilearn.masking import unmask

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
mask_filename = '../input/trends-assessment-prediction/fMRI_mask.nii'
#subject_filename = '../input/trends-assessment-prediction/fMRI_train/10004.mat'
subject_filename = '../input/trends-assessment-prediction/fMRI_test/10030.mat'
# smri_filename = 'ch2better.nii'
mask_niimg = nl.image.load_img(mask_filename)

def load_subject(filename, mask_niimg):
    """
    Load a subject saved in .mat format with
        the version 7.3 flag. Return the subject
        niimg, using a mask niimg as a template
        for nifti headers.
        
    Args:
        filename    <str>            the .mat filename for the subject data
        mask_niimg  niimg object     the mask niimg object used for nifti headers
    """
    subject_data = None
    with h5py.File(subject_filename, 'r') as f:
        subject_data = f['SM_feature'][()]
    # It's necessary to reorient the axes, since h5py flips axis order
    subject_data = np.moveaxis(subject_data, [0,1,2,3], [3,2,1,0])
    subject_niimg = nl.image.new_img_like(mask_niimg, subject_data, affine=mask_niimg.affine, copy_header=True)
    return subject_niimg


# In[ ]:


#example...

subject_niimg = load_subject(subject_filename, mask_niimg)
print("Image shape is %s" % (str(subject_niimg.shape)))
num_components = subject_niimg.shape[-1]
print("Detected {num_components} spatial maps".format(num_components=num_components))


# In[ ]:


type(subject_niimg)


# So now we have one subject's image as a Nifti1Image, which allows us to use other functions from nilearn and nibabel libraries designed to work with this image file type.  Note that the image is X, Y, Z, N where N is the number of fMRI independent components.  Since the group ICA was spatially constrained the idea is that component #1 for each subject matches the same template.  Based on the component numbers (in the label file) there were probably about 100 ICs generated.  Some of these were probably movement-related or noise components, discarded prior to generating the fnc table data.

# In[ ]:


maskData = apply_mask(subject_niimg, mask_niimg)
type(maskData)


# This call to apply_mask turns the 4D image into a 2D numpy array!

# In[ ]:


print('Total number of voxels:' + str(53*63*52))
print('Number of voxels in standardized brain mask:' + str(maskData.shape[1]))


# In[ ]:


df_train_scores = pd.read_csv('../input/trends-assessment-prediction/train_scores.csv')
df_train_scores['age_bins'] = pd.cut(x=df_train_scores['age'], bins=[10, 20, 30, 40, 50, 60, 70, 80, 90], 
                                     labels=['teens','twenties','thirties','forties','fifties','sixties','seventies','eighties'])
skf = StratifiedKFold(n_splits=25, shuffle=True, random_state=5272020)
for train_index, test_index in skf.split(df_train_scores, df_train_scores['age_bins']):
     print("TRAIN length:", len(train_index), "TEST length:", len(test_index))


# We can't load all the images in a notebook, but you can get a sense of the average effect with a few hundred subjects.  Let's look at component #5

# In[ ]:


#this is just a test, so let's try component 5 (ADN) for fun:
myComp = 5
#initialize np array for the test subjects:
sMat = np.zeros(shape=(len(test_index), maskData.shape[1]))


# In[ ]:


i = 0
for id in test_index:
    subject_filename = '../input/trends-assessment-prediction/fMRI_train/' + str(df_train_scores['Id'].iloc[id]) + '.mat'
    subject_niimg = load_subject(subject_filename, mask_niimg)
    maskData = apply_mask(subject_niimg, mask_niimg)
    sMat[i,]= maskData[myComp,]
    i += 1


# By applying the mask, we get data for each subject for this component into a 2D array, and now we can run a one-sample t-test to find the voxels in the maps that are different from 0.  We can then use unmask to go back to 3D space and use nilearn functions to plot the test statistics on a brain image underlay.

# In[ ]:


t = stats.ttest_1samp(sMat, 0, axis=0)
tmap = unmask(t.statistic, mask_niimg).get_fdata()

t_img = nib.Nifti1Image(tmap, header=mask_niimg.header, affine=mask_niimg.affine)
nlplt.plot_stat_map(t_img, title="IC %d" % myComp, threshold=20.2, colorbar=True)


# So what can you do with this?  Well, there are clearly two regions that define this network.  We can define these as two, distinct regions of interest (ROIs) and calculate summary statistics that could serve as additional features.

# In[ ]:


#generate the binary structure:
struct = generate_binary_structure(3,3)
labeled_array, num_features = label(tmap>20, struct)
    
#label the clusters
L_img = nib.Nifti1Image(labeled_array, header=mask_niimg.header, affine=mask_niimg.affine)

nlplt.plot_roi(L_img, colorbar=True, cmap='Paired')

num_features


# In[ ]:


affine = mask_niimg.affine
label_img = nib.Nifti1Image(labeled_array, affine)
clustMask = apply_mask(label_img, mask_niimg)


# In[ ]:


clustMask.shape


# In[ ]:


RightHemMean = np.mean(sMat[:,clustMask==1], axis=1)
LeftHemMean = np.mean(sMat[:,clustMask==2], axis=1)


# In[ ]:


r = stats.linregress(RightHemMean, LeftHemMean)

print(r)


# In[ ]:


plt.plot(RightHemMean, LeftHemMean, 'o', label='original data')
plt.plot(RightHemMean, r.intercept + r.slope*RightHemMean, 'r', label='fitted line')
plt.legend()
plt.show()


# So, left and right hemisphere ROI means are correlated, which makes some sense given they are part of the same IC, but what about our targets?

# In[ ]:


X =  np.zeros(shape=len(test_index))
i = 0
for id in test_index:
    X[i]= df_train_scores['age'].iloc[id]
    i += 1


# In[ ]:


r = stats.linregress(RightHemMean, X)

print(r)
plt.plot(RightHemMean, X, 'o', label='original data')
plt.plot(RightHemMean, r.intercept + r.slope*RightHemMean, 'r', label='fitted line')
plt.legend()
plt.show()


# OK, that's not going to get you a gold, but I thought it was worth sharing a few steps to get into the .mat files, which are either underutilized or the secret weapon in this comp.  Happy to receive feedback about issues with my code and fix problems!!

# In[ ]:




