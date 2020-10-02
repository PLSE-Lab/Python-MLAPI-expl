#!/usr/bin/env python
# coding: utf-8

# <p>
# <b>ADNI notebook</b><br>
# 
# <p>
# The purpose of this notebook is to work with MRI images from the Alzheimer's Disease Neuroimaging Initiative (http://adni.loni.usc.edu/).
# The data compiled here are:
# - 3013 T1-weighted structural MRI scans from 321 subjects (112 of whom are cognitively normal at at least one timepoint, 129 mild cognitive impairment, 150 Alzheimer's disease)
# - I divided the scans into 2109 scans in the training set, 435 in the validation set, 469 in the test set. Note that 1) there are some individuals with one scan in the training set and different scan in the validation set. 2) the exact same set of individuals are in the validation and test sets. This probably isn't the ideal way to allocate individuals to the train/valid/test sets, even though they're different scans (at different timepoints) in the two datasets.
# - each scan has 62 2D axial slices
# - demographic data: clinical diagnosis, age, sex, genotype
# 
# <p>
# <b>
# <font color="red">IMPORTANT NOTE:</font><br>
# Before you can work with this data, you have to accept the terms of the ADNI Data Use Agreement here: http://adni.loni.usc.edu/wp-content/uploads/how_to_apply/ADNI_Data_Use_Agreement.pdf<br>
# Note this term in particular:<br>
# "I will require anyone on my team who utilizes these data, or anyone with whom I share
# these data to comply with this data use agreement."</b>
# 
# <p>
# Additional post-processing:
# - skull stripping to remove non-brain voxels (more detail below)
# - realignment of each scan to standard space coordinates with 2mm isotropic resolution (linear affine transform)
# - exclusion of slices at the extreme ends the z-axis (keep 62/91 slices)
# - padding/trimming images to obtain 96x96 square images
# - splitting of images into train, validation, and test sets; some subjects have scans in multiple sets, some subjects are only represented in one set

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




