#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_cell_magic('capture', '', 'from nilearn.image import mean_img\nfrom nilearn.plotting import plot_roi, plot_epi')


# In[ ]:


bold_file = '../input/sub-01_ses-retest_task-fingerfootlips_bold_space-mni152nlin2009casym_preproc.nii/sub-01_ses-retest_task-fingerfootlips_bold_space-MNI152NLin2009cAsym_preproc.nii'
mean_image_nii = mean_img(bold_file)
plot_epi(mean_image_nii, display_mode='x', draw_cross=False, cut_coords=5)
plot_epi(mean_image_nii, display_mode='y', draw_cross=False, cut_coords=5)
plot_epi(mean_image_nii, display_mode='z', draw_cross=False, cut_coords=5)
get_ipython().run_line_magic('pinfo', 'plot_epi')


# In[ ]:


brain_mask_file = '../input/sub-01_ses-retest_task-fingerfootlips_bold_space-mni152nlin2009casym_brainmask.nii/sub-01_ses-retest_task-fingerfootlips_bold_space-MNI152NLin2009cAsym_brainmask.nii'
plot_roi(brain_mask_file, mean_image_nii, display_mode='x', draw_cross=False, cut_coords=5)
plot_roi(brain_mask_file, mean_image_nii, display_mode='y', draw_cross=False, cut_coords=5)
plot_roi(brain_mask_file, mean_image_nii, display_mode='z', draw_cross=False, cut_coords=5)


# In[ ]:




