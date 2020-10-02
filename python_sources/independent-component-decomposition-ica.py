#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_cell_magic('capture', '', 'from nilearn.decomposition import CanICA\nfrom nilearn.plotting import plot_prob_atlas\nfrom nilearn.image import iter_img\nfrom nilearn.plotting import plot_stat_map, plot_glass_brain, show\nfrom glob import glob')


# In[ ]:


func_filenames = glob("../input/sub-*/*_preproc.nii")
canica = CanICA(n_components=10, smoothing_fwhm=6., threshold=3., 
                verbose=10, random_state=0)
canica.fit(func_filenames)


# In[ ]:


components_img = canica.components_img_


# In[ ]:


for i, cur_img in enumerate(iter_img(components_img)):
    plot_glass_brain(cur_img, title="IC %d" % i)
    show()


# In[ ]:




