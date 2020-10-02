#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import tempfile
from dipy.segment.mask import median_otsu
from dipy.core.gradients import gradient_table
from dipy.reconst.shm import CsaOdfModel
from dipy.direction import peaks_from_model
from dipy.direction import ProbabilisticDirectionGetter
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel
from dipy.direction import peaks_from_model
from dipy.tracking.local import LocalTracking, ThresholdTissueClassifier
from dipy.tracking import utils
from dipy.reconst import peaks, shm
from dipy.viz import window, actor
from dipy.viz.colormap import line_colors
from dipy.tracking.streamline import Streamlines
from dipy.tracking.eudx import EuDX
from nilearn.plotting import plot_anat, plot_roi, plot_stat_map
from nilearn.image import index_img, iter_img, new_img_like, math_img
from IPython.display import Image
from xvfbwrapper import Xvfb
import nibabel as nb
import pylab as plt
import numpy as np


# In[ ]:


# helper function for plotting woth dipy and VTK on a headless system
def show_image(actor, size=(1000,1000)):
    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_filename = os.path.join(tmp_dir, 'tmp.png')
        with Xvfb() as xvfb:
            ren = window.Renderer()
            ren.add(actor)
            window.record(ren, n_frames=1, out_path=temp_filename, size=size)
            window.clear(ren)
        return Image(filename=temp_filename) 


# This tutorial has been adopted from materials available at http://nipy.org/dipy/examples_index.html
# # Exploring the data
# Lets start by exploring the diffusion weighted data. We will start by loading the NIfTI file.

# In[ ]:


img = nb.load('../input/hardi150.nii/HARDI150.nii')
data = img.get_data()
data.shape


# The file is four dimensional. The fourth dimension corresponds to the different diffusion orientation probed during the scan. In addition to the NIfTI file we will need two more files - one with diffusion weights and one with orientations.

# In[ ]:


gtab = gradient_table('../input/HARDI150.bval', '../input/HARDI150.bvec')
(gtab.bvals == 0).sum()


# As you can see the first 10 volumes are not diffusion weighted and the rest are probing diffusion at different orientations described in the `bvecs` file.

# In[ ]:


gtab.bvecs.shape


# In[ ]:


show_image(actor.point(gtab.gradients, window.colors.blue, point_radius=100))


# Lets have a look at the data and plot the first volume.

# In[ ]:


i = 0
cur_img = index_img(img, i)
plot_anat(cur_img, cut_coords=(0,0,2), draw_cross=False, figure=plt.figure(figsize=(18,4)), cmap='magma', 
              vmin=0, vmax=1600, title="bval = %g, bvec=%s"%(gtab.bvals[i], str(np.round(gtab.bvecs[i,:],2))))

i = 38
cur_img = index_img(img, i)
plot_anat(cur_img, cut_coords=(0,0,2), draw_cross=False, figure=plt.figure(figsize=(18,4)), cmap='magma', 
              vmin=0, vmax=400, title="bval = %g, bvec=%s"%(gtab.bvals[i], str(np.round(gtab.bvecs[i,:],2))))

i = 70
cur_img = index_img(img, i)
plot_anat(cur_img, cut_coords=(0,0,2), draw_cross=False, figure=plt.figure(figsize=(18,4)), cmap='magma', 
              vmin=0, vmax=400, title="bval = %g, bvec=%s"%(gtab.bvals[i], str(np.round(gtab.bvecs[i,:],2))))


# As you can see the diffusion unweighted volume (also called `b0` since the `b` value is zero) is brighter than diffusion weighted volumes. The other volumes have properties that depend on the diffusion orientation.
# 
# **Excercise: plot other diffusion weighted and unweighted volumes.**

# # Fitting a model of diffusion signal

# In[ ]:


csa_model = CsaOdfModel(gtab, sh_order=8)


# In[ ]:


data_small = data[30:50, 65:85, 38:39]
csa_fit_small = csa_model.fit(data_small)


# In[ ]:


csa_odf_small = csa_fit_small.odf(peaks.default_sphere)


# ## Ploting orientation probability distributions

# In[ ]:


fodf_spheres_small = actor.odf_slicer(csa_odf_small, sphere=peaks.default_sphere, scale=0.9, norm=False, colormap='plasma')
show_image(fodf_spheres_small)


# ## Ploting principal orientations

# In[ ]:


csd_peaks_small = peaks_from_model(model=csa_model,
                                   data=data_small,
                                   sphere=peaks.default_sphere,
                                   relative_peak_threshold=.5,
                                   min_separation_angle=25,
                                   parallel=True)

fodf_peaks_small = actor.peak_slicer(csd_peaks_small.peak_dirs, csd_peaks_small.peak_values)
show_image(fodf_peaks_small)


# **Excercise: change the `sh_order` parameter**

# ## Fitting model in the whole brain

# In[ ]:


labels_img = nb.load("../input/aparc-reduced.nii/aparc-reduced.nii")
plot_roi(math_img("(labels == 1) | (labels == 2)", labels=labels_img), index_img(img, 0),figure=plt.figure(figsize=(18,4)),)


# In[ ]:


labels = labels_img.get_data()
white_matter = (labels == 1) | (labels == 2)
csa_model = shm.CsaOdfModel(gtab, 6)
csa_fit = csa_model.fit(data)
csa_peaks = peaks.peaks_from_model(model=csa_model,
                                   data=data,
                                   sphere=peaks.default_sphere,
                                   relative_peak_threshold=.8,
                                   min_separation_angle=45,
                                   mask=white_matter)


# In[ ]:


gfa_img = nb.Nifti1Image(csa_peaks.gfa, img.affine)
plot_stat_map(gfa_img, index_img(img,0),figure=plt.figure(figsize=(18,4)))


# # Reconstructing white matter tracks 

# ## Picking the seed

# In[ ]:


classifier = ThresholdTissueClassifier(csa_peaks.gfa, .25)


# In[ ]:


plot_roi(math_img("x == 2", x=labels_img), index_img(img, 0), figure=plt.figure(figsize=(18,4)))


# In[ ]:


seed_mask = labels == 2
seeds = utils.seeds_from_mask(seed_mask, density=[2, 2, 2], affine=img.affine)


# ## Deterministic tracking

# In[ ]:


# Initialization of LocalTracking. The computation happens in the next step.
streamlines_generator = LocalTracking(csa_peaks, classifier, seeds, img.affine, step_size=.5)

# Generate streamlines object
streamlines = Streamlines(streamlines_generator)

color = line_colors(streamlines)
streamlines_actor = actor.line(streamlines, line_colors(streamlines))
show_image(streamlines_actor)


# ## Probabilistic tracking

# In[ ]:


prob_dg = ProbabilisticDirectionGetter.from_shcoeff(csa_fit.shm_coeff,
                                                    max_angle=30.,
                                                    sphere=peaks.default_sphere)

streamlines_generator = LocalTracking(prob_dg, classifier, seeds, img.affine,
                                      step_size=.5, max_cross=1)

# Generate streamlines object.
streamlines = Streamlines(streamlines_generator)
streamlines_actor = actor.line(streamlines, line_colors(streamlines))
show_image(streamlines_actor)


# **Excercise: repeat both types of tracking - are the results the same each time?**

# # Connectivity analysis

# ## Whole brain white matter tracking

# In[ ]:


seeds = utils.seeds_from_mask(white_matter, density=2)
streamline_generator = EuDX(csa_peaks.peak_values, csa_peaks.peak_indices,
                            odf_vertices=peaks.default_sphere.vertices,
                            a_low=.05, step_sz=.5, seeds=seeds)
affine = streamline_generator.affine

streamlines = Streamlines(streamline_generator, buffer_size=512)

show_image(actor.line(streamlines, line_colors(streamlines)))


# In[ ]:


len(streamlines)


# ## Filtering streamlines

# In[ ]:


cc_slice = labels == 2
cc_streamlines = utils.target(streamlines, cc_slice, affine=affine)
cc_streamlines = Streamlines(cc_streamlines)

other_streamlines = utils.target(streamlines, cc_slice, affine=affine,
                                 include=False)
other_streamlines = Streamlines(other_streamlines)
assert len(other_streamlines) + len(cc_streamlines) == len(streamlines)


# In[ ]:


len(cc_streamlines)


# ## Connectivity matrix

# In[ ]:


plot_roi(labels_img, index_img(img, 0), figure=plt.figure(figsize=(18,4)))


# In[ ]:


np.unique(np.array(labels))


# In[ ]:


plot_roi(math_img("x == 0", x=labels_img), index_img(img, 0), figure=plt.figure(figsize=(18,4)))


# In[ ]:


plot_roi(math_img("x == 1", x=labels_img), index_img(img, 0), figure=plt.figure(figsize=(18,4)))


# In[ ]:


M, grouping = utils.connectivity_matrix(cc_streamlines, labels, affine=affine,
                                        return_mapping=True,
                                        mapping_as_streamlines=True)
M[:3, :] = 0
M[:, :3] = 0
plt.imshow(np.log1p(M), interpolation='nearest')


# **Excercise: estimate connectivity matrix for all streamlines (not just those going through CC)**

# What is the strongest connection?

# In[ ]:


np.argmax(M)
from numpy import unravel_index
new_M = M.copy()
#new_M[11,54] = 0
#new_M[54,11] = 0
unravel_index(new_M.argmax(), new_M.shape)


# ## Bundle density map

# In[ ]:


from nilearn.plotting import plot_stat_map
source_region = 32
target_region = 75
lr_superiorfrontal_track = grouping[source_region, target_region]
shape = labels.shape
dm = utils.density_map(lr_superiorfrontal_track, shape, affine=affine)
dm_img = nb.Nifti1Image(dm.astype("int16"), img.affine)
pl = plot_stat_map(dm_img, index_img(img,0), figure=plt.figure(figsize=(18,4)))
pl.add_contours(math_img("x == %d"%source_region, x=labels_img))
pl.add_contours(math_img("x == %d"%target_region, x=labels_img))


# **Excercise: create density maps for other tracks**

# Looking for more data to play with? Check out https://www.kaggle.com/openneuro/ds001378
