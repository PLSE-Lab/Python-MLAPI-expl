#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from json import load
import urllib.request, json 
from pandas.io.json import json_normalize
import seaborn as sns
import pylab as plt
import numpy as np
from matplotlib.lines import Line2D
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


sns.set_context('poster')


# #### Load QC metric data archived from the MRIQC WebAPI

# In[ ]:


df_webapi = pd.read_csv('../input/mriqc-data-cleaning/bold.csv')


# #### Load QC metric data from CHDS pilot scans

# In[ ]:


df_chds = pd.read_csv('../input/chds-pilot-qc/group_bold.tsv', sep='\t')


# #### Filter WebAPI data to include only scans that roughly match the CHDS parameters

# In[ ]:


# Get scans with 1.25 < TR < 1.75
tr_match_idx = (df_webapi['bids_meta.RepetitionTime'] >= 1.25).values * (df_webapi['bids_meta.RepetitionTime'] <= 1.75).values


# In[ ]:


# Select only resting scans
rest_idx = (df_webapi['bids_meta.TaskName'].str.contains('rest')).values


# In[ ]:


# Select only 3T data
threeT_idx = (df_webapi['bids_meta.MagneticFieldStrength'] == 3.0).values


# In[ ]:


# Get scans with 2mm < voxel size < 3.5mm (in-plane resolution)
res_x_idx = (df_webapi['spacing_x'] >= 2.0).values * (df_webapi['spacing_x'] <= 3.5).values
res_y_idx = (df_webapi['spacing_y'] >= 2.0).values * (df_webapi['spacing_y'] <= 3.5).values
res_match_idx = res_y_idx * res_x_idx


# In[ ]:


match_idx = tr_match_idx * threeT_idx * res_match_idx


# ### Temporal Signal-to-Noise (tSNR).
# Higher values are better.

# In[ ]:


fig, ax = plt.subplots(1,2, figsize=(20,10))
sns.stripplot(x='bids_meta.MultibandAccelerationFactor', y='tsnr', data=df_webapi[match_idx], jitter=0.4, alpha=0.7, size=4, ax=ax[0])
for i in df_chds.iterrows():
    if "2p5mm" in i[1].bids_name:
        ax[0].axhline(i[1].tsnr, alpha=0.7, c='C8')
    elif "3p0mm" in i[1].bids_name:
        ax[0].axhline(i[1].tsnr, alpha=0.7, c='C9')
sns.stripplot(x='bids_meta.MultibandAccelerationFactor', y='tsnr', data=df_webapi, jitter=0.4, alpha=0.7, size=4, ax=ax[1])
for i in df_chds.iterrows():
    if "2p5mm" in i[1].bids_name:
        ax[1].axhline(i[1].tsnr, alpha=0.7, c='C8')
    elif "3p0mm" in i[1].bids_name:
        ax[1].axhline(i[1].tsnr, alpha=0.7, c='C9')
                
legend_elements = [Line2D([0], [0], color='C8', lw=4, label='2.5mm, MB3'),
                   Line2D([0], [0], color='C9', lw=4, label='3.0mm, MB2')]
plt.legend(handles=legend_elements, loc=(1.04, 0.5))
ax[0].set_title("Matched for field strength, \nTR, and in-plane resolution.\n")
ax[1].set_title("All EPI scans.\n")
ax[0].set_xlabel("Multiband Accelleration Factor")
ax[1].set_xlabel("Multiband Accelleration Factor")
ax[0].set_ylim(0,100)
ax[1].set_ylim(0,100)
plt.tight_layout()


# ### Foreground-Background Energy Ratio (FBER).
# "Mean energy of image values within the head relative to outside the head.
# 
# Higher values are better."

# In[ ]:


fig, ax = plt.subplots(1,2, figsize=(20,10))
sns.stripplot(x='bids_meta.MultibandAccelerationFactor', y='fber', data=df_webapi[match_idx], jitter=0.4, alpha=0.7, size=4, ax=ax[0])
for i in df_chds.iterrows():
    if "2p5mm" in i[1].bids_name:
        ax[0].axhline(i[1].fber, alpha=0.7, c='C8')
    elif "3p0mm" in i[1].bids_name:
        ax[0].axhline(i[1].fber, alpha=0.7, c='C9')
sns.stripplot(x='bids_meta.MultibandAccelerationFactor', y='fber', data=df_webapi, jitter=0.4, alpha=0.7, size=4, ax=ax[1])
for i in df_chds.iterrows():
    if "2p5mm" in i[1].bids_name:
        ax[1].axhline(i[1].fber, alpha=0.7, c='C8')
    elif "3p0mm" in i[1].bids_name:
        ax[1].axhline(i[1].fber, alpha=0.7, c='C9')
                
legend_elements = [Line2D([0], [0], color='C8', lw=4, label='2.5mm, MB3'),
                   Line2D([0], [0], color='C9', lw=4, label='3.0mm, MB2')]
plt.legend(handles=legend_elements, loc=(1.04, 0.5))
ax[0].set_title("Matched for field strength, \nTR, and in-plane resolution.\n")
ax[1].set_title("All EPI scans.\n")
ax[0].set_xlabel("Multiband Accelleration Factor")
ax[1].set_xlabel("Multiband Accelleration Factor")
ax[0].set_ylim(0,4000)
ax[1].set_ylim(0,15000)
plt.tight_layout()


# ### Entropy Focus Criterion (EFC).
# "Uses the Shannon entropy of voxel intensities as an indication of ghosting and blurring induced by head motion.
# 
# A range of low values is better, with EFC = 0 for all the energy concentrated in one pixel."

# In[ ]:


fig, ax = plt.subplots(1,2, figsize=(20,10))
sns.stripplot(x='bids_meta.MultibandAccelerationFactor', y='efc', data=df_webapi[match_idx], jitter=0.4, alpha=0.7, size=4, ax=ax[0])
for i in df_chds.iterrows():
    if "2p5mm" in i[1].bids_name:
        ax[0].axhline(i[1].efc, alpha=0.7, c='C8')
    elif "3p0mm" in i[1].bids_name:
        ax[0].axhline(i[1].efc, alpha=0.7, c='C9')
sns.stripplot(x='bids_meta.MultibandAccelerationFactor', y='efc', data=df_webapi, jitter=0.4, alpha=0.7, size=4, ax=ax[1])
for i in df_chds.iterrows():
    if "2p5mm" in i[1].bids_name:
        ax[1].axhline(i[1].efc, alpha=0.7, c='C8')
    elif "3p0mm" in i[1].bids_name:
        ax[1].axhline(i[1].efc, alpha=0.7, c='C9')
                
legend_elements = [Line2D([0], [0], color='C8', lw=4, label='2.5mm, MB3'),
                   Line2D([0], [0], color='C9', lw=4, label='3.0mm, MB2')]
plt.legend(handles=legend_elements, loc=(1.04, 0.5))
ax[0].set_title("Matched for field strength, \nTR, and in-plane resolution.\n")
ax[1].set_title("All EPI scans.\n")
ax[0].set_xlabel("Multiband Accelleration Factor")
ax[1].set_xlabel("Multiband Accelleration Factor")
plt.tight_layout()


# ### Ghost to Signal Ratio (GSR).
# Lower values are better.

# In[ ]:


fig, ax = plt.subplots(1,2, figsize=(20,10))
sns.stripplot(x='bids_meta.MultibandAccelerationFactor', y='gsr_y', data=df_webapi[match_idx], jitter=0.4, alpha=0.7, size=4, ax=ax[0])
for i in df_chds.iterrows():
    if "2p5mm" in i[1].bids_name:
        ax[0].axhline(i[1].gsr_y, alpha=0.7, c='C8')
    elif "3p0mm" in i[1].bids_name:
        ax[0].axhline(i[1].gsr_y, alpha=0.7, c='C9')
sns.stripplot(x='bids_meta.MultibandAccelerationFactor', y='gsr_y', data=df_webapi, jitter=0.4, alpha=0.7, size=4, ax=ax[1])
for i in df_chds.iterrows():
    if "2p5mm" in i[1].bids_name:
        ax[1].axhline(i[1].gsr_y, alpha=0.7, c='C8')
    elif "3p0mm" in i[1].bids_name:
        ax[1].axhline(i[1].gsr_y, alpha=0.7, c='C9')
                
legend_elements = [Line2D([0], [0], color='C8', lw=4, label='2.5mm, MB3'),
                   Line2D([0], [0], color='C9', lw=4, label='3.0mm, MB2')]
plt.legend(handles=legend_elements, loc=(1.04, 0.5))
ax[0].set_title("Matched for field strength, \nTR, and in-plane resolution.\n")
ax[1].set_title("All EPI scans.\n")
ax[0].set_xlabel("Multiband Accelleration Factor")
ax[1].set_xlabel("Multiband Accelleration Factor")
plt.tight_layout()


# ### AFNI Quality Index (AQI).
# Taken from `3dTqual`. Lower values are better.

# In[ ]:


fig, ax = plt.subplots(1,2, figsize=(20,10))
sns.stripplot(x='bids_meta.MultibandAccelerationFactor', y='aqi', data=df_webapi[match_idx], jitter=0.4, alpha=0.7, size=4, ax=ax[0])
for i in df_chds.iterrows():
    if "2p5mm" in i[1].bids_name:
        ax[0].axhline(i[1].aqi, alpha=0.7, c='C8')
    elif "3p0mm" in i[1].bids_name:
        ax[0].axhline(i[1].aqi, alpha=0.7, c='C9')
sns.stripplot(x='bids_meta.MultibandAccelerationFactor', y='aqi', data=df_webapi, jitter=0.4, alpha=0.7, size=4, ax=ax[1])
for i in df_chds.iterrows():
    if "2p5mm" in i[1].bids_name:
        ax[1].axhline(i[1].aqi, alpha=0.7, c='C8')
    elif "3p0mm" in i[1].bids_name:
        ax[1].axhline(i[1].aqi, alpha=0.7, c='C9')
                
legend_elements = [Line2D([0], [0], color='C8', lw=4, label='2.5mm, MB3'),
                   Line2D([0], [0], color='C9', lw=4, label='3.0mm, MB2')]
plt.legend(handles=legend_elements, loc=(1.04, 0.5))
ax[0].set_title("Matched for field strength, \nTR, and in-plane resolution.\n")
ax[1].set_title("All EPI scans.\n")
ax[0].set_xlabel("Multiband Accelleration Factor")
ax[1].set_xlabel("Multiband Accelleration Factor")
ax[0].set_ylim(0,0.04)
ax[1].set_ylim(0,0.1)
plt.tight_layout()


# ### AFNI Outlier Ratio (AOR).
# Taken from `3dToutcount`. Lower values are better.

# In[ ]:


fig, ax = plt.subplots(1,2, figsize=(20,10))
sns.stripplot(x='bids_meta.MultibandAccelerationFactor', y='aor', data=df_webapi[match_idx], jitter=0.4, alpha=0.7, size=4, ax=ax[0])
for i in df_chds.iterrows():
    if "2p5mm" in i[1].bids_name:
        ax[0].axhline(i[1].aor, alpha=0.7, c='C8')
    elif "3p0mm" in i[1].bids_name:
        ax[0].axhline(i[1].aor, alpha=0.7, c='C9')
sns.stripplot(x='bids_meta.MultibandAccelerationFactor', y='gsr_y', data=df_webapi, jitter=0.4, alpha=0.7, size=4, ax=ax[1])
for i in df_chds.iterrows():
    if "2p5mm" in i[1].bids_name:
        ax[1].axhline(i[1].aor, alpha=0.7, c='C8')
    elif "3p0mm" in i[1].bids_name:
        ax[1].axhline(i[1].aor, alpha=0.7, c='C9')
                
legend_elements = [Line2D([0], [0], color='C8', lw=4, label='2.5mm, MB3'),
                   Line2D([0], [0], color='C9', lw=4, label='3.0mm, MB2')]
plt.legend(handles=legend_elements, loc=(1.04, 0.5))
ax[0].set_title("Matched for field strength, \nTR, and in-plane resolution.\n")
ax[1].set_title("All EPI scans.\n")
ax[0].set_xlabel("Multiband Accelleration Factor")
ax[1].set_xlabel("Multiband Accelleration Factor")
ax[0].set_ylim(0,0.02)
ax[1].set_ylim(0,0.1)
plt.tight_layout()


# In[ ]:




