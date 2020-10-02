#!/usr/bin/env python
# coding: utf-8

# # Introduction About the Lower back pain
# ![](https://media3.giphy.com/media/xT9DPj0h2lVvVWQo24/giphy.gif)
# 
# ### What is Lower Back Pain ?
# * Low back pain is a universal human experience -- almost everyone has it at some point. The lower back, which starts below the ribcage, is called the lumbar region. Pain here can be intense and is one of the top causes of missed work. Fortunately, low back pain often gets better on its own. When it doesn't, there are effective treatments.
# 
# ### Common symptoms experienced by people with low back pain
# * Low back pain is defined as pain and discomfort below the the costal margin and above the inferior gluteal folds, with or without referred leg pain. 
# * It may be experienced as **aching, burning, stabbing, sharp or dull, well-defined, or vague with intensity ranging from mild to severe.**  
# * The pain may begin suddenly or develop gradually. Non-specific low back pain is defined as low back pain not attributed to recognisable, known **specific pathology (e.g. infection, tumour, osteoporosis, ankylosing spondylitis, fracture, inflammatory process, radicular syndrome or cauda equina syndrome).** 
# 
# ### Causes of Lower Back Pain
# 
# ![](https://www.consumerhealthdigest.com/wp-content/uploads/2017/06/causes-backpain-info.jpg)
# 
# ### Low back pain subtypes
# * **Chronic back pain (CLBP)** is defined as low back pain persisting for longer than 7-12 weeks, or after the period of healing or recurring back pain that intermittently affects an individual over a long period of time.
# * **Acute back pain** is defined as low back pain lasting for less than 12 weeks.
# * **Subacute pain** is defined low back pain lasting between six weeks and three months.
# 
# ### Visual Information
# ![](http://www.aspetar.com/journal/upload/images/2015111212013.png)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from IPython.display import IFrame,YouTubeVideo
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
# Any results you write to the current directory are saved as output.


# ## Lower Back Pain Explained

# In[ ]:


YouTubeVideo("zfs9oyA3pKg",width=800, height=480)


# ## 1 . What is Pelvic incidence?(Column-1)
# 
# *  Pelvic incidence is defined as the angle between the line perpendicular to the sacral plate at its midpoint and the line connecting this point to the femoral heads axis.
# 
# ![](https://www.hindawi.com/journals/ari/2014/594650.fig.002.jpg)

# In[ ]:


IFrame("https://public.tableau.com/shared/94CHXD9P2?:embed=y&:showVizHome=no", width=1100, height=800)
# https://public.tableau.com/views/SpeakerStatistics/SpeakerStatistics?:embed=y&:showVizHome=no'
# https://public.tableau.com/shared/94CHXD9P2?:display_count=yes


# ## 2 . What is Pelvic tilt ?(Column-2)
# * Pelvic tilt is the orientation of the pelvis in respect to the thighbones and the rest of the body. The pelvis can tilt towards the front, back, or either side of the body.
# 
# ![](https://upload.wikimedia.org/wikipedia/commons/thumb/a/a0/PostureFoundationGarments04fig1.png/330px-PostureFoundationGarments04fig1.png)

# In[ ]:


IFrame("https://public.tableau.com/views/LBP-PelvicTilt/LBP2?:embed=y&:showVizHome=no", width=1100, height=800)
# https://public.tableau.com/views/LBP-PelvicTilt/LBP2?:embed=y&:display_count=yes&publish=yes


# ## What is lumbar lordosis angle?(Column-3)
# * The mean values of lumbar lordotic angle (LLA), lumbosacral angle (LSA) and sacral inclination angle (SIA) were 33.2 +/- 12.1 degrees, 11.4 +/- 4.7 degrees and 26.4 +/- 10 degrees, respectively. ... The normal value of LLA can be defined as 20-45 degrees with a range of 1 SD.
# 
# ![](https://www.researchgate.net/profile/Chakib_Ayoub/publication/51710697/figure/fig1/AS:202877927727105@1425381275104/Measurement-of-lumbar-lordotic-angle-a-Using-Cobbs-method-tangent-lines-are-drawn.png)

# In[ ]:


IFrame("https://public.tableau.com/views/LBP-lumbarlordosisangle/LPB3?:embed=y&:showVizHome=no", width=1100, height=800)
# https://public.tableau.com/views/LBP-lumbarlordosisangle/LPB3?:embed=y&:display_count=yes&publish=yes


# ## What is sacral_slope? (Column-4)
# * The sacral slope ( SS ) is measured between the tangent line to the superior endplate of S1 and the horizontal plane ( a ). Sacral inclination ( SI ) is defined as the angle between the vertical plane and the tangential line to the sacral dorsum ( b ).
# ![](https://www.researchgate.net/profile/Christine_Tardieu/publication/263776876/figure/fig2/AS:203169045979142@1425450683397/A-geometric-construction-by-complementary-angles-reveals-that-the-morphological-parameter.png)

# In[ ]:


IFrame("https://public.tableau.com/views/LBP-sacral_slope/LPB4?:embed=y&:showVizHome=no", width=1100, height=800)
# https://public.tableau.com/views/LBP-sacral_slope/LPB4?:embed=y&:display_count=yes&publish=yes


# ## What is pelvic_radius? (Column-5)
# 
# * HA bisector of line joining the centres of both femoral heads, pelvic radius line from HA extended through posterior S1 end plate, PRnn (where nn vertebral name) angle between the pelvic radius and the superior end plate of the named vertebrae, i.e. PRT12 angle between pelvic radius and the T12 vertebrae.
# 
# ![](https://openi.nlm.nih.gov/imgs/512/336/3348686/PMC3348686_or-2012-1-e11-g002.png)

# In[ ]:


IFrame("https://public.tableau.com/views/LBP-pelvic_radius/LPB5?:embed=y&:showVizHome=no", width=1100, height=800)
# https://public.tableau.com/views/LBP-pelvic_radius/LPB5?:embed=y&:display_count=yes&publish=yes


# In[ ]:




