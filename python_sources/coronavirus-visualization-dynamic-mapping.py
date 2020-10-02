#!/usr/bin/env python
# coding: utf-8

# # About Coronavirus

# **Coronaviruses** are a group of viruses that cause diseases in mammals and birds. In humans, the viruses cause respiratory infections which are typically mild including the common cold but rarer forms like SARS and MERS can be lethal. In cows and pigs they may cause diarrhea, while in chickens they can cause an upper respiratory disease. There are no vaccines or antiviral drugs that are approved for prevention or treatment. [More in Wikipedia](https://en.wikipedia.org/wiki/Coronavirus)

# The 2019-nCoV is a contagious coronavirus that hailed from Wuhan, China. This new strain of virus has striked fear in many countries as cities are quarantined and hospitals are overcrowded. This dataset will help us understand how 2019-nCoV is spread aroud the world.

# ![Coronavirus](https://media.nbcnewyork.com/2019/09/Travel-Shuts-Down-at-Ground-Zero-of-Coronavirus-to-Contain-Spread-of-New-Disease.jpg)

# # About data

# The data was opened by [Johns Hopkins University](https://www.jhu.edu/).
# 
# Their dataset is transformed into a format that is easier for kaggle to handle.

# ### Content
# 
# 1. [Import](#1.-Import)
# 2. [Read data](#2.-Read-data)
# 3. [Visualization with Tableau Public](#3.-Visualization-with-Tableau-Public)
# 4. [Research & Visualization with Python](#4.-Research-&-Visualization-with-Python)

# # 1. Import

# In[ ]:


import os


# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt # Data Visulizations


# # 2. Read data

# In[ ]:


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


df = pd.read_csv('../input/2019-coronavirus-dataset-01212020-01262020/2019_nCoV_20200121_20200128.csv')


# In[ ]:


df.shape


# In[ ]:


df.head()


# # 3. Visualization with Tableau Public
# 
# If you have problem with visualization, you can reload page or see the Dashboard in [the link](https://public.tableau.com/views/DashboardCoronavirus/DashboardCoronavirus?:retry=yes&:increment_view_count=no&:embed_code_version=3&:loadOrderID=0&:display_count=y&:origin=viz_share_link).

# In[ ]:


get_ipython().run_cell_magic('html', '', "<div class='tableauPlaceholder' id='viz1580309334676' style='position: relative'>\n    <noscript>\n        <a href='#'>\n            <img alt=' ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Da&#47;DashboardCoronavirus&#47;DashboardCoronavirus&#47;1_rss.png' style='border: none' />\n        </a>\n    </noscript>\n    <object class='tableauViz'  style='display:none;'>\n        <param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> \n        <param name='embed_code_version' value='3' /> \n        <param name='site_root' value='' />\n        <param name='name' value='DashboardCoronavirus&#47;DashboardCoronavirus' />\n        <param name='tabs' value='yes' />\n        <param name='toolbar' value='yes' />\n        <param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Da&#47;DashboardCoronavirus&#47;DashboardCoronavirus&#47;1.png' /> \n        <param name='animate_transition' value='yes' />\n        <param name='display_static_image' value='yes' />\n        <param name='display_spinner' value='yes' />\n        <param name='display_overlay' value='yes' />\n        <param name='display_count' value='yes' />\n    </object>\n</div>                \n<script type='text/javascript'>                    \n    var divElement = document.getElementById('viz1580309334676');                    \n    var vizElement = divElement.getElementsByTagName('object')[0];                    \n    if ( divElement.offsetWidth > 800 ) { vizElement.style.minWidth='1200px';vizElement.style.maxWidth='100%';vizElement.style.minHeight='850px';vizElement.style.maxHeight=(divElement.offsetWidth*0.75)+'px';} else if ( divElement.offsetWidth > 500 ) { vizElement.style.minWidth='1200px';vizElement.style.maxWidth='100%';vizElement.style.minHeight='850px';vizElement.style.maxHeight=(divElement.offsetWidth*0.75)+'px';} else { vizElement.style.width='100%';vizElement.style.minHeight='1350px';vizElement.style.maxHeight=(divElement.offsetWidth*1.77)+'px';}                     \n    var scriptElement = document.createElement('script');                    \n    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    \n    vizElement.parentNode.insertBefore(scriptElement, vizElement);                \n</script>")


# # 4. Research & Visualization with Python

# In[ ]:


# Country and State wise Explorations. Sorted

data = pd.DataFrame(df[df['Last Update'] == '1/31/2020 19:00'])
data= pd.DataFrame(df.groupby(['Country/Region'])['Confirmed','Suspected','Recovered','Death'].agg('sum')).reset_index()
data = data.sort_values(by=['Confirmed'], ascending=False, na_position='first')
data.head(19)


# In[ ]:


# Province of China. Sorted

china = pd.DataFrame(df[df['Last Update'] == '1/31/2020 19:00'])
china= df[df['Country/Region'] == 'Mainland China']
china_data= pd.DataFrame(china.groupby(['Province/State'])['Confirmed','Suspected','Recovered','Death'].agg('sum')).reset_index()
china_data = china_data.sort_values(by=['Confirmed'], ascending=False, na_position='first')
china_data.head(35)


# In[ ]:




