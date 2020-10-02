#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# > **Loading train and test dataset....****

# In[ ]:


train=pd.read_csv("../input/train.csv")
test=pd.read_csv("../input/test.csv")


# > **Taking a look at train and test datasets using info() and desribe()****

# In[ ]:


train.info()


# > ****No feature has any missing value in train  dataset****

# In[ ]:


train.describe()


# In[ ]:


train.head()


# In[ ]:


test.info()


# > **No missing value in test dataset****

# In[ ]:


test.describe()


# In[ ]:


test.head()


# In[ ]:


train.shape


# In[ ]:


test.shape


# **Checking Distribution of target class****

# In[ ]:


train['Cover_Type'].value_counts()


# No class imbalance present in target...

# **CORRELATION MATRIX********

# In[ ]:


get_ipython().run_cell_magic('html', '', "<div class='tableauPlaceholder' id='viz1536511440051' style='position: relative'><noscript><a href='#'><img alt='Sheet 4 ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;CQ&#47;CQG4SX4GY&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='path' value='shared&#47;CQG4SX4GY' /> <param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;CQ&#47;CQG4SX4GY&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='filter' value='publish=yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1536511440051');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>")


# > **Visualizing Relation of various features with TARGET****

# In[ ]:


get_ipython().run_cell_magic('html', '', "<div class='tableauPlaceholder' id='viz1536593745587' style='position: relative'><noscript><a href='#'><img alt='Elevation and CoverType ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;fo&#47;forest1&#47;Sheet1&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='forest1&#47;Sheet1' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;fo&#47;forest1&#47;Sheet1&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1536593745587');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>")


# In[ ]:


get_ipython().run_cell_magic('html', '', "<div class='tableauPlaceholder' id='viz1536593802868' style='position: relative'><noscript><a href='#'><img alt='Aspect And CoverType ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;fo&#47;forest1&#47;Sheet3&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='forest1&#47;Sheet3' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;fo&#47;forest1&#47;Sheet3&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1536593802868');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>")


# In[ ]:


get_ipython().run_cell_magic('html', '', "<div class='tableauPlaceholder' id='viz1536593831870' style='position: relative'><noscript><a href='#'><img alt='Hillshade and CoverType ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;fo&#47;forest1&#47;Sheet4&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='forest1&#47;Sheet4' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;fo&#47;forest1&#47;Sheet4&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1536593831870');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>")


# In[ ]:


get_ipython().run_cell_magic('html', '', "<div class='tableauPlaceholder' id='viz1536594078045' style='position: relative'><noscript><a href='#'><img alt='WildernessArea ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;fo&#47;forest1&#47;Sheet5&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='forest1&#47;Sheet5' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;fo&#47;forest1&#47;Sheet5&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1536594078045');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>")


# In[ ]:


get_ipython().run_cell_magic('html', '', "<div class='tableauPlaceholder' id='viz1536594109807' style='position: relative'><noscript><a href='#'><img alt='Horizaontal distances  ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;fo&#47;forest1&#47;Sheet6&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='forest1&#47;Sheet6' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;fo&#47;forest1&#47;Sheet6&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1536594109807');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>")


# In[ ]:


get_ipython().run_cell_magic('html', '', "<div class='tableauPlaceholder' id='viz1536594142347' style='position: relative'><noscript><a href='#'><img alt='Slope ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;fo&#47;forest1&#47;Sheet7&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='forest1&#47;Sheet7' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;fo&#47;forest1&#47;Sheet7&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1536594142347');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>")


# In[ ]:


get_ipython().run_cell_magic('html', '', "<div class='tableauPlaceholder' id='viz1536594177024' style='position: relative'><noscript><a href='#'><img alt='VerticalDistance ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;fo&#47;forest1&#47;Sheet8&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='forest1&#47;Sheet8' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;fo&#47;forest1&#47;Sheet8&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1536594177024');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>")


# In[ ]:


get_ipython().run_cell_magic('html', '', "<div class='tableauPlaceholder' id='viz1536597186908' style='position: relative'><noscript><a href='#'><img alt='Different Soils  ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;fo&#47;forest1&#47;Sheet9&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='forest1&#47;Sheet9' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;fo&#47;forest1&#47;Sheet9&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1536597186908');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>")


# > **Bit of Feature Engineering....**

# Creating wild12,wild24,wild13,wild14,wild,23,wild34 using various WildernessArea

# In[ ]:


train['wild12']=train['Wilderness_Area1']+train['Wilderness_Area2']
train['wild13']=train['Wilderness_Area1']+train['Wilderness_Area3']
train['wild14']=train['Wilderness_Area1']+train['Wilderness_Area4']
train['wild23']=train['Wilderness_Area2']+train['Wilderness_Area3']
train['wild24']=train['Wilderness_Area2']+train['Wilderness_Area4']
train['wild34']=train['Wilderness_Area3']+train['Wilderness_Area4']
test['wild12']=test['Wilderness_Area1']+test['Wilderness_Area2']
test['wild13']=test['Wilderness_Area1']+test['Wilderness_Area3']
test['wild14']=test['Wilderness_Area1']+test['Wilderness_Area4']
test['wild23']=test['Wilderness_Area2']+test['Wilderness_Area3']
test['wild24']=test['Wilderness_Area2']+test['Wilderness_Area4']
test['wild34']=test['Wilderness_Area3']+test['Wilderness_Area4']


# In[ ]:


train['slope_hydrology']=(train['Horizontal_Distance_To_Hydrology']**2+train['Vertical_Distance_To_Hydrology']**2)**0.5
test['slope_hydrology']=(test['Horizontal_Distance_To_Hydrology']**2+test['Vertical_Distance_To_Hydrology']**2)**0.5
train['mean_distance']=train['Horizontal_Distance_To_Fire_Points']+train['Horizontal_Distance_To_Hydrology']+train['Horizontal_Distance_To_Roadways']
test['mean_distance']=test['Horizontal_Distance_To_Fire_Points']+test['Horizontal_Distance_To_Hydrology']+train['Horizontal_Distance_To_Roadways']
train['FR']=train['Horizontal_Distance_To_Fire_Points']+train['Horizontal_Distance_To_Roadways']
test['FR']=test['Horizontal_Distance_To_Fire_Points']+test['Horizontal_Distance_To_Roadways']
train['FH']=train['Horizontal_Distance_To_Fire_Points']+train['Horizontal_Distance_To_Hydrology']
test['FH']=test['Horizontal_Distance_To_Fire_Points']+test['Horizontal_Distance_To_Hydrology']
train['HR']=train['Horizontal_Distance_To_Hydrology']+train['Horizontal_Distance_To_Roadways']
test['HR']=train['Horizontal_Distance_To_Hydrology']+train['Horizontal_Distance_To_Roadways']


# In[ ]:


train['Diff_Hill_9-noon']=train['Hillshade_9am']-train['Hillshade_Noon']
train['Diff_Hill_noon-3']=train['Hillshade_Noon']-train['Hillshade_3pm']
train['Diff_Hill_9-3']=train['Hillshade_9am']-train['Hillshade_3pm']
test['Diff_Hill_9-noon']=test['Hillshade_9am']-test['Hillshade_Noon']
test['Diff_Hill_noon-3']=test['Hillshade_Noon']-test['Hillshade_3pm']
test['Diff_Hill_9-3']=test['Hillshade_9am']-test['Hillshade_3pm']


# In[ ]:


#train['soil1-10']=train['Soil_Type1']+train['Soil_Type2']+train['Soil_Type3']+train['Soil_Type4']+train['Soil_Type5']+train['Soil_Type6']+train['Soil_Type7']+train['Soil_Type8']+train['Soil_Type9']+train['Soil_Type10']
#train['soil11-20']=train['Soil_Type11']+train['Soil_Type12']+train['Soil_Type13']+train['Soil_Type14']+train['Soil_Type15']+train['Soil_Type16']+train['Soil_Type17']+train['Soil_Type18']+train['Soil_Type19']+train['Soil_Type20']
#train['soil21-30']=train['Soil_Type21']+train['Soil_Type22']+train['Soil_Type23']+train['Soil_Type24']+train['Soil_Type25']+train['Soil_Type26']+train['Soil_Type27']+train['Soil_Type28']+train['Soil_Type29']+train['Soil_Type30']
#train['soil31-40']=train['Soil_Type31']+train['Soil_Type32']+train['Soil_Type33']+train['Soil_Type34']+train['Soil_Type35']+train['Soil_Type36']+train['Soil_Type37']+train['Soil_Type38']+train['Soil_Type39']+train['Soil_Type40']
#test['soil1-10']=test['Soil_Type1']+test['Soil_Type2']+test['Soil_Type3']+test['Soil_Type4']+test['Soil_Type5']+test['Soil_Type6']+test['Soil_Type7']+test['Soil_Type8']+test['Soil_Type9']+test['Soil_Type10']
#test['soil11-20']=test['Soil_Type11']+test['Soil_Type12']+test['Soil_Type13']+test['Soil_Type14']+test['Soil_Type15']+test['Soil_Type16']+test['Soil_Type17']+test['Soil_Type18']+test['Soil_Type19']+test['Soil_Type20']
#test['soil21-30']=test['Soil_Type21']+test['Soil_Type22']+test['Soil_Type23']+test['Soil_Type24']+test['Soil_Type25']+test['Soil_Type26']+test['Soil_Type27']+test['Soil_Type28']+test['Soil_Type29']+test['Soil_Type30']
#test['soil31-40']=test['Soil_Type31']+test['Soil_Type32']+test['Soil_Type33']+test['Soil_Type34']+test['Soil_Type35']+test['Soil_Type36']+test['Soil_Type37']+test['Soil_Type38']+test['Soil_Type39']+test['Soil_Type40']


# Visualizing the new Features and There relation

# In[ ]:


get_ipython().run_cell_magic('html', '', "<div class='tableauPlaceholder' id='viz1536949151064' style='position: relative'><noscript><a href='#'><img alt='Hydrology_Slope ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Fe&#47;FeatureEngineering&#47;Sheet1&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='FeatureEngineering&#47;Sheet1' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Fe&#47;FeatureEngineering&#47;Sheet1&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1536949151064');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>")


# In[ ]:


get_ipython().run_cell_magic('html', '', "<div class='tableauPlaceholder' id='viz1536949173517' style='position: relative'><noscript><a href='#'><img alt='Mean_Horizontal_Distance ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Fe&#47;FeatureEngineering&#47;Sheet2&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='FeatureEngineering&#47;Sheet2' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Fe&#47;FeatureEngineering&#47;Sheet2&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1536949173517');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>")


# In[ ]:


get_ipython().run_cell_magic('html', '', "<div class='tableauPlaceholder' id='viz1536949195772' style='position: relative'><noscript><a href='#'><img alt='Firepoint+Hydrology distance ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Fe&#47;FeatureEngineering&#47;Sheet3&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='FeatureEngineering&#47;Sheet3' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Fe&#47;FeatureEngineering&#47;Sheet3&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1536949195772');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>")


# In[ ]:


get_ipython().run_cell_magic('html', '', "<div class='tableauPlaceholder' id='viz1536949216700' style='position: relative'><noscript><a href='#'><img alt='Hydrology+Roadways distance ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Fe&#47;FeatureEngineering&#47;Sheet4&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='FeatureEngineering&#47;Sheet4' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Fe&#47;FeatureEngineering&#47;Sheet4&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1536949216700');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>")


# In[ ]:


get_ipython().run_cell_magic('html', '', "<div class='tableauPlaceholder' id='viz1536949238972' style='position: relative'><noscript><a href='#'><img alt='Fireplace+Roadways distance ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Fe&#47;FeatureEngineering&#47;Sheet5&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='FeatureEngineering&#47;Sheet5' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Fe&#47;FeatureEngineering&#47;Sheet5&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1536949238972');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>")


# In[ ]:


get_ipython().run_cell_magic('html', '', "<div class='tableauPlaceholder' id='viz1536949256302' style='position: relative'><noscript><a href='#'><img alt='hillshade_diff 9am-3pm ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Fe&#47;FeatureEngineering&#47;Sheet6&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='FeatureEngineering&#47;Sheet6' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Fe&#47;FeatureEngineering&#47;Sheet6&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1536949256302');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>")


# In[ ]:


get_ipython().run_cell_magic('html', '', "<div class='tableauPlaceholder' id='viz1536949283631' style='position: relative'><noscript><a href='#'><img alt='hillshade_diff_noon-3pm ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Fe&#47;FeatureEngineering&#47;Sheet7&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='FeatureEngineering&#47;Sheet7' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Fe&#47;FeatureEngineering&#47;Sheet7&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1536949283631');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>")


# In[ ]:


get_ipython().run_cell_magic('html', '', "<div class='tableauPlaceholder' id='viz1536949329291' style='position: relative'><noscript><a href='#'><img alt='hillshade_diff_9am-noon ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Fe&#47;FeatureEngineering&#47;Sheet8&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='FeatureEngineering&#47;Sheet8' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Fe&#47;FeatureEngineering&#47;Sheet8&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1536949329291');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>")


# In[ ]:


get_ipython().run_cell_magic('html', '', "<div class='tableauPlaceholder' id='viz1536599004792' style='position: relative'><noscript><a href='#'><img alt='Wild12(Wilderness 1+2) And Wild34(Wilderness 3+4) ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;fo&#47;forest1&#47;Sheet9&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='forest1&#47;Sheet9' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;fo&#47;forest1&#47;Sheet9&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1536599004792');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>")


# In[ ]:


get_ipython().run_cell_magic('html', '', "<div class='tableauPlaceholder' id='viz1536599039365' style='position: relative'><noscript><a href='#'><img alt='wild14 and wild23 ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;fo&#47;forest1&#47;Sheet10&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='forest1&#47;Sheet10' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;fo&#47;forest1&#47;Sheet10&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1536599039365');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>")


# In[ ]:


get_ipython().run_cell_magic('html', '', "<div class='tableauPlaceholder' id='viz1536599069635' style='position: relative'><noscript><a href='#'><img alt='wild24 and wild13 ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;fo&#47;forest1&#47;Sheet11&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='forest1&#47;Sheet11' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;fo&#47;forest1&#47;Sheet11&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1536599069635');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>")


# Certain Relations Do Exists.....

# Taking Elevation:aspect and Elevation:slope as features

# In[ ]:


train['Ele:Aap']=train['Elevation']/train['Aspect']
train['Ele:Slp']=train['Elevation']/train['Slope']
test['Ele:Aap']=test['Elevation']/test['Aspect']
test['Ele:Slp']=test['Elevation']/test['Slope']
train['Ele:Aap']=train['Ele:Aap'].replace(float('inf'),0)
test['Ele:Aap']=test['Ele:Aap'].replace(float('inf'),0)
train['Ele:Slp']=train['Ele:Slp'].replace(float('inf'),0)
test['Ele:Slp']=test['Ele:Slp'].replace(float('inf'),0)


# In[ ]:


get_ipython().run_cell_magic('html', '', "<div class='tableauPlaceholder' id='viz1536601237830' style='position: relative'><noscript><a href='#'><img alt='Sheet 12 ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;fo&#47;forest1&#47;Sheet12&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='forest1&#47;Sheet12' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;fo&#47;forest1&#47;Sheet12&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='filter' value='publish=yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1536601237830');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>")


# In[ ]:


get_ipython().run_cell_magic('html', '', "<div class='tableauPlaceholder' id='viz1536950701510' style='position: relative'><noscript><a href='#'><img alt='Sheet 9 ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Fe&#47;FeatureEngineering&#47;Sheet9&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='FeatureEngineering&#47;Sheet9' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Fe&#47;FeatureEngineering&#47;Sheet9&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='filter' value='publish=yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1536950701510');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>")


# Removing Unnecessary Features

# In[ ]:


Target=train['Cover_Type']
train=train.drop(['Cover_Type','Id'],axis=1)
test=test[train.columns]


# > **PredictionTime****
# Importing Necessary modules
# The Given problem is Multiclass classification
# After spending time with many algorithm,i would be using XGBClassifier 

# In[ ]:


from sklearn.model_selection import GridSearchCV as grc,train_test_split as tts
from sklearn.ensemble import RandomForestClassifier as rfc,VotingClassifier as vc
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier as orc
from sklearn.linear_model import LogisticRegression as lr
from xgboost import XGBClassifier as xgb
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier as knn


# **Diving train data into train and validation data in the ratio 7:3**
# 

# In[ ]:


xtrain,xtest,ztrain,ztest=tts(train,Target,train_size=0.7)
#xtrain1,xtest1,ztrain1,ztest1=tts(train1,Target,train_size=0.7)


# Tuned parameters for RandomforestClassifier

# In[ ]:


xg={'n_estimators':[200],'learning_rate':[0.05],'subsample':[0.6],'colsample_bytree':[0.7],'max_depth':[13]}
model=grc(xgb(),xg)


# Fitting the model over train data

# In[ ]:


model.fit(train,Target)


# In[ ]:


model.best_params_


# Checking Accuracy score

# **Making Prediction and preparing result csv file for submission****

# In[ ]:


from sklearn.metrics import accuracy_score as acs
print(acs(ztest,model.predict(xtest)))


# In[ ]:


predictedType=model.predict(test)


# In[ ]:


result=pd.DataFrame(predictedType)


# In[ ]:


result.index=pd.read_csv('../input/test.csv')['Id']


# In[ ]:


result.columns=['Cover_Type']


# In[ ]:


result.to_csv('result.csv')


# In[ ]:


result


# In[ ]:




