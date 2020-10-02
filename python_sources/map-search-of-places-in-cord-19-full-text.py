#!/usr/bin/env python
# coding: utf-8

# ## Search for Geographically Relevant Articles
# This hosted map allows you to search for possible place names mentioned in the CORD-19 full text articles (2020-03-20 release). 
# 
# Each point represents a geocoded placename that may be represented in the text. Clicking a point will give a list of sentences that may mention that place and each sentence provides full metadata on the containing article and a link to it. It is our hope that this interactive map will provide a useful exploratory tool for data scientists and COVID-19 researchers who are seeking out specific geographic datasets.
# 
# In its current form, the map and underlying data have not been cleaned. We have released it, as is, in order to disseminate it as quickly as possible. Over the coming days, we intend to clean the data and refine the map.
# 
# The full spatial and textual data used to generate the map is available for research and use in the [Spatial Data for CORD-19 (COVID-19 ORDC)]( https://www.kaggle.com/charlieharper/spatial-data-for-cord19-covid19-ordc) dataset on Kaggle.
# 
# This map is publicly available on the web at https://arcg.is/yb1uu

# In[ ]:


from IPython.display import IFrame
IFrame(src='https://arcg.is/yb1uu', width=600, height=500)


# In[ ]:




