#!/usr/bin/env python
# coding: utf-8

# **Covid and BCG_Immunisation Data Analysis and Visualisation, As of 9th May 2020**
# 
# For full jupyter notebook go to 
# * covid_bcg notebook : https://github.com/priyanka7255/public_repo/blob/master/covid/EDAAll.ipynb
# * Day0 Plots for covid deaths : https://github.com/priyanka7255/public_repo/blob/master/covid/EDADay0Plots-DeathPerMillion.ipynb
# 
# Data sources
# * WHO Immunization data source : https://apps.who.int/immunization_monitoring/globalsummary
# * John Hopkins covid data : https://github.com/CSSEGISandData/COVID-19

# **Below are some charts, detailed notebook links above**
# 
# The graphs below show bcg_immunisation and covid deaths/cases. In the graph we see that countries with high death/confirmed cases have low BCG Immunisation rate. This is not suggesting any casaution, but further investigation can confirm if there is correlation. I have just attemted to visualise the data. Detailed notebook at https://github.com/priyanka7255/public_repo/blob/master/covid/EDAAll.ipynb
# 
# Also note, two clinical trials addressing this question are underway, and WHO will evaluate the evidence when it is available. Details at https://www.who.int/news-room/commentaries/detail/bacille-calmette-gu%C3%A9rin-(bcg)-vaccination-and-covid-19

# In[ ]:


from IPython.display import Image 

for i in range(1,4):
    pil_img = Image(filename='/kaggle/input/chart'+str(i)+'.png')
    display(pil_img)


# I have also created charts to show the covid progression for various countries since day0. day0 is the day since specific number of cases are forund for each country. The charts specify day0 in the chart title. Detailed notebook at https://github.com/priyanka7255/public_repo/blob/master/covid/EDADay0Plots-DeathPerMillion.ipynb

# In[ ]:


for i in range(4,8):
   pil_img = Image(filename='/kaggle/input/chart'+str(i)+'.png')
   display(pil_img)
   

