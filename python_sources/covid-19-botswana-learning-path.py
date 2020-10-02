#!/usr/bin/env python
# coding: utf-8

# # BOTSWANA

# # How many cases of COVID-19 does Botswana really have?
# > Reported Botswana cases counts are based on an etestimated  number of suspected cases that turned out to be mostly negative.Updated information coming from the MOH has confirmed six (6) cases that tested positive. Since information is extracted from various sources not everyone is tested, the numbers will be somewhat biased. 
# 
# - Author: Thato Seeletso Mmusi
# - Country: Botswana

# SHORT SYNOPSIS OF NOTEBOOK

# The following world map illustrates all countries in the world affected by COVID-19. As seen on the map, Botswana now has four confirmed cases - 6 cases tested positive with one death. In the following notebook I will model the most basic data on Botswana and illustrate it graphically, followed by a look at what is going on in countries surrounding Botswana as well as a summary of the pandemic effects on world leading countries.

# <img src="https://camo.githubusercontent.com/1fa430f92b9a149023e599a08111de7588fc6682/68747470733a2f2f73746f726167652e677569646f7474692e6465762f636f76696431392f6d61702f776f726c642e676966" style="" /> 

# DISCLAIMER map sourced from https://github.com/emanuele-guidotti/COVID19 by Emanuele Guidotti 

# Below is the latest information I could extract from various sources in Botswana. I cross checked it with the information on a flyer updatevfrom the Ministry of Health  on the BWGovernment Facebook page.
# 
# - <a href="https://www.moh.gov.bw/" >Ministry of Health - Botswana</a>
# - <a href="https://www.facebook.com/BotswanaGovernment/photos/a.617504571665538/2860234297392543/?type=3&eid=ARCv5y5VQ3ZX9zkob078fNi6aI_VCls2CIWk6ESk9IJ2NPhme-C7JlVKEKIzjboPUsUSNVU-zhxUYqOz&__xts__%5B0%5D=68.ARBvq0OXssAWyyYcw-x_BvUvtndgvRu6c5nI2jBkZXzf_XOOAqLhTC7e6f-tdwwvVvOxUHU7egM5pJJOW19HOCLF5tJaMoLiUpn-PU4ml3RQjrtSpygSY4AoH2Mx6iAz4EGaZ_OUOv-iKoMtM72VOeOOYWOE0-stk_4jND2ruLesSvHs5llrliQvig3IntPzJnoQxMYwA_YGZOLlqeLd_YvWYcj9tB_05Efa9cfF85QidnPhG6nXWaUo62jyEIxmNSPi_fPhQDO1rqPrv_onzTsOAAu3L_T62pLujG2imLFcDBWv4r6MSCW-XNG3DxiSSpwKqppG7gWFjlyiYB7LqC9g6w&__tn__=EHH-R">BWGovernment - Botswana</a> 
# 
#     Below is the electronic of such data
#     
#     <img src="img/bwGovernment_update.jpg" style="width:50%; height:50%;" />
#     figure 2.1
#     
#     As of 30 March 2020 Botswana recorded it first three (3) confirmed cases.
#     
#     <img src="img/BwGovernment 31-03-2020.jpg" style="width:50%; height:50%;" />
#     figure 2.1.1
#     
#     <img src="https://media-exp1.licdn.com/dms/image/C4D12AQEsCL4Pd-swOQ/article-inline_image-shrink_1000_1488/0?e=1591833600&v=beta&t=9nQLkKSL67cuq00snTWJK5V_m2ZER_VuPoiIqIAVA4Y" style="width:50%; height:50%;" />
#     figure 2.1.2
#     
#     The information illustrated by picture figure 2.1.2 above was first released on the 3rd April 2020 by the Ministry of   
#     Health in Botswana. 
#  
#     It was updated in the evening of same day to 2001 people quarantined, 670 people tested, 572 tested negative, 4 tested    
#     positive with 1 death.
#     
#     Latest updates as of 06/04/2020
#     
#     BREAKING: 2 new confirmed Coronavirus positive cases in Botswana
# 
#    Current stats:
#    966 tests
#    804 negative
#    6 confirmed cases
#    156 pending tests
#    1 death
#    
#    source: https://web.facebook.com/MmegiOnline/?tn-str=k%2AF
# 
# All patients are in good health at Sir Ketumile Masire Hospital
# 
#     
#     NB: Data is not easily available locally(in most cases one has to acquire it themselves, and in some instances ask for   
#     permission), there is still plenty of red tape acquiring it. 

# To work with the only publicily avaialble data in Botswana I will start by importing the following <a href="">pandas</a> convention
# NB: Please note throughout the notebook I will be using the same conventions as follows: 
# 

# In[ ]:


from pandas import Series, DataFrame
import pandas as pd


# If you pause and take a look above you will realize I have imported two data structures namely the "Series" & "DataFrame"  from pandas package. 
# 
# 
# - So, the reader wiil come across pd. more often in the code, it would just be the shortcut to the pandas library.
# - In addition because the "Series" and "DataFrame" are used more so often it is easier if i import them into the local    
#   namespace.
#   
# I will work with the available Botswana data using the two structures as follows:

# # Series

# I will start manipulating the data using the Series Workhorse of the pandas library

# A Series is a one-dimensional object containing an array of data and an associated array of data labels, called its index.
# - An array like object containing any <a href="">NumPy</a> data type
# - The simplest Series is created from only an array of data:

# Below I reproduce the Government data as given on above illustrated figure 2.1.2

# In[ ]:


obj = Series([2001, 966, 804, 6, 1, 156])
obj 


# The string representation of a Series is displayed interactively:
# 
# - On the left is the index 
# - On the right are the values of string representation. 
# 
# Note: in the above example I did not specify an index for the data,so by default
# one consisting of the integers 0 through N - 1 (where N is the length of the data) is
# created. 

# Before I can assign each data point to an index to identify the series I think it is best here to illustrate the fact that a series is array-like object by doing the following:
#       Get the array representation and index object of the Series through 
#         - its values
#         - index attributes

# In[ ]:


obj.values


# In[ ]:


obj.index


# To further expand on the available data that I have already created a Series data structure, I will now  to create  index identifying each data point:

# In[ ]:



object_2 = Series([2001, 966, 804, 6, 1, 156], index=['Number Quarantined Cases', 'Number of Suspected Cases', 'Number of Cases that tested Negative', 'Number of Cases that tested Positive', 'Deaths','Pending Resuts'])
object_2


# In[ ]:


object_2.plot(kind='bar')


# Next I will do the same thing I did aboe using DataFrame

#  # DataFrame

# A DataFrame represents a tabular, data structure containing an ordered
# collection of columns, each of which can be a different value type (numeric,
# string, boolean).The DataFrame has both a row and column index;

# In[ ]:


data = {'Measures Taken': ['Quarantined', 'Suspected/Tested Cases', 'Negative Tests', 'Positive Tests','Deaths', 'Awaiting Results'],
'Cases Counted': [2001, 966, 804, 6, 1, 156]}
frame = DataFrame(data)
frame


# In[ ]:


frame2 = DataFrame(data, columns=['Measures Taken', 'Cases Counted'],
index=['Quarantined', 'Suspected/Tested Cases', 'Negative Tests', 'Positive Tests', 'Deaths','Awaiting Results'])

frame2


# In[ ]:


frame['Measures Taken'][0]


# In[ ]:


frame2.plot(kind='bar')


# But looking at the information I am working with I think it is cluttered and and could be arranged or done in a better way... I will try illustrating this by usng a pie chart to show the Total Suspected cases as an overall picture of Total tested, those who tested positive, those who tested negative and those with pending results.

# In[ ]:


frame3 = DataFrame(data, columns=['Measures Taken', 'Cases Counted'],
index=['Quarantined', 'Suspected/Tested Cases', 'Negative Tests', 'Positive Tests', 'Deaths', 'Awaiting Results'])

frame3


# In[ ]:


df = pd.DataFrame({'Suspected Cases': [1, 804, 6, 156]},
                  index=['Deaths','Tested Negative', 'Tested Positive', 'Awaiting Results'])
plot = df.plot.pie(y='Suspected Cases', figsize=(5, 5))


# In[ ]:




