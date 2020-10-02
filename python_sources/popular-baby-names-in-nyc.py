#!/usr/bin/env python
# coding: utf-8

# # Finding Popular Baby Names in NYC

# ### Libraries & Data Load

# In[ ]:


import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np

# Visualisation Libraries
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import warnings
import re

from wordcloud import WordCloud, STOPWORDS 

pd.plotting.register_matplotlib_converters()
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('seaborn-darkgrid')
pd.set_option('display.max_columns', 50)
warnings.filterwarnings("ignore")
pd.options.display.float_format = '{:.2f}'.format


# In[ ]:


url = '../input/most-popular-baby-names-in-nyc/most_popular_baby_names_by_sex_and_mother_s_ethnic_group_new_york_city.csv'
data = pd.read_csv(url, header='infer')


# ### Data Exploration

# In[ ]:


data.shape


# In[ ]:


#Checking for null / missing values
data.isna().sum()


# In[ ]:


data.head()


# In[ ]:


data.groupby('ethcty').size()


# As we can observe, the following ethnicities are duplicates but have different names
# * ASIAN AND PACI & ASIAN AND PACIFIC ISLANDER
# * BLACK NON HISP & BLACK NON HISPANIC
# * WHITE NON HISP & WHITE NON HISPANIC
# 
# Hence replacing:
# * ASIAN AND PACI with ASIAN AND PACIFIC ISLANDER
# * BLACK NON HISP with BLACK NON HISPANIC
# * WHITE NON HISP with WHITE NON HISPANIC
# 

# In[ ]:


data = data.replace(to_replace="ASIAN AND PACI", value='ASIAN AND PACIFIC ISLANDER')
data = data.replace(to_replace="BLACK NON HISP", value='BLACK NON HISPANIC')
data = data.replace(to_replace="WHITE NON HISP", value='WHITE NON HISPANIC')


# In[ ]:


#Converting the Gender & Ethnicity Column to category
data['gndr'] = data['gndr'].astype('category')
data['ethcty'] = data['ethcty'].astype('category')


# In[ ]:


data[['cnt','rnk']].describe().transpose()


# ### Analysis - Bivariate Analysis [Year & Ethnicity]

# In[ ]:


# Create a function to find top 5 names in the specified year & ethnicity

def FindPopularNames(eth, yr):
    cnt_qnt = data['cnt'].quantile(0.75)  #Finding the 75% percentile
    rnk_qnt = data['rnk'].quantile(0.02)  #Finding the 2% percentile
    
    eth_list = list(data.ethcty.unique())
    yr_list = list(data.brth_yr.unique())
    
    #print(eth_list)
    #print(yr_list)
    
    # creating a sub dataset
    if (eth not in eth_list) or (yr not in yr_list):
        raise Exception("Sorry, your input value is not in the standard ethnicity/year list. Please try again!")
    else:
        xx_ml = data[(data['cnt'] > cnt_qnt) & (data['rnk'] <= rnk_qnt) & (data['brth_yr'] == yr) & (data['ethcty'] == eth)  & (data['gndr'] == 'MALE')]
        xx_fml = data[(data['cnt'] > cnt_qnt) & (data['rnk'] <= rnk_qnt) & (data['brth_yr'] == yr) & (data['ethcty'] == eth)  & (data['gndr'] == 'FEMALE')]
    
    xx_male_name = ' '
    xx_fmale_name = ' '
    stopwords = set(STOPWORDS) 
    
    for Mname in xx_ml['nm']:
        xx_male_name = xx_male_name + Mname + ' '

    for Fname in xx_fml['nm']:
        xx_fmale_name = xx_fmale_name + Fname + ' '
    
    
    fig, ax = plt.subplots(1, 2, figsize=(20, 6))
    
    Male_WC = WordCloud(width = 1000, height = 1000, background_color ='white', stopwords = stopwords, min_font_size = 10).generate(xx_male_name)
    FMale_WC = WordCloud(width = 1000, height = 1000, background_color ='white', stopwords = stopwords, min_font_size = 10).generate(xx_fmale_name)
    
    # plot the WordCloud image                        
    #plt.figure(figsize = (6, 6), facecolor = None) 
    #plt.imshow(wordcloud) 
    ax[0].imshow(Male_WC)
    ax[0].axis("off") 
    ax[0].set_title(f'Popular "{eth}" Male Baby Names in NYC in {yr}', fontsize='18', pad=30)
    
    ax[1].imshow(FMale_WC)
    ax[1].axis("off")
    ax[1].set_title(f'Popular "{eth}" Female Baby Names in NYC in {yr}', fontsize='18', pad=30)
    
    plt.tight_layout(pad = 0) 
    plt.show() 
    
    


# ### Popular Hispanic Baby Names in 2011

# In[ ]:


FindPopularNames('HISPANIC',2011)


# ### Popular White Non Hispanic Baby Names in 2014

# In[ ]:


FindPopularNames('WHITE NON HISPANIC',2014)


# ### Popular BLACK NON HISPANIC Baby Names in 2013

# In[ ]:


FindPopularNames('BLACK NON HISPANIC',2013)


# ### Popular ASIAN AND PACIFIC ISLANDER Baby Names in 2012

# In[ ]:


FindPopularNames('ASIAN AND PACIFIC ISLANDER', 2012)

