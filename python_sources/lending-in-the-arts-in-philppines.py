#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# pip install squarify


# ## The status of lending from Kiva in the Arts in Philippines

# #### Import the necessary libraries

# In[ ]:


#Basic exploratory libraries
import numpy as np
import pandas as pd

#Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import squarify as sq

#Word cloud libraries
from os import path
from PIL import Image
from wordcloud import WordCloud, ImageColorGenerator

#Library to supress unnecessary warnings
import warnings
warnings.filterwarnings('ignore')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


#  

# #### Read in the required data

# In[ ]:


kiva_loan = pd.read_csv('/kaggle/input/data-science-for-good-kiva-crowdfunding/kiva_loans.csv')
kiva_themes = pd.read_csv('/kaggle/input/data-science-for-good-kiva-crowdfunding/loan_theme_ids.csv')
kiva_mpi = pd.read_csv('/kaggle/input/data-science-for-good-kiva-crowdfunding/kiva_mpi_region_locations.csv')

kiva_lt = kiva_loan.merge(kiva_themes, on = 'id')


#  

# #### Get the subset of data for Philippines

# In[ ]:


philippines_loan = kiva_lt[kiva_lt['country'] == 'Philippines']
philippines_mpi = kiva_mpi[kiva_mpi['country'] == 'Philippines']

philippines = philippines_loan.merge(philippines_mpi, on = 'country')
philippines.head(1)


# In[ ]:


philippines.columns


#  

# Once obtained, I combined it all to one data frame that has just the columns I need to analyse lending in the arts in Philippines, then get just that subset of data.

# In[ ]:


philippines = philippines[['region_x', 'sector', 'activity', 'Loan Theme Type', 'use', 'date', 'loan_amount',
                           'funded_amount', 'borrower_genders', 'lender_count', 'term_in_months',
                           'repayment_interval', 'geo', 'lat', 'lon']]
philippines.columns.values[0] = 'region'
philippines.columns.values[3] = 'theme'


# In[ ]:


arts = philippines[philippines['sector'] == 'Arts']
arts.head(2)


# In[ ]:


arts.columns


# In[ ]:


arts.count()


#  

# #### Doing initial exploration of data in the arts

# Higher number of lenders gave towards the smaller amounts of loans. The smaller amounts were also higher in number compared to the higher amounts.

# In[ ]:


sns.set_context('talk')
sns.jointplot(x = 'loan_amount', y = 'lender_count', data = arts)


#  

# Looking at the individual distribution plots of both lender_count and loan_amount, it confirms that the lower amounts are the ones that have more people giving to them, and also the ones that are most requested.
# 
# It is, therefore, important to develop lending programmes better suited to those seeking lower amounts of funding.

# In[ ]:


plt.figure(figsize = (20,5))
plt.xticks(rotation = 75)

sns.distplot(arts['lender_count'], bins = 10)


# In[ ]:


plt.figure(figsize = (20,5))
plt.xticks(rotation = 75)

sns.distplot(arts['loan_amount'], bins = 10)


#  

# #### Narrowing it down further to look at gender activity and loan amounts

# In[ ]:


arts_activity = arts.groupby(['borrower_genders', 'activity'])['loan_amount'].sum().sort_values(ascending = False).reset_index()
arts_activity


#  

# The following treemap shows that the top three activities funded were weaving, crafts and arts.

# In[ ]:


sizes = arts_activity['loan_amount']
label = np.array(arts_activity['activity']) + '\n' + sizes.astype('str')

plt.style.use('ggplot')
plt.figure(figsize = (20,15))

sq.plot(sizes = sizes, label = label, alpha = 0.6, text_kwargs={'fontsize':10})


#  

# Broken down by gender, it shows that women by far outnumber the men that are funded in the arts in Philippines. In the bottom four activities (knitting, embroidery, musical instruments and patchwork), there were no men that received loans.

# In[ ]:


male = len(arts[arts['borrower_genders'] == 'male'])
female = len(arts[arts['borrower_genders'] == 'female'])

print('There are {} females versus {} males who have received loans in the arts in Philippines. A difference of {}.'.format(female, male, (female - male)))


#  

# In[ ]:


fig = px.bar(arts_activity, x = 'activity', y = 'loan_amount',
            hover_data = ['loan_amount', 'borrower_genders'], color = 'loan_amount',
            labels = {'loan_amount': 'Loan Amount'}, height = 500)

fig.show()


# Looking at repayment interval shows that majority do so irregularly.

# In[ ]:


arts_repayment = arts.groupby(['borrower_genders', 'activity', 'repayment_interval'])['loan_amount'].sum().sort_values(ascending = False).reset_index()
arts_repayment


# In[ ]:


fig = px.bar(arts_repayment, x = 'repayment_interval', y = 'loan_amount',
            hover_data = ['loan_amount', 'borrower_genders', 'activity'], color = 'loan_amount',
            labels = {'loan_amount': 'Loan Amount'}, height = 500)

fig.show()


# Mapping the distribution of loan amounts in the arts in Philippines.

# In[ ]:


mapbox_access_token = 'pk.eyJ1IjoiZ2l0b25nYSIsImEiOiJjazBueDZsN2cwNGE3M21xcnl0bGg0cWUxIn0.9-jwOGyzRkCFbcPfafeoMw'

size = arts['loan_amount'] / 100

fig = go.Figure(go.Scattermapbox(
    lat = arts['lat'],
    lon = arts['lon'],
    mode = 'markers',
    marker = go.scattermapbox.Marker(
        size = size,
        color = 'red',
            ),
    text = arts['region'],
        )
    )

fig.update_layout(
    autosize = True,
    hovermode = 'closest',
    mapbox = go.layout.Mapbox(
        accesstoken = mapbox_access_token,
        bearing = 0,
        center = go.layout.mapbox.Center(
            lat = 12.2445, 
            lon = 125.0388,
            ),
        pitch = 0,
        zoom = 4,
        )
    )

fig.show()


# A word cloud shows that the biggest use of the money is to buy supplies and raw materials for business needs.

# In[ ]:


text = " ".join(review for review in arts['use'])
wordcloud = WordCloud(max_words=1000, background_color="white").generate(text)

plt.figure(figsize=[15,10])
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# ## Conclusion

# Women by far make up the bulk of those lent money in the arts in Philippines. They use the money mainly to buy raw materials and supplies for their businesses.
# 
# The top 2 activities are in weaving and crafts, which tend to belong to functional arts (baskets, furniture, mats). It means that majority of those that borrow make items to sell for functional use.
# 
# Most of those that borrow pay back their loans irregularly, which probably points to the unpredictable sales cycles they face for their products.
# 
# Kiva should, therefore, come up with loans tailor made for women in the arts in Philippines that would best suit their needs and irregular sales cycles.
