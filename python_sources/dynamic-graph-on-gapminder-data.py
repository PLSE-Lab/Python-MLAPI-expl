#!/usr/bin/env python
# coding: utf-8

# ### Dynamic data visualization 
# 
# <p> As a passionate statistitians you may have seen the TED talk of Hans Rosling where he shows how the the world have been chainging in the last 70 years. Specifically Rosling talks about how child mortality is not anymore the same that it was back then and how the countries of the world have improved in providing better health services for their citizens. </p>
# 
# <p>Then he displays a graphics, a dynamic graphics of how child mortality (children that survive to reach 5 years old) with respect to children per woman in the different countries of the world has changed in the last 70 years. </p>
#     
# <p>So why this notebook ? Here are some points: </p>
#     
# - I love that presentation in particular the dynamic graph and I wanted to find a way to reproduce it in a Jupyter Notebook. 
# 
# - Here I share the code. 
#     
# - If you haven't seen it then here is the video on Youtube : https://www.youtube.com/watch?v=hVimVzgtD6w    
# 
# </p>
# 

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input/gapminder'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import re
import mailbox

import scipy.stats
from IPython import display 
from ipywidgets import interact, widgets


# ### Load the Gapminder dataset 
# <p> If you want to know more about Gapminder foundation here is the link on Wikipedia : https://en.wikipedia.org/wiki/Gapminder_Foundation

# In[ ]:


gapminder = pd.read_csv('/kaggle/input/gapminder.csv')
gapminder.head()


# ### Display scatterplot 
# 
# - The graphics show the situation was back in 1960 with respect to babies per woman and children surviving to age 5.
# 
# - Each point represents a country. Interestingly there are two clusters which back in 1960 represented the rich world and the poor world.  

# In[ ]:


with plt.rc_context():
      plt.rc("figure", figsize=(16,6))
      gapminder[gapminder['year'] == 1960].plot.scatter('babies_per_woman', 'age5_surviving')
      plt.show()


# ### Display the graphics with one more dimension: population per country.
# - the following graphics displays population per country: the larger is the cycle and the larger is the population of the country represented by the circle
# - moreover, the colors represent the continents : Africa = skyblue, Europe = gold, America = palegreen and Asia = coral
# - the two big circles of coral color are China and India

# In[ ]:


def plotyear(year):
    data = gapminder[gapminder['year'] == year]
    
    area = 0.000005 *data['population']
    colors = data['region'].map({'Africa': 'skyblue', 'Europe':'gold', 'America':'palegreen', 'Asia':'coral'})
    
    
    data.plot.scatter('babies_per_woman', 'age5_surviving', s=area, c=colors, 
                         linewidths = 1, edgecolors = 'k', figsize=(16,8))
    plt.show()
plotyear(1960) 


# ### Make the graphics dynamic. 
# - by using the method **interact** imported from ipywidgets  it is possibile to make a dynamic representation 
# - moving the cursor it is possible to se how the graphics evolves in time from 1950 back in 2015.
# - more the cursor slowly or the graphics will move very fast  

# In[ ]:


interact(plotyear, year = widgets.IntSlider(min=1948, max=2015, step = 1, value= 1960))


# ### Conclusions. 
# <p> The interesting thing to see is that all countries move towards less babies per woman (typically 2 babies) and towards survival rate above 95%. 
# Final recomendation: read Gosling's book **Factfulness** it will give you many insights of how the world has been chainging.     

# In[ ]:




