#!/usr/bin/env python
# coding: utf-8

# ## Crop Yield & Rainfall Data India - Analysis

# In[ ]:


# Generic Libraries
import numpy as np
import pandas as pd

# Visualisation Libraries
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import re

pd.plotting.register_matplotlib_converters()
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('seaborn-darkgrid')
pd.set_option('display.max_columns', 50)
warnings.filterwarnings("ignore")
pd.options.display.float_format = '{:.1f}'.format


# ### Data Load

# In[ ]:


url = '../input/crop-yield-per-state-and-rainfall-data-of-india/out.csv'

data = pd.read_csv(url, header='infer')


# ### Data Exploration

# In[ ]:


data.shape


# In[ ]:


data.isna().sum()


# In[ ]:


data.info()


# In[ ]:


data.head()


# We'll drop the "Unnamed: 0" column as it not relevant for our data analysis

# In[ ]:


data = data.drop('Unnamed: 0', axis=1)


# In[ ]:


#Creating a data backup
data_backup = data.copy()


# ### Finding Correlation
# 
# Under this analysis, we'd like to know if there is a relation (positive/negative) between the Area, Production & Rainfall 

# In[ ]:


APR_df = data[['Area','Production','Rainfall']]

corr = APR_df.corr()
plt.figure(figsize=(8, 8))
g = sns.heatmap(corr, annot=True, cmap = 'PuBuGn_r', square=True, linewidth=1, cbar_kws={'fraction' : 0.02})
g.set_yticklabels(g.get_yticklabels(), rotation=0, horizontalalignment='right')
g.set_title("Correlation between Area, Production & Rainfall", fontsize=14)
#bottom, top = g.get_ylim()
#g.set_ylim(bottom + 0.5, top + 0.5)
plt.show()


# #### Observation:
# 
# *   There is a positive co-relation between Area - Production & Rainfall - Production
# 
# 

# ### Data Visualisation & Analysis

# In[ ]:


data.head()


# In[ ]:


#Creating a subset of the original data for visualisation
sub_data = data.drop(data[data.Season == 'Whole Year '].index)


# In[ ]:


# Create a function that returns Bar Graphs for Production & Rainfall in States per seasons
def seasonal_view(state):
    """
    Creating 2 seperate pivot-tables with mean Production & Rainfall
    """
    ptable_prod = sub_data[(sub_data['State'] == state)].pivot_table(values='Production',index='Year', columns='Season', aggfunc= 'mean', fill_value= 0.0)
    ptable_rain = sub_data[(sub_data['State'] == state)].pivot_table(values='Rainfall',index='Year', columns='Season', aggfunc= 'mean', fill_value= 0.0)


    fig = plt.figure()
    plt.subplots_adjust(hspace = 5)
    sns.set_palette('deep')
    
    """
    Draw a Line Graph on First subplot. - Rainfall
    """
    
    year_labels = ptable_rain.index.tolist()
    season_labels = ptable_rain.columns.tolist()
    
    ax1 = ptable_rain.plot(kind='line', figsize=(15,6))
    # Add some text for labels, title and custom x-axis tick labels, etc.
    plt.title(f'{state.capitalize()} Seasonal Annual (mean) Rainfall', fontsize=16)
    plt.ylabel("Rainfall", fontsize=13)
    plt.legend(prop={'size':10}, loc='best',bbox_to_anchor=(0.4, 0., 0.75, 0.5) )
    


    """
    Draw a Line Graph on First subplot. - Production
    """
    
    ax2 = ptable_prod.plot(kind='line',figsize=(15,6))
    plt.title(f'{state.capitalize()} Seasonal Annual (mean) Production', fontsize=16)
    plt.ylabel("Production", fontsize=13)
    plt.legend(prop={'size':10}, loc='best',bbox_to_anchor=(0.4, 0., 0.75, 0.5))

    fig.tight_layout()
    plt.show()


# ## State Wise Production & Rainfall Visualisation

# In[ ]:


seasonal_view('Bihar')


# In[ ]:


seasonal_view('Punjab')


# In[ ]:


seasonal_view('Kerala')


# In[ ]:


seasonal_view('Jharkhand')


# In[ ]:


seasonal_view('Uttarakhand')


# In[ ]:


seasonal_view('Odisha')


# In[ ]:


seasonal_view('Chhattisgarh')

