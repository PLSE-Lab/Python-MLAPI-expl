#!/usr/bin/env python
# coding: utf-8

# # Scatter Plot of AUC with CCLE Data
# ## Abazeed Lab at Cleveland Clinic Lerner Research Institute
# 
# ## To skip the code and see the visualization, please scroll down to the bottom of this notebook.

# In[1]:


import numpy as np
import pandas as pd
import plotly
import plotly.plotly as py
import plotly.graph_objs as go


# ### Loading and Processing the Data

# In[2]:


raw_data = pd.read_csv('../input/CCLE.rpkm.v2.gct', sep='\t', skiprows=2)
data = raw_data.set_index('Description').drop('Name', axis=1).T
radsen_table_raw = pd.read_csv('../input/RadSen_CTD2_new.xlsx', sep='\t', skiprows=2)
radsen_table = radsen_table_raw.set_index('Label').drop('Description', axis=1).T

joined_data = pd.concat([data, radsen_table[['AUC', 'Normalized_AUC']]], axis=1, sort=False)
joined_data.dropna(axis=0, how='any', subset=['AUC', 'Normalized_AUC'], inplace=True)
joined_data['AUC'] = joined_data['AUC'].astype('float')
joined_data['Normalized_AUC'] = joined_data['Normalized_AUC'].astype('float')
joined_data = joined_data.dropna()  # ask Mo if he is ok with losing 11 rows (cancer types) since they have nan
# and we cannot find correlation. 532 rows go down to 521
cell_line_names = pd.Series(joined_data.index, index=joined_data.index)
joined_data['cancer_type'] = cell_line_names.str.split('_', n=1, expand=True).loc[:,1]


# ##### 
# 1. Calculating Correlations of Gene Expressions with AUC
# 2. Ordering the Genes in Order of their Correlation with AUC

# In[3]:


correlations = joined_data.iloc[:, :-3].corrwith(joined_data['AUC']).sort_values(ascending=False)
gene_names = list(joined_data.columns[:-3])


# In[4]:


correlations.hist();


# In[5]:


np.random.uniform(5)


# In[6]:


['PROSTATE', 'CENTRAL_NERVOUS_SYSTEM', 'URINARY_TRACT', 'KIDNEY',
       'THYROID', 'SKIN', 'SOFT_TISSUE', 'SALIVARY_GLAND', 'OVARY',
       'BONE', 'PLEURA', 'STOMACH', 'ENDOMETRIUM', 'PANCREAS', 'BREAST',
       'UPPER_AERODIGESTIVE_TRACT', 'LARGE_INTESTINE', 'LUNG',
       'AUTONOMIC_GANGLIA', 'OESOPHAGUS', 'FIBROBLAST', 'LIVER',
       'BILIARY_TRACT']


# In[7]:


cancer_types = joined_data['cancer_type'].unique()


# In[8]:


unique_colors = 10*np.linspace(0, 1, len(cancer_types))


# In[9]:


cancer_type_to_color_map = {}
for index, cancer_type in enumerate(cancer_types):
    cancer_type_to_color_map[cancer_type] = unique_colors[index]


# In[10]:


len(joined_data['cancer_type'].unique())


# In[11]:


colors_of_data = joined_data['cancer_type'].map(cancer_type_to_color_map)


# Remove Cases with less than 20 data points

# # Visualization

# In[12]:


go.Scatter(legendgroup)


# In[ ]:


def plot_gene(gene_name):
    x_values = list(joined_data[gene_name])
    y_values = list(joined_data['AUC'])
    plotly.offline.init_notebook_mode(connected=False)
    plotly.offline.iplot({ "data": [go.Scatter(x=x_values, y=y_values, mode = 'markers', text = list(joined_data.index),
                                              marker= dict(color= colors_of_data, colorscale = 'Jet'),
                                              )],
                         "layout": go.Layout(title=gene_name + ' corr=' + str(correlations.loc[gene_name])[:5],
                                             hovermode='closest', sho)
                         }
                        )


# In[ ]:


plot_gene('ARFGEF2')


# ## Choose the Gene of Interest in the Menu Ordered by Increasing Correlation with AUC

# Note that the graph may take a few minutes to appear/refresh after selecting a gene since there is a lot of data and processing involved.

# 
# 1. ### x and y axis labels (rsem normalized rna seq... vs bla bla )
# ### make the different disease types/organs different colors
# 

# In[ ]:


correlations.head()

