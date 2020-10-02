#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import plotly
import plotly.plotly as py
import plotly.graph_objs as go


# In[ ]:


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


# In[ ]:


correlations = joined_data.iloc[:, :-3].corrwith(joined_data['AUC']).sort_values(ascending=False)
gene_names = list(joined_data.columns[:-3])


# In[ ]:


# preparing the colors for data points of different cancer types
cancer_types = joined_data['cancer_type'].unique()
cancer_type_to_color_map = {}
unique_colors = 10*np.linspace(0, 1, len(cancer_types))
for index, cancer_type in enumerate(cancer_types):
    cancer_type_to_color_map[cancer_type] = unique_colors[index]
colors_of_data = joined_data['cancer_type'].map(cancer_type_to_color_map)


# In[ ]:


def plot_gene(gene_name, cancer_type='all'):
    
    if gene_name not in gene_names:
        print('Could not find such gene: ' + gene_name)
    elif cancer_type != 'all' and cancer_type not in cancer_types:
        print("Could not find such cancer type, please enter, in 'quotes' one of the following cancer               tissue types in capital letters and in quotes")
        for cancer_type in cancer_types:
            print(cancer_type)
    else :
        plotly.offline.init_notebook_mode(connected=False)
        if cancer_type=='all':
            data = joined_data
            x_values = list(data[gene_name] )
            y_values = list(data['AUC'] )
            plotly.offline.iplot({ "data": [go.Scatter(x=x_values, y=y_values, mode = 'markers', text = list(data.index),
                                                           marker= dict(color= colors_of_data, colorscale = 'Jet'), )],
                                      "layout": go.Layout(title=gene_name + ' corr=' + str(correlations.loc[gene_name])[:5],
                                                          hovermode='closest', 
                                                          xaxis=go.layout.XAxis(title=go.layout.xaxis.Title(
                                                              text='rsem normalized rna expression level' )
                                                                               ),
                                                          yaxis=go.layout.YAxis(title=go.layout.yaxis.Title(text = 'AUC') )
                                                         )
                                     }
                                    )
        else:

            data = joined_data.loc[joined_data['cancer_type']==cancer_type]
            x_values = list(data[gene_name] )
            y_values = list(data['AUC'])
            correlation = str(np.corrcoef(data[gene_name], data['AUC'])[0,1])[:5]
            plotly.offline.iplot({ "data": [go.Scatter(x=x_values, y=y_values, mode = 'markers', text = list(data.index),
                                                      )
                                           ],
                                  "layout": go.Layout(title=gene_name + ' corr=' + correlation + ' ' + cancer_type,
                                                      hovermode='closest', 
                                                      xaxis=go.layout.XAxis(title=go.layout.xaxis.Title(
                                                          text='rsem normalized rna expression level' ) ),
                                                      yaxis=go.layout.YAxis(title=go.layout.yaxis.Title(text = 'AUC') )
                                                     )
                                 }
                                )


# In[ ]:


plot_gene('ARFGEF2')


# In[ ]:


plot_gene('TP53', 'LUNG')


# In[ ]:





# In[ ]:




