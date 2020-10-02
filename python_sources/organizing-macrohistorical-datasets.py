#!/usr/bin/env python
# coding: utf-8

# # ***Data Wrangling***
# First of all, I would like to thanks Sohier Dane for uploading this wonderful dataset from the National Bureau of Economic Research (NBER). <br><br>Since there is an impressive quantity of data, this kernel will be mostly about organizing the database and creating features. I will also do some basic visualizations, though most of my visualizations will be on another kernel.<br><br>**UPDATE : This kernel is done, I will no longer update it. If you want to take a look at my recent project, please visit my second kernel with NBER dataset!**

# **Import and read CSV files**

# In[ ]:


import pandas as pd
import numpy as np
data_df = pd.read_csv('../input/data.csv')
doc_df = pd.read_csv('../input/documentation.csv')


# >'doc_df' is a dataframe containing the variable specifications (ID, description, documentation, chapter), and 'data_df' is a dataframe containing the data.

# **Quick View**

# In[ ]:


print(doc_df.info())
doc_df.head()


# In[ ]:


print(data_df.info())
data_df.head()


# **1. doc_df has a 'documentation' column. I will create several features from this documentation. <br> 2. I will also merge data and documentation. This will be done by keeping the original documentation dataframe (doc_df) and creating a 'data' column, populated as a dictionary as follow  :<br><br>  {year_x1 : value_x1, year_x2 : value x_2, ...}**

# In[ ]:


pd.options.mode.chained_assignment = None

label_list = ['var', 'md', 'loc', 'width', 'dk', 'col', 'nber_series', 'area_covered','units',
      'annual_coverage', 'quarterly_coverage', 'monthly_coverage', 'seasonal_adjustment',
      'source', 'notes', 'data']

for x in range(len(label_list)):
    doc_df[label_list[x]] = None

for x in range(len(doc_df)):
    if "VAR" in doc_df['documentation'][x]:
        doc_df['var'][x] = ' '.join(doc_df['documentation'][x].split('VAR')[1].split('MD=')[0].split())
    if 'MD=' in doc_df['documentation'][x]:
        doc_df['md'][x] = ''.join(doc_df['documentation'][x].split('MD=')[1:]).split()[0]
    if 'LOC' in doc_df['documentation'][x]:
        doc_df['loc'][x] = ''.join(doc_df['documentation'][x].split('LOC')[1:]).split()[0]
    if 'WIDTH' in doc_df['documentation'][x]:
        doc_df['width'][x] = ''.join(doc_df['documentation'][x].split('WIDTH')[1:]).split()[0]
    if 'DK' in doc_df['documentation'][x]:
        doc_df['dk'][x] = ''.join(doc_df['documentation'][x].split('DK')[1:]).split()[0]
    if 'COL' in doc_df['documentation'][x]:
        doc_df['col'][x] = ''.join(doc_df['documentation'][x].split('COL')[1:]).split()[0]
    if 'NBER SERIES:  ' in doc_df['documentation'][x]:
        doc_df['nber_series'][x] = ''.join(doc_df['documentation'][x].split('NBER SERIES:  ')[1].split('\nAREA COVERED:')[0])
    if 'AREA COVERED:  ' in doc_df['documentation'][x]:
        doc_df['area_covered'][x] = doc_df['documentation'][x].split('AREA COVERED:  ')[1].split('\nUNITS:')[0]
    if 'UNITS:  ' in doc_df['documentation'][x]:
        doc_df['units'][x] = ''.join(doc_df['documentation'][x].split('UNITS:  ')[1].split('\nANNUAL COVERAGE:')[0])
    if 'ANNUAL COVERAGE:  ' in doc_df['documentation'][x]:
        doc_df['annual_coverage'][x] = ''.join(doc_df['documentation'][x].split('ANNUAL COVERAGE:  ')[1].split('\nQUARTERLY COVERAGE:')[0])
    if 'QUARTERLY COVERAGE:  ' in doc_df['documentation'][x]:
        doc_df['quarterly_coverage'][x] = ''.join(doc_df['documentation'][x].split('QUARTERLY COVERAGE:  ')[1].split('\nMONTHLY COVERAGE:')[0])
    if 'MONTHLY COVERAGE:  ' in doc_df['documentation'][x]:
        doc_df['monthly_coverage'][x] = ''.join(doc_df['documentation'][x].split('MONTHLY COVERAGE:  ')[1].split('\nSEASONAL ADJUSTMENT:')[0])
    if 'SEASONAL ADJUSTMENT:  ' in doc_df['documentation'][x]:
        doc_df['seasonal_adjustment'][x] = ''.join(doc_df['documentation'][x].split('SEASONAL ADJUSTMENT:  ')[1].split('\nSOURCE')[0])
    if 'SOURCE:  ' in doc_df['documentation'][x]:
        doc_df['source'][x] = ''.join(doc_df['documentation'][x].split('SOURCE:  ')[1].split('\nNOTES')[0])
    if 'NOTES:  ' in doc_df['documentation'][x]:
        doc_df['notes'][x] = ''.join(doc_df['documentation'][x].split('NOTES:  ')[1:])
        
    data = data_df[data_df['Variable'] == doc_df['file_name'][x]]
    time = pd.to_datetime(data['Date'])
    keys = list(time.values)
    vals = list(data['Value'].values)

    data_dict = {}
    for y in range(len(data)):
        data_dict[keys[y]] = vals[y]
    doc_df['data'][x] = data_dict


# **Vertical display to show new features**

# In[ ]:


doc_df.head().transpose()


# **View unstructured data (description, source, notes)**

# In[ ]:


print(
    '{}\n\n{}\n{}'.format(
        doc_df.description[0],
        doc_df.source[0],
        doc_df.notes[0]
    )
)


# **To create a data grid with a specific variable, we just have to put the 'data' column, wich is dictionary, in a DataFrame. We can also put the 'description' column as index. **

# In[ ]:


pd.DataFrame(
    data = doc_df.data[100],
    index = [doc_df.description[100]]
            ).transpose()


# **Create a function to search keywords and countries in the merged dataframe**

# In[ ]:


def keyw_in_country(keyw, country=0, doc_df=doc_df):
    df_list = []
    if type(keyw) == str:
        keyw = [keyw]
    if type(country) == str:
        country = [country]
        
    for y in range(len(keyw)):
        df_list.append(doc_df[
            list(
                map(
                    lambda x: keyw[y].lower() in str(doc_df['description'][x]).lower(),
                    range(len(doc_df))))].reset_index()
                      )
        
    new_df = pd.concat(df_list).reset_index()
    
    if country == 0:
        return(new_df)
    else:
        df_list = []
        
    for z in range(len(country)):
        df_list.append(new_df[
        list(
            map(
                lambda x: country[z].lower() in str(new_df['area_covered'][x]).lower(), 
                range(len(new_df))))]
          )
    return(pd.concat(df_list))


# **Ex. Search Wine and Wheat in France and Germany**

# In[ ]:


keyw_in_country(
    keyw = ['Wine','Wheat'],
    country = ['France', 'Germany']
) 


# **Create a function to plot a basic 'time-linear' variable**

# In[ ]:


def plot_variable(var_label, col, file_name, graph_title):
    
    import plotly.graph_objs as go
    from plotly import tools
    from plotly.offline import init_notebook_mode, plot, iplot
    init_notebook_mode()
    
    if type(var_label) == str:
        var_label = [var_label]
    if type(col) == str:
        col = [col]
    
    data = []
    
    for x in range(len(var_label)):
        df = doc_df[doc_df['file_name'] == var_label[x]].reset_index()
        trace_high = go.Scatter(
            x=pd.to_datetime(list(df.data[0].keys())), 
            y=list(df.data[0].values()),
            name = df.description[0],
            line = dict(color = col[x]),
            opacity = 0.8)
        data.append(trace_high)
    
    layout = dict(
        title = graph_title,
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1,
                         label='1m',
                         step='month',
                         stepmode='backward'),
                    dict(count=6,
                         label='6m',
                         step='month',
                         stepmode='backward'),
                    dict(step='all')
                ])
            ),
            rangeslider=dict(),
            type='date'
        )
    )

    fig = dict(data=data, layout=layout)
    iplot(fig, filename = file_name)


# **Ex. Visualize London Silver Price (take file_name from a previous keyw_in_country)**

# In[ ]:


keyw_in_country(keyw='Silver', country = 'London')


# In[ ]:


plot_variable(
    var_label = 'a04018', 
    col = '#aeb9ba', 
    file_name = "London Silver", 
    graph_title = 'London Silver Price')


# **Ex. Visualize U.S. Monetary Gold Stock**

# In[ ]:


gold_df = keyw_in_country(keyw='Gold Stock', country = 'U.S.')
gold_df


# In[ ]:


plot_variable(
    var_label=['m14076a','m14076b', 'm14076c'],
    col=['#b79e0e', '#e8c70d', '#ffd800'],
    file_name = 'gold_stock',
    graph_title = 'U.S. Monetary Gold Stock')


# **PROBLEM!**
# <br><br>
# **See units feature**

# In[ ]:


gold_df.units


# In[ ]:


old_dict = gold_df.data[2]
new_dict = {}
for x in range(len(old_dict)):
    new_dict[list(old_dict.keys())[x]] = list(old_dict.values())[x] / 1000
    
doc_df = doc_df.set_value(3155,'data', new_dict)
# FOR CONSISTENCY, I ALSO CHANGE UNITS COLUMNS ACCORDINGLY 
doc_df = doc_df.set_value(3155,'units', 'BILLIONS OF DOLLARS')


# In[ ]:


keyw_in_country(keyw='Gold Stock', country = 'U.S.').units


# In[ ]:


plot_variable(
    var_label=['m14076a','m14076b', 'm14076c'],
    col=['#ccbd70', '#f9e161', '#ffd800'],
    file_name = 'gold_stock',
    graph_title = 'U.S. Monetary Gold Stock')


# > For more information on why gold stock skyrocketed in 1933-1934 : <br><br>
# >>https://en.wikipedia.org/wiki/Gold_Reserve_Act

# 

# **Visualize Nones and adjust our wrangling algorithm accordingly. Re-visualize Nones.**

# In[ ]:


import missingno as msno

msno.matrix(df=doc_df, figsize=(20, 14), color=(0.31, 0.158, 0.69))

for x in range(len(doc_df)):
    if 'NBER SERIES: ' in doc_df['documentation'][x]:
        doc_df['nber_series'][x] = ''.join(
            doc_df['documentation'][x].split('NBER SERIES: ')[1].split('\nAREA COVERED')[0]
        ).split()[0]

    if 'AREA COVERED: ' in doc_df['documentation'][x]:
        doc_df['area_covered'][x] = ''.join(
            doc_df['documentation'][x].split('AREA COVERED: ')[1].split('\nUNITS')[0]
        ).split()[0]

    if 'UNITS: ' in doc_df['documentation'][x]:
        doc_df['units'][x] = ''.join(
            doc_df['documentation'][x].split('UNITS: ')[1].split('\nANNUAL COVERAGE')[0]
        ).split()[0]

    if 'ANNUAL COVERAGE: ' in doc_df['documentation'][x]:
        doc_df['annual_coverage'][x] = ''.join(
            doc_df['documentation'][x].split('ANNUAL COVERAGE: ')[1].split('\nQUARTERLY COVERAGE')[0]
        ).split()[0]

    if 'QUARTERLY COVERAGE: ' in doc_df['documentation'][x]:
        doc_df['quarterly_coverage'][x] = ''.join(
            doc_df['documentation'][x].split('QUARTERLY COVERAGE: ')[1].split('\nMONTHLY COVERAGE')[0]
        ).split()[0]

    if 'MONTHLY COVERAGE: ' in doc_df['documentation'][x]:
        doc_df['monthly_coverage'][x] = ''.join(
            doc_df['documentation'][x].split('MONTHLY COVERAGE: ')[1].split('\nSEASONAL ADJUSTMENT')[0]
        ).split()[0]
    
    if 'SEASONAL ADJUSTMENT: ' in doc_df['documentation'][x]:
        doc_df['seasonal_adjustment'][x] = ''.join(
            doc_df['documentation'][x].split('SEASONAL ADJUSTMENT: ')[1].split('\nSOURCE')[0]
        ).split()[0]

    if 'SOURCE: ' in doc_df['documentation'][x]:
        doc_df['source'][x] = ''.join(
            doc_df['documentation'][x].split('SOURCE: ')[1].split('\nNOTES')[0]
        )

    if 'NOTES: ' in doc_df['documentation'][x]:
        doc_df['notes'][x] = ''.join(
            doc_df['documentation'][x].split('NOTES: ')[1:]
        )

msno.matrix(df=doc_df, figsize=(20, 14), color=(0.31, 0.458, 0.89))


# In[ ]:


doc_df.to_csv(path_or_buf='nber.csv')


# **Final Look**

# In[ ]:


doc_df.head(3).transpose()

