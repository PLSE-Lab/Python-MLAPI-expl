#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
from IPython.display import HTML


# In[2]:


HTML('''<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
The raw code for this IPython notebook is by default hidden for easier reading.
To toggle on/off the raw code, click <a href="javascript:code_toggle()">here</a>.''')


# # Table of content:
# ## [Introduction](#intro)
# ## [Method](#method)
# ## [Additional data](#add-data)
# ## [Localisation of borrowers](#localisation)
# ### [Localisation of borrowers by country](#localisation-country)
# ### [Localisation of borrowers by sector](#localisation-sector)
# ## [Welfare assessment](#welfare-assessment)
# ## [Classification of loan use](#loan-use-classification)

# In[ ]:





# # Introduction <a class="anchor" id="intro"></a>

# # Method <a class="anchor" id="method"></a>
# -  absolute values
# -  relative values

# In[ ]:





# # Additional data <a class="anchor" id="add-data"></a>
# -  World Bank: poverty data, 

# In[3]:


import sys  
#reload(sys)  
#sys.setdefaultencoding('utf8')

import pandas as pd
from pandas-datareader import wb
pd.set_option("display.max_colwidth",200)

import matplotlib.pyplot as plt
import seaborn as sns


import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as ff


# In[4]:


#%% define function to plot figures from series

def plot_world_map_from_series(df, filename = None):
    
    df = df.reset_index()
    # by default, first columns is country, second columns is value
    
    data = [ dict(
        type = 'choropleth',
        locations = df.ix[:,0],
        locationmode = 'country names',
        z = df.ix[:,1].astype('float'),
#        text = df.ix[:,0].str.cat( df.ix[:,1].astype('str'), sep = ' '),
        text = df.ix[:,0], 
#        colorscale = 'Blues',
        autocolorscale = True,
        reversescale = False,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) ),
        colorbar = dict(
            tickprefix = '',
            #title = df.columns[1]
        )
      ) ]

    layout = dict(
        title = df.columns[1],
        geo = dict(
            showframe = False,
            showcoastlines = False,
            projection = dict(
                type = 'Mercator'
            )
        )
    )

    if filename == None:
        filename = df.columns[1]

    fig = dict( data=data, layout=layout )
    return py.iplot( fig, validate = True, filename = filename)
    
#%%
def plot_barh_from_series(df, filename = None):
    
    df = df.reset_index()
    # by default, first columns is country, second columns is value
    
    trace = go.Bar(
        y= df.ix[:,0],
        x=df.ix[:,1],
        orientation = 'h',
        marker=dict(
            color=df.ix[:,1],
            autocolorscale = True,
            reversescale = False
        ),
    )
    
    layout = go.Layout(
        title= df.columns[1],
        width=800,
        height=1200,
        )
    data = [trace]
    
    fig = go.Figure(data=data, layout=layout)
    
    if filename == None:
        filename = df.columns[1]
    
    return py.iplot(fig, filename= filename)
#%%
def plot_correlation_matrix(corr, xcols = None, ycols = None, filename = None, title = None):
    # corr is the correlation matrix obtained from a dataframe using pandas
    
    if xcols == None:
        xcols = corr.columns.tolist()
    if ycols == None:
        ycols = corr.columns.tolist()
    
    layout = dict(
        title = title,
        width = 800,
        height = 800,
#        margin=go.Margin(l=100, r=10, b=50, t=50, pad=5),
        margin=go.Margin(l=250, r=50, b=50, t=250, pad=4),
        yaxis= dict(tickangle=-30,
                    side = 'left',
                    ),
        xaxis= dict(tickangle=-30,
                    side = 'top',
                    ),
    )
    fig = ff.create_annotated_heatmap(
        z=corr.values,
        x= xcols,
        y= ycols,
        colorscale='Portland',
        reversescale=True,
        showscale=True,
        font_colors = ['#efecee', '#3c3636'])
    fig['layout'].update(layout)
    
    if filename == None:
        filename = 'correlation matrix'
    return py.iplot(fig, filename= filename)


# In[5]:


#%% load data from kiva
data_kvloans = pd.read_csv("../input/kiva_loans.csv")


# In[6]:


data_kvmpi = pd.read_csv("../input/kiva_mpi_region_locations.csv")
data_kvmpi.dropna(axis= 0, thresh = 2, inplace = True)


# In[7]:


data_kvloans.head()


# In[8]:


data_kvloans.tail()


# In[9]:


print(data_kvloans.columns.tolist())


# In[10]:


# column of year when loan funded
data_kvloans['year']  = pd.to_datetime(data_kvloans['date']).dt.year.astype(str)

# change name of Cote d'Ivoire
data_kvloans['country'] = data_kvloans['country'].str.replace("Cote D'Ivoire","Cote d'Ivoire")


# In[11]:


idlist = ['SP.POP.TOTL', 'NY.GDP.PCAP.CD']
idname = ['Total population', 'GDP per capita (current US$)']

data_wb = wb.download( indicator= idlist, country='all', start='2014', end='2017')
data_wb.columns = idname


# In[12]:


data_wb.head(10)


# In[13]:


data_wb.tail(10)


# data poverty

# In[15]:


idlist = ['SI.DST.FRST.10', 'SI.DST.FRST.20', 'SI.POV.GAPS','SI.POV.NAGP', 'SI.POV.UMIC.GP', 'SI.POV.DDAY', 'SI.POV.LMIC', 'SI.POV.UMIC', 'SI.POV.NAHC' ]

data_poverty = wb.download( indicator= idlist, country='all', start='2007', end='2017')

data_poverty.columns = wb.get_indicators().set_index('id').ix[idlist]['name'].values

data_poverty.head()
data_poverty.tail()


# # Localisation of borrowers <a class="anchor" id="localisation"></a>

# In[16]:


#%% put all necessary data into a single frame
df1 = data_kvloans.groupby(by = ['country', 'year'])['loan_amount'].mean()
df1.name = 'Mean loan amount'

df2 = data_kvloans.groupby(by = ['country', 'year'])['loan_amount'].sum()
df2.name = 'Total loan amount'

# number of loans in each country in each years
df3 = data_kvloans.groupby(by = ['country', 'year'])['loan_amount'].size()
df3.name = '# loans'

df = pd.concat( [data_wb, df1, df2, df3 ], axis = 1, join = 'outer' )

df['# loans in 10000 inhabitants'] = 10000.*df['# loans'] /df['Total population']

df['Mean loan amount / GDP per capita (current US$) (%)'] = 100.* df['Mean loan amount'] / df['GDP per capita (current US$)']


# In[17]:


df.head()


# In[18]:


df.tail(10)


# Negative correlation between number of loans and GDP/capita and mean loan amount

# In[19]:


plot_correlation_matrix( df.corr().round(2))


# In[20]:


HTML('''<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
The raw code for this IPython notebook is by default hidden for easier reading.
To toggle on/off the raw code, click <a href="javascript:code_toggle()">here</a>.''')


# # Localisation of borrowers by country <a class="anchor" id="localisation-country"></a>

# mean number of loans per year

# In[21]:


dfaux = df.reset_index().groupby('country')['# loans'].mean()
dfaux = dfaux.fillna(value = 0)
plot_barh_from_series( dfaux.sort_values(ascending = False).head(40) )

l1 = dfaux.sort_values(ascending = False).index.tolist()


# In[22]:


df.ix[ l1[:5]]


# In[23]:


plot_world_map_from_series( dfaux )


# total population

# In[24]:


dfaux = df.reset_index().groupby('country')['Total population'].mean().loc[l1]

plot_barh_from_series( dfaux.head(30) )


# number of loans per 10000 inhabitants

# In[25]:


dfaux = df.reset_index().groupby('country')['# loans in 10000 inhabitants'].mean().loc[l1]

plot_barh_from_series( dfaux.head(30) )


# plot world map without Samoa (an outlier in the middle of the Pacific ocean) and El Salvador
# 
# localisation of loans in South America, South East Asia, certain parts of Africa

# In[26]:


dfaux = df.reset_index().groupby('country')['# loans in 10000 inhabitants'].mean()
# remove Samoa (too different)
dfaux = dfaux[ ~dfaux.index.isin([ 'Samoa', 'El Salvador'] ) ]
# replace nan with 0 (no loan)
dfaux = dfaux.fillna(value = 0)
plot_world_map_from_series( dfaux )


# In[27]:


df.ix[ ['Samoa', 'El Salvador']]


# number of loans vs. size of poor population in the country

# In[28]:


# join mean values of two dataframes (over different periods, assumption that the trend stays the same for poverty data)
df_pov = pd.concat([df.reset_index().groupby('country').mean(), data_poverty.reset_index().groupby('country').mean(), data_kvmpi.groupby('country')['MPI'].mean() ], axis = 1, join = 'outer')


# In[29]:


df_pov[u'# loans in 10000  poor inhabitants'] = 10000. * df_pov[u'# loans'] / (df_pov[u'Total population'] * df_pov[u'Poverty headcount ratio at national poverty lines (% of population)'] / 100.)


# In[30]:


plot_barh_from_series( df_pov.loc[l1[:30],u'# loans in 10000  poor inhabitants'])


# not different from above figure (# loans in 10000 inhabitants)

# plot_world_map_from_series( df_pov[u'# loans in 10000  poor inhabitants'] , filename = '# loans in 10000 poors')

# In[31]:


dfaux = df.reset_index().groupby('country')['Mean loan amount'].mean()
dfaux = dfaux.fillna(value = 0.)
dfaux.drop(['Virgin Islands (U.S.)'], inplace = True)


# In[32]:


plot_barh_from_series( dfaux.ix[l1[:30] ] )


# In[33]:


plot_world_map_from_series( dfaux.loc[ ~(dfaux.index == "Cote d'Ivoire")]) # remove Cote d'Ivoire


# In[34]:


data_kvloans.loc[  data_kvloans['country'] == "Cote d'Ivoire"]


# mean loan amount / GDP per capita

# In[35]:


dfaux = df.reset_index().groupby('country')['Mean loan amount / GDP per capita (current US$) (%)'].mean().loc[l1]
dfaux = dfaux.fillna(value = 0)

plot_barh_from_series( dfaux.ix[l1[:70]] , filename = 'mean loan vs. gdp per capita barh')


# In[36]:


plot_world_map_from_series( dfaux, filename = 'mean loan vs. gdp per capita map')


# strong localisation of mean loan amount / GDP per capita in Africa

# In[37]:


xcols = ['Total population','GDP per capita (current US$)',u'Income share held by lowest 10%',u'Income share held by lowest 20%',u'Poverty gap at $1.90 a day (2011 PPP) (%)',u'Poverty gap at national poverty lines (%)',u'Poverty gap at $5.50 a day (2011 PPP) (% of population)',u'Poverty headcount ratio at $1.90 a day (2011 PPP) (% of population)',u'Poverty headcount ratio at $3.20 a day (2011 PPP) (% of population)',u'Poverty headcount ratio at $5.50 a day (2011 PPP) (% of population)',u'Poverty headcount ratio at national poverty lines (% of population)','MPI']
ycols = ['Total loan amount', '# loans','# loans in 10000 inhabitants', '# loans in 10000  poor inhabitants', 'Mean loan amount', 'Mean loan amount / GDP per capita (current US$) (%)']
plot_correlation_matrix( df_pov.corr().loc[ycols, xcols].round(2), xcols = xcols, ycols = ycols, filename = 'correlation matrix poverty' )


# The poorer the population is, the smaller the total number (as well as its ratio in 10000 poor inhabitants) and total amount of KIVA loans  become. 
# 
# However, for those in poor countries who receive KIVA loans, the mean values of loans is positively correlated with the poverty level, thus these loans represent a higher percentage with respect to the country GDP per capita.
# 
# As a summary, people at lower welfare levels get less KIVA loans (in total number of loans, in total amount of loans). These loans represent an important ratio with respect to their average income.

# In[38]:


HTML('''<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
The raw code for this IPython notebook is by default hidden for easier reading.
To toggle on/off the raw code, click <a href="javascript:code_toggle()">here</a>.''')


# # Localisation of borrowers by sector <a class="anchor" id="localisation-sector"></a>

# In[39]:


dfkv = data_kvloans[ [u'country', u'year', u'activity', u'sector', u'loan_amount']]


# In[40]:


trace = go.Pie(sort = False,labels=dfkv.groupby('sector').size().index.tolist(), values=list(dfkv.groupby('sector').size().values))
fig = {
    'data': [trace],
    'layout': {'title': '# loans by sector'}
     }

py.iplot(fig)


# In[41]:


trace = go.Pie(sort = False,labels=dfkv.groupby('sector')['loan_amount'].sum().index.tolist(), values=list(dfkv.groupby('sector')['loan_amount'].sum().values))
fig = {
    'data': [trace],
    'layout': {'title': 'Total loan amount by sector'}
     }

py.iplot(fig)


# In[42]:


trace = go.Pie(sort = False, labels=dfkv.groupby('sector')['loan_amount'].mean().index.tolist(), values=list(dfkv.groupby('sector')['loan_amount'].mean().values))
fig = {
    'data': [trace],
    'layout': {'title': 'Mean loan amount by sector'}
     }

py.iplot(fig)


# conclusions: agriculture, food, retail mean loan amount close to other sectors, however, number of loans much greater , thus total loan amount greater. no localisation in sector?
# 
# mean loan amount on all sectors are close (even smaller than), whether for entertainment, agriculture, food, retail, clothing, health or other stuffs
# 
# mean values can still be biased (few big loans might strongly increase mean value). consider also median values

# In[43]:


#%% % of each sector in each country

dfaux = dfkv.groupby(['country', 'sector']).sum()
dfaux = dfaux.unstack(level = -1)
dfaux.fillna(value = 0., inplace = True)


dfa = 100 * dfaux / dfaux.sum(axis = 0)

sectors = ['Agriculture', 'Arts', 'Clothing', 'Construction', 'Education', 'Entertainment', 'Food', 'Health', 'Housing', 'Manufacturing', 'Personal Use', 'Retail', 'Services', 'Transportation', 'Wholesale']

for sector in sectors:
    print('Sector ' + sector + ' : Loan amount per country (% of worldwide total loan amount in this sector')

    dfaux = dfa.xs(sector, axis = 1, level = 1)
    dfaux.columns = [ 'Loan amount (% of worldwide total loan in the sector)']
    
    country_list = data_wb.index.get_level_values('country').unique().tolist()
    
    for i in country_list:
        if not(i in dfaux.index.tolist()):
            dfaux = dfaux.append( pd.DataFrame( index = [i], data = [0.0], columns = dfaux.columns ) )
    dfaux.drop(['Virgin Islands (U.S.)'], inplace = True)
    plot_world_map_from_series( dfaux )


# In[44]:


sectors = ['Agriculture', 'Arts', 'Clothing', 'Construction', 'Education', 'Food', 'Health', 'Housing', 'Manufacturing', 'Retail', 'Services', 'Transportation', 'Wholesale']


for sector in sectors:

    print('Sector ' + sector + ' : Distribution of loans in terms of number of loans, mean loan amount, mean amount / GDP per capita')

    # total loan amount per country per sector
    df1 = dfkv[ dfkv['sector'] == sector ].groupby(by = ['country', 'year'])['loan_amount'].sum()
    df1.name = 'Total loan amount'
    # mean loan amount per country per sector
    df2 = dfkv[ dfkv['sector'] == sector ].groupby(by = ['country', 'year'])['loan_amount'].mean()
    df2.name = 'Mean loan amount'
    
    # number of loans per country per sector
    df3 = dfkv[ dfkv['sector'] == sector ].groupby(by = ['country', 'year'])['loan_amount'].size()
    df3.name = '# loans'
    
    
    
    df = pd.concat([data_wb, df1, df2, df3 ], axis = 1, join = 'outer')

    df['# loans in 10000 inhabitants'] = 10000.*df['# loans'] /df['Total population']
    
    df['Mean loan amount / GDP per capita (current US$) (%)'] = 100.* df['Mean loan amount'] / df['GDP per capita (current US$)']
    

    # Mean loan amount /year
    dfaux = df.reset_index().groupby('country')['Mean loan amount'].mean()
    dfaux = dfaux.fillna(value = 0.)
    dfaux.drop(['Virgin Islands (U.S.)'], inplace = True)
    plot_barh_from_series( dfaux.sort_values(ascending = False).head(10) )
    plot_world_map_from_series( dfaux)
    
    # Mean loan amount / gdp / year
    dfaux = df.reset_index().groupby('country')['Mean loan amount / GDP per capita (current US$) (%)'].mean()
    dfaux = dfaux.fillna(value = 0)
    dfaux.drop(['Virgin Islands (U.S.)'], inplace = True)
    plot_world_map_from_series( dfaux, filename = 'Mean loan vs. gdp per capita')
    
    # mean number of loans per year
    dfaux = df.reset_index().groupby('country')['# loans'].mean()
    dfaux = dfaux.fillna(value = 0)
    dfaux.drop(['Virgin Islands (U.S.)'], inplace = True)
    plot_world_map_from_series( dfaux, filename = 'Mean number of loans per year' )
    
    # total number of loans / 10000 habitants in all years
    dfaux = df.reset_index().groupby('country')['# loans in 10000 inhabitants'].mean()
    # remove Samoa (too different)
    dfaux = dfaux[ ~dfaux.index.isin([ 'Samoa'] ) ]
    # replace nan with 0 (no loan)
    dfaux = dfaux.fillna(value = 0)
    dfaux.drop(['Virgin Islands (U.S.)'], inplace = True)
    plot_world_map_from_series( dfaux , filename = 'N# loans in 10000 inhabitants')


# In[45]:


HTML('''<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
The raw code for this IPython notebook is by default hidden for easier reading.
To toggle on/off the raw code, click <a href="javascript:code_toggle()">here</a>.''')


# # Welfare assessment <a class="anchor" id="welfare-assessment"></a>

# ## Classification of loan use

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# 

# In[ ]:




