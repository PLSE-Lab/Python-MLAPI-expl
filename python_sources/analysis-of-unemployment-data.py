#!/usr/bin/env python
# coding: utf-8

# Youth unemployment is an evergrowing monster thats looking at us right in our faces and their is very less that is being done to alleviate the problem. The problem has aggravated in the recent times with the economic slowdown taking jobs of numerous people and drying up new opportunities for them with regular news of companies focussing on restructring their organization and countless layoff's as a consequence. If this problem is not dealt with, with firm hand and a proper direction then our youths will be having hardships thus impacting their future.
# 
# This little analysis takes a first look at the unemployment data collated by World bank from around the world and collected over a 5 year period from 2010 to 2014. The data includes individual nations as well as nations grouped according to income level and supranational groups such as Arab World, EU etc. A lot more manipulations could be done on the data for a lot in-depth analysis.
# 
# **P.S.:This is my first hand at general analysis using Python other than the specialized scientific data (Pressure, Temperature, Velocity, Turbulence etc. data using MATLAB etc.) that I am used too so any long lines of codes or any thing that can be done in short cuts should please be forgiven. A lot of work in here is inspired from the works of Anisotropic (https://www.kaggle.com/arthurtok/d/sovannt/world-bank-youth-unemployment/generation-unemployed-interactive-plotly-visuals) and whose beautifully done analysis is an inspiration for this work. Do check out that too**
# 
# **Comments/Advice/Suggestions are welcome.**

# The first step is the import of relevant libraries and tools to be used for analysis

# In[ ]:


# Import the relevant libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

get_ipython().run_line_magic('matplotlib', 'inline')


# Read the file and take a first look at the various columns and the values contained therein. Also do check the size of the matrix/dataframe imported that you are going to work on.

# In[ ]:


df=pd.read_csv('../input/API_ILO_country_YU.csv')
df.head(5)


# In[ ]:


df.shape


#  A look into the various columns and the country names is a preliminary first step into analyzing the data. Looking into column names gives an idea of what might be contained in the data and a look into the countries here will help us determine if any countries are repeated. Since only 219 rows are there a manual check of each row is possible in this case. Also we have to check whether all the data is only for individual nations or also from Supranational organizations (like UN, EU, SAARC, G8, G20 to name a few) or from banking institutions (IBD,ADB,IMF,income categorization etc.). This will help us in the analysis as it won't make sense to compare countries with organization groups (this is akin to apple vs oranges). A detailed analysis will look into the data vis-a-vis at the level of individual countries and as part of an organization. A certain extent of the success of these organizations can be judged from the unemployment statistics as to how successful these organizations in living up to the aspirations of the youth, in negotiating with each other for providing job opportunities to the young population of the group members. Though it will only provide one facet or one metric of success/failure of an organization yet it will bring into fore the effects of these organizations.
#  
# The process that will be followed in this project is to first divide the original datasets into a number of constituent smaller datasets each catering to one group viz. Individual Counteries, Supranational Groups and finally the income groups. Then we can have a peek on how unemployment has affected the world around us and which regions/counteries are most and least afflicted by it 
# and which have been most and least successful in tiding over the unemployment crisis. 

# In[ ]:


df.columns


# If we take a quick glance at the imported dataset it is revealed that a lot of enteries in the 'Country Name' and 'Country Code' columns of the dataset are actually supranational entities (EU, Euro Area, Arab World etc.) and groups based on Income/Banking Institutions classifications (IDA, IBRD, High Income) apart from several other enteries. For our analysis purposes we will segregate the original dataset acording to the type of enteries i.e. Country, Supranational and Income Groups. For the purpose of this segregation we will make a list of all 'non-country' entities and then drop them from the initial dataset to obtain a country only data. The other datasets too will then be devised using similar lists. 

# In[ ]:


non_country_list=['Arab World','Central Europe and the Baltics','Caribbean small states','East Asia & Pacific (excluding high income)',
                 'Early-demographic dividend', 'East Asia & Pacific','Europe & Central Asia (excluding high income)',
                 'Europe & Central Asia','Euro area','European Union','Fragile and conflict affected situations','High income',
                 'Heavily indebted poor countries (HIPC)','IBRD only', 'IDA & IBRD total', 'IDA total','IDA blend','IDA only',
                 'Latin America & Caribbean (excluding high income)','Latin America & Caribbean','Least developed countries: UN classification', 
                 'Low income','Lower middle income', 'Low & middle income','Late-demographic dividend','Middle East & North Africa',
                 'Middle income','Middle East & North Africa (excluding high income)','North America','OECD members','Other small states',
                 'Pre-demographic dividend','Post-demographic dividend','South Asia','Sub-Saharan Africa (excluding high income)',
                 'Sub-Saharan Africa','Small states','East Asia & Pacific (IDA & IBRD countries)',
                 'Europe & Central Asia (IDA & IBRD countries)','Latin America & the Caribbean (IDA & IBRD countries)',
                 'Middle East & North Africa (IDA & IBRD countries)','South Asia (IDA & IBRD)',
                 'Sub-Saharan Africa (IDA & IBRD countries)','Upper middle income','World']


# In[ ]:


df_non_country=df[df['Country Name'].isin(non_country_list)]


# In[ ]:


df_non_country.head()


# In[ ]:


df_non_country.shape


# In[ ]:


index=df_non_country.index


# In[ ]:


df_country=df.drop(index)


# In[ ]:


df_country.head()


# In[ ]:


df_country.shape


# # Exploratory Analysis of the data
# 
# We will begin with an exploratory analysis of the data by making several basic statistical plots such as box plots, scatter plots etc. of the data from individual countries and having a quick glance at how the various countries compare against one another. This first look into the data will reveal several basic information about the data as to the max-min values, the average and median values etc. to name a few.

# # Box Plots
# 
# The box plot of all the individual counteries for the 5 year duration during which the data is collected is shown below.

# In[ ]:


x_data = ['2010', '2011','2012', '2013','2014']

y0 = df_country['2010']
y1 = df_country['2011']
y2 = df_country['2012']
y3 = df_country['2013']
y4 = df_country['2014']

y_data = [y0,y1,y2,y3,y4]

colors = ['rgba(93, 164, 214, 0.5)', 'rgba(255, 144, 14, 0.5)', 'rgba(44, 160, 101, 0.5)',
          'rgba(255, 65, 54, 0.5)', 'rgba(207, 114, 255, 0.5)']

traces = []

for xd, yd, color in zip(x_data, y_data, colors):
        traces.append(go.Box(
            y=yd,
            name=xd,
            boxpoints='all',
            whiskerwidth=0.2,
            fillcolor=color,
            marker=dict(
                size=2,
            ),
            boxmean=True,    
            line=dict(width=1),
        ))

layout = go.Layout(
    title='Distribution of Unemployment Data',
    xaxis=dict(
        title='Year'
    ),
    yaxis=dict(
        title='Unemployment Rate (%)',
        autorange=True,
        showgrid=True,
        zeroline=False,
        dtick=5,
        gridcolor='rgb(255, 255, 255)',
        gridwidth=1,
#        zerolinecolor='rgb(255, 255, 255)',
#        zerolinewidth=2,
    ),
    margin=dict(
        l=40,
        r=30,
        b=80,
        t=100,
    ),
    paper_bgcolor='rgb(243, 243, 243)',
    plot_bgcolor='rgb(243, 243, 243)',
    showlegend=False
)

fig = go.Figure(data=traces, layout=layout)
py.iplot(fig)


# ** Inferences from Box Plots **
# 
# The box plot along with the scatter data helps us to visualize the data in a better way. From the box plots it can be easily seen that the mean of the data over the years remains almost constant with 18.285 % in 2010 to 18.2546 % in 2014. Aside from that it is readily observed that median in each case is lower than the average suggesting that unemployment in majority of the countries is lower than the average unemployment rate. The median range however varies from 16.05 % (2010) to 14.3 % (2014) peaking in 2010 thus suggesting that overall in majority of the countries the unemployment rate has gone down, though only marginally. Most of the data is in the IQR (Inter Quartile Range) throughout the years that data has been collected with only a small number of cases each year that lie beyond the upper bound suugesting presence of severe unemployment in countries represented by these. These are usually the outliers in the data evidenced from their being small in numbers (out of 174 countries only around 5 of these have unemployment rates outside the upper bound in 2010)

# 
# 
# # Scatter Plots
# 
# Next I am going to have a deeper look into data by making scatter plots from the data where we will be comparing countries such that companies having higher rate of unemployment will have a bigger bubble than the one suffering from a lower unemployment rate. Although the plots have been made for all the years that the data has been collected, discussion will only be done for 2010 and 2014 as similar observations could be made from the other plots.  

# In[ ]:


l=[]
trace0= go.Scatter(
        y= df_country['2010'],
        mode= 'markers',
        name='Unemployment (%)',
        marker= dict(size= df_country['2010'].values,
                    line= dict(width=1),
                    color= df_country['2010'].values,
                    opacity= 0.7,
                    colorscale='Portland',
                    showscale=True),
        text= df_country['Country Name'].values) # The hover text goes here... 
l.append(trace0);

layout= go.Layout(
    title= 'Scatter plot of unemployment rates in 2010',
    hovermode= 'closest',
    xaxis= dict(
#        title= 'Pop',
        ticklen= 5,
        zeroline= False,
        gridwidth= 2,
    ),
    yaxis=dict(
        title= 'Unemployment Rate (%)',
        ticklen= 5,
        gridwidth= 2,
    ),
    showlegend= False,
)
fig= go.Figure(data=l, layout=layout)
py.iplot(fig)

# Pre-defined color scales - 'pairs' | 'Greys' | 'Greens' | 'Bluered' | 'Hot' | 'Picnic' | 'Portland' | 'Jet' | 
# 'RdBu' | 'Blackbody' | 'Earth' | 'Electric' | 'YIOrRd' | 'YIGnBu'

# Chose Portland because it seems to be the best colorscale


# In[ ]:


l1=[]
trace1= go.Scatter(
        y= df_country['2011'],
        mode= 'markers',
        name='Unemployment (%)',
        marker= dict(size= df_country['2011'].values,
                    line= dict(width=1),
                    color= df_country['2011'].values,
                    opacity= 0.7,
                    colorscale='Portland',
                    showscale=True),
        text= df_country['Country Name'].values) # The hover text goes here... 
l1.append(trace1);

layout= go.Layout(
    title= 'Scatter plot of unemployment rates in 2011',
    hovermode= 'closest',
    xaxis= dict(
#        title= 'Pop',
        ticklen= 5,
        zeroline= False,
        gridwidth= 2,
    ),
    yaxis=dict(
        title= 'Unemployment Rate (%)',
        ticklen= 5,
        gridwidth= 2,
    ),
    showlegend= False
)
fig= go.Figure(data=l1, layout=layout)
py.iplot(fig,filename='scatter_plot2011')

# Pre-defined color scales - 'pairs' | 'Greys' | 'Greens' | 'Bluered' | 'Hot' | 'Picnic' | 'Portland' | 'Jet' | 
# 'RdBu' | 'Blackbody' | 'Earth' | 'Electric' | 'YIOrRd' | 'YIGnBu'

# Chose Portland because it seems to be the best colorscale


# In[ ]:


l2=[]
trace2= go.Scatter(
        y= df_country['2012'],
        mode= 'markers',
        name='Unemployment (%)',
        marker= dict(size= df_country['2012'].values,
                    line= dict(width=1),
                    color= df_country['2012'].values,
                    opacity= 0.7,
                    colorscale='Portland',
                    showscale=True),
        text= df_country['Country Name'].values) # The hover text goes here... 
l2.append(trace2);

layout= go.Layout(
    title= 'Scatter plot of unemployment rates in 2012',
    hovermode= 'closest',
    xaxis= dict(
#        title= 'Pop',
        ticklen= 5,
        zeroline= False,
        gridwidth= 2,
    ),
    yaxis=dict(
        title= 'Unemployment Rate (%)',
        ticklen= 5,
        gridwidth= 2,
    ),
    showlegend= False
)
fig= go.Figure(data=l2, layout=layout)
py.iplot(fig,filename='scatter_plot2012')

# Pre-defined color scales - 'pairs' | 'Greys' | 'Greens' | 'Bluered' | 'Hot' | 'Picnic' | 'Portland' | 'Jet' | 
# 'RdBu' | 'Blackbody' | 'Earth' | 'Electric' | 'YIOrRd' | 'YIGnBu'

# Chose Portland because it seems to be the best colorscale


# In[ ]:


l3=[]
trace3= go.Scatter(
        y= df_country['2013'],
        mode= 'markers',
        name='Unemployment (%)',
        marker= dict(size= df_country['2013'].values,
                    line= dict(width=1),
                    color= df_country['2013'].values,
                    opacity= 0.7,
                    colorscale='Portland',
                    showscale=True),
        text= df_country['Country Name'].values) # The hover text goes here... 
l3.append(trace3);

layout= go.Layout(
    title= 'Scatter plot of unemployment rates in 2013',
    hovermode= 'closest',
    xaxis= dict(
#        title= 'Pop',
        ticklen= 5,
        zeroline= False,
        gridwidth= 2,
    ),
    yaxis=dict(
        title= 'Unemployment Rate (%)',
        ticklen= 5,
        gridwidth= 2,
    ),
    showlegend= False
)
fig= go.Figure(data=l3, layout=layout)
py.iplot(fig,filename='scatter_plot2013')

# Pre-defined color scales - 'pairs' | 'Greys' | 'Greens' | 'Bluered' | 'Hot' | 'Picnic' | 'Portland' | 'Jet' | 
# 'RdBu' | 'Blackbody' | 'Earth' | 'Electric' | 'YIOrRd' | 'YIGnBu'

# Chose Portland because it seems to be the best colorscale


# In[ ]:


l4=[]
trace4= go.Scatter(
        y= df_country['2014'],
        mode= 'markers',
        name='Unemployment (%)',
        marker= dict(size= df_country['2014'].values,
                    line= dict(width=1),
                    color= df_country['2014'].values,
                    opacity= 0.7,
                    colorscale='Portland',
                    showscale=True),
        text= df_country['Country Name'].values) # The hover text goes here... 
l4.append(trace4);

layout= go.Layout(
    title= 'Scatter plot of unemployment rates in 2014',
    hovermode= 'closest',
    xaxis= dict(
#        title= 'Pop',
        ticklen= 5,
        zeroline= False,
        gridwidth= 2,
    ),
    yaxis=dict(
        title= 'Unemployment Rate (%)',
        ticklen= 5,
        gridwidth= 2,
    ),
    showlegend= False
)
fig= go.Figure(data=l4, layout=layout)
py.iplot(fig,filename='scatter_plot2014')

# Pre-defined color scales - 'pairs' | 'Greys' | 'Greens' | 'Bluered' | 'Hot' | 'Picnic' | 'Portland' | 'Jet' | 
# 'RdBu' | 'Blackbody' | 'Earth' | 'Electric' | 'YIOrRd' | 'YIGnBu'

# Chose Portland because it seems to be the best colorscale


# In[ ]:





# **Observations**
# 
# As evidenced from the scatter plots it is observed that several countries like South Africa, Bosnia and Herzegovina which were already suffering from high unemployment rates in 2010 are still among the countries with highest unemployment rates even in 2014. However, the conditions of some european countries has been going from bad to worse year on year. Greece which was badly hit in the economic slowdown and had to be bailed out by European Union seems to be badly hit by rising unemployment along with Spain and Italy among others. Even from our scatter plots we can see that majority of countries still are in the lower ranges of unemployment suggesting that all is not lost for the youths everywhere.

# # Horizontal Bar Plots-All Countries & Best/Worst Performers

# Merely knowing where each country stood over the years or with respect to each other is not enough. We should also be able to deduce from the data how unemployment has changed over the years in a country. This will tell us how the government's/economies of respective countries have managed the crisis of unemployment looming in their backyard and how they have countered it in shorter term (2 yr periods) and on a longer term (5 year period). To this end 3 additional columns have been added to the data for the accounting the percentage change in a country's unemployment rate from 2010 to 2012, change from 2012 to 2014 and ultimately the change in unemployment from 2010 to 2014.

# In[ ]:


df_country['2014-2012 change']=df_country['2014']-df_country['2012']


# In[ ]:


df_country['2012-2010 change']=df_country['2012']-df_country['2010']


# In[ ]:


df_country.head()


# In[ ]:


# Tried with Plotly now going with seaborn
twoyearchange201412_bar, countries_bar1 = (list(x) for x in zip(*sorted(zip(df_country['2014-2012 change'], df_country['Country Name']), 
                                                             reverse = True)))

twoyearchange201210_bar, countries_bar2 = (list(x) for x in zip(*sorted(zip(df_country['2012-2010 change'], df_country['Country Name']), 
                                                             reverse = True)))

# Another direct way of sorting according to values is creating distinct sorted dataframes as in below commented ways and then
# passing their values directly as in below mentioned code to achieve the same effect as by above mentioned method.

# df_country_sorted=df_country.sort(columns='2014-2012 change',ascending=False)
# df_country_sorted.head()


sns.set(font_scale=1) 
fig, axes = plt.subplots(1,2,figsize=(20, 50))
colorspal = sns.color_palette('husl', len(df_country['2014']))
sns.barplot(twoyearchange201412_bar, countries_bar1, palette = colorspal,ax=axes[0])
sns.barplot(twoyearchange201210_bar, countries_bar2, palette = colorspal,ax=axes[1])
axes[0].set(xlabel='%age change in Youth Unemployment Rates', title='Net %age change in Youth Unemployment Rates between 2012-2014')
axes[1].set(xlabel='%age change in Youth Unemployment Rates', title='Net %age change in Youth Unemployment Rates between 2010-2012')
fig.savefig('output.png')


# In[ ]:


df_country['2014-2010 change']=df_country['2014']-df_country['2010']


# In[ ]:


def top_successful_1(df,n=10,column='2014-2010 change'):
    return df.sort_index(by=column,ascending=True).head(n)


# In[ ]:


def top_failure_1(df,n=10,column='2014-2010 change'):
    return df.sort_index(by=column,ascending=False).head(n)


# In[ ]:


top15=top_successful_1(df_country,n=15)


# In[ ]:


bottom15=top_failure_1(df_country,n=15)


# In[ ]:


sns.set(font_scale=1.4) 
fig, axes = plt.subplots(1,2,figsize=(25, 20))
colorspal = sns.color_palette('husl', len(top15['2014']))
sns.barplot(top15['2014-2010 change'], top15['Country Name'], palette = colorspal,ax=axes[0])
sns.barplot(bottom15['2014-2010 change'], bottom15['Country Name'], palette = colorspal,ax=axes[1])
axes[0].set(xlabel='%age change in Youth Unemployment Rates', title='Top 15 Performers in Controlling Unemployment between 2010-14')
axes[1].set(xlabel='%age change in Youth Unemployment Rates', title='Bottom 15 Performers in Controlling Unemployment between 2010-14')
fig.savefig('output1.png')


# **Inferences from Graphs**
# 
# Although the net change in unemployment rates over 2 year periods (2010-'12 & 2012-'14) has been plotted for all countries, I am mainly going to keep this discussion focussed on the long term change (2010-'14) and that to on countries which are either most successful or an utter failure in dealing with rising unemployment rates. These long term rates is basically a mirror to the willingness/unwillingness of the particular country's govt. to achieve sustainable growth in employment opportunities to counter the rising unemployment. 
# 
# > **Top Successful Countries**
# 
# > The horizontal bar plot on the left hand side shows the top 15 countries which were most successful in dealing with unemployment in their countries over the period of 2010-14. As can be seen most of these countries are in europe with estonia being the most successful in dealing with unemployment having a negative growth rate of -16.29 % although it's unemployment rate still stood at 17 % which is still higher than most other countries. the top 5 countries are Estonia, Latvia, Lithuania, Moldova and Ghana. A special mention to the unemployment rate in USA which is able to achieve a reduction of about 5 % in it's unemployment rate from 2010-14.  
# 
# 
# > **Least Successful Countries**
# 
# > Here again the list of least successful countries is dominated by european countries with countries like Greece, Spain and Italy reeling under high rates of unemployment among it's people. Most of these european countries have seen a double digit increase in their unemployment rates with situations reaching critical proportions in countries like greece with an unemployment rate of ~54 % in 2014 and spain suffering under a 57 % unemployment rate. All the countries in this list present a state of gloomy future for their youths with the respective authorities unable to rein in the unemployment monster. 
# 
# Note: A higher value in the top 15 successful and bottom 15 failures in no way suggest that there are aren't other countries where unemployment isn't as dangerous as in these countries. Take for example South africa which was already having an unemployment rate of 50.8 % in 2010 which jumped to 52.6 % in 2014, making their 5 year change a lowly ~3 %, as compared to 22 % increase for greece taking it to 53.9 %. What this bar plots indicate are the countries most capable/ill-equipped to deal with unemployment.

# # Visualization on Maps
# 
# A picture is worth a thousand words (not sure it was exactly this) cannot be more apt when you can associate the names of places with their location on a map. A map containing all the data is much more visually appealing then rows and rows of endless data. A small effort is made here to present all the possible data on a world map. 

# In[ ]:


# Plotting 2010 World Unemployment Data Geographically
data = [ dict(
        type = 'choropleth',
        locations = df_country['Country Code'],
        z = df_country['2010'],
        text = df_country['Country Name'],
        colorscale = 'Reds',
        autocolorscale = False,
        reversescale = False,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) ),
        colorbar = dict(
            autotick = False,
            title = 'Unemployment (%)'),
      ) ]

layout = dict(
    title = 'Unemployment around the globe in 2010',
    geo = dict(
        showframe = True,
        showcoastlines = True,
        showocean = True,
        #oceancolor = 'rgb(0,255,255)',
        oceancolor = 'rgb(222,243,246)',
        projection = dict(
            type = 'Mercator'
        )
    )
)

fig = dict( data=data, layout=layout )
py.iplot( fig, validate=False,filename='world2010')

# Colorscale Sets the colorscale and only has an effect if `marker.color` is set to a numerical array. 
# Alternatively, `colorscale` may be a palette name string of the following list: Greys, YlGnBu, Greens, YlOrRd, 
# Bluered, RdBu, Reds, Blues, Picnic, Rainbow, Portland, Jet, Hot, Blackbody, Earth, Electric, Viridis


# In[ ]:


# Plotting 2014 World Unemployment Data Geographically
data = [ dict(
        type = 'choropleth',
        locations = df_country['Country Code'],
        z = df_country['2014'],
        text = df_country['Country Name'],
        colorscale = 'Reds',
        autocolorscale = False,
        reversescale = False,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) ),
        colorbar = dict(
            autotick = False,
            title = 'Unemployment (%)'),
      ) ]

layout = dict(
    title = 'Unemployment around the globe in 2014',
    geo = dict(
        showframe = True,
        showcoastlines = True,
        showocean = True,
        #oceancolor = 'rgb(0,255,255)',
        oceancolor = 'rgb(222,243,246)',
        projection = dict(
            type = 'Mercator'
        )
    )
)

fig = dict( data=data, layout=layout)
py.iplot( fig, validate=False,filename='world2014')
# Colorscale Sets the colorscale and only has an effect if `marker.color` is set to a numerical array. 
# Alternatively, `colorscale` may be a palette name string of the following list: Greys, YlGnBu, Greens, YlOrRd, 
# Bluered, RdBu, Reds, Blues, Picnic, Rainbow, Portland, Jet, Hot, Blackbody, Earth, Electric, Viridis


# **YEAR 2010 & YEAR 2014**
# 
# The unemployment rates around the globe has been shown on the maps above using a Red colour scale with dark hues of the color indicating a higher unemployment rate and the lighter hues indicating a lower rate. A quick glance at the maps show that african nations particularly in the southern & northern african region have been suffering from high rates of unemployment as compared to the rest of the countries in the continent in both 2010 and 2014. Asia with two of the world's biggest developing economies i.e. India and China seem to be doing something right and has kept the unemployment rate on the lower side. Russian Federation and United States have some good news as they have witnessed a reduction in the unemployment rates. A matter of concern as observed from the maps is the large increase in unemployment rates across europe particularly southern Europe i.e. the areas of Greece, Italy, Spain etc. All facts included the developing nations seems to be doing better than the developed nations in their quest for controlling unemployment among it's citizens.

# In[ ]:


# Plotting 2014 World Unemployment Data Geographically
data = [ dict(
        type = 'choropleth',
        locations = df_country['Country Code'],
        z = df_country['2014-2010 change'],
        text = df_country['Country Name'],
        colorscale = 'RdBu',
        autocolorscale = False,
        reversescale = False,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) ),
        colorbar = dict(
            autotick = False,
            title = 'Unemployment (%)'),
      ) ]

layout = dict(
    title = 'Net Change in Unemployment around the globe over the 5 year period (2010-14)',
    geo = dict(
        showframe = True,
        showcoastlines = True,
        showocean = True,
        #oceancolor = 'rgb(0,255,255)',
        oceancolor = 'rgb(222,243,246)',
        projection = dict(
            type = 'Mercator'
         )
    )    
)

fig = dict( data=data, layout=layout )
py.iplot( fig, validate=False,filename='WorldChange')

# Colorscale Sets the colorscale and only has an effect if `marker.color` is set to a numerical array. 
# Alternatively, `colorscale` may be a palette name string of the following list: Greys, YlGnBu, Greens, YlOrRd, 
# Bluered, RdBu, Reds, Blues, Picnic, Rainbow, Portland, Jet, Hot, Blackbody, Earth, Electric, Viridis


# **Change in Net Employment Rates over the 5 year period**
# 
# Change in Net unemployment over the 5 year period from 2010-'14 suggest either a small positive increment or a net decrease in unemployment rates in majority of the countries surveyed. Apart from some North African states and Southern European states which seems to have seen a drastic increase in unemployment levels the rest of the world seems to be doing just fine. The South Asian and Oceania region seems to be doing just great with countries like India, China and Australia registering a net increase of just 0.19 % , 1.5 % and 1.7 % respectively over the 5 year period. Even countries like Russia and USA seems to be on the track of recovery with unemployment rates dropping by 3.9 % and ~5% respectively in these countries. However, this net change survey has to be considered keeping in mind the entire perspective as some countries like RSA though seeing only a change of +1.8% over the 5 year period is still one of the countries suffering from extremely high unemployment rates (>50%). All in all the combination of above 3 maps paint a somewhat mixed picture displaying a rampant rise in unemployment rates in some countries while the majority of countries are somewhat successful in mitigating/managing the menace of unemployment.

# # Non Country Data Exploration/Supranational Group/Income Group Exploration

# In[ ]:


supranational_groups=['Arab World','Caribbean small states','East Asia & Pacific','European Union','Latin America & Caribbean',
                      'Middle East & North Africa','North America','OECD members','Other small states','South Asia',
                      'Sub-Saharan Africa','World']


# In[ ]:


df_supranational=df_non_country[df_non_country['Country Name'].isin(supranational_groups)]


# In[ ]:


df_supranational=df_supranational[['Country Name','2010','2011','2012','2013','2014']]
df_supranational.head()


# In[ ]:


df_supranational=df_supranational.set_index('Country Name')


# In[ ]:


df_supranational=(df_supranational.T).copy()
df_supranational.head()


# In[ ]:


income_groups=['High income','Heavily indebted poor countries (HIPC)','Least developed countries: UN classification', 
                 'Low income','Lower middle income', 'Low & middle income','Middle income','Upper middle income','World']


# In[ ]:


df_income=df_non_country[df_non_country['Country Name'].isin(income_groups)]


# In[ ]:


df_income=df_income[['Country Name','2010','2011','2012','2013','2014']]


# In[ ]:


df_income=df_income.set_index('Country Name')


# In[ ]:


df_income=(df_income.T).copy()
df_income


# In[ ]:


# Supranational Group Unemployment Comparison

supranational_groups=['Arab World','Caribbean small states','East Asia & Pacific','European Union','Latin America & Caribbean',
                      'Middle East & North Africa','North America','OECD members','Other small states','South Asia',
                      'Sub-Saharan Africa','World']

years=df_supranational.index

traces=[]

for i in range(len(supranational_groups)):
    traces.append(go.Scatter(
                  x=years,
                  y=df_supranational.iloc[:,i],
                  name=supranational_groups[i],
                  mode='lines+markers',
                  line = dict(
                              width = 3,
                              dash = 'dashdot')
        ))

layout = go.Layout(
    title='Unemployment Over the Years in different Regions of the World',
    yaxis=dict(title='Unemployment Rate (%)',
               zeroline=True,
               showline=True,
               showgrid=False,
               showticklabels=True,
               linecolor='rgb(0,0,0)',
               linewidth=2,
               tickmode='auto',
               tickwidth=2,
               ticklen=5,
               nticks=8,
               tickfont=dict(
                            family='Arial',
                            size=12,
                            color='rgb(82, 82, 82)',
                            ),
               ticks='outside'),
    
    xaxis=dict(title='Years',
               showline=True,
               showgrid=False,
               showticklabels=True,
               linecolor='rgb(0,0,0)',
               linewidth=2,
               autotick=False,
               tickwidth=2,
               ticklen=5,
               tickfont=dict(
                            family='Arial',
                            size=12,
                            color='rgb(82, 82, 82)',
                            ),
               ticks='outside',
               tickmode='array',
               tickvals=['2009','2010', '2011', '2012', '2013', '2014','2015'])
)

fig = go.Figure(data=traces, layout=layout)
py.iplot(fig)


# **Trends from Supranational/Regional Groups**
# 
# A simple scatter plot displaying the unemployment rates in bigger regional groups is shown above with a view to compare them with each other and also the world unemployment rate. Thw World Unemployment rate is taken as the baseline for comparison and it can be seen from the graphs above that Asian (East asia & Pacific and South Asia & Latin America) has lower unemployment rate than the average world unemployment rates. These regions are mostly composed of developing economies thereby cementing the fact that these nations are what is driving the world forward then their developed counterparts after the economic slowdown which seems to be excessively harsh on European Union and Middle East Countries (Oil price slump may have also contributed to rising unemployment in Middle East). North America has seen a steady decrease in unemployment rates since 2010 finally achieving a value lower than the world world unemployment rate in 2014. Over the years, the overall increase in unemployment around the world is almost negligible from 13.78 % (2010) to 13.98 % (2014).

# In[ ]:


# Income Group Unemployment Comparison

income_groups=['High income','Heavily indebted poor countries (HIPC)','Least developed countries: UN classification', 
                 'Low income','Lower middle income', 'Low & middle income','Middle income','Upper middle income','World']

years_income=df_income.index

traces=[]

for i in range(len(income_groups)):
    traces.append(go.Scatter(
                  x=years_income,
                  y=df_income.iloc[:,i],
                  name=income_groups[i],
                  mode='lines+markers',
                  line = dict(
                              width = 3,
                              dash = 'dashdot')
        ))

layout = go.Layout(
    title='Unemployment Over the Years among Various Income Groups',
    yaxis=dict(title='Unemployment Rate (%)',
               zeroline=True,
               showline=True,
               showgrid=False,
               showticklabels=True,
               linecolor='rgb(0,0,0)',
               linewidth=2,
               tickmode='auto',
               tickwidth=2,
               ticklen=5,
               nticks=8,
               tickfont=dict(
                            family='Arial',
                            size=12,
                            color='rgb(82, 82, 82)',
                            ),
               ticks='outside'),
    
    xaxis=dict(title='Years',
               showline=True,
               showgrid=False,
               showticklabels=True,
               linecolor='rgb(0,0,0)',
               linewidth=2,
               autotick=False,
               tickwidth=2,
               ticklen=5,
               tickfont=dict(
                            family='Arial',
                            size=12,
                            color='rgb(82, 82, 82)',
                            ),
               ticks='outside',
               tickmode='array',
               tickvals=['2009','2010', '2011', '2012', '2013', '2014','2015'])
)

fig = go.Figure(data=traces, layout=layout)
py.iplot(fig)


# **Trends from Income based Country Groups**
# 
# As done in case of regional groups, a second segregation is also done on the basis  of income levels of country and a similar comparison is carried out between them and the world unemployment rates. As can be clearly seen from the scatter plot, it is the high income countries which are suffering from a higher unemployment rates when compared to their lesser developed/developing counterparts who are able to provide more opportunities to their workforce and keep a check on rising unemployment rates. This supports our earlier seen trend of increasing unemployment in middle east and european countries which are considered developed as compared to the low income/medium income economies of Asia such as India, China etc where unemployment rates were lower than the world unemployment rates.  
# 

# # Conclusion

# The analysis of the world unemployment data presents a stark contrast between european countries and lower/medium income developing countries. The data suggests a world maintaining an almost constant unemployment rate over the years, with europe presenting a complex situation in itself. It is home to countries both most successful and least successful in dealing with unemployment. Whereas the situation in Greece, Spain has deteriorated, the success in handling unemployment in countries such as Estonia, Lithuania and Latvia is a silver lining in these times for europe. More or less the situation in Asian giants China and India has remained same over the years. The Americans and Russians have seemed to 'pulled up their socks' and now working towards reducing unemployment in their countries with some degree of success.   
# 
# All in all this is just an attempt to present a very grave problem faced by our youth and general population and the data doesn't seem to give any hope that things will improve in short term from here on, as the world unemployment rate is holding almost steady (increase of ~ +0.2 %) and the deteriorating condition in several European Countries. There is greater need to address the underlying problem of unemployment (underskilled workforce etc.) and an honest attempt should be made by concerned authorities before the situation spirals further out of hand.
