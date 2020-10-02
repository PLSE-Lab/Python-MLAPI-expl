#!/usr/bin/env python
# coding: utf-8

# This kernel is my way to learn Plotly visulizations and I hope this will help other new kagglers to learn as well. I will mentions links to the plotly websites before all the visulizations so that it is easy to view the plotly reference. I have hidden the code input so if you want to see the code click on the code button to view the code.<br>
# **So Let's Begin...** 
# 
# 

# Some context about the Dataset: This dataset is a record of every building or building unit (apartment, etc.) sold in the New York City property market over a 12-month period.
# 
# References: I have used the below notebooks for reference, they have been a great learning resource for me so do check these notebooks.
# 
# * [Plotly Tutorials for Beginners](https://www.kaggle.com/kanncaa1/plotly-tutorial-for-beginners)
# * [How to become a property tycoon in New York](https://www.kaggle.com/akosciansky/how-to-become-a-property-tycoon-in-new-york)
# * [NYC Property sales top to bottom approach](https://www.kaggle.com/thomsshmims/nyc-property-sales-top-to-bottom-approach)
# * [A Visual and Insightful journey donorschoose](https://www.kaggle.com/shivamb/a-visual-and-insightful-journey-donorschoose)

# **Lets import the libraries and read the csv file into the dataframe**

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from datetime import date
import seaborn as sns
import random 
import warnings
import operator
warnings.filterwarnings("ignore")
import os
print(os.listdir("../input"))


# In[ ]:


import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as ff


# In[ ]:


df = pd.read_csv("../input/nyc-rolling-sales.csv")
df.head()


# I am changing the names of the columns to lower case to make it easier to work with and also replacing the spaces with underscores

# In[ ]:


df.columns = ['Unnamed: 0', 'borough', 'neighborhood','building_class category','tax_class_at_present', 'block', 'lot', 'ease_ment','building_class_at_present', 'address', 'apartment_number', 'zip_code',
       'residential_units', 'commercial_units', 'total_units','land_square_feet', 'gross_square_feet', 'year_built','tax_class_at_time_of_sale', 'building_class_at_time_of_sale',
       'sale_price', 'sale_date']
df.info()


# The datatypes of the columns are not right, so the info function does not show null values, so we need to change them to appropriate datatypes and then check for null values.
# So let's change the datatypes of columns

# In[ ]:


# deleting the Unnamed column
del df['Unnamed: 0']

# SALE PRICE, LAND and GROSS SQUARE FEET is object type but should be numeric data type columns hence converting them to numeric
df['sale_price'] = pd.to_numeric(df['sale_price'], errors='coerce')
df['land_square_feet'] = pd.to_numeric(df['land_square_feet'], errors='coerce')
df['gross_square_feet']= pd.to_numeric(df['gross_square_feet'], errors='coerce')

# Both TAX CLASS attributes should be categorical
df['tax_class_at_time_of_sale'] = df['tax_class_at_time_of_sale'].astype('category')
df['tax_class_at_present'] = df['tax_class_at_present'].astype('category')


# Changing the data type of the data object and adding some more columns to the dataset related to time of Sale.

# In[ ]:


#SALE DATE is object but should be datetime
df['sale_date']    = pd.to_datetime(df['sale_date'], errors='coerce')
df['sale_year']    = df['sale_date'].dt.year
df['sale_month']   = df['sale_date'].dt.month
df['sale_quarter'] = df['sale_date'].dt.quarter
df['sale_day']     = df['sale_date'].dt.day
df['sale_weekday'] = df['sale_date'].dt.weekday


# <h2>Removing the Duplicate values</h2>

# In[ ]:


# as seen in the other Kernels there are some duplicate values that need to be deleted 
# lets check and delete those values
print("Number of duplicates in the given dataset = {0}".format(sum(df.duplicated(df.columns))))
df = df.drop_duplicates(df.columns, keep='last')
print("Number of duplicates in the given dataset after cleanup = {0}".format(sum(df.duplicated(df.columns))))


# <h2>Missing values</h2>
# There are a few columns with missing values, so let's check all columns and find out what percentage values are missing from each column and then plot a bar chart using plotly.
# 
# Link for the Bar chart plot : https://plot.ly/python/bar-charts/

# In[ ]:


# lets find out the percentage of non null values in each column and plot them to get a better view
variables = df.columns
count = []

for variable in variables:
    length = df[variable].count()
    count.append(length)
    
count_pct = np.round(100 * pd.Series(count) / len(df), 2)


# In[ ]:


dataa = [go.Bar(
            y= df.columns,
            x = count_pct,
            width = 0.7,
            opacity=0.6, 
            orientation = 'h',
            marker=dict(
                color='rgb(158,202,225)',
                line=dict(
                    color='rgb(8,48,107)',
                    width=1.5,
                )
            ),
        )]
layout = go.Layout(
    title='Percentage of non-null values in each column',
    autosize = False,
    width=800,
    height=800,
    margin=go.Margin(
        l=250,
        r=50,
        b=100,
        t=100,
        pad=4
    ),
)

fig = go.Figure(data=dataa, layout = layout)
py.iplot(fig, filename='barplottype')


# we see that there are 3 columns i.e sale_price, gross_square_feet and land_square_feet that have null values. so we must remove those rows.

# In[ ]:


# as there are rows where sale price is null, we should remove them from our dataset
df = df[df['sale_price'].notnull()]


# In the dataset we have sale_prices with values 0 or some very small numbers, which are cases or transfers of deeds between parties: for example, parents transferring ownership to their home to a child after moving out for retirement, and according to the below kernel there are also outliers that in the above 3 columns which can be easily seen in the scatter plots so, we select a range for the values in these columns and remove all the other rows. 
# 
# Reference : https://www.kaggle.com/akosciansky/how-to-become-a-property-tycoon-in-new-york

# In[ ]:


df = df[(df['sale_price'] > 100000) & (df['sale_price'] < 5000000)]

# Removes all NULL values
df = df[df['land_square_feet'].notnull()] 
df = df[df['gross_square_feet'].notnull()] 

# Keeps properties with fewer than 20,000 Square Feet, which is about 2,000 Square Metres
df = df[df['gross_square_feet'] < 20000]
df = df[df['land_square_feet'] < 20000]


# After removing the *null values*, now let's plot the bar chart again to see if we have any more null values.

# In[ ]:


# lets find out the percentage of non null values in each column and plot them to get a better view
variables = df.columns
count = []

for variable in variables:
    length = df[variable].count()
    count.append(length)
    
count_pct = np.round(100 * pd.Series(count) / len(df), 2)

dataa = [go.Bar(
            y= df.columns,
            x = count_pct,
            width = 0.7,
            opacity=0.6, 
            orientation = 'h',
            marker=dict(
                color='rgb(158,202,225)',
                line=dict(
                    color='rgb(8,48,107)',
                    width=1.5,
                )
            ),
        )]
layout = go.Layout(
    title='Percentage of non-null values in each column',
    autosize = False,
    width=800,
    height=800,
    margin=go.Margin(
        l=250,
        r=50,
        b=100,
        t=100,
        pad=4
    ),
    yaxis=dict(
        title='column'
    )
)

fig = go.Figure(data=dataa, layout = layout)
py.iplot(fig, filename='barplottype')


# Well, now that we have removed all the null values from our dataset so let's move forward.

# In[ ]:


print("Length of dataset after cleanup = {0}".format(len(df)))


# <h2>Target variable Analysis : Sale Price</h2>
# For target variable analysis of a continuous variable I mostly use the following plots,
# 1. **Scatter plot**: Sort the values and then plot the continuous variable to get to know the range of values and look for any outliers.
# 2. **Histogram**: To know the distribution of the values, I also check the skewness of the variable and apply log to the column if necessary. 
# 
# 
# Link for Scatter plot: https://plot.ly/python/line-and-scatter/
# 
# Link for Histogram plot: https://plot.ly/python/histograms/

# In[ ]:


# Create a trace
trace = go.Scatter(
    x = np.sort(df['sale_price']),
    y = np.arange(len(df)),
    mode = 'markers'
)
layout = go.Layout(
    title='Sale Prices',
    autosize = True,
    paper_bgcolor='rgb(243, 243, 243)',
    plot_bgcolor='rgb(243, 243, 243)'
)
d = [trace]

# Plot and embed in ipython notebook!
fig = go.Figure(data=d, layout = layout)
py.iplot(fig, filename='basic-scatter')


# In[ ]:


trace1 = go.Histogram(
    x = df['sale_price'],
    name = 'sale_price'
)
dat = [trace1]
# Plot!
#py.iplot(dat, filename='Distplot with Normal Curve')
from scipy.stats import skew
print("Skewness of Sale Price attribute is : {0}".format(skew(df['sale_price'])))

trace2 = go.Histogram(
    x = np.log(df['sale_price']), 
    name = 'log(sale_price)'
)
dat = [trace1]
# Plot!
#py.iplot(dat, filename='Distplot with Normal Curve')
print("Skewness of Sale Price attribute after applying log is : {0}".format(skew(np.log(df['sale_price']))))

fig = tls.make_subplots(rows=2, cols=1, subplot_titles=('Sale Prices', 'Sale Prices after applying log'));
fig.append_trace(trace1, 1, 1);
fig.append_trace(trace2, 2, 1);
fig['layout'].update(height=600, width=800,title='Histogram Plot for Sale prices of Houses');
py.iplot(fig, filename='simple-subplot')


# In[ ]:


# creating a different new copy of dataset
df2 = pd.read_csv("../input/nyc-rolling-sales.csv")
del df2['Unnamed: 0']
df2.columns = ['borough', 'neighborhood','building_class_category','tax_class_at_present', 'block', 'lot', 'ease_ment','building_class_at_present', 'address', 'apartment_number', 'zip_code',
       'residential_units', 'commercial_units', 'total_units','land_square_feet', 'gross_square_feet', 'year_built','tax_class_at_time_of_sale', 'building_class_at_time_of_sale',
       'sale_price', 'sale_date']
# lets rename boroughs and do some visualization on it
df2['borough'][df2['borough'] == 1] = 'Manhattan'
df2['borough'][df2['borough'] == 2] = 'Bronx'
df2['borough'][df2['borough'] == 3] = 'Brooklyn'
df2['borough'][df2['borough'] == 4] = 'Queens'
df2['borough'][df2['borough'] == 5] = 'Staten Island'

df2['sale_price'] = pd.to_numeric(df2['sale_price'], errors='coerce')
df2['land_square_feet'] = pd.to_numeric(df2['land_square_feet'], errors='coerce')
df2['gross_square_feet']= pd.to_numeric(df2['gross_square_feet'], errors='coerce')

# Both TAX CLASS attributes should be categorical
df2['tax_class_at_time_of_sale'] = df2['tax_class_at_time_of_sale'].astype('category')
df2['tax_class_at_present'] = df2['tax_class_at_present'].astype('category')


# <h2>Boroughs</h2>
# Let's see how are properties distributed across the boroughs using the pie chart and the bar chart.
# 
# Link for Pie chart: https://plot.ly/python/pie-charts/
# 
# Link for Bar charts: https://plot.ly/python/bar-charts/

# In[ ]:


# distribution of houses in each borough
boroughs = ['Manhattan','Bronx','Brooklyn','Queens','Staten Island']
property_count = []
for b in boroughs:
    property_count.append(len(df2.borough[df2.borough == b]))

fig = {
  "data": [
    {
      "values": property_count,
      "labels": boroughs,
      "hoverinfo":"label+percent",
      "hole": .3,
      "type": "pie"
    }],
  "layout": {
        "title":"Percentage of Properties in Boroughs",
    }
}
py.iplot(fig, filename='donut')


dataa = [go.Bar(
            y= boroughs,
            x = property_count,
            width = 0.7,
            opacity=0.6, 
            orientation = 'h',
            marker=dict(
                color='rgb(158,202,225)',
                line=dict(
                    color='rgb(8,48,107)',
                    width=1.5,
                )
            ),
        )]
layout = go.Layout(
    title='Number of Housing Properties in each Bourough',
    autosize = False,
    width=800,
    height=500,
    margin=go.Margin(
        l=250,
        r=50,
        b=100,
        t=100,
        pad=4
    ),
)

fig = go.Figure(data=dataa, layout = layout)
py.iplot(fig, filename='barplottype')


# You can hover over the chart to see the actual number of properties in each borough.(I really like the interactive plotly charts!!)

# <h2>Sale Prices in each Borough</h2>
# 
# For a bivariate analysis between a continuous variable and a categorical variable we mostly use box plots or voilin plots, these plots give us a lot of information such as mean, median, min, max, q1, q3. Hence ploting box plots for sale_price vs boroughs 
# 
# Link for Box plots : https://plot.ly/python/box-plots/

# In[ ]:


# average price of a house in each borough
# box plot
df2 = df2[df2['sale_price'].notnull()]
df2 = df2[(df2['sale_price'] > 100000) & (df2['sale_price'] < 5000000)]

trace0 = go.Box(
    y=df2.sale_price[df2.borough == 'Manhattan' ],
    name = 'Manhattan',
    marker = dict(
        color = 'rgb(12, 12, 140)',
    )
)
trace1 = go.Box(
    y=df2.sale_price[df2.borough ==  'Bronx' ],
    name = 'Bronx',
    marker = dict(
        color = 'rgb(8,81,156)',
    )
)
trace2 = go.Box(
    y=df2.sale_price[df2.borough ==  'Brooklyn' ],
    name = 'Brooklyn',
    marker = dict(
        color = 'rgb(12, 12, 140)',
    )
)
trace3 = go.Box(
    y=df2.sale_price[df2.borough ==  'Queens' ],
    name = 'Queens',
    marker = dict(
        color = 'rgb(12, 128, 128)',
    )
)
trace4 = go.Box(
    y=df2.sale_price[df2.borough ==  'Staten Island' ],
    name = 'Staten Island',
    marker = dict(
        color = 'rgb(12, 12, 140)',
    )
)

dat = [trace0, trace1, trace2, trace3, trace4]
layout = go.Layout(
    title='Housing Prices in Boroughs',
    xaxis=dict(
        title='Borough'
    ),
    yaxis=dict(
        title='Sale Price'
    ),
    paper_bgcolor='rgb(243, 243, 243)',
    plot_bgcolor='rgb(243, 243, 243)'
)

fig = go.Figure(data=dat, layout=layout)
py.iplot(fig)


# According to the above plot, max price of property in each borough is almost the same but the median price of the houses in Manhattan is the highest, followed by Brooklyn.

# <h2>Sale Price of properties in each Month</h2>
# Let's try to plot the sale prices of properties in each month, for this plot I have used the rainbow plot from the plotly reference :p

# In[ ]:


# Rainbow plot
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
colors = ['hsl('+str(h)+',50%'+',50%)' for h in np.linspace(0, 360, 12)]
data = [{
    'y' : df.sale_price[df.sale_month == ind],
    'type':'box',
    'name' : months[ind - 1],
    'marker':{'color': colors[ind - 1]}
} for ind in range(1,13)]

layout = go.Layout(
    title='Housing Prices in each Borough',
    xaxis=dict(
        title='Month'
    ),
    yaxis=dict(
        title='Sale Price'
    ),
    paper_bgcolor='rgb(243, 243, 243)',
    plot_bgcolor='rgb(243, 243, 243)'
)

fig = go.Figure(data=data, layout=layout)
#dat = [trace0, trace1, trace2, trace3, trace4]
py.iplot(fig)


# Mean or Median Sale prices do not change much in any month (as expected). This may be because the data is only collected for 12 months or we might have seen a pattern. 

# <h2>Average Land square feet in each Borough</h2>
# To plot average land square feet in each borough I am using bubble chart, In this type of chart the size of the bubble indicates the magnitude of the variable. This plot can be used in place of bar chart sometimes.
# 
# Link for Bubble Chart: https://plot.ly/python/bubble-charts/ 

# In[ ]:


# average LAND SQUARE FEET in each borough box plot
d3 = pd.DataFrame(df.groupby(['borough']).mean()).reset_index()
d3['borough'][d3.borough == 1] = 'Manhattan'
d3['borough'][d3.borough == 2] = 'Bronx'
d3['borough'][d3.borough == 3] = 'Brooklyn'
d3['borough'][d3.borough == 4] = 'Queens'
d3['borough'][d3.borough == 5] = 'Staten Island'
total = d3.land_square_feet.sum()
trace0 = go.Scatter(
    x=d3.borough,
    y=d3.land_square_feet,
    mode='markers',
    marker=dict(
        size=[((x/total)*300) for x in d3.land_square_feet],
        color=['rgb(93, 164, 214)', 'rgb(255, 144, 14)',  'rgb(44, 160, 101)', 'rgb(255, 65, 54)', 'rgb(255, 15, 54)'],
    )
)

data = [trace0]
layout = go.Layout(
    title='Average Land square feet of properties in each Borough',
    xaxis=dict(
        title='Borough',
        gridcolor='rgb(255, 255, 255)',
    ),
    yaxis=dict(
        title='Land square feet',
        gridcolor='rgb(255, 255, 255)',
    ),
    paper_bgcolor='rgb(243, 243, 243)',
    plot_bgcolor='rgb(243, 243, 243)'
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='bubblechart-size')


# <h2>Average Gross square feet in each Borough</h2>

# In[ ]:


# average GROSS SQUARE FEET in each borough box plot
total = d3.gross_square_feet.sum()
trace0 = go.Scatter(
    x=d3.borough,
    y=d3.gross_square_feet,
    mode='markers',
    marker=dict(
        size=[((x/total)*300) for x in d3.gross_square_feet],
        color=['rgb(93, 164, 214)', 'rgb(255, 144, 14)',  'rgb(44, 160, 101)', 'rgb(255, 65, 54)', 'rgb(255, 15, 54)'],
    )
)

layout = go.Layout(
    title='Average Gross square feet in each Borough',
    xaxis=dict(
        title='Boroughs',
    ),
    yaxis=dict(
        title='Average gross square feet',
    ),
    paper_bgcolor='rgb(243, 243, 243)',
    plot_bgcolor='rgb(243, 243, 243)',
)


data = [trace0]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='gross-square-feet-borough-bubble-chart')


# <h2>Number of properties sold in each borough in each month</h2>

# In[ ]:


p = pd.DataFrame(df.groupby(['borough', 'sale_month']).sale_price.count()).reset_index()
colors = ['hsl('+str(h)+',50%'+',50%)' for h in np.linspace(0, 360, 12)]
data = [
    {
        'x': months,
        'y': [boroughs[ind]] * 12,
        'showlegend' : False,
        'mode': 'markers',
        'text': [ x for x in p.sale_price[p.borough == ind+1]],
        'marker': {
            'color': colors,
            'size': [(x/np.sum(p.sale_price[p.borough == ind+1])*400) for x in p.sale_price[p.borough == ind+1]],
        }
    } for ind in range(5)
]

layout = go.Layout(
    title='Number of properties sold in each borough in each month',
    xaxis=dict(
        title='Month',
        gridcolor='rgb(255, 255, 255)',
    ),
    yaxis=dict(
        title='Borough',
        gridcolor='rgb(255, 255, 255)',
    ),
    paper_bgcolor='rgb(243, 243, 243)',
    plot_bgcolor='rgb(243, 243, 243)',
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='temp')


# In[ ]:


from collections import Counter
# print(Counter(data2['BUILDING CLASS CATEGORY']))
c = dict(Counter(df2['building_class_category']))
import operator
sorted_category = sorted(c.items(), key=operator.itemgetter(1))
cat_name  = []
cat_value = []
for tup in sorted_category:
    cat_name.append(tup[0])
    cat_value.append(tup[1])


# <h2>Building Class</h2>
# Let's plot a bar chart to see what class do majority of the buildings belong to..

# In[ ]:


# plot to see what class do majority of the buildings belong to 
dataa = [go.Bar(
            y= cat_name,
            x = cat_value,
            width = 0.7,
            opacity=0.6, 
            orientation = 'h',
            marker=dict(
                color='rgb(158,202,225)',
                line=dict(
                    color='rgb(8,48,107)',
                    width=1.5,
                )
            ),
        )]
layout = go.Layout(
    title='What building class do majority of apartments belong to?',
    autosize = False,
    width=800,
    height=800,
    margin=go.Margin(
        l=350,
        r=50,
        b=100,
        t=100,
        pad=4
    ),
    xaxis=dict(
        title='Number of housing properties',
    ),
    yaxis=dict(
        title='Building class',
    ),
)

fig = go.Figure(data=dataa, layout = layout)
py.iplot(fig, filename='barplottype')


# According to the above plot most of the properties are **One Family Dwellings**

# <h2>Sale Price and Tax class</h2>
# * Class 1: Includes most residential property of up to three units (such as one-,
# two-, and three-family homes and small stores or offices with one or two
# attached apartments), vacant land that is zoned for residential use, and most
# condominiums that are not more than three stories.
# * Class 2: Includes all other property that is primarily residential, such as
# cooperatives and condominiums.
# * Class 3: Includes property with equipment owned by a gas, telephone or electric
# company
# *  Class 4: Includes all other properties not included in class 1,2, and 3, such as
# offices, factories, warehouses, garage buildings, etc
# 
# I am using voilin plots for this.
# 
# Link for voilin plots : https://plot.ly/python/violin/

# In[ ]:


tax_class = ["tax_class_1","tax_class_2","tax_class_4"]
classes = ["1","2","4"]
data = []
for i in range(3):
    trace = {
            "type": 'violin',
            "x": tax_class[i],
            "y": df.sale_price[df.tax_class_at_present.str.contains(classes[i])],
            "name": tax_class[i],
            "box": {
                "visible": True
            },
            "meanline": {
                "visible": True
            }
        }
    data.append(trace)

        
fig = {
    "data": data,
    "layout" : {
        "title": "Sale Prices of houses according to tax class",
        "yaxis": {
            "title": "Sale Price",
            "zeroline": False,
        },
        "xaxis": {
            "title": "Tax classes"
        }
    }
}


py.iplot(fig, filename='tax-class-sale-price-voilin-plots', validate = False)


# <h2>Average House Price in the top 20 neighborhoods</h2>
# Lets select the top 20 neighbourhoods and see the average price in them.

# In[ ]:


from collections import Counter
neighborhoods = list(dict(Counter(df.neighborhood).most_common(20)).keys())

avg_sale_prices = []
for i in neighborhoods:
    avg_price = np.mean(df.sale_price[df.neighborhood == i])
    avg_sale_prices.append(avg_price)
    
dataa = [go.Bar(
            y= neighborhoods,
            x = avg_sale_prices,
            width = 0.7,
            opacity=0.6, 
            orientation = 'h',
            marker=dict(
                color='rgb(158,202,225)',
                line=dict(
                    color='rgb(8,48,107)',
                    width=1.5,
                )
            ),
        )]
layout = go.Layout(
    title='Average House Price in the top 20 neighborhoods',
    autosize = True,
    margin=go.Margin(
        l=250,
        r=50,
        b=100,
        t=100,
        pad=4
    ),
    xaxis=dict(
        title='Sale Price',
    ),
    yaxis=dict(
        title='Neighborhood',
    ),
)

fig = go.Figure(data=dataa, layout = layout)
py.iplot(fig, filename='barplottype')


# <h2>Correlation Plot </h2>
# Finally a correlation plot for all the numeric variables. This is not a plotly plot but still a use plot to know the effect of other varaibles on the sale price of the properties.

# In[ ]:


# Compute the correlation matrix
d= df[['sale_price', 'total_units','gross_square_feet',  'land_square_feet', 'residential_units', 
         'commercial_units', 'borough', 'block', 'lot', 'zip_code', 'year_built',]]
corr = d.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, 
            square=True, linewidths=.5, annot=True, cmap=cmap)
plt.yticks(rotation=0)
plt.title('Correlation Matrix of all Numerical Variables')
plt.show()


# For other plotly charts you can check their website : https://plot.ly/ or have a look at some other notebooks on Kaggle and learn from them.
# Finally, If you like this notebook, then please upvote it... 
# Thanks.
