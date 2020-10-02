#!/usr/bin/env python
# coding: utf-8

# # Introduction 
# ![](https://miro.medium.com/max/757/1*eYEoP5hF-IyQ4SCLEs1cbQ.png)
# ### In this kernel notebook I will be focusing on initially covering the new Pandas_Bokeh Data visualisation followed by a exploratory data analysis ,a case study about Karnataka Education using Bokeh.
# 
# **Pandas-Bokeh** provides a Bokeh plotting backend for Pandas, GeoPandas and Pyspark DataFrames, similar to the already existing Visualization feature of Pandas. Importing the library adds a complementary plotting method plot_bokeh() on DataFrames and Series.
# 
# 
# With **Pandas-Bokeh**, creating stunning, interactive, HTML-based visualization is as easy as calling:
# 
# df.plot_bokeh()
# 
# 
# **Pandas-Bokeh** also provides native support as a Pandas Plotting backend for Pandas >= 0.25. When **Pandas-Bokeh** is installed, switchting the default Pandas plotting backend to Bokeh can be done via:
# 
# pd.set_option('plotting.backend', 'pandas_bokeh')
# 
# ![](https://miro.medium.com/max/1962/0*lfsR26JXj4o_QMWI.gif)
# 
# 
# Now its time to first install Pandas_Bokeh using PIP command.

# In[ ]:


get_ipython().system('pip install pandas-bokeh')


# ### Import Libraries
# 

# In[ ]:


import numpy as np
import pandas as pd
import pandas_bokeh
pandas_bokeh.output_notebook()
pd.set_option('plotting.backend', 'pandas_bokeh')
# Create Bokeh-Table with DataFrame:
from bokeh.models.widgets import DataTable, TableColumn
from bokeh.models import ColumnDataSource


# # Import Data

# 
# 
# ## Plot types
# 
# 
# #### Pandas & Pyspark DataFrames
# * Line plot
# * Point plot
# * Step plot
# * Scatter plot
# * Bar plot
# * Histogram
# * Area plot
# * Pie plot
# * Map plot
# 
# #### Geoplots (Point, Line, Polygon) with GeoPandas
# 
# 
# ### Lineplot
# 
# This simple lineplot in Pandas-Bokeh already contains various interactive elements:
# 
# * a pannable and zoomable (zoom in plotarea and zoom on axis) plot
# * by clicking on the legend elements, one can hide and show the individual lines
# * a Hovertool for the plotted lines
# 
# Consider the following simple example:
# 
# We will be importing the time series data about the power usage in various states in India. All of the values are measured in **MU(millions of units)**. **The date ranges from 28/10/2019 to 23/05/2020.**

# In[ ]:


df = pd.read_csv('../input/state-wise-power-consumption-in-india/dataset_tk.csv')
df_long = pd.read_csv('../input/state-wise-power-consumption-in-india/long_data_.csv')


# Firstly creating Date column and dropping the unwanted column and reformatting the date column

# In[ ]:


df["Date"]=df["Unnamed: 0"]
df['Date'] = pd.to_datetime(df.Date, dayfirst=True)
df = df.drop(["Unnamed: 0"], axis = 1) 


# In[ ]:


df.info()


# Now let us divide the states based on 5 regions namely 
# 
# 1. Northern Region
# 
# 2. Southern Region
# 
# 3. Eastern Region
# 
# 4. Western Region
# 
# 5. North Eastern Region
# 

# In[ ]:


df['NR'] = df['Punjab']+ df['Haryana']+ df['Rajasthan']+ df['Delhi']+df['UP']+df['Uttarakhand']+df['HP']+df['J&K']+df['Chandigarh']
df['WR'] = df['Chhattisgarh']+df['Gujarat']+df['MP']+df['Maharashtra']+df['Goa']+df['DNH']
df['SR'] = df['Andhra Pradesh']+df['Telangana']+df['Karnataka']+df['Kerala']+df['Tamil Nadu']+df['Pondy']
df['ER'] = df['Bihar']+df['Jharkhand']+ df['Odisha']+df['West Bengal']+df['Sikkim']
df['NER'] =df['Arunachal Pradesh']+df['Assam']+df['Manipur']+df['Meghalaya']+df['Mizoram']+df['Nagaland']+df['Tripura']


# In[ ]:


df_line = pd.DataFrame({"Northern Region": df["NR"].values,
                        "Southern Region": df["SR"].values,
                        "Eastern Region": df["ER"].values,
                        "Western Region": df["WR"].values,
                        "North Eastern Region": df["NER"].values},index=df.Date)

df_line.plot_bokeh(kind="line",title ="India - Power Consumption Regionwise",
                   figsize =(1000,800),
                   xlabel = "Date",
                   ylabel="MU(millions of units)"
                   )


# ### In the above data visualisation which is completely intereactive. You can click on any index regions and check the data .Is it an interesting data visualisation ???????
# 
# ##### Let us look at some other types of LinePlot
# 
# #### Bar Type:
# 

# In[ ]:


df_line.plot_bokeh(kind="bar",title ="India - Power Consumption Regionwise",figsize =(1000,800),xlabel = "Date",ylabel="MU(millions of units)")


# #### Point Type:

# In[ ]:


df_line.plot_bokeh(kind="point",title ="India - Power Consumption Regionwise",figsize =(1000,800),xlabel = "Date",ylabel="MU(millions of units)")


# #### Histogram Type:

# In[ ]:


df_line.plot_bokeh(kind="hist",title ="India - Power Consumption Regionwise",
                   figsize =(1000,800),
                   xlabel = "Date",
                   ylabel="MU(millions of units)"
                )


# #### Lineplot with rangetool

# In[ ]:


df_line = pd.DataFrame({"Northern Region": df["NR"].values,
                        "Southern Region": df["SR"].values,
                        "Eastern Region": df["ER"].values,
                        "Western Region": df["WR"].values,
                        "North Eastern Region": df["NER"].values},index=df.Date)

df_line.plot_bokeh(kind="line",title ="India - Power Consumption Regionwise",
                   figsize =(1000,800),
                   xlabel = "Date",
                   ylabel="MU(millions of units)",rangetool=True
                   )


# ### Pointplot
# 
# If you just wish to draw the date points for curves, the pointplot option is the right choice. It also accepts the kwargs of bokeh.plotting.figure.scatter like marker or size:

# In[ ]:


df_line.plot_bokeh.point(
    x=df.Date,
    xticks=range(0,1),
    size=5,
    colormap=["#009933", "#ff3399","#ae0399","#220111","#890300"],
    title=" Point Plot - India Power Consumption",
    fontsize_title=20,
    marker="x",figsize =(1000,800))


# ### Stepplot
# 
# With a similar API as the line- & pointplots, one can generate a stepplot. Additional keyword arguments for this plot type are passes to bokeh.plotting.figure.step, e.g. mode (before, after, center), see the following example
# 

# In[ ]:


df_line.plot_bokeh.step(
    x=df.Date,
    xticks=range(-1, 1),
    colormap=["#009933", "#ff3399","#ae0399","#220111","#890300"],
    title="Step Plot - India Power Consumption",
    figsize=(1000,800),
    fontsize_title=20,
    fontsize_label=20,
    fontsize_ticks=20,
    fontsize_legend=8,
    )


# ### Scatterplot
# 
# A basic scatterplot can be created using the kind="scatter" option. For scatterplots, the x and y parameters have to be specified and the following optional keyword argument is allowed:
# 
# category: Determines the category column to use for coloring the scatter points
# 
# kwargs**: Optional keyword arguments of bokeh.plotting.figure.scatter
# 
# Note, that the pandas.DataFrame.plot_bokeh() method return per default a Bokeh figure, which can be embedded in Dashboard layouts with other figures and Bokeh objects (for more details about (sub)plot layouts and embedding the resulting Bokeh plots as HTML click here).
# 
# In the example below, we use the building grid layout support of Pandas-Bokeh to display both the DataFrame (using a Bokeh DataTable) and the resulting scatterplot:

# In[ ]:


df = pd.read_csv("../input/iris/Iris.csv")
df = df.sample(frac=1)


# In[ ]:


df.head()


# In[ ]:


data_table = DataTable(
    columns=[TableColumn(field=Ci, title=Ci) for Ci in df.columns],
    source=ColumnDataSource(df),
    height=300,
)

# Create Scatterplot:
p_scatter = df.plot_bokeh.scatter(
    x="PetalLengthCm",
    y="SepalWidthCm",
    category="Species",
    title="Iris DataSet Visualization",
    show_figure=False
)

# Combine Table and Scatterplot via grid layout:
pandas_bokeh.plot_grid([[data_table, p_scatter]], plot_width=400, plot_height=350)


# ### Barplot
# 
# The barplot API has no special keyword arguments, but accepts optional kwargs of bokeh.plotting.figure.vbar like alpha. It uses per default the index for the bar categories (however, also columns can be used as x-axis category using the x argument).
# 
# Let us look at an example

# In[ ]:


data = {
    'Cars':
    ['Maruti Suzuki', 'Honda', 'Toyota', 'Hyundai', 'Benz', 'BMW'],
    '2018': [20000, 15722, 4340, 38000, 2890, 412],
    '2019': [19000, 13700, 340, 31200, 290, 234],
    '2020': [23456, 15891, 440, 36700, 890, 417]
}
df = pd.DataFrame(data).set_index("Cars")

p_bar = df.plot_bokeh.bar(
    ylabel="Price per Unit", 
    title="Car Units sold per Year", 
    alpha=0.6)


# Using the stacked keyword argument you also make stacked barplots as shown below

# In[ ]:


stacked_bar = df.plot_bokeh.bar(
    ylabel="Price per Unit", 
    title="Car Units sold per Year", 
    stacked=True,
    alpha=0.6)


# Also horizontal versions of the above barplot are supported with the keyword kind="barh" or the accessor plot_bokeh.barh. You can still specify a column of the DataFrame as the bar category via the x argument if you do not wish to use the index.

# In[ ]:


#Reset index, such that "Cars" is now a column of the DataFrame:
df.reset_index(inplace=True)

#Create horizontal bar (via kind keyword):
p_hbar = df.plot_bokeh(
    kind="barh",
    x="Cars",
    ylabel="Price per Unit", 
    title="Car Units sold per Year", 
    alpha=0.6,
    legend = "bottom_right",
    show_figure=False)

#Create stacked horizontal bar (via barh accessor):
stacked_hbar = df.plot_bokeh.barh(
    x="Cars",
    stacked=True,
    ylabel="Price per Unit", 
    title="Car Units sold per Year", 
    alpha=0.6,
    legend = "bottom_right",
    show_figure=False)

#Plot all barplot examples in a grid:
pandas_bokeh.plot_grid([[p_bar, stacked_bar],
                        [p_hbar, stacked_hbar]], 
                       plot_width=450)


# Now let us look at a more practical example of housing prices problem to understand it better.

# In[ ]:


df = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv",index_col='SalePrice')
numeric_features = df.select_dtypes(include=[np.number])
p_bar = numeric_features.plot_bokeh.bar(
    ylabel="Sale Price", 
    figsize=(1000,800),
    title="Housing Prices", 
    alpha=0.6)


# 

# # Bokeh Introduction
# 
# Bokeh is an interactive visualization library that targets modern web browsers for presentation. Its goal is to provide elegant, concise construction of versatile graphics, and to extend this capability with high-performance interactivity over very large or streaming datasets. Bokeh can help anyone who would like to quickly and easily create interactive plots, dashboards, and data applications.
# 
# For this kernel we will taking an example of Karnataka State(India) Education dataset for our exploratory data analysis.
# 
# # EDA -Bokeh - Karnataka Education
# 
# An NGO organisation takes initiatives to improve primary education in  India and want to carry out this program in Karnataka. It wants to target districts that fall behind in areas such as 
# 
# - Education Infrastructure
# 
# - Education Awareness
# 
# - Demographic features
# 
# Identify such districts that could be targeted in its first phase.
# 
# The source data for this exercise is obtained from data.gov.in
# 
# ## Goal :
# 
# The goal of this notebook was primarily to:
# 
# 1.       Explain the data, define your target and come up with features that can be used for modelling.
# 
# 2.       Create a model based on your features and come up with the list of target districts
# 
# 3.       Detailed analysis to include all components such as 
# 
#         - Data fetch
#         - Data cleansing
#         - Exploratory data analysis
#         - Summary and Data Visualization
#         
# 

# # Import Libraries

# In[ ]:


get_ipython().system('pip install pandas-bokeh')


# In[ ]:


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
# Import Bokeh Library for output
from bokeh.io import output_notebook
output_notebook()
from bokeh.models import ColumnDataSource
from bokeh.models import HoverTool
from bokeh.models import LinearInterpolator,CategoricalColorMapper
from bokeh.io import show
from bokeh.plotting import figure
from bokeh.palettes import Spectral8


# ## Data Fetching

# In[ ]:


data = pd.read_csv('../input/karnataka-state-education/Town-wise-education - Karnataka.csv')


# ## Exploratory Data Analysis

# In[ ]:


data.info()


# Let us have a quick glance of what the data looks like by observing the first and last few rows

# In[ ]:


data.head()


# In[ ]:


data.tail()


# Lets us examine the shape of this dataset

# In[ ]:


data.shape


# This means that we have 812 dimensions(rows) and 46 features (columns) in this dataset.
# Now let us explore the data types of the dataset

# In[ ]:


data.info()


# The above observation shows that there are categorical and numerical features in the dataset.Let us explore further...
# 
# Now let us find out how many unique categories are available from the above categorical features.
# Let us examine if there are any nulls in the dataset

# In[ ]:


data.isnull().sum()


# Let us also look at the entire metrics including the inter quartile range,mean,standard deviation for all the features
# 

# In[ ]:


data.describe(include = 'all')


# Now let us look in detail the categorical features.For this basically extract all the categorical features into a dataframe object

# In[ ]:


categorical_features = data.select_dtypes(include=[np.object])
categorical_features.info()


# #### Let us observe the unique categories for all the object varables

# In[ ]:


for column_name in data.columns:
    if data[column_name].dtypes == 'object':
        data[column_name] = data[column_name].fillna(data[column_name].mode().iloc[0])
        unique_category = len(data[column_name].unique())
        print("Feature '{column_name}' has '{unique_category}' unique categories".format(column_name = column_name,
                                                                                         unique_category=unique_category))


# So based on the above results it is evident that 'Table Name' and 'Total/Rural/Urban' categorical features are redundant in nature which can be eliminated as it has no significance.
# 
# We can also eliminate 'State Code' as we are dealing with only Karnataka.
# 
# From above observations we can conclude that we do not need to do any imputation as there are no missing values.
# 
# Before we jump into data visualisations and explore further let us observe the general metrics using the describe function
# 
# ok now let me explore more about the data in detail and come up with some basic observations

# ## Data Cleansing
# 
# Now let us drop some of the columns as discussed above which has no importance in our EDA such as 
# - 'Table Name'
# - 'State Code'
# - 'Total/Rural/Urban'

# In[ ]:


data.drop('Table Name',axis =1,inplace = True)

data.drop('State Code',axis =1,inplace = True)

data.drop('Total/ Rural/ Urban',axis =1,inplace = True)


# Let us further get more insights of the data by observing first few records say for a district & town 

# In[ ]:


data.head()


# #### Few Data Observations:
# 
# The general observation observed in each of the district are as follows
# - 29 unique records for age group for each town code
#     - 'All Ages' category is a summation of all ages 
# - All of the below 12 categories are depicted in the form of persons ,male and female where persons is summation of male and female
#     - Illiterate
#     - Literate
#     - Educational Level - Literate without Educational Level
#     - Educational Level - Below Primary 
#     - Educational Level - Primary
#     - Educational Level - Middle
#     - Educational Level - Matric/Secondary
#     - Educational Level - Higher Secondary/Intermediate Pre-University/Senior Secondary
#     - Educational Level - Non-technical Diploma or Certificate Not Equal to Degree
#     - Educational Level - Technical Diploma or Certificate Not Equal to Degree
#     - Educational Level - Graduate & Above
#     - Unclassified
# Note : Also we have 'Total - Persons', 'Total - Male','Total - Female' which do not have any significance to our analysis as these are summation of all the above categories person/male/female wise for each of the above category
# 
# #### Current Focus :
# 
#  - Since our main focus of this exercise is to improve the 'Primary Education'. So the analysis going forward  as per my assumption that the following categories fall in the need of primary education and rest not
#     - Illiterate 
#     - Educational Level - Literate without Educational Level
#     - Educational Level - Below Primary 
#     - Educational Level - Primary
#     - Unclassified
#     
#    Which means that the following features can be eliminated from the dataset
#    
#     Total - Persons                                                                              
#     Total - Males                                                                                
#     Total - Females                                                                              
#     Literate - Persons                                                                           
#     Literate - Males                                                                             
#     Literate - Females                                                                           
#     Educational Level - Middle Persons                                                           
#     Educational Level - Middle Males                                                             
#     Educational Level - Middle Females                                                           
#     Educational Level - Matric/Secondary Persons                                                 
#     Educational Level - Matric/Secondary Males                                                   
#     Educational Level - Matric/Secondary Females                                                 
#     Educational Level - Higher Secondary/Intermediate Pre-University/Senior Secondary Persons    
#     Educational Level - Higher Secondary/Intermediate Pre-University/Senior Secondary Males      
#     Educational Level - Higher Secondary/Intermediate Pre-University/Senior Secondary Females    
#     Educational Level - Non-technical Diploma or Certificate Not Equal to Degree Persons         
#     Educational Level - Non-technical Diploma or Certificate Not Equal to Degree Males           
#     Educational Level - Non-technical Diploma or Certificate Not Equal to Degree Females         
#     Educational Level - Technical Diploma or Certificate Not Equal to Degree Persons             
#     Educational Level - Technical Diploma or Certificate Not Equal to Degree Males               
#     Educational Level - Technical Diploma or Certificate Not Equal to Degree Females             
#     Educational Level - Graduate & Above Persons                                                 
#     Educational Level - Graduate & Above Males                                                   
#     Educational Level - Graduate & Above Females 
#     
#  
#  #### Key Note:
# 
# For each district code & town code we have the total of all age groups as 'All Ages' category in 'Age Group' feature .In my opinion it is irrelevant as we need to focus on age groups which need primary education.So lets go ahead and remove these rows in the dataset

# In[ ]:


data = data[data['Age-Group'] != 'All ages']


# Now let us look in detail the numerical features that need to be dropped as they do not contribute to the primary education.
# 

# In[ ]:


columns = [ 'Total - Persons',
           'Total - Males',
           'Total - Females',
           'Literate - Persons',
           'Literate - Males',
           'Literate - Females',
           'Educational Level - Middle Persons',
           'Educational Level - Middle Males',
           'Educational Level - Middle Females',
           'Educational Level - Matric/Secondary Persons',
           'Educational Level - Matric/Secondary Males',
           'Educational Level - Matric/Secondary Females',
           'Educational Level - Higher Secondary/Intermediate Pre-University/Senior Secondary Persons',
           'Educational Level - Higher Secondary/Intermediate Pre-University/Senior Secondary Males',
           'Educational Level - Higher Secondary/Intermediate Pre-University/Senior Secondary Females',
           'Educational Level - Non-technical Diploma or Certificate Not Equal to Degree Persons',
           'Educational Level - Non-technical Diploma or Certificate Not Equal to Degree Males',
           'Educational Level - Non-technical Diploma or Certificate Not Equal to Degree Females',
           'Educational Level - Technical Diploma or Certificate Not Equal to Degree Persons',
           'Educational Level - Technical Diploma or Certificate Not Equal to Degree Males',
           'Educational Level - Technical Diploma or Certificate Not Equal to Degree Females',
           'Educational Level - Graduate & Above Persons',
           'Educational Level - Graduate & Above Males',
           'Educational Level - Graduate & Above Females']                                                                            

data.drop(columns,axis =1,inplace = True)


# ## Exploratory Data Analysis:
# 
# In this section I am going to visualize data with respect to categories focused on improving 'Primary Education'.
# 
# - Illiterate 
# - Educational Level - Literate without Educational Level
# - Educational Level - Below Primary 
# - Educational Level - Primary
# - Unclassified
# 
#  I have used Bokeh interactive data visualisation library to visualise data .Please note that the mouse hover over function is enabled to see the data visualisations for each district based on the above mentioned categories depicted below .Also please note the size of the circle depicts the size of the feature .
# 
# The visualisations depict district and age wise data representations for each of the above mentioned groups including total count as well as male and female counts.
# 
# #### Illeiterate :
# 
# In this section you can observe the data visualisation of the following features with respect to district and age group 
# 
# - Illiterate - Persons
# - Illiterate - Males
# - Illiterate - Females

# In[ ]:


source = ColumnDataSource(dict(
    x = data['District Code'],
    y = data['Illiterate - Persons'],
    area = data['Area Name'],
    illerate = data['Illiterate - Persons'],
    illerate_male = data['Illiterate - Males'],
    illerate_female = data['Illiterate - Females'],
    age = data['Age-Group']
)       
)

size_mapper = LinearInterpolator(
    x = [data['Illiterate - Persons'].min(),data['Illiterate - Persons'].max()],
    y = [2,100]
)

color_mapper = CategoricalColorMapper(
    factors = list(data['Area Name'].unique()),
    palette = Spectral8
)

PLOT_OPTS = dict(height = 800,width = 800,x_range = (1,30),y_range=(10,100000))

p = figure(title = 'Illiteracy District/Area Wise',
           toolbar_location = 'above',
           tools = [HoverTool(
               tooltips = [('Area ','@area'),
                           ('Illerate - Total Persons ','@illerate'),
                           ('Illerate - Total Males ','@illerate_male'),
                           ('Illerate - Total Females ','@illerate_female'),
                           ('Age Group ','@age'),
                        ],show_arrow = False)],
           x_axis_label = 'District Code',
           y_axis_label = 'No of Illiterates',
           **PLOT_OPTS)

p.circle(x='x',
         y='y', 
         size = {'field': 'illerate','transform':size_mapper},
         color = {'field': 'area','transform':color_mapper},
         alpha = 0.7,
         legend = 'area',
         source = source)
p.legend.location = (0,-50)
p.right.append(p.legend[0])
p.legend.border_line_color = None
show(p,notebook_handle=True)


# #### Educational Level - Literate without Educational Level :
# 
# In this section you can observe the data visualisation of the following features with respect to district and age group 
# - Educational Level - Literate without Educational Level - Persons
# - Educational Level - Literate without Educational Level - Males
# - Educational Level - Literate without Educational Level - Females

# In[ ]:


source = ColumnDataSource(dict(
    x = data['District Code'],
    y = data['Educational Level - Literate without Educational Level Persons'],
    area = data['Area Name'],
    illerate = data['Educational Level - Literate without Educational Level Persons'],
    illerate_male = data['Educational Level - Literate without Educational Level Males'],
    illerate_female = data['Educational Level - Literate without Educational Level Females'],
    age = data['Age-Group']
)       
)

size_mapper = LinearInterpolator(
    x = [data['Educational Level - Literate without Educational Level Persons'].min(),
         data['Educational Level - Literate without Educational Level Persons'].max()],
    y = [2,100]
)

color_mapper = CategoricalColorMapper(
    factors = list(data['Area Name'].unique()),
    palette = Spectral8
)

PLOT_OPTS = dict(height = 800,width = 800,x_range = (1,30),y_range=(10,6000))

p = figure(title = 'Educational Level - Literate without Educational Level (District vs. Age Wise)',
           toolbar_location = 'above',
           tools = [HoverTool(
               tooltips = [('Area ','@area'),
                           ('Educational Level - Literate without Educational Level - Total Persons ','@illerate'),
                           ('Educational Level - Literate without Educational Level - Total Males ','@illerate_male'),
                           ('Educational Level - Literate without Educational Level - Total Females ','@illerate_female'),
                           ('Age Group ','@age'),
                        ],show_arrow = False)],
           x_axis_label = 'District Code',
           y_axis_label = 'No of Educational Level - Literate without Educational Level',
           **PLOT_OPTS)

p.circle(x='x',
         y='y', 
         size = {'field': 'illerate','transform':size_mapper},
         color = {'field': 'area','transform':color_mapper},
         alpha = 0.7,
         legend = 'area',
         source = source)
p.legend.location = (0,-50)
p.right.append(p.legend[0])
p.legend.border_line_color = None
show(p,notebook_handle=True)


# #### Educational Level - Below Primary:
# 
# In this section you can observe the data visualisation of the following features with respect to district and age group 
# - Educational Level - Below Primary - Persons
# - Educational Level - Below Primary - Males
# - Educational Level - Below Primary - Females

# In[ ]:


source = ColumnDataSource(dict(
    x = data['District Code'],
    y = data['Educational Level - Below Primary Persons'],
    area = data['Area Name'],
    illerate = data['Educational Level - Below Primary Persons'],
    illerate_male = data['Educational Level - Below Primary Males'],
    illerate_female = data['Educational Level - Below Primary Females'],
    age = data['Age-Group']
)       
)

size_mapper = LinearInterpolator(
    x = [data['Educational Level - Below Primary Persons'].min(),
         data['Educational Level - Below Primary Persons'].max()],
    y = [2,50]
)

color_mapper = CategoricalColorMapper(
    factors = list(data['Area Name'].unique()),
    palette = Spectral8
)

PLOT_OPTS = dict(height = 800,width = 800,x_range = (1,30),y_range=(10,100000))

p = figure(title = 'Educational Level - Below Primary (District vs. Age Wise)',
           toolbar_location = 'above',
           tools = [HoverTool(
               tooltips = [('Area ','@area'),
                           ('Educational Level - Below Primary Total Persons ','@illerate'),
                            ('Educational Level - Below Primary Total Males ','@illerate_male'),
                           ('Educational Level -  Below Primary Total Females ','@illerate_female'),
                           ('Age Group ','@age'),
                        ],show_arrow = False)],
           x_axis_label = 'District Code',
           y_axis_label = 'No of Educational Level - Below Primary Persons',
           **PLOT_OPTS)

p.circle(x='x',
         y='y', 
         size = {'field': 'illerate','transform':size_mapper},
         color = {'field': 'area','transform':color_mapper},
         alpha = 0.7,
         legend = 'area',
         source = source)
p.legend.location = (0,-50)
p.right.append(p.legend[0])
p.legend.border_line_color = None
show(p,notebook_handle=True)


# #### Educational Level - Primary:
# 
# In this section you can observe the data visualisation of the following features with respect to district and age group 
# - Educational Level - Primary - Persons
# - Educational Level - Primary - Males
# - Educational Level - Primary - Females

# In[ ]:


source = ColumnDataSource(dict(
    x = data['District Code'],
    y = data['Educational Level - Primary Persons'],
    area = data['Area Name'],
    illerate = data['Educational Level - Primary Persons'],
    illerate_male = data['Educational Level - Primary Males'],
    illerate_female = data['Educational Level - Primary Females'],
    age = data['Age-Group']
)       
)

size_mapper = LinearInterpolator(
    x = [data['Educational Level - Primary Persons'].min(),
         data['Educational Level - Primary Persons'].max()],
    y = [2,50]
)

color_mapper = CategoricalColorMapper(
    factors = list(data['Area Name'].unique()),
    palette = Spectral8
)

PLOT_OPTS = dict(height = 800,width = 800,x_range = (1,30),y_range=(10,100000))

p = figure(title = 'Educational Level - Primary (District vs. Age Wise)',
           toolbar_location = 'above',
           tools = [HoverTool(
               tooltips = [('Area ','@area'),
                           ('Educational Level -  Primary Total Persons ','@illerate'),
                           ('Educational Level -  Primary Total Male ','@illerate_male'),
                           ('Educational Level -  Primary Total Female ','@illerate_female'),
                           ('Age Group ','@age'),
                        ],show_arrow = False)],
           x_axis_label = 'District Code',
           y_axis_label = 'No of Educational Level - Primary Persons',
           **PLOT_OPTS)

p.circle(x='x',
         y='y', 
         size = {'field': 'illerate','transform':size_mapper},
         color = {'field': 'area','transform':color_mapper},
         alpha = 0.7,
         legend = 'area',
         source = source)
p.legend.location = (0,-50)
p.right.append(p.legend[0])
p.legend.border_line_color = None
show(p,notebook_handle=True)


# #### Unclassified:
# 
# In this section you can observe the data visualisation of the following features with respect to district and age group 
# - Unclassified - Persons
# - Unclassified - Males
# - Unclassified - Females

# In[ ]:


source = ColumnDataSource(dict(
    x = data['District Code'],
    y = data['Unclassified - Persons'],
    area = data['Area Name'],
    illerate = data['Unclassified - Persons'],
    illerate_male = data['Unclassified - Males'],
    illerate_female = data['Unclassified - Females'],
    age = data['Age-Group']
)       
)

size_mapper = LinearInterpolator(
    x = [data['Unclassified - Persons'].min(),
         data['Unclassified - Persons'].max()],
    y = [1,100]
)

color_mapper = CategoricalColorMapper(
    factors = list(data['Area Name'].unique()),
    palette = Spectral8
)

PLOT_OPTS = dict(height = 800,width = 800,x_range = (1,30),y_range=(10,400))

p = figure(title = 'Unclassified -  (District vs. Age Wise)',
           toolbar_location = 'above',
           tools = [HoverTool(
               tooltips = [('Area ','@area'),
                           ('Unclassified - Total Persons ','@illerate'),
                           ('Unclassified - Total Males ','@illerate_male'),
                           ('Unclassified - Total Females ','@illerate_female'),
                           ('Age Group ','@age'),
                        ],show_arrow = False)],
           x_axis_label = 'District Code',
           y_axis_label = 'Unclassified - Persons',
           **PLOT_OPTS)

p.circle(x='x',
         y='y', 
         size = {'field': 'illerate','transform':size_mapper},
         color = {'field': 'area','transform':color_mapper},
         alpha = 0.7,
         legend = 'area',
         source = source)
p.legend.location = (0,-50)
p.right.append(p.legend[0])
p.legend.border_line_color = None

show(p,notebook_handle=True)


# Now let us look at summary total count of all categories with respect to total persons,total males & total females for each of the current features of our focus as show below.We are going to create three new features for the same namely
# - Total
# - Total_Males
# - Total_Females

# In[ ]:


data['Total']=data['Illiterate - Persons']+data['Educational Level - Below Primary Persons']+data['Educational Level - Literate without Educational Level Persons']+data['Educational Level - Primary Persons']+data['Unclassified - Persons']
data['Total_Males']=data['Illiterate - Males']+data['Educational Level - Below Primary Males']+data['Educational Level - Literate without Educational Level Males']+data['Educational Level - Primary Males']+data['Unclassified - Males']
data['Total_Females']=data['Illiterate - Females']+data['Educational Level - Below Primary Females']+data['Educational Level - Literate without Educational Level Females']+data['Educational Level - Primary Females']+data['Unclassified - Females']


# Now let us visualise with the above new features created to get a summary holistic view of the entire analysis.
# 
# #### Summary (Total):
# 
# In this section you can observe the data visualisation of the following features with respect to district and age group 
# - Total 
# - Total - Males
# - Total - Females

# In[ ]:


source = ColumnDataSource(dict(
    x = data['District Code'],
    y = data['Total'],
    area = data['Area Name'],
    illerate = data['Total'],
    illerate_male = data['Total_Males'],
    illerate_female = data['Total_Females'],
    age = data['Age-Group']
)       
)

size_mapper = LinearInterpolator(
    x = [data['Total'].min(),
         data['Total'].max()],
    y = [5,100]
)

color_mapper = CategoricalColorMapper(
    factors = list(data['Area Name'].unique()),
    palette = Spectral8
)

PLOT_OPTS = dict(height = 800,width = 800,x_range = (1,30),y_range=(10,120000))

p = figure(title = 'Summary -  (District vs. Age Wise)',
           toolbar_location = 'above',
           tools = [HoverTool(
               tooltips = [('Area ','@area'),
                           ('Summary','@illerate'),
                           ('Total Males ','@illerate_male'),
                           ('Total Females ','@illerate_female'),
                           ('Age Group ','@age'),
                        ],show_arrow = False)],
           x_axis_label = 'District Code',
           y_axis_label = 'Total Population needing Primary Education',
           **PLOT_OPTS)

p.circle(x='x',
         y='y', 
         size = {'field': 'illerate','transform':size_mapper},
         color = {'field': 'area','transform':color_mapper},
         alpha = 0.7,
         legend = 'area',
         source = source)
p.legend.location = (0,-50)
p.right.append(p.legend[0])
p.legend.border_line_color = None
show(p,notebook_handle=True)


# ## Conclusion:
#     
# - The above summary data visualisation depicts that the target districts that need to be focused with respect to primary education as part of the Phase 1 NGO initiative
#     - Hubli Darwad 
#     - Mysore
#     - Bangalore
#     - Belguam
#     - Gulbarga
#     - Bellary
#     - Davanagiri
#     - Mangalore
# - Also it is observed that age groups of 0-6 years and 30-45 years need more attention for most of the cases
# Scope of improvement :
# To perform more detailed analysis and understand the reasons and come up with a predictive model.
# Due to time constraints of doing this exercise this part is left for further exercise .

# ## If you like this  kernel Greatly Appreciate to <font color="red">UPVOTE</font> .  Thank you
# 
# 
