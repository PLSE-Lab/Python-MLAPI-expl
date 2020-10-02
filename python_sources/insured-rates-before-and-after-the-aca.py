#!/usr/bin/env python
# coding: utf-8

# <h1 align="center"> Visualizing Changes in Healthcare Insurance Coverage (2010-2015)
# <h3 align="center"> by: Martin Cisneros

# # Contents  
#    
# ## Introduction
# [Introduction](#intro)  
# 
# ## Data Wrangling
# [Import Basic Libraries](#libs1)  
# [Retrieve Health Insurance Coverage Data](#kaggle)  
# [Clean the Data](#clean)  
# [Mutate Full State Name for the Abbreviated State Names](#abbr) 
# 
# ## Visualizations 
# [Import Visualization Libraries](#libs2)  
# [USA Choropleth Map - Insured Rate Changes by State](#usmap)   
# [Box Plot - Medicaid State Expansion](#box)  
# [Scatter Plots - Relationship between Insured Rate Changes and Other Variables](#scatter)  
# [USA Choropleth Map - Medicaid Expansion by State](#usmap2) 
# 
# ## Conclusion
# [Conclusion](#conclusion) 
# 
# 
# ## Appendix
# [Predictive Analysis](#model)  

# <a id='intro'></a>
# # Introduction

# The _Affordable Care Act (ACA)_ is the name for the health care reform law which addresses health insurance coverage, costs, and preventive care. The ACA was signed into law on March 23, 2010. 
# 
# The following questions will be addressed: 
# 
# -   How has the Affordable Care Act changed the rate of citizens with health insurance coverage? 
# -   Which states observed the greatest increase in their insured rate? 
# -   Did those states expand Medicaid program coverage and/or implement a health insurance marketplace? 
# 
# On March 24th 2017, House Republican leaders pulled legislation to repeal the _Affordable Care Act_. In light of these recent developments, let's take a look at ACA's impact on US health care thus far.

# # Data Wrangling

# <a id='libs1'></a>
# ### Import Basic Libraries

# In[ ]:


import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')


# <a id='kaggle'></a>
# ### Retrieve Health Insurance Coverage Data

# In[ ]:


filename1 = '../input/states.csv'
ACA = pd.read_csv(filename1)

ACA.head(1)


# <a id='clean'></a>
# ### Clean the Affordable Care Act Data

# Let's check the data types for potential cleaning. 

# In[ ]:


ACA.dtypes


# In[ ]:


# Convert %'s to float64
ACA['Uninsured Rate Change (2010-2015)'] = ACA['Uninsured Rate Change (2010-2015)'].replace('%','',regex=True).astype('float')
ACA['Uninsured Rate (2010)'] = ACA['Uninsured Rate (2010)'].replace('%','',regex=True).astype('float')
ACA['Uninsured Rate (2015)'] = ACA['Uninsured Rate (2015)'].replace('%','',regex=True).astype('float')

# Convert $ to int64
convertdollartoint = lambda x: int(x.strip('$'))
ACA['Average Monthly Tax Credit (2016)'] = ACA['Average Monthly Tax Credit (2016)'].map(convertdollartoint)

# Remove white spaces from original data's State column
ACA['State'] = ACA['State'].str.strip()


# In[ ]:


# Remove the USA entry 
ACA = ACA[ACA['State'] != 'United States']


# In[ ]:


# Sign correction to create an 'Insured Rate Change' column
ACA['Insured Rate Change (2010-2015)'] = ACA['Uninsured Rate Change (2010-2015)'].apply(lambda x: x*-1)
#ACA


# In[ ]:


# Create a new column for .pretty Medicaid Expansion information
def convert_bool(bool):
        if bool == True:
            bool = 'Yes'
        elif bool == False:
            bool = 'No'
        return bool

MedicaidExpansion = ACA['State Medicaid Expansion (2016)']
ACA['MedicaidExpansion'] = MedicaidExpansion.apply(convert_bool)


# <a id='abbr'></a>
# ###  Mutate Full State Name for the Abbreviated State Names

# We need US State abbreviations for Map visualization tool. 

# In[ ]:


# Create a new dictionary with State Name and Abbreviation mappings

states_abr = ["AK", "AL", "AR", "AZ", "CA", "CO", "CT", "DC", "DE", "FL", "GA",
                                         "HI", "IA", "ID", "IL", "IN", "KS", "KY", "LA", "MA", "MD", "ME",
                                         "MI", "MN", "MO", "MS",  "MT", "NC", "ND", "NE", "NH", "NJ", "NM",
                                         "NV", "NY", "OH", "OK", "OR", "PA", "PR", "RI", "SC", "SD", "TN",
                                         "TX", "UT", "VA", "VT", "WA", "WI", "WV", "WY"]

state_full = ["Alaska","Alabama","Arkansas","Arizona","California","Colorado",
                                       "Connecticut","District of Columbia","Delaware","Florida","Georgia",
                                       "Hawaii","Iowa","Idaho","Illinois","Indiana","Kansas","Kentucky",
                                       "Louisiana","Massachusetts","Maryland","Maine","Michigan","Minnesota",
                                       "Missouri","Mississippi","Montana","North Carolina","North Dakota",
                                       "Nebraska","New Hampshire","New Jersey","New Mexico","Nevada",
                                       "New York","Ohio","Oklahoma","Oregon","Pennsylvania","Puerto Rico",
                                       "Rhode Island","South Carolina","South Dakota","Tennessee","Texas",
                                       "Utah","Virginia","Vermont","Washington","Wisconsin",
                                       "West Virginia","Wyoming"]

state_hash = dict(zip(states_abr, state_full))

states_abr_df = pd.DataFrame([[key,value] for key,value in state_hash.items()],columns=["abbr","name"])


# In[ ]:


# Join to ACA dataset

def combine_dfs(ACA, states_abr_df):
        return pd.merge(ACA, states_abr_df, left_on = 'State', right_on = 'name', how = 'left')

ACA = combine_dfs(ACA, states_abr_df)


# View full dataset ready for analysis. 

# In[ ]:


#ACA.describe()
#ACA


# # Visualizations

# <a id='libs2'></a>
# ### Import Visualization Libraries

# In[ ]:


import matplotlib.pyplot as plt
from altair import Chart, X, Y, Axis, Color, Scale, Legend
import plotly.offline as offline
import plotly.graph_objs as go

offline.init_notebook_mode()
get_ipython().run_line_magic('matplotlib', 'inline')


# <a id='usmap'></a>
# ### USA Choropleth Map - Insured Rate Changes by State

# To visualize which states observed the greatest increase in their insured rate, let's look at a map of the US with a color coded scale respective to insured rate changes. I used Plotly for states heat map: https://plot.ly/python/choropleth-maps/

# In[ ]:


scl = [[0.0, 'rgb(242,240,247)'],[0.2, 'rgb(218,218,235)'],[0.4, 'rgb(188,189,220)'],            [0.6, 'rgb(158,154,200)'],[0.8, 'rgb(117,107,177)'],[1.0, 'rgb(29, 22, 71)']]


ACA['text'] = ACA['State']+'<br>'+    'Medicaid Expansion? '+ACA['MedicaidExpansion']
    
data = [ dict(
        type='choropleth',
        colorscale = scl,
        autocolorscale = False,
        locations = ACA['abbr'],
        z = ACA['Insured Rate Change (2010-2015)'] ,
        locationmode = 'USA-states',
        text = ACA['text'],
        marker = dict(
            line = dict (
                color = 'rgb(2, 2, 2)',
                width = 1
            ) ),
        colorbar = dict(
            title = "% Change ('10-'15)")
        ) ]

layout = dict(
        title = 'Insured Rate Change (2010-2015)',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
        width= 800,
        margin = dict(
            l=0,
            r=50,
            b=100,
            t=100,
            pad=4)
             )
    
fig = dict(data=data, layout=layout)
fig1 = offline.iplot(fig, filename='d3-cloropleth-map')


# States with the highest insured rate change between 2010 and 2015 are: Nevada, Oregon, California, Kentucky, New Mexico, and West Virginia. All expanded Medicaid programs. 

# <a id='box'></a>
# ### Box Plots - Medicaid State Expansion Impact on Insured Rate Changes

# Did those states with highest insured rate changes expand Medicaid program coverage and/or implement a health insurance marketplace?

# In[ ]:


# Create new dataframes for medicaid expansion data
expansion_yes = ACA[ACA['State Medicaid Expansion (2016)'] == True]
expansion_no = ACA[ACA['State Medicaid Expansion (2016)'] == False]

expansion = [expansion_yes['Insured Rate Change (2010-2015)'],expansion_no['Insured Rate Change (2010-2015)']] 


# In[ ]:


# Summary statistics for medicaid expansion lists 
print(expansion_yes['Insured Rate Change (2010-2015)'].describe())
print(expansion_no['Insured Rate Change (2010-2015)'].describe())


# There were 32 states with expanded Medicaid programs and 19 without.

# In[ ]:


USA_Average = ACA.mean()['Insured Rate Change (2010-2015)']
print(USA_Average)


# In[ ]:


fig = plt.figure(1, figsize=(9, 6))
ax = fig.add_subplot(111)
bp = ax.boxplot(expansion, patch_artist=True)
plt.xlabel('Medicaid Expansion')
plt.ylabel('% Insured Increase')
plt.title('Medicaid Expansion and Insured Rate Changes')
ax.set_xticklabels(['Yes', 'No'])

# Remove top axes and right axes ticks
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()

#  Change outline color, fill color and linewidth of the boxes
for box in bp['boxes']:
    # change outline color
    box.set( color='#1d1647', linewidth=1)
    # change fill color
    box.set( facecolor = '#7570b3' )

# Change color and linewidth of the whiskers
for whisker in bp['whiskers']:
    whisker.set(color='#1d1647', linewidth=1)

# Change color and linewidth of the caps
for cap in bp['caps']:
    cap.set(color='#1d1647', linewidth=1)

# Change color and linewidth of the medians
for median in bp['medians']:
    median.set(color='#ffffff', linewidth=1)

# Change the style of fliers and their fill
for flier in bp['fliers']:
    flier.set(marker='o', color='#ffffff', alpha=0.5)


# For the purpose of simplifying the following commentary, let's refer to states with expanded programs as _Expanders_ and states without expansions as _Non-expanders_. From the box and whisker plot we can deduce the following: 
# 
# - _Expanders_ had a **slightly higher average** increase in insured rate changes than _Non-expanders_.
# - The spread of insured rate change in _Expanders_ is **much larger** than the spread in _Non-expanders_. 
# - Roughly **25%** of _Expanders_ had higher insured rate increases than **all** _Non-expanders_.
# - Skewness for _Expanders_ is **positive**. Skewness for _Non-expanders_ is **symmetrical** (if anything just slightly negative). 
# 

# <a id='scatter'></a>
# ### Scatter Plots - Relationship between Insured Rate Changes and Other Variables

# Let's look at how Insured Rate Changes relate to variables such as: 
# - Marketplace Tax Credits (2016)
# - Employer Health Insurance Coverage (2015)
# - Marketplace Health Insurance Coverage (2016)
# - Medicaid Enrollment Change (2013-2016)
# 
# I am using the declarative statistical visualization library known as Altair: https://altair-viz.github.io/

# In[ ]:


MedicaidEnrollmentChange = Chart(ACA).mark_point(filled=True).encode(
    color = Color('MedicaidExpansion', 
    legend=Legend(title='Medicaid Expansion?'),
    scale=Scale(domain=['Yes', 'No'],
                range=['#4840ad', '#db3f66'])),
    x=X('Medicaid Enrollment Change (2013-2016)', axis=Axis(title='Medicaid Enrollment Change (2013-2016)')),
    y=Y('Insured Rate Change (2010-2015)', axis=Axis(title='Insured Rate Change (2010-2015)'))
)

MedicaidEnrollmentChange


# In[ ]:


MarketplaceTaxCredits = Chart(ACA).mark_point(filled=True).encode(
    color = Color('MedicaidExpansion', 
    legend=Legend(title='Medicaid Expansion?'),
    scale=Scale(domain=['Yes', 'No'],
                range=['#4840ad', '#db3f66'])),
    x=X('Marketplace Tax Credits (2016)', axis=Axis(title='Marketplace Tax Credits (2016)')),
    y=Y('Insured Rate Change (2010-2015)', axis=Axis(title='Insured Rate Change (2010-2015)'))
)

MarketplaceTaxCredits


# In[ ]:


EmployerHealthInsuranceCoverage = Chart(ACA).mark_point(filled=True).encode(
    color = Color('MedicaidExpansion', 
    legend=Legend(title='Medicaid Expansion?'),
    scale=Scale(domain=['Yes', 'No'],
                range=['#4840ad', '#db3f66'])),
    x=X('Employer Health Insurance Coverage (2015)', axis=Axis(title='Employer Health Insurance Coverage (2015)')),
    y=Y('Insured Rate Change (2010-2015)', axis=Axis(title='Insured Rate Change (2010-2015)'))
)

EmployerHealthInsuranceCoverage


# In[ ]:


MarketplaceHealthInsuranceCoverage = Chart(ACA).mark_point(filled=True).encode(
    color = Color('MedicaidExpansion', 
    legend=Legend(title='Medicaid Expansion?'),
    scale=Scale(domain=['Yes', 'No'],
                range=['#4840ad', '#db3f66'])),
    x=X('Marketplace Health Insurance Coverage (2016)', axis=Axis(title='Marketplace Health Insurance Coverage (2016)')),
    y=Y('Insured Rate Change (2010-2015)', axis=Axis(title='Insured Rate Change (2010-2015)'))
)

MarketplaceHealthInsuranceCoverage


# In[ ]:


AVGMonthlyTaxCredit = Chart(ACA).mark_point(filled=True).encode(
    color = Color('MedicaidExpansion', 
    legend=Legend(title='Medicaid Expansion?'),
    scale=Scale(domain=['Yes', 'No'],
                range=['#4840ad', '#db3f66'])),
    x=X('Average Monthly Tax Credit (2016)', axis=Axis(title='Average Monthly Tax Credit (2016)')),
    y=Y('Insured Rate Change (2010-2015)', axis=Axis(title='Insured Rate Change (2010-2015)'))
)

AVGMonthlyTaxCredit


# <a id='usmap2'></a>
# ### USA Choropleth Map - Medicaid Expansion by State

# In[ ]:


expansion_yes['text'] = expansion_yes['State']+'<br>'+            'Insured Rate Change '+expansion_yes['Insured Rate Change (2010-2015)'].astype('str')+'%'

expansion_no['text'] = expansion_no['State']+'<br>'+    'Insured Rate Change '+expansion_no['Insured Rate Change (2010-2015)'].astype('str')+'%'
    
trace1 = dict(
    type='choropleth',
    z=[1]*32,
    autocolorscale=False,
    colorscale=[[0, 'rgb(255,255,255)'], [1, 'rgb(117,112,179)']],
    showscale=False,
    hoverinfo='text',
    locationmode='USA-states',
    locations = expansion_yes['abbr'],
    name='Medicaid Expansion',
    text = expansion_yes['text'],
    zauto=False,
    zmax=1,
    zmin=0,

)
trace2 = dict(
    type='choropleth',
    z=[1]*19,
    autocolorscale=False,
    colorscale=[[0, 'rgb(255,255,255)'], [1, 'rgb(221,86,120)']],
    hoverinfo='text',
    locationmode='USA-states',
    locations = expansion_no['abbr'],
    name='No Expansion',
    showscale=False,
    text = expansion_no['text'],
    zauto=False,
    zmax=1,
    zmin=0,
)

data = ([trace1, trace2])
layout = dict(
        title = 'Current Status of State Medicaid Expansion Decisions',
        autosize=False,
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255,255,255)'),
        images=list([
            dict(
                x=1,
                y=0.6,
                sizex=0.19,
                sizey=0.4,
                source='http://i.imgur.com/eukX4DD.png', 
                xanchor='right',
                xref='paper',
                yanchor='bottom',
                yref='paper'
            )
        ]),
        width= 800,
        margin = dict(
            l=0,
            r=50,
            b=100,
            t=100,
            pad=4)
    )

fig = dict(data=data, layout=layout)
fig2 = offline.iplot(fig, filename='d3-cloropleth-map')


# <a id='conclusion'></a>
# # Conclusion

# We set out to answer the following questions: 
# 
# - How has the Affordable Care Act changed the rate of citizens with health insurance coverage?
# - Which states observed the greatest increase in their insured rate?
# - Did those states expand Medicaid program coverage and/or implement a health insurance marketplace?
# 
# 
# The Affordable Care Act increased the rate of citizens with health insurance coverage by an average of **5.43%** across all states. The states with the greatest increase in their insured rate were **Nevada, Oregon, California, Kentucky, New Mexico, and West Virginia**. 
# 
# A central goal of the ACA was to reduce the number of uninsured by increasing access to affordable coverage options through Medicaid and the Health Insurance Marketplace. On June 28, 2012, the U.S. Supreme Court issued its decision regarding the constitutionality of the individual mandate and state-level Medicaid expansion mandate. The ruling made the Medicaid expansion optional for states. Roughly **25%** of states with Medicaid expansion had higher insured rate increases than **all** states without Medicaid expansion. 
# 
# Since there is no deadline for states to implement the Medicaid expansion, future decisions by those **19** states without Medicaid expansion programs will contribute to the continued reduction of their uninsured rates. 

# <a id='model'></a>
# # Appendix

# The following analysis was conducted with the attempt to predict what will happen to the nationwide insured rate in the upcoming years; however, small sample size (51 states) and the lack of independent predictor variables limit the scope for target variable prediction (insured rate) through linear and random forest regression. As expected, both predictive models result in extremely high error. 
# 
# [Import Machine Learning Libraries](#libs3)  
# [Data Treatment](#treatment)  
# [K-means Clustering](#kmeans)  
# [Plotting Clusters through PCA Dimensionality Reduction](#pca)  
# [Identifying Target and Predictor Variables](#variables)  
# [Fitting a Linear Regression](#regression)  
# [Random Forest Regression](#forest)  

# <a id='libs3'></a>
# ### Import ML Libraries

# In[ ]:


# Tree Based Modeling 

from sklearn.preprocessing import Imputer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor


# <a id='treatment'></a>
# ### Prepare Data for Analysis

# In[ ]:


# Identify the variables with missing values 

ACA.isnull().any()


# In[ ]:


# Impute missing values with mean

imp=Imputer(missing_values="NaN", strategy="mean" )

imp.fit(ACA[["Medicaid Enrollment (2013)"]])
ACA["Medicaid Enrollment (2013)"]=imp.transform(ACA[["Medicaid Enrollment (2013)"]]).ravel()

imp.fit(ACA[["Medicaid Enrollment Change (2013-2016)"]])
ACA["Medicaid Enrollment Change (2013-2016)"]=imp.transform(ACA[["Medicaid Enrollment Change (2013-2016)"]]).ravel()


# In[ ]:


# Split data into train and test sets

train=ACA.sample(frac=0.7,random_state=200)
test=ACA.drop(train.index)

train['Type']='Train'
test['Type']='Test'

model = pd.concat([train,test],axis=0) 


# <a id='kmeans'></a>
# ### K-means Clustering

# In[ ]:


# Initialize the model with 2 parameters -- number of clusters and random state.
kmeans_model = KMeans(n_clusters=4, random_state=1)

# Get only the numeric columns from games.
good_columns = model._get_numeric_data()

# Fit the model using the good columns.
kmeans_model.fit(good_columns)

# Get the cluster assignments.
labels = kmeans_model.labels_


# <a id='pca'></a>
# ### Plotting Clusters through PCA Dimensionality Reduction

# In[ ]:


# Create a PCA model.
pca_2 = PCA(2)

# Fit the PCA model on the numeric columns from earlier.
plot_columns = pca_2.fit_transform(good_columns)

# Make a scatter plot of each game, shaded according to cluster assignment.
plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=labels)

# Show the plot.
plt.show()


# <a id='variables'></a>
# ### Identifying Target and Predictor Variables

# In[ ]:


# Find correlations between our continuous target variable
model.corr()["Insured Rate Change (2010-2015)"].sort_values(ascending=0)


# Many of these variables are not independent of the target variable or will cause multicollinearity issues. 

# In[ ]:


# Get all the columns from the dataframe.
columns = model.columns.tolist()

# Filter the columns to include the ones we want.
columns = [c for c in columns if c in ["State Medicaid Expansion (2016)", "Marketplace Tax Credits (2016)"]]

# Store the variable we'll be predicting on.
target = "Insured Rate Change (2010-2015)"


# In[ ]:


# Generate the training set.  Set random_state to be able to replicate results.
train = ACA.sample(frac=0.7, random_state=1)

# Select anything not in the training set and put it in the testing set.
test = ACA.loc[~ACA.index.isin(train.index)]

# Print the shapes of both sets.
print(train.shape)
print(test.shape)


# <a id='regression'></a>
# ### Fitting a Linear Regression

# In[ ]:


# Initialize the model class.
model1 = LinearRegression()

# Fit the model to the training data.
model1.fit(train[columns], train[target])


# In[ ]:


# Generate our predictions for the test set.
predictions1 = model1.predict(test[columns])

# Compute error between our test predictions and the actual values.
mean_squared_error(predictions1, test[target])


# <a id='forest'></a>
# ### Random Forest Regression

# In[ ]:


# Initialize the model with some parameters.
model2 = RandomForestRegressor(n_estimators=100, min_samples_leaf=10, random_state=1)

# Fit the model to the data.
model2.fit(train[columns], train[target])

# Make predictions.
predictions2 = model2.predict(test[columns])

# Compute the error.
mean_squared_error(predictions2, test[target])


# On average, our predictions are 3.9-4.5% away from the actual values! Almost half of the full range of actual values. Too high. 
