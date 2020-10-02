#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, ShuffleSplit

from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go
from plotly import tools

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

init_notebook_mode(connected=True)


# In[ ]:


# from https://gist.github.com/satra/aa3d19a12b74e9ab7941
from scipy.spatial.distance import pdist, squareform
#from numbapro import jit, float32

def distcorr(X, Y):
    """ Compute the distance correlation function
    
    >>> a = [1,2,3,4,5]
    >>> b = np.array([1,2,9,4,4])
    >>> distcorr(a, b)
    0.762676242417
    """
    X = np.atleast_1d(X)
    Y = np.atleast_1d(Y)
    if np.prod(X.shape) == len(X):
        X = X[:, None]
    if np.prod(Y.shape) == len(Y):
        Y = Y[:, None]
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    n = X.shape[0]
    if Y.shape[0] != X.shape[0]:
        raise ValueError('Number of samples must match')
    a = squareform(pdist(X))
    b = squareform(pdist(Y))
    A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()
    
    dcov2_xy = (A * B).sum()/float(n * n)
    dcov2_xx = (A * A).sum()/float(n * n)
    dcov2_yy = (B * B).sum()/float(n * n)
    dcor = np.sqrt(dcov2_xy)/np.sqrt(np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy))
    return dcor


# In[ ]:


df = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
assert(all(df.isnull().sum()) == False)
assert(all(test.isnull().sum()) == False)


# In[ ]:


features = list(set(df.columns.tolist()) - set(['formation_energy_ev_natom', 'bandgap_energy_ev', 'id']))
targets = ['formation_energy_ev_natom', 'bandgap_energy_ev']


# # Exploratory data analysis
# Gernerally within exploratory data analysis we treyt to see what the data can tell us before the actual task of of modelling or hypothesis testing. We will focus in this notebook on the features found in the train.csv file. Intially we will perform descriptive univariate analysis. Next we will study how the response variables change given a set of featuers and/or their subsets.

# ## Descriptive univariate analysis
# 
# 

# ### Pearson Correlation
# A simple methood that aids in the understanding of the linear correlation within features (and targets). The resulting value is between [-1, 1] . A result of 1 indicates a perfect positive correlation, while a result of -1 means a perfective negatiuve correlation while a result of 0 means that there is no lieaer correlation ebtween the two variables. 
# The correlation between the features as as follows: 

# In[ ]:


corr_df = df[features].corr()

data1 = [go.Heatmap(x=features,
                    y=features,
                    z=corr_df.values,
                   showscale = True,
                   zmin=-1, zmax=1)]

layout = dict(title="Pearsons correlation heatmap- Features", 
                xaxis=dict(
        title='Features',
        titlefont=dict(
            size=18,
        ),
        showticklabels=False,
        ticks="",
        tickangle=90,
        tickfont=dict(
            size=4,
        ),
                
    ),
    yaxis=dict(
        title='Features',
        titlefont=dict(
            size=18,
        ),
        showticklabels=False,
        ticks="",
        tickangle=0,
        tickfont=dict(
            size=0,
        ),
    ),
    width = 750, height = 750,
    autosize = False )

figure = dict(data=data1,layout=layout)
iplot(figure)


# The correlation between the features and the targets are as follows:

# In[ ]:


corr = df[list(set(features)|set(['formation_energy_ev_natom','bandgap_energy_ev']))].corr()

data2 = [go.Heatmap(x=features,
                   y=targets,
                   z=corr.loc[targets, features].values)]

layout2 = dict(title="Correlation heatmap", 
                xaxis=dict(
        title='Features',
        titlefont=dict(
            size=18,
        ),
        showticklabels=False,
        tickangle=90,
        ticks="",
        tickfont=dict(
            size=4,
        ),
                
    ),
    yaxis=dict(
        title='Targets',
        titlefont=dict(
            size=18,
        ),
        showticklabels=False,
        tickangle=0,
        ticks="",
        tickfont=dict(
            size=4,
        ),
    ),
    width = 750, height = 275,
    autosize = False )

figure2 = dict(data=data2,layout=layout2)
iplot(figure2)


# We can see from the above visualizations the following: 
# *  Lattice_vector_1 is highly correlated with lattice_angle_beta_degree and could provide a good opportunity for germinating a new feature from their combination. A logical step from here is to plot both these variables against each other, possibly mapping the response variable on top to generate new features. Finally cross-validation could help in validating the hypothesis that the addition/removal of these features could be statistically significant in improving the prediction score.
# * Bandgap Energy is highly correlated to percent_atom_al and further investigations should be made in the modeling phase to e.g. give more importance to this variable. 
# * Formation Energy is  correlated to lattice_vector_3_ang while its rather weakly correlated to the other features. 

# It is worth noting that generally Pearsons correlation can be very helpful in identifying the direction of dependence (negative or positive). However, special care should be given in the interpretation of the correlation results. This is because the Pearson correlation only measures linear dependence. Furthermore, the Pearson correlation of 0 does not indicate independance. In order to evaluate the linear/nonlinear dependence/independence, it could be helpful to visualize the ** Distance correlation** which we do in the following. 
# 

# ### Distance correlation:
# Distance correlation is a robust method of correlation estmatioan that addresses the mentioned shorcomings of Pearsons  correlation. Namely, identifying nonlinear relationships as evaluating independence. 

# In[ ]:


features_corr_df = pd.DataFrame(data=np.zeros((len(features),len(features))), columns=features, index=features)

for column in features:
    for feature in features:
        features_corr_df.loc[feature, column] = distcorr(df[feature].values, df[column].values)
    
data2 = [go.Heatmap(x=features_corr_df.columns.tolist(),
                    y=features_corr_df.columns.tolist(),
                    z=features_corr_df.values,
                   showscale = True,
                   colorscale = 'Viridis')]

layout = dict(title="Distance correlation heatmap- Features", 
                xaxis=dict(
        title='Features',
        titlefont=dict(
            size=18,
        ),
        showticklabels=False,
        ticks="",
        tickangle=90,
        tickfont=dict(
            size=4,
        ),
                
    ),
    yaxis=dict(
        title='Features',
        titlefont=dict(
            size=18,
        ),
        showticklabels=False,
        ticks="",
        tickangle=0,
        tickfont=dict(
            size=4,
        ),
    ),
    width = 750, height = 750,
    autosize = False )

figure = dict(data=data2,layout=layout)
iplot(figure)


# In[ ]:


target_features_corr_df = pd.DataFrame(data=np.zeros((len(targets),len(features))), 
                                       columns=features, 
                                       index=targets)

for feature in features:
    for target in targets:
        target_features_corr_df.loc[target, feature] = distcorr(df[target].values, df[feature].values)
        
data2 = [go.Heatmap(x=features,
                   y=targets,
                   z=target_features_corr_df.values,
                   colorscale = 'Viridis')]

layout2 = dict(title="Target-Features distance correlation heatmap", 
                xaxis=dict(
        title='Features',
        titlefont=dict(
            size=18,
        ),
        showticklabels=False,
        tickangle=90,
        ticks="",
        tickfont=dict(
            size=4,
        ),
                
    ),
    yaxis=dict(
        title='Targets',
        titlefont=dict(
            size=18,
        ),
        showticklabels=False,
        tickangle=0,
        ticks="",
        tickfont=dict(
            size=4,
        ),
    ),
    width = 750, height = 275,
    autosize = False )

figure2 = dict(data=data2,layout=layout2)
iplot(figure2)


# We can see from the above visualizations the following: 
# *  A few other features appear to be correlated, and could introduce new opportunities for tests (improved scoring given new features)
# * Bandgap Energy is highly correlated to percent_atom_al and percent_atom_in as seen in Pearson correlation.
# * Formation Energy is  correlated to lattice_vector_3_ang , percent_atom_in and percent_atom_ga, special care is to be made when using the different modeling techniques to evaluate this correlation. 

# ### Model-based features analysis
# In the same way that Pearson's correlation coefficient is equivalent to the standardized regression coefficient in a predictive linear regression model, we can build any predictive model to measure the performance of each individual feature in predicting the target variable. In the following we visualize the results of fitting a Random Forest Regressors using one feature at a time to predict one target at a time. We utilize cross-validation and a shallow tree structure to avoid over-fitting.

# In[ ]:


modelbased_corr_df = pd.DataFrame(data=np.zeros((len(targets),len(features))), 
                                  columns=features, 
                                  index=targets)

reg = RandomForestRegressor(n_estimators=20, max_depth=4, n_jobs=-1)
for target in targets:
    y = df[target].values
    for feature in features:
        X = df[feature].values
        score = cross_val_score(reg, 
                                X.reshape(-1, 1), 
                                y.ravel(), 
                                scoring='r2', 
                                cv=ShuffleSplit(n_splits= 5, test_size=.2))
        modelbased_corr_df.loc[target, feature] = round(np.mean(score),3)

data2 = [go.Heatmap(x=modelbased_corr_df.columns.tolist(),
                    y=modelbased_corr_df.index.tolist(),
                    z=modelbased_corr_df.values,
                    colorscale = 'Viridis')]

layout2 = dict(title="Target-Features model-based correlation heatmap", 
                xaxis=dict(
        title='Features',
        titlefont=dict(
            size=18,
        ),
        showticklabels=False,
        tickangle=90,
        ticks="",
        tickfont=dict(
            size=4,
        ),
                
    ),
    yaxis=dict(
        title='Targets',
        titlefont=dict(
            size=18,
        ),
        showticklabels=False,
        tickangle=0,
        ticks="",
        tickfont=dict(
            size=4,
        ),
    ),
    width = 750, height = 250,
    autosize = False )

figure2 = dict(data=data2,layout=layout2)
iplot(figure2)


# We can see from the above visualization the following: 
# * Subgroup could be a more meaningful indicator of Formation energy than bandgap energy.
# * The no. of total atoms is not a very helpful feature. 
# * More interesting set of features pop up for predicting the bandgap energy especially the lattice vector angles. 
# * Formation energy seems to benefit from the lattice_vector[1,2,3]ang features, and should be evaluated in the multivariate feature analysis step. 

# ## Multivariate feature analysis

# ### Lattice vectors vs. Bandgap Energy

# In[ ]:


x = df['lattice_vector_1_ang'].values
y = df['lattice_vector_2_ang'].values
z = df['lattice_vector_3_ang'].values


trace1 = go.Scatter3d(
    x=x,
    y=y,
    z=z,
    mode='markers',
    marker=dict(
        size=12,
        color=df['bandgap_energy_ev'].values,                # set color to an array/list of desired values
        colorscale='Jet',   # choose a colorscale
        opacity=0.5
    )
)

data = [trace1]
layout = go.Layout(
    showlegend=True,
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0
    )
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)


# In[ ]:


x = df['lattice_vector_1_ang'].values
y = df['lattice_vector_2_ang'].values
z = df['lattice_vector_3_ang'].values


trace1 = go.Scatter3d(
    x=x,
    y=y,
    z=z,
    mode='markers',
    marker=dict(
        size=12,
        color=df['formation_energy_ev_natom'].values,                # set color to an array/list of desired values
        colorscale='Jet',   # choose a colorscale
        opacity=0.5
    )
)

data = [trace1]
layout = go.Layout(
    showlegend=True,
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0
    )
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)


# We can see from the above visualization the following: 
# * Unfortunately it seems that Formation energy does not benefit from the lattice_vector[1,2,3]ang features in a visible manner. Special care should be used when building the prediction model if there is any benefit from using these features.  

# ### Atom percentages vs. Formation energy
# To understand how these atom percentages of AL, IN and GA interact with each other and the targets, we visualize them using ternary plots. Ternary plots are suitable when chemical composition  is to be visualized as it allows to three variables to be plotted in a 2D graph. The targets are conveniently then visualized on the triangle plot using a heat-map.
# > A ternary plot, ternary graph, triangle plot, simplex plot, Gibbs triangle or de Finetti diagram is a barycentric plot on three variables which sum to a constant. It graphically depicts the ratios of the three variables as positions in an equilateral triangle. It is used in physical chemistry, petrology, mineralogy, metallurgy, and other physical sciences to show the compositions of systems composed of three species. [[Wikipedia](https://en.wikipedia.org/wiki/Ternary_plot)]

# In[ ]:



def makeAxis(title, tickangle): 
    return {
      'title': title,
      'titlefont': { 'size': 20 },
      'tickangle': tickangle,
      'tickfont': { 'size': 15 },
      'tickcolor': 'rgba(0,0,0,0)',
      'ticklen': 5,
      'showline': True,
      'showgrid': True
    }

data = [{ 
    'type': 'scatterternary',
    'mode': 'markers',
    'a': df['percent_atom_al'].values,
    'b': df['percent_atom_in'].values,
    'c': df['percent_atom_ga'].values,
    'hoverinfo': "a+b+c+name+text",
    'hovertext': "A: AL, B: IN, C: GA",
    'text': df['formation_energy_ev_natom'].values,
    'name': "",
    #'text': "a: AL, b: IN, c: GA",
    'marker': {
        'symbol': 100,
        'size': 12,
        'color': df['formation_energy_ev_natom'].values,                # set color to an array/list of desired values
        'colorscale':'Jet',   # choose a colorscale
        'opacity': 0.8,
        'line': { 'width': 2.5 },
        'showscale': True,
        'label': 'formation_energy_ev_natom'
    },
    }]

layout = {
    'ternary': {
        'sum': 1,
        'aaxis': makeAxis('AL at. %', 0),
        'baxis': makeAxis('<br>IN at. %', 45),
        'caxis': makeAxis('<br>GA at. %', -45),
    },
    'annotations': [{
      'showarrow': False,
      'text': 'Formation Energy at different (AL, IN, GA) at. % ',
        'x': 0.5,
        'y': 1.3,
        'font': { 'size': 15 }
    }]
}

fig = {'data': data, 'layout': layout}
iplot(fig, validate=False)


# We can see from the above visualization the following: 
# *  There seems to be lower values of Formation energy at the edges where IN at. % is 0.
# *  The same is true for 0 values of Al at. %.
# *  Values of Formation Energy seen to increase around the values where GA at. % are minimal.
# *  There seems to be a gradient of increasing Formation Energy from the boundaries where Al at. %  IN at. % are minimal to the boundary where GA at. % are minimal.

# ### Atom percentages vs. Bandgap energy

# In[ ]:


data = [{ 
    'type': 'scatterternary',
    'mode': 'markers',
    'a': df['percent_atom_al'].values,
    'b': df['percent_atom_in'].values,
    'c': df['percent_atom_ga'].values,
#     'hoverinfo': 'text',
#     'hovertext': '',
    'text': "a: AL, b: IN, c: GA",
    'marker': {
        'symbol': 100,
        'size': 12,
        'color': df['bandgap_energy_ev'].values,                # set color to an array/list of desired values
        'colorscale':'Jet',   # choose a colorscale
        'opacity': 0.8,
        'line': { 'width': 2.5 },
        'showscale': True
    },
    }]

layout = {
    'ternary': {
        'sum': 1,
        'aaxis': makeAxis('AL at. %', 0),
        'baxis': makeAxis('<br>IN at. %', 45),
        'caxis': makeAxis('<br>GA at. %', -45),
    },
    'annotations': [{
      'showarrow': False,
      'text': 'Bandgap Energy at different (AL, IN, GA) at. %',
        'x': 0.5,
        'y': 1.3,
        'font': { 'size': 15 }
    }]
}

fig = {'data': data, 'layout': layout}
iplot(fig, validate=False)


# We can see from the above visualization the following: 
# *  There seems to be lower values of Bandgap Energy at the edges where IN at. % and GA at. %  are close to 0.
# *  Values of Bandgap Energy seen to increase around the values where IN at. % and GA at. %  are close to 0 and Al at. % is maximal.
# *  There seems to be a gradient of increasing Bandgap Energy from the boundaries where Al at. %  IN at. % are minimal to the boundary where GA at. % are minimal.

# ### No. of Total Atoms vs. Form & Bandgap energy

# In[ ]:


data = []
for number_of_total_atoms in df['number_of_total_atoms'].value_counts().index.tolist():
    y0 = df[df['number_of_total_atoms']==number_of_total_atoms]['formation_energy_ev_natom'].values
    data.append(go.Box(y=y0, name=str(number_of_total_atoms), boxpoints = 'suspectedoutliers',boxmean='sd'))
    
    layout = go.Layout(
        title = "Number of total atoms vs. Formation energy",
        yaxis=dict( title = 'Formation energy'),
        xaxis=dict( title = 'Number of total Atoms'))
    
iplot(go.Figure(data=data,layout=layout))


# Looking at the boxplot above, it seems that there is a need for some preprocessing needed for the observations where the number of atms = 10. Indeed a quick look at the Formation Energy values for that case yeilds:

# In[ ]:


df[df['number_of_total_atoms']==10]['formation_energy_ev_natom']


# Observation no. [1235, 1983] seem to be outliers and it would be benificial to remove them from the training set. 

# In[ ]:


data = []
for number_of_total_atoms in df['number_of_total_atoms'].value_counts().index.tolist():
    y0 = df[df['number_of_total_atoms']==number_of_total_atoms]['bandgap_energy_ev'].values
    data.append(go.Box(y=y0, name=str(number_of_total_atoms), boxpoints = 'suspectedoutliers',boxmean='sd'))
    layout = go.Layout(
        title = "Number of total atoms vs. Bandgap Energy",
        yaxis=dict( title = 'Bandgap Energy'),
        xaxis=dict( title = 'Number of total Atoms'))
iplot(go.Figure(data=data,layout=layout))


# No real pattern can be observed from the above plots. Generally however, it seems that the Band-gap Energy and Formation Energies are very similar given the different number of atoms. This is expected as the e.g. Distance Correlation was 0.158, 0.142 between No. of total atoms and  Band-gap Energy and Formation Energies respectively.

# ### Space-group vs. Form & Bandgap energy

# In[ ]:


data = []
for spacegroup in df['spacegroup'].value_counts().index.tolist():
    y0 = df[df['spacegroup']==spacegroup]['formation_energy_ev_natom'].values
    data.append(go.Box(y=y0, name=str(spacegroup), boxpoints = 'suspectedoutliers',boxmean='sd'))
    layout = go.Layout(
        title = "Spacegroup vs. Formation Energy",
        yaxis=dict( title = 'Formation Energy'),
        xaxis=dict( title = 'Spacegroup'))
iplot(go.Figure(data=data,layout=layout))    


# In[ ]:


data = []
for spacegroup in df['spacegroup'].value_counts().index.tolist():
    y0 = df[df['spacegroup']==spacegroup]['bandgap_energy_ev'].values
    #y0 = np.log(y0+1)
    data.append(go.Box(y=y0, name=str(spacegroup), boxpoints = 'suspectedoutliers',boxmean='sd'))
    layout = go.Layout(
        title = "Spacegroup vs. Bandgap Energy",
        yaxis=dict( title = 'Bandgap Energy'),
        xaxis=dict( title = 'Spacegroup'))
iplot(go.Figure(data=data,layout=layout))


# We can see from the above visualizations the following: 
# *  Space group variables don't show a real trend and are rather overlapping in case of the Bandgap energy target.

# # End!
# Next, I will try to spend some time calculating features from the train folder and visualising them accordingly! I hope this helped a bit in undersatnding the features in the train.csv file! Thank you for reading :) 

# In[ ]:




