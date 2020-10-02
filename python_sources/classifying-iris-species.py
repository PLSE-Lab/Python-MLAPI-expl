#!/usr/bin/env python
# coding: utf-8

# # Classifying Iris Species

# Let's assume that a hobby botanist is interested in distinguishing the species of some iris flowers that she has found. She has collected some measurements associated with each iris: the length and width of the petals and the length and width of the sepals, all measured in centimetres. <br>
# <br>
# She also has the measurements of some irises that have been previously identified by an expert botanist as belonging to the species setosa, versicolor, or virginica. For these measurements, she can be certain of which species each iris belongs to. Let's assume that these are the only species our hobby botanist will encounter in the wild. <br>
# <br>
# Attribute Information: <br>
# 
# 1. sepal length in cm <br>
# 2. sepal width in cm <br>
# 3. petal length in cm <br>
# 4. petal width in cm <br>
# 5. class: <br>
# -- Iris Setosa <br>
# -- Iris Versicolour <br>
# -- Iris Virginica <br>

# <img src='https://miro.medium.com/max/1000/1*Hh53mOF4Xy4eORjLilKOwA.png' width=650 align='center'/>

# > ### Importing libraries

# In[ ]:


# Importing numpy, pandas and Series + DataFrame:
import numpy as np
import pandas as pd
from pandas import Series, DataFrame

# Imports for plotly:
import plotly.graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots

# Imports for plotting:
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Loading the Data

# In[ ]:


# Importing and loading the iris dataset
iris = pd.read_csv('../input/Iris.csv')


# In[ ]:


# Show first 5 rows of dataset:
header = ff.create_table(iris.head())

header.show()


# In[ ]:


print(iris['Species'].unique())


# There are three iris species in this dataset: setosa, versicolor and virginica. The dataset consists of 6 columns called: Id, SepalLengthCm, SepalWidthCm, PetalLengthCM, PetalWidthCm and Species.

# In[ ]:


# Function to describe variables
def desc(df):
    d = pd.DataFrame(df.dtypes,columns=['Data_Types'])
    d = d.reset_index()
    d['Columns'] = d['index']
    d = d[['Columns','Data_Types']]
    d['Missing'] = df.isnull().sum().values    
    d['Uniques'] = df.nunique().values
    return d


descr = ff.create_table(desc(iris))

descr.show()


# We can see from the table above, that we have no missing values, and the lenght of dataset is 150 rows/samples.

# ## Exploratory Data Analysis (EDA) 

# In[ ]:


# Distritution of Species:

s_df = pd.DataFrame(iris.groupby(['Species'])['Species'].count())

data=go.Bar(x = s_df.index
           , y = s_df.Species
           ,  marker=dict( color=['#0e9aa7', '#f6cd61', '#fe8a71'])
           )



layout = go.Layout(title = 'Distribution of Iris Species'
                   , xaxis = dict(title = 'Species')
                   , yaxis = dict(title = 'Volume')
                  )

fig = go.Figure(data,layout)
fig.show()


# ### Histograms

# In[ ]:


setosa = iris[iris.Species == 'Iris-setosa']
versicolor = iris[iris.Species == 'Iris-versicolor']
virginica = iris[iris.Species == 'Iris-virginica']


# In[ ]:


# Histogram data for Sepal Length

hist_data  = [setosa.SepalLengthCm, versicolor.SepalLengthCm, virginica.SepalLengthCm]

group_labels = ['setosa', 'versicolor', 'virginica']
colors = ['#0e9aa7', '#f6cd61', '#fe8a71']

# Create distplot with curve_type set to 'normal'
fig = ff.create_distplot(hist_data, group_labels, colors=colors,
                         bin_size=.1, show_rug=False)
# Add title
fig.update_layout(title_text='Histogram for Sepal Length'
                  , xaxis = dict(title = 'lenght (cm)')
                  , yaxis = dict(title = 'count')
                 )


fig.show()


# In[ ]:


# Histogram data for Sepal Width

hist_data  = [setosa.SepalWidthCm, versicolor.SepalWidthCm, virginica.SepalWidthCm]

group_labels = ['setosa', 'versicolor', 'virginica']
colors = ['#0e9aa7', '#f6cd61', '#fe8a71']

# Create distplot with curve_type set to 'normal'
fig = ff.create_distplot(hist_data, group_labels, colors=colors,
                         bin_size=.1, show_rug=False)
# Add title
fig.update_layout(title_text='Histogram for Sepal Width'
                  , xaxis = dict(title = 'width (cm)')
                  , yaxis = dict(title = 'count')
                 )

fig.show()


# In[ ]:


# Histogram data for Petal Length

hist_data  = [setosa.PetalLengthCm, versicolor.PetalLengthCm, virginica.PetalLengthCm]

group_labels = ['setosa', 'versicolor', 'virginica']
colors = ['#0e9aa7', '#f6cd61', '#fe8a71']

# Create distplot with curve_type set to 'normal'
fig = ff.create_distplot(hist_data, group_labels, colors=colors,
                         bin_size=.1, show_rug=False)
# Add title
fig.update_layout(title_text='Histogram for Petal Length'
                  , xaxis = dict(title = 'lenght (cm)')
                  , yaxis = dict(title = 'count')
                 )

fig.show()


# In[ ]:


# Histogram data for Petal Width

hist_data  = [setosa.PetalWidthCm, versicolor.PetalWidthCm, virginica.PetalWidthCm]

group_labels = ['setosa', 'versicolor', 'virginica']
colors = ['#0e9aa7', '#f6cd61', '#fe8a71']

# Create distplot with curve_type set to 'normal'
fig = ff.create_distplot(hist_data, group_labels, colors=colors,
                         bin_size=.1, show_rug=False)
# Add title
fig.update_layout(title_text='Histogram for Petal Width'
                  , xaxis = dict(title = 'width (cm)')
                  , yaxis = dict(title = 'count')
                 )

fig.show()


# We can see that we can clearly distinguis setosa species according to petal parameters. Virginica and versicolor parameters are overlaping.

# ### Scatter Plots

# In[ ]:


# Scattergraph for Iris Sepal (length vs width):

fig = go.Figure()

fig.add_trace(go.Scatter(
      x=setosa.SepalLengthCm
    , y=setosa.SepalWidthCm
    , name='setosa'
    , mode='markers'
    , marker_color='#0e9aa7'
))

fig.add_trace(go.Scatter(
      x=versicolor.SepalLengthCm
    , y=versicolor.SepalWidthCm
    , name='versicolor'
    , mode='markers'
    , marker_color='#f6cd61'
))

fig.add_trace(go.Scatter(
      x=virginica.SepalLengthCm
    , y=virginica.SepalWidthCm
    , name='virginica'
    , mode='markers'
    , marker_color='#fe8a71'
))

# Set options common to all traces with fig.update_traces
fig.update_traces(mode='markers'
                 # , marker_line_width=2
                  , marker_size=10)

fig.update_layout(title='Iris Sepal (length vs width)'
                  , xaxis = dict(title = 'length (cm)')
                  , yaxis = dict(title = 'width (cm)')
                 )


fig.show()


# In[ ]:


# Scattergraph for Iris Petal (length vs width):

fig = go.Figure()

fig.add_trace(go.Scatter(
      x=setosa.PetalLengthCm
    , y=setosa.PetalWidthCm
    , name='setosa'
    , mode='markers'
    , marker_color='#0e9aa7'
))

fig.add_trace(go.Scatter(
      x=versicolor.PetalLengthCm
    , y=versicolor.PetalWidthCm
    , name='versicolor'
    , mode='markers'
    , marker_color='#f6cd61'
))

fig.add_trace(go.Scatter(
      x=virginica.PetalLengthCm
    , y=virginica.PetalWidthCm
    , name='virginica'
    , mode='markers'
    , marker_color='#fe8a71'
))

# Set options common to all traces with fig.update_traces
fig.update_traces(mode='markers'
                 # , marker_line_width=2
                  , marker_size=10)

fig.update_layout(title='Iris Petal (length vs width)'
                  , xaxis = dict(title = 'length (cm)')
                  , yaxis = dict(title = 'width (cm)')
                 )


fig.show()


# ### Scatterplot Matrix

# In[ ]:


# Plotting the features of our dataset (this gives us density graphs and scatter plots): 

columns = list(iris.columns)[1:] # remove id column

sns.set(style="ticks")
sns.pairplot(iris[columns], hue='Species',palette=['#0e9aa7', '#f6cd61', '#fe8a71'], height = 2.8)
plt.show()


# ### Box Plots

# In[ ]:


# Box plot for Sepal Length:

fig = go.Figure()
fig.add_trace(go.Box(y=setosa.SepalLengthCm, name='setosa', marker_color='#0e9aa7'))
fig.add_trace(go.Box(y=virginica.SepalLengthCm, name='virginica', marker_color='#fe8a71'))
fig.add_trace(go.Box(y=versicolor.SepalLengthCm, name = 'versicolor',  marker_color='#f6cd61'))

fig.update_layout(title='Sepal Length'
                  , xaxis = dict(title = 'species')
                  , yaxis = dict(title = 'length (cm)')
                 )

fig.show()


# In[ ]:


# Box plot for Sepal Width:

fig = go.Figure()
fig.add_trace(go.Box(y=setosa.SepalWidthCm, name='setosa', marker_color='#0e9aa7'))
fig.add_trace(go.Box(y=virginica.SepalWidthCm, name='virginica', marker_color='#fe8a71'))
fig.add_trace(go.Box(y=versicolor.SepalWidthCm, name = 'versicolor',  marker_color='#f6cd61'))

fig.update_layout(title='Sepal Width'
                  , xaxis = dict(title = 'species')
                  , yaxis = dict(title = 'length (cm)')
                 )

fig.show()


# In[ ]:


# Box plot for Petal Length:

fig = go.Figure()
fig.add_trace(go.Box(y=setosa.PetalLengthCm, name='setosa', marker_color='#0e9aa7'))
fig.add_trace(go.Box(y=virginica.PetalLengthCm, name='virginica', marker_color='#fe8a71'))
fig.add_trace(go.Box(y=versicolor.PetalLengthCm, name = 'versicolor',  marker_color='#f6cd61'))

fig.update_layout(title='Petal Length'
                  , xaxis = dict(title = 'species')
                  , yaxis = dict(title = 'length (cm)')
                 )

fig.show()


# In[ ]:


# Box plot for Petal Width:

fig = go.Figure()
fig.add_trace(go.Box(y=setosa.PetalWidthCm, name='setosa', marker_color='#0e9aa7'))
fig.add_trace(go.Box(y=virginica.PetalWidthCm, name='virginica', marker_color='#fe8a71'))
fig.add_trace(go.Box(y=versicolor.PetalWidthCm, name = 'versicolor',  marker_color='#f6cd61'))

fig.update_layout(title='Sepal Width'
                  , xaxis = dict(title = 'species')
                  , yaxis = dict(title = 'length (cm)')
                 )

fig.show()


# ## Training and testing data

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


# Defining target set y, and a training set X:
y = iris.Species
X = iris.drop('Species', axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 101)


# ### Building the model: K-Nearest Neighbors

# In[ ]:


# The most important parameter of k-Nearest Neighbors classifier is the number of neighbors, which we will set to 1:
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 10)


# In[ ]:


# Fitting the data with knn model:
knn.fit(X_train, y_train)


# ### Evaluating the model 

# In[ ]:


# Using the predict method on KNN to predict values for X_test:
y_pred = knn.predict(X_test)


# In[ ]:


print('Test set score {:.2f}'.format(knn.score(X_test,y_test)))


# In[ ]:


# Importing classification_method and confusion_matrix:
from sklearn.metrics import classification_report, confusion_matrix


# In[ ]:


# Printing out classification report:
print(classification_report(y_test,y_pred))


# In[ ]:


z = confusion_matrix(y_test, y_pred)

x = ['setosa', 'versicolor', 'virginica']
y = ['setosa', 'versicolor', 'virginica']

# change each element of z to type string for annotations
z_text = [[str(y) for y in x] for x in z]

# set up figure 
fig = ff.create_annotated_heatmap(z, x=x, y=y, annotation_text=z_text, colorscale='Portland')

# add title
fig.update_layout(title_text='<i><b>Confusion matrix</b></i>',
                  #xaxis = dict(title='x'),
                  #yaxis = dict(title='x')
                 )

# add custom xaxis title
fig.add_annotation(dict(font=dict(color="black",size=14),
                        x=0.5,
                        y=-0.15,
                        showarrow=False,
                        text="Predicted value",
                        xref="paper",
                        yref="paper"))

# add custom yaxis title
fig.add_annotation(dict(font=dict(color="black",size=14),
                        x=-0.35,
                        y=0.5,
                        showarrow=False,
                        text="Real value",
                        textangle=-90,
                        xref="paper",
                        yref="paper"))

# adjust margins to make room for yaxis title
fig.update_layout(margin=dict(t=50, l=200))

# add colorbar
fig['data'][0]['showscale'] = True
fig.show()


# In[ ]:


# Printing out classification report:
print(classification_report(y_test,y_pred))


# ### Model Improvement  - choosing the best K-value
# 

# In[ ]:


# Creating a for loop that trains various KNN models with different K values:
# Keeping a track of the error_rate for each of these models with a list
error_rate = []

# Will take some time
for i in range(1,11):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))


# In[ ]:


# Line graph for k vs. error rate:

x = list(range(1,11))

fig = go.Figure()
fig.add_trace(go.Scatter(x=x
                         , y=error_rate
                         , mode='lines'
                         , name='Error Rate line'
                        )
             )

fig.add_trace(go.Scatter(x=x
                         , y=error_rate
                         , mode='markers'
                         , name='Error Rate point'
                        )
             )

fig.update_layout(title='Line graph for K value vs. Error Rate'
                  , xaxis_title='K'
                  , yaxis_title='Error Rate'
                 )

fig.show()


# ### Retraining the Model with new K-value
#  

# In[ ]:


# NOW WITH K=6
knn = KNeighborsClassifier(n_neighbors=6)

knn.fit(X_train,y_train)
pred = knn.predict(X_test)

print('WITH K=6')
print('\n')

print(classification_report(y_test,pred))


# In[ ]:


z = confusion_matrix(y_test, pred)

x = ['setosa', 'versicolor', 'virginica']
y = ['setosa', 'versicolor', 'virginica']

# change each element of z to type string for annotations
z_text = [[str(y) for y in x] for x in z]

# set up figure 
fig = ff.create_annotated_heatmap(z, x=x, y=y, annotation_text=z_text, colorscale='Portland')

# add title
fig.update_layout(title_text='<i><b>Confusion matrix</b></i>',
                  #xaxis = dict(title='x'),
                  #yaxis = dict(title='x')
                 )

# add custom xaxis title
fig.add_annotation(dict(font=dict(color="black",size=14),
                        x=0.5,
                        y=-0.15,
                        showarrow=False,
                        text="Predicted value",
                        xref="paper",
                        yref="paper"))

# add custom yaxis title
fig.add_annotation(dict(font=dict(color="black",size=14),
                        x=-0.35,
                        y=0.5,
                        showarrow=False,
                        text="Real value",
                        textangle=-90,
                        xref="paper",
                        yref="paper"))

# adjust margins to make room for yaxis title
fig.update_layout(margin=dict(t=50, l=200))

# add colorbar
fig['data'][0]['showscale'] = True
fig.show()

