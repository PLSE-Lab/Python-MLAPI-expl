#!/usr/bin/env python
# coding: utf-8

# # Iris Dataset Visualization
# Libraries used for visualization:
# 1. Numpy
# 2. Pandas
# 3. Matplotlib
# 4. Seaborn

# In[ ]:


# Importing modules
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import parallel_coordinates
from pandas.plotting import andrews_curves
import plotly.graph_objs as go
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
from mpl_toolkits.mplot3d import Axes3D


# ## Exploring the Iris Dataset

# In[ ]:


# Set default Seaborn style
sns.set()

# Loading the Iris dataset, which is in the "../input/" directory
iris = pd.read_csv("../input/Iris.csv") # the Iris dataset is now a Pandas DataFrame

# Viewing the dataset
iris.head()


# In[ ]:


# Shape of the dataset
print('Shape of the dataset:' , iris.shape)

# Number of columns in the data
print('Number of columns: ', len(iris.columns))

# Number of examples of each Species
iris["Species"].value_counts()


# In[ ]:


# Describing data
iris.describe()


# In[ ]:


iris.isnull().sum()


# The above result concludes that the Iris Dataset has no missing values.

# ##  Extracting data

# In[ ]:


# Extacting petal lengths of species 
versicolor_petal_length = iris[ iris["Species"]  == "Iris-versicolor"].iloc[:,3].values
virginica_petal_length = iris[ iris["Species"]  == "Iris-virginica"].iloc[:,3].values
setosa_petal_length = iris[ iris["Species"]  == "Iris-setosa"].iloc[:,3].values

# Extacting petal widths of species 
versicolor_petal_width = iris[ iris["Species"]  == "Iris-versicolor"].iloc[:,4].values
virginica_petal_width = iris[ iris["Species"]  == "Iris-virginica"].iloc[:,4].values
setosa_petal_width = iris[ iris["Species"]  == "Iris-setosa"].iloc[:,4].values


# ## Plotting
# ### Seaborn Multi-plot grid for plotting conditional relationships
# The plot below shows relation between Sepal length and width for different species

# In[ ]:


# Seaborn Facet Grid plot
sns.FacetGrid(iris, hue = "Species", height = 10 )     .map(plt.scatter, "SepalLengthCm", "SepalWidthCm", s = 150 )     .add_legend() 


# ### Swarmplot using Seaborn
# The following plot shows the distribution of Petal length for different species

# In[ ]:


# Resetting plots
sns.reset_orig()
sns.set()

# Create bee swarm plot with Seaborn's default settings
sns.swarmplot(x = "Species", y = "PetalLengthCm", data = iris)

# Label the axes
_ = plt.xlabel('Species')
_ = plt.ylabel('Petal length (cm)')

# Show the plot
plt.show()


# ### Empirical distribution function
# Helps visualize all the datapoints

# In[ ]:


def ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements."""
    # Number of data points: n
    n = len(data)

    # x-data for the ECDF: x
    x = np.sort(data)

    # y-data for the ECDF: y
    y = np.arange(1, n + 1 ) / n

    return x, y

# Compute ECDFs
x_set, y_set = ecdf(setosa_petal_length)
x_vers, y_vers = ecdf(versicolor_petal_length)
x_virg, y_virg = ecdf(virginica_petal_length)

# Plot all ECDFs on the same plot
_ = plt.plot(x_set, y_set, marker = ".", linestyle = "none")
_ = plt.plot(x_vers, y_vers, marker = ".", linestyle = "none")
_ = plt.plot(x_virg, y_virg, marker = ".", linestyle = "none")

# Annotate the plot
plt.legend(('setosa', 'versicolor', 'virginica'), loc='lower right')
_ = plt.xlabel('petal length (cm)')
_ = plt.ylabel('ECDF')

# Display the plot
plt.show()


# ### Boxplot
# Boxplots are useful for identifying outliers. The plot below indicates that Setosa has 3 outliers and Versicolor 1.

# In[ ]:


# Create box plot with Seaborn's manual settings
sns.set(font_scale=1.6, rc={'figure.figsize':(12, 10)})
sns.boxplot(x = "Species", y = 'PetalLengthCm', data = iris)

# Label the axes
_ = plt.xlabel('Species')
_ = plt.ylabel('Petal Length (cm)')

# Show the plot
plt.show()


# ****Visual Exploratory data analysis (EDA)****
# 

# Paired plot using Seaborn

# In[ ]:


# Setting default settings of Seaborn
sns.set()

# Plotting
sns.pairplot(iris[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species']], hue="Species", diag_kind="kde")


# ### Violin plot using Seaborn

# In[ ]:


plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
sns.violinplot(x='Species',y='PetalLengthCm',data=iris)
plt.subplot(2,2,2)
sns.violinplot(x='Species',y='PetalWidthCm',data=iris)
plt.subplot(2,2,3)
sns.violinplot(x='Species',y='SepalLengthCm',data=iris)
plt.subplot(2,2,4)
sns.violinplot(x='Species',y='SepalWidthCm',data=iris)


# ### Histogram plots

# In[ ]:


exclude = ['Id']
iris.loc[:, iris.columns.difference(exclude)].hist() 
plt.figure(figsize=(15,10))
plt.show()


# ### Parallel Coordinates
# Parallel coordinates plots each feature on a separate column & then draws lines connecting the features for each data sample

# In[ ]:


data = iris.drop(['Id'],axis=1)
# Make the plot
plt.figure(figsize=(15,10))
parallel_coordinates(data, 'Species', colormap=plt.get_cmap("Set1"))
plt.title("Iris data class visualization according to features (setosa, versicolor, virginica)")
plt.xlabel("Features of data set")
plt.ylabel("cm")
plt.savefig('graph.png')
plt.show()


# ### Andrew Curves
# Andrews Curves involve using attributes of samples as coefficients for Fourier series and then plotting these

# In[ ]:


andrews_curves(iris.drop("Id", axis=1), "Species")


# ### Basic 3D Scatter Plot (Plotly)
# 

# In[ ]:


# Data of Setosa
iris_setosa = iris[iris['Species'] == "Iris-setosa"]

#  Data of Versicolor
iris_versicolor = iris[iris['Species'] == "Iris-versicolor"]

#  Data of Virginica
iris_virginica = iris[iris['Species'] == "Iris-virginica"]

# set1 =  Setosa
trace1 = go.Scatter3d(
    x=iris_setosa.SepalLengthCm,
    y=iris_setosa.SepalWidthCm,
    z=iris_setosa.PetalLengthCm,
    mode='markers',
    name = "iris_setosa",
    marker=dict(
        color='rgb(217, 100, 100)',
        size=12,
        line=dict(
            color='rgb(255, 255, 255)',
            width=0.1
        )
    )
)
# set2 =  Versicolor
trace2 = go.Scatter3d(
    x=iris_versicolor.SepalLengthCm,
    y=iris_versicolor.SepalWidthCm,
    z=iris_versicolor.PetalLengthCm,
    mode='markers',
    name = "iris_versicolor",
    marker=dict(
        color='rgb(0, 128, 225)',
        size=12,
        line=dict(
            color='rgb(204, 204, 204)',
            width=0.1
        )
    )
)

# set3 =  Virginica
trace3 = go.Scatter3d(
    x=iris_virginica.SepalLengthCm,
    y=iris_virginica.SepalWidthCm,
    z=iris_virginica.PetalLengthCm,
    mode='markers',
    name = "iris_virginica",
    marker=dict(
        color='rgb(54, 170, 127)',
        size=12,
        line=dict(
            color='rgb(204, 204, 204)',
            width=0.1
        )
    )
)

data = [trace1, trace2, trace3]
layout = go.Layout(
    title = ' 3D plot of Setosa, Versicolor and Virginica',
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0
    ),
    scene = dict(
    xaxis = dict(title='SepalLengthCm'),
    yaxis = dict(title='SepalWidthCm'),
    zaxis = dict(title='PetalLengthCm'),),
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)


# In[ ]:


global grafico #figure
#Function scatter_plot group data by argument name, plot and edit labels
def scatter_plot(x_label,y_label,z_label,clase,c,m,label):
    x = iris[ iris['Species'] == clase ][x_label] #groupby Name column x_label
    y = iris[ iris['Species'] == clase ][y_label]
    z = iris[ iris['Species'] == clase ][z_label]
    # s: size point; alpha: transparent 0, opaque 1; label:legend
    grafico.scatter(x,y,z,color=c, edgecolors='k',s=50, alpha=0.9, marker=m,label=label)
    grafico.set_xlabel(x_label)
    grafico.set_ylabel(y_label)
    grafico.set_zlabel(z_label)
    return 

grafico = plt.figure().gca(projection='3d')  #new figure
scatter_plot('SepalLengthCm','SepalWidthCm','PetalLengthCm','Iris-virginica','g','o','Iris-virginica')
scatter_plot('SepalLengthCm','SepalWidthCm','PetalLengthCm','Iris-versicolor','b','o','Iris-versicolor')
scatter_plot('SepalLengthCm','SepalWidthCm','PetalLengthCm','Iris-setosa','r','o','Iris-setosa')
plt.legend()
plt.show()


# In[ ]:


data = []
clusters = []
colors = ['rgb(228,26,28)','rgb(55,126,184)','rgb(77,175,74)']

for i in range(len(iris['Species'].unique())):
    name = iris['Species'].unique()[i]
    color = colors[i]
    x = iris[ iris['Species'] == name ]['SepalLengthCm']
    y = iris[ iris['Species'] == name ]['SepalWidthCm']
    z = iris[ iris['Species'] == name ]['PetalLengthCm']
    
    trace = dict(
        name = name,
        x = x, y = y, z = z,
        type = "scatter3d",    
        mode = 'markers',
        marker = dict( size=3, color=color, line=dict(width=0) ) )
    data.append( trace )

layout = dict(
    width=800,
    height=550,
    autosize=False,
    title='Iris dataset',
    scene=dict(
        xaxis=dict(
            title = "SepalLengthCm",
            gridcolor='rgb(255, 255, 255)',
            zerolinecolor='rgb(255, 255, 255)',
            showbackground=True,
            backgroundcolor='rgb(230, 230,230)'
        ),
        yaxis=dict(
            title = "SepalWidthCm",
            gridcolor='rgb(255, 255, 255)',
            zerolinecolor='rgb(255, 255, 255)',
            showbackground=True,
            backgroundcolor='rgb(230, 230,230)'
        ),
        zaxis=dict(
            title = "PetalWidthCm",
            gridcolor='rgb(255, 255, 255)',
            zerolinecolor='rgb(255, 255, 255)',
            showbackground=True,
            backgroundcolor='rgb(230, 230,230)'
        ),
        aspectratio = dict( x=1, y=1, z=0.7 ),
        aspectmode = 'manual'        
    ),
)

fig = dict(data=data, layout=layout)

# IPython notebook
iplot(fig, filename='pandas-3d-iris', validate=False)

