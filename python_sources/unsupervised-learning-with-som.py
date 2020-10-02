#!/usr/bin/env python
# coding: utf-8

# # Unsupervised learning with Self-Organizing Maps: MNIST
# 
# In this notebook, I will provide a short example of how Kohonen Self-Organizing Maps (SOM) can be used for dimensionality reduction and unsupervised learning. I will use the MNIST dataset provided by Kaggle to train a SOM and project the handwritten digits to a two-dimensional map that (hopefully) preserves the topological property of the original dataset. Although a number of alternative methods can be used to obtain competitive results in less time (in particular supervised learning with CNN is known to be extremely effective), the visual nature of the dataset and the easy interpretation of its elements (common handwritten digits) make it ideal to explore the capabilities of this method and understand how SOM work.
# 
# This notebook was inspired by [Interactive Intro to Dimensionality Reduction](https://www.kaggle.com/arthurtok/interactive-intro-to-dimensionality-reduction), by Anisotropic, where a number of common dimensionality reduction and clustering methods are presented. 

# ## What are SOM?
# 
# In brief, Self-organizing maps are a type of artificial neural network based on competitive learning (at variance to error-correcting learning typical of other NNs). The idea is to iteratively adapt a connected two-dimensional matrix of vectors (or nodes) to the higher-dimensional topology of the input dataset. 
# At each cycle, a node is selected and its elements (the weights) are updated, together with those of its neighbors, to approach a randomly chosen datapoint from the training set. The competitive element comes into play during the update stage, since the closest node (according to a chosen metric) to the extracted datapoint is selected for the weights update at each iteration.
# 
# SOMs are particularly suited for cases where low-dimensional manifolds are hidden in higher dimensions and are often used together and/or competing with other dimensionality reduction methods and in particular Principal Component Analysis (PCA) for which it could be seen as a non-linear generalization: an exhaustive explanation of SOM's advantages and disadvantages, however, is beyond the scope of this notebook, but there are plenty of resources online for those who would like to know more.

# ## Preparing the Data

# In[ ]:


#As usual we start importing a number of libraries that will be come in handy later on
import numpy as np 
import pandas as pd 
import seaborn as sns
from imageio import imwrite
#from scipy.misc import imsave
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.cbook import get_sample_data
get_ipython().run_line_magic('matplotlib', 'inline')

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

from sklearn.preprocessing import StandardScaler
from PIL import Image, ImageChops


# We will use here SimpSOM, a python library for SOM freely available on [PyPI](https://pypi.python.org/pypi/SimpSOM) and [Github](https://github.com/fcomitani/SimpSOM). 

# In[ ]:


import SimpSOM as sps


# To help the convergence of the map it is a good idea to limit the number of points on which the training will be done. We should thus choose a limited number of landmark points that possibly well represent the distribution of the entire population in the high dimensionality space. In our case we can assume that the distribution is uniform across the ten digits, we can then pick randomly a subset of these for the training. It is important to note, however, that although a higher number of training points does, in theory, increase the accuracy of the mapping by taking into account more variability in the images, it could also make the mapping process more complicated and hinder its convergence. Sometimes you just need a little less data!   
# 
# Here, we will select only 500 points. Let's check that indeed they are equally distributed. It's a good idea to also fix the random seed for reproducibility and debugging purposes since the final results of the mapping will be dependent on its value.
# 
# As shown in other kernels, the images data is unfolded into 28\*28 arrays, with each element containing a single black and white pixel color value. We will keep it this way, and normalize the arrays to improve the map learning.

# In[ ]:


np.random.seed(0)

#We load here the data from the provided training set, we randomly select 500 landmark points and separate the labels. 
train = pd.read_csv('../input/train.csv')
train = train.sample(n=500, random_state=0)
labels = train['label']
train = train.drop("label",axis=1)

#Let's plot the distribution and see if the distribution is uniform
sns.distplot(labels.values,bins=np.arange(-0.5,10.5,1))

#Then we normalize the data, a crucial step to the correct functioning of the SOM algorithm
trainSt = StandardScaler().fit_transform(train.values)


# The resulting distribution is not great, but it's what we can expect by taking only 500 points. It will have to do.
# 
# ## Training the Map
# 
# Now we can proceed in setting up a map and train it with our landmark points.
# 
# First, we need to build a network, it can be as big as one wants, but it should contain at least enough nodes to map all our landmark points separated, possibly more. Obviously, a bigger map allows to distinguish more subtle topological features, but comes at a cost, as the computation time increases with each added node.
# 
# Again, given the limited resources available on interactive kernels, we will build a 40x40 map and we will train it for only 5000 epochs (a good rule of thumb is to cycle for at least 10 times the number of training points) with a relatively high initial learning rate of 0.1. We activate periodic boundary conditions, to avoid artifacts at the borders and we chose PCA as weights initialization.
# The nodes weights can be randomly chosen either from a uniform distribution within the minimum and maximum values of our training dataset (good thing we normalized it beforehand!) or from the space spanned by the two first PCA vectors. We chose here the latter as it may help the map converge faster.
# 
# With these parameters, it will still take around 6 hours on Kaggle's CPU.

# In[ ]:


#We build a 40x40 network and initialise its weights with PCA 
net = sps.somNet(40, 40, trainSt, PBC=True, PCI=True)

#Now we can train it with 0.1 learning rate for 10000 epochs
net.train(0.1, 10000)

#We print to screen the map of the weights differences between nodes, this will help us identify cluster centers 
net.diff_graph(show=True,printout=True)


# Nice! We now have a trained map, but we need to understand what it means. In the plot above the color, the scale goes from dark blue to yellow for nodes whose neighbors weights are less or more distant (or simply different) respectively. This means that darker areas represent cluster basins where all neighbor nodes are similar, while yellow areas are high-difference border regions between basins. Just think of mountains and valleys! 
# 
# ## Visualizing the Results
# 
# What we need to do next is simply project our data (any MNIST data) on the map and see where each image gets mapped.In our case, the interpretation of the data is relatively easy, since all our pictures are labeled with the corresponding category. In pure unsupervised learning projects, where labels are not known, this can be difficult and a simple visual inspection may be problematic. A good approach can be to extract the weighted vectors from the basin center nodes and have a look at them as 28x28 images. Here we can do it for a few basins and see that indeed they are archetypes of specific digits: it works!

# In[ ]:


#Here we first define a few useful functions
def autocrop(fileName):
    im = Image.open(fileName)
    im=im.crop((0,100,2900,im.size[1]))
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)

def posMap(x,y):
     if y%2==0:
        return [y, x*2/np.sqrt(3)*3/4]
     else:
        return [y+0.5, x*2/np.sqrt(3)*3/4]
    
def posCount(x,y):
     return y*40+x

def posCountR(x):
     return [np.int(x%40),np.int(x/40)]


# In[ ]:


#Let's print a few trained nodes' weights and see how good they are
listNodes=[[20,0],[23,11],[1,6],[13,37],[7,33],[18,31]]
listCount=[posCount(20,0), posCount(23,11), posCount(1,6), posCount(13,37), posCount(7,33), posCount(18,31)]

i=0
for node in net.nodeList:
    if i in listCount:
        print('Node\'s position: {:d} {:d}'.format(posCountR(i)[1], posCountR(i)[0]) )
        plt.imshow(np.asarray(node.weights).reshape(28,28))
        plt.axis('off')
        plt.show()
    i+=1


# As expected, some of them are quite similar to specific digits archetypes, while others still contain generic and undefined shapes, either because they lay closer to a border region between different categorical clusters, or simply because in need of more training.
# Let's now map our training points to see where they ended up by matching them with their closest node. We map the results interactively with plotly, hovering on each point with the mouse we can see the original label. To help visualize the clusters, each label has been colored differently in the scatter plot.

# In[ ]:


projData=net.project(trainSt[:500])


# In[ ]:


#We first save a cropped version of the original map to superimpose and then we add the scatterpoints
cropped = autocrop('nodesDifference.png')
cropped.save('cropped.png')


# In[ ]:


#And here we prepare the plotly graph. 
trace0 = go.Scatter(
    x = [x for x,y in projData],
    y = [y for x,y in projData],
#    name = labels,
    hoveron = [str(n) for n in labels],
    text = [str(n) for n in labels],
    mode = 'markers',
    marker = dict(
        size = 8,
        color = labels,
        colorscale ='Jet',
        showscale = False,
        opacity = 1
    ),
    showlegend = False

)
data = [trace0]

layout = go.Layout(
    images= [dict(
                  source= "cropped.png",
                  xref= "x",
                  yref= "y",
                  x= -0.5,
                  y= 39.5*2/np.sqrt(3)*3/4,
                  sizex= 40.5,
                  sizey= 40*2/np.sqrt(3)*3/4,
                  sizing= "stretch",
                  opacity= 0.5,
                  layer= "below")],
    width = 800,
    height = 800,
    hovermode= 'closest',
    xaxis= dict(
        range=[-1,41],
        zeroline=False,
        showgrid=False,
        ticks='',
        showticklabels=False
    ),
    yaxis=dict(
        range=[-1,41],
        zeroline=False,
        showgrid=False,
        ticks='',
        showticklabels=False
    ),
    showlegend= True
)


fig = dict(data=data, layout=layout)
py.iplot(fig, filename='styled-scatter')


# Nice! The algorithm is able to clearly separate in clusters some of our categories. Their centers are mostly found in the "basins" where the weights difference is minimal (in dark blue). The topology of the high dimensionality space is also somehow respected, with similar shapes mapped closed together (e.g. handwritten 7 and 9) and others far apart (such 1 and 0).  Unfortunately, however, the accuracy is not great, as there are a number of outliers. This issue could be easily solved by fine-tuning the algorithm parameters, with further training and a better representative set of landmark points. Introducing more variability in the training set, by adding random rotations and translations, could really make the difference.

# 
# 
# ## Conclusion
# 
# 

# Self-Organizing Maps represent an interesting alternative to more commonly used unsupervised learning and dimensionality reduction algorithms. Visualizing the topology of the dataset can be helpful in identifying hidden patterns and relationships between clusters of datapoints. The algorithm can be easily extended and used in conjunction with other methods (such as PCA to initialize the weights or k-means to cluster the projected data on top of the map) to tackle more challenging problems. 
