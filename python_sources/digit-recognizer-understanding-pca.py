#!/usr/bin/env python
# coding: utf-8

# Let's assume we have a ${n}$-dimensional dataset where ${n}$ is a very large number. Training an algorithm in a such high dimensional feature space can be computationally very intensive.
# Some of these features might be correlated. How do we capture the essence of the data without having to go through every single dimension? Is it possible to represent the same data in lower dimensions ? 
# That's where dimensionality reduction methods like PCA help.  Once the mathematical definition of PCA is stated, we will work on trying to derive why the  PCA is defined the way it is using basic principles.
# 
# The definition of PCA is as follows:  
# The principal components of ${X}$ ${\in}$ ${R^{n}}$ are defined as  the components of 
# ${Z}$=${A'}$${X^{*}}$         where ${X^{*}}$ is the standardized  version of the original ${X}$   and  ${A}$ is the matrix that consists of the eigenvectors of the correlation matrix of  ${X^{*}}$
# 
# 
# Ok now to work up from some of the basics 
# 
# Let's assume a single  sample in our data  is ${X}$ = ${[  x_{1 },  x_{2},...,x_{n} ]^{T}}$  , an ${n}$ dimensional vector where each $x_{i}$ is the component along the ${i}$-th feature axis (basis) .    
# The features ${x_{i}}$ help distinguish one data point from another in this ${n}$-dimensional space.  Ok but how?  
# 
#  
# 
# **Variance**  
# 
# This is a measure of how spread out the individual values  in a distribution are from their mean value.  
# The more variance a particular feature exhibits i.e along a certain feature axis, the better the data can be separated along this axis.    
# 
# I demonstrate this claim with the simple case of ten data points where each point is described using 2 features. (2D data).  
# 
# The distribution of this data is such that the mean along feature ${x_{1}}$  i.e. ${\mu _{x_{1}}}=0$ and the mean along feature ${x_{2}}$ i.e. ${\mu _{x_{2}}}=3$.  
# I will change the variance along each feature axes  from 0 (no deviation from the mean) to a high value and we will see why variance helps identify the ten points nicely in the latter case.  
# 
# First let's import all relevant libraries.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly.offline as py
from numpy import linalg as LA
py.init_notebook_mode(connected=True)

import plotly.graph_objs as go
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/"))

# Any results you write to the current directory are saved as output.


# Next create 10 2-D points with ${\mu_{x_{1}}=0}$ and ${\mu_{x_{2}}=3}$.  
# Here I am defining a function for creating the distribution.  
# 
# 

# In[ ]:


#var_x1=0, var_x2=3
mu1,mu2,N=0,3,10
def plot_gaussian(sigma1,sigma2):     #var = sigma*sigma
    x1=np.random.normal(mu1,sigma1,N)    # gaussian distributed feature x1
    x2=np.random.normal(mu2,sigma2,N)    # gaussian distributed feature x2
    trace=go.Scatter(
    x=x1,
    y=x2,
    mode='markers',
    )
    data=[trace]
    layout=go.Layout(
        xaxis=dict(
                range=[-100,100]
        ),
        yaxis=dict(
                range=[-100,100]
        ),
    )
    fig=dict(data=data,layout=layout)
    py.iplot(fig)


# Case 1: ${var_{x_{1}}=0}$, ${var_{x_{2}}=0}$

# In[ ]:


plot_gaussian(sigma1=0,sigma2=0)


# no variance along ${x_{1}}$ and ${x_{2}}$ implies all 10 points lie at the point (0,3) i.e. all data lies on the mean $({\mu_{x_{1}}},{\mu_{x_{2}}})$. **There is no way to separate out the data points** in this case of zero variance along both the axes . Essentially implying all 10 data points are the same. (e.g.  10 t-shirts all green all marked "L". Although in essence each one is different but when plotted just using these 2 features every t shirt is a green L )   
# 
# Next we change the variance along ${x_{1}}$ 

# In[ ]:


plot_gaussian(sigma1=10,sigma2=0)


# In this case  we see that all 10 points are separated because of the variance along feature ${x_{1}}$ but  there is no way to distinguish the data along the ${x_{2}}$ axis since there is no variance along ${x_{2}}$.  (10 t shirts, now showing the whole spectrum of colors (${x_{1}}$) but all sized "L" (${x_{2}}$) )  
# 
# 
# Next we consider the case where data is varying along both the axes.  However to stress my point of larger variance along a feature implying better distinction along the feature , I am distributing the data such that there is more variance along ${x_{1}}$ as compared to ${x_{2}}$ (10 shirts with a higher color range and a small size range S,M,L )
# 

# In[ ]:


plot_gaussian(sigma1=10,sigma2=2)


# We note that **the direction along which there is  a large variation** helps distinguish the data better as compared to a direction where the variation isn't as large.  (easier to pick a red shirt out of 4 M sized tshirts). 
# 
# Ok, so why was this exercise important?  
# Because this ends up as one of the constraints that we will use while deriving the PCA formulation .  
# 
# PCA is essentially a  transformation of the existing feature coordinate system to a new coordinate system, the feature vectors of which satisfy the following constraints:  
# 
# 1. The directions of the new vectors will  be ones where the variance of the data is maximized. (the variance argument we noted above)
# 
# 2. The new transformed axes should be orthogonal.
# 
# 3. Yet another constraint we might impose is that these basis vectors be of magnitude 1 i.e normalized. (This is to keep a constant scaling)
# 
# So out of all the possible vectors in this n-dimensional space we need to find a set of ${n}$-dimensional vectors that satisfy the above 3 constraints.  
# 
# 
# If our original basis vector set is ${\vec{B}=\{\vec{b_{1}},\vec{b_{2}},\cdots,\vec{b_{n}}\}}$ where each ${\vec{b_{i}}}$ is a unit vector lying along the ${i}$-th feature axis then a single data sample ${\vec{x}=x_{1}{\vec{b_{1}}} + x_{2}{\vec{b_{2}}} + \cdots +x_{n}{\vec{b_{n}}}   }$ where ${x_{i}}$ is it's ${i}$-th feature  
# Similarly any vector ${\vec{\alpha_{i}}}$ in this space can then be expressed as  ${\vec{\alpha_{i}}=\alpha_{1}\vec{b_{1}}+\alpha_{2}\vec{b_{2}}+\cdots+\alpha_{n}\vec{b_{n}}  }$    
# *Note that ${\vec\alpha_{i}}$ is not amongst the samples of our data. It is a direction in the space spanned by our basis vectors. i.e it is a vector that can be "reached" using a linear combination of our basis vectors*  
# 
# Now we need this ${\vec{\alpha_{i}}}$ to fulfill all 3 of the above constraints. i.e.  
# 1. Maximize variance of ${\vec x}$ when ${\vec x}$ is projected along ${\vec{\alpha_{i}}}$ i.e maximize the variance which can be expressed as  
# ${var(\vec\alpha_{i}
# \vec{x})=\vec {\alpha_{i}}  {\sum} \vec {\alpha_{i}}^{T} }$ where ${\sum}$ is the covariance matrix of the sample ${\vec{x}}$
# 2.  The new feature axes are orthogonal to each other. i.e . No one feature cannot be expressed as a linear combination of the remaining features.  This is expressed as ${<\vec{\alpha_{i}}^{T}.\vec{\alpha_{j}}>=0}$ i.e the inner product is zero . In a 2D sense, this implies that the dot product of the two new feature vectors are zero. No component of one feature is projected on another (think of the spatial ${\hat{i},\hat{j}}$ vectors that lie along the perpendicular x-y axes) .   
# This also implies that the i-th Principal Component  ${\vec{\alpha_{i}}^{T}\vec{x}}$ is uncorrelated to the j-th Principal Component  ${\vec{\alpha_{j}}^{T}\vec{x}\space\implies\space cov[{\vec{\alpha_{i}}^{T}\vec{x}},{\vec{\alpha_{j}}^{T}\vec{x}}]=0}$
# 3.  ${||\vec{\alpha_{i}}||=\vec {\alpha}^{T} \vec {\alpha}=1}$
# 
# Let's find the first such vector ${\vec\alpha_{1}}$  
# We use the method of [Lagrange multipliers](https://en.wikipedia.org/wiki/Lagrange_multiplier) to find the correct ${\vec \alpha_{1}}$  that fulfills all the above contraints.
# 
#  So the equation we need to solve is  ${\space\space }$ ${\vec{\alpha_{1}}^{T} \sum \vec {\alpha_{1}}   -\lambda_{1}(\vec {\alpha_{1}}^{T} \vec {\alpha_{1}} -1)=0}$    ${\space\space\space}$where ${\lambda_{1} }$ is the Lagrange multiplier  
# 
# ${\vec{\alpha_{1}}^{T} \sum \vec {\alpha_{1}}   -\lambda_{1}(\vec {\alpha_{1}}^{T} \vec {\alpha_{1}} -1)=0}$  
# ${ Differentiating \space w.r.t. \space {\vec{\alpha_{1}}^{T}}}$  
# ${\sum\vec{\alpha_{1}} -\lambda_{1}\vec{\alpha_{1}}=0}$  
# which translates to ${\sum\vec{\alpha_{1}}=\lambda_{1}\vec\alpha_{1}}$ ${\space}$ i.e. ${\vec\alpha_{1}}$ is an eigenvector of the covariance matrix of ${x}$ i.e  ${\sum}$  with the eigenvalue ${\lambda_{1}}$  
# Also since ${\sum\vec{\alpha_{1}}=\lambda_{1}\vec\alpha_{1}}$ ${\space\implies\space}$  ${\vec{\alpha_{1}}^{T}\sum\vec{\alpha_{1}}=\lambda_{1}\vec{\alpha}^{T}\vec\alpha_{1}}$ ${\space\implies\space}$ ${\vec{\alpha_{1}}^{T}\sum\vec{\alpha_{1}}=\lambda_{1} \space(since \space \vec {\alpha_{1}}^{T} \vec {\alpha_{1}}=1 )}$ ${\space\implies\space}$ ${var(\vec\alpha_{1}
# \vec{x})=\lambda_{1}}$   
# 
# i.e **${\lambda_{1}}$ is the variance of the data ${\vec{x}}$ when measured along ${\vec\alpha_{1}}$.** 
# If there are p-eigenvectors for ${\sum}$, we need the one that gives maximum variance i.e ${\lambda_{1}}$ needs to be as large as possible. **Therefore ${\vec\alpha_{1}}$ corresponds to the eigenvector of ${\sum}$ with the largest eigenvalue**
# 
# 
# Inorder to find the next vector ${\alpha_{2}}$ we again use the same technique. However note that while the previous find of ${\alpha_{1}}$ took into account only two of the three constraints (max variance and normalization), for finding the next vector, we include the third constraint i.e orthogonality ${\alpha_{2}^T \alpha_{1}=0}$  
# Also from #2 ${cov[{\vec{\alpha_{1}}^{T}\vec{x}},{\vec{\alpha_{2}}^{T}\vec{x}}]=0}$  
# Now ${cov[{\vec{\alpha_{1}}^{T}\vec{x}},{\vec{\alpha_{2}}^{T}\vec{x}}]=\vec\alpha_{1}^{T}\sum\vec\alpha_{2}=\vec\alpha_{2}^{T}\sum\vec\alpha_{1}=\vec\alpha_{2}^{T}\lambda_{1}\vec\alpha_{1}=\lambda_{1}\vec\alpha_{2}^{T}\alpha_{1}=\lambda_{1}\vec\alpha_{1}^{T}\alpha_{2}=0}$  
# Thus any of the equations   
# 
# ${\vec\alpha_{1}^{T}\sum\vec\alpha_{2}=0\space,\space \vec\alpha_{2}^{T}\sum\vec\alpha_{1}=0}$  
# ${\space\space\vec\alpha_{2}^{T}\alpha_{1}=0\space,\space\space\vec\alpha_{1}^{T}\alpha_{2}=0}$  
# satisfies ${cov[{\vec{\alpha_{1}}^{T}\vec{x}},{\vec{\alpha_{2}}^{T}\vec{x}}]=0}$   
# 
# Ok now to find ${\vec\alpha_{2}}$ using Lagrange multipliers.
# ${\vec{\alpha_{2}}^{T} \sum \vec {\alpha_{2}}   -\lambda_{2}(\vec {\alpha_{2}}^{T} \vec {\alpha_{2}} -1)-\phi(\vec\alpha_{2}^{T}\vec\alpha_{1}-0)=0}$  where ${\lambda_{2}}$ and ${\phi}$ are Lagrange multipliers  
# and since  ${\vec\alpha_{2}^{T}\alpha_{1}=0}$ the equation reduces to ${\vec{\alpha_{2}}^{T} \sum \vec {\alpha_{2}}   -\lambda_{2}(\vec {\alpha_{2}}^{T} \vec {\alpha_{2}} -1)=0 \implies \sum\alpha_{2}=\lambda_{2}\alpha_{2}}$  
# i.e ${\lambda_{2}}$ is the eigenvalue of the eigenvector ${\vec\alpha_{2}}$ of the covariance matrix ${\sum}$ of ${\vec{x}}$  and since ${\lambda_{1}}$ is the already the largest eigenvalue , which makes ${\lambda_{2}}$ the next largest.   
# Similarly we can find ${\alpha_{3},\alpha_{4}}$..etc
# 
# We can therefore find a set of ${p}$ eigenvectors  ${\{\alpha_{1},\alpha_{2},\cdots,\alpha_{p}\}}$ such that their corresponding eigenvalues are in the order ${\lambda_{1} > \lambda_{2} > \cdots > \lambda_{p}}$
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 

#  We would like  that the new features (basis vectors) in this lower dimensional space to be orthogonal. i.e. non-redundant. No one feature can be expressed as a linear combination of the remaining features in this new vector space. (Consider the opposite case, where all features are redundant. i.e imagine ending up with a high dimensional vector e.g. the house price described with features of floor size in cms,inches and meters. No new information is added by the inclusion of the second and third feature in this case that the first one doesn't already provide. Orthogonality of features brings more information that leads to a better separation of data )   
# Next step would be to transform the data into this new coordinate system where the basis vectors are the eigenvectors we just found.    
# 
# 
# We can put all the ${p}$ eigenvectors into a transformation matrix ${T}$ and then transform the original data ${X}$ to ${X'}$ using    ** ${X'=XT}$**
# 
# Let's see the whole process using some dummy data.

# 

# Here I have created some dummy 2-D data where each point that shows a correlation between its two feature axes.

# In[ ]:




from  numpy.random import multivariate_normal as mvr_gauss
from sklearn.preprocessing import StandardScaler

number_samples=5000
mu=np.array([6.0,4.0])   # define the means of the data distributed along both the axes, I chose mu-x=6 and mu-y=4
#now the desired covariance matrix
#the diagonal elements of the covariance matrix indicates the variance along the feature 
# while the off-diagonal elements shows the variance between the two features. i.e the covariance
cvr=np.array([
    [15,-4],                           
    [-4,4]
])

#Generate the random samples
xf=mvr_gauss(mu,cvr,number_samples)    # xf is now 5000 samples each with 2 features i.e a 5000 X 2 matrix

stdsc=StandardScaler()                #standardization of data
xf_std=stdsc.fit_transform(xf)

#Create a plotly trace
trace=go.Scatter(
    x=xf_std[:,0],              #xf[:,0] is the feature1
    y=xf_std[:,1],              #xf[:,1] is the feature2
    mode='markers',
        
)
data=[trace]

#layout for naming the axes
layout=go.Layout(
            title='example of 2D correlated data', autosize=False,
        
            xaxis=dict(title= 'feature 1',range=[-10,20]),
            yaxis=dict(title= 'feature 2')#,range=[-10,20]),
         )


fig=dict(data=data,layout=layout)
py.iplot(fig)


# 
# Our first task it to find the eigenvectors of the covariance matrix of our data.  
# 

# In[ ]:



pxd=pd.DataFrame(xf_std)

#calculate the covariance matrix of the dataframe
sig=pxd.cov()

#calculate the eigenvectors and eigenvalues
eigvals,eigvecs=LA.eig(sig)


# Now to plot the eigenvectors on the data

# In[ ]:


#Create a plotly trace
import plotly.figure_factory as ff
from plotly import tools

trace1=go.Scatter(
    x=xf_std[:,0],              #xf_std[:,0] is the feature1
    y=xf_std[:,1],              #xf_std[:,1] is the feature2
    mode='markers',
    opacity=0.5,
    name='data'
    
)

#fig=tools.make_subplots(1,1)
mu_x=0
mu_y=0
x0,y0=[mu_x,mu_x],[mu_y,mu_y]
u,v=eigvecs[:,0]*5,eigvecs[:,1]*5

scale=1
fig=ff.create_quiver(x0,y0,u,v,scale,arrow_scale=0.03)
'''fig['layout']['autosize']=False
fig['layout']['xaxis']['range']=[-8,8]
fig['layout']['yaxis']['range']=[-8,8]
'''
fig.data[0].name='eigenvectors'
#layout for naming the axes
'''layout=go.Layout(
            title='example of 2D correlated data',
            autosize= False,
            xaxis=dict(title= 'feature 1'),
            yaxis=dict(title= 'feature 2'),
            
         )'''
figg=tools.make_subplots(rows=1,cols=1)
#fig.append_trace(trace1,1,1)
figg.add_trace(fig.data[0],1,1)
figg.add_trace(trace1,1,1)
figg.layout.autosize=False
figg.layout.xaxis.range=[-6,6]
figg.layout.yaxis.range=[-6,6]

#fig.layout=layout

#fig=dict(data=data,layout=layout)
py.iplot(figg)


# Now to transform the data to the new coordinate system where the eigenvectors are now the axes. 
# We will create a transformation matrix ${T=[\alpha_{1},\alpha_{2}]}$ where ${\alpha_{1},\alpha_{2}}$ are the two eigenvectors in our example.  
# Then transform the data using ${X'=XT}$

# In[ ]:


t=np.array(eigvecs).T
xf_p=np.dot(xf_std,t)


# In[ ]:


#Create a plotly trace
import plotly.figure_factory as ff
from plotly import tools

trace1=go.Scatter(
    x=xf_std[:,0],              #xf_std[:,0] is the feature1
    y=xf_std[:,1],              #xf_std[:,1] is the feature2
    mode='markers',
    opacity=0.5,
    name='original standardized data'
        
)
#Transformed data
trace_transformed=go.Scatter(
    x=xf_p[:,0],
    y=xf_p[:,1],
    mode='markers',
    opacity=0.8,
    name='transformed data' 
    
)

#fig=tools.make_subplots(1,1)
mu_x=0
mu_y=0
x0,y0=[mu_x,mu_x],[mu_y,mu_y]
u,v=eigvecs[:,0]*5,eigvecs[:,1]*5
figg=tools.make_subplots(rows=1,cols=2)
scale=1
fig=ff.create_quiver(x0,y0,u,v,scale,arrow_scale=0.03)
fig['layout']['autosize']=False
fig.data[0]['name']='eigenvectors'

#layout for naming the axes
'''layout=go.Layout(
            title='example of 2D correlated data',
            autosize= True,
            xaxis=dict(title= 'feature 1'),
            yaxis=dict(title= 'feature 2'),
            
         )
'''

figg.add_trace(fig.data[0],1,1)
figg.add_trace(trace1,1,1)
figg.add_trace(trace_transformed,1,2)
figg.layout['xaxis2']['range']=[-6,6]
figg.layout['yaxis2']['range']=[-6,6]
figg.layout['xaxis2']['title']='eigenvector 1'
figg.layout['yaxis2']['title']='eigenvector 2'
figg.layout['xaxis']['title']='feature 1'
figg.layout['yaxis']['title']='feature 2'

py.iplot(figg)


# The data is more variant on the RHS diagram along the vertical axis (direction of eigenvector with the largest ${\lambda}$)  
# 
# **Why we standardize and the correlation matrix**  
# PCA can be defined using the covariance matrix and also using the correlation matrix of ${X}$. Since the two are related as follows:  
# ${corr_{ij}=\dfrac{cov_{ij}}{(\sigma_{i}\sigma_{j})}}$  where ${\sigma_{i}}$ and ${\sigma_{j}}$ are the standard deviations for the ${i}$-th and ${j}$-th features respectively.
# 
# If we are working with the covariance matrix of X, then the entries of the covariance matrix of X changes as the units used to express X changes. i.e. imagine 2 simple vector samples consisting of height and weight of a person ${{A}=\matrix{174      & 85  \\
#     175       & 88  \\}}$ where the height is in cms and the weight in kgs . Now the same samples when  expressed in units of meters and kgs becomes ${{A}=\matrix{1.74      & 85  \\
#     1.75       & 88  \\}}$    
# 
# 
# The covariance matrix in the first case is  ${{s_{1}}=\matrix{0.5      & 0.5  \\ 
#     0.5       & 0.5  \\}}$ and for the second case its ${{s_{2}}=\matrix{0.00005      & 0.005  \\
#     0.005       & 0.500  \\}}$ which leads to two different sets of eigenvectors.   
#    
#    Next lets consider the case of the correlation matrix . You will find that the correlation matrix in either of the two above cases comes down to   ${{corr}=\matrix{1.0      & 1.0  \\
#     1.0  & 1.0  \\}}$      
#     
#    When we standardize the data, we are bringing it to zero mean and unit variance .  
#    The covariance matrix of such data is the same as it's correlation matrix based on the above relation between the two.  
#    Therefore upon standardizing the data , we end up with the same set of eigenvectors from correlation matrix as we do from the covariance matrix.

# **PCA using scikit-learn on high - dim data ** 
# 
# Now that some of the basics are covered, we move to using the [PCA functionality offered by scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)

# In[ ]:


from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
#constants
NUMBER_OF_TRAINING_IMGS=5000

#load the MNIST dataset
labeled_images=pd.read_csv('../input/train.csv') 
images=labeled_images.iloc[0:NUMBER_OF_TRAINING_IMGS,1:] # first NUMBER_OF_TRAINING_IMGS rows,column 2 onwards.
labels=labeled_images.iloc[0:NUMBER_OF_TRAINING_IMGS,:1] #first NUMBER_OF_TRAINING_IMGS rows, first column. 

#split into train-test
train_images,test_images,train_labels,test_labels=train_test_split(images,labels,test_size=0.2,random_state=13)

#standardize the data
#stdsc=StandardScaler()
stdsc=MinMaxScaler()
train_images_std=stdsc.fit_transform(train_images)
test_images_std=stdsc.transform(test_images)
#perform PCA on training data,getting all possible eigenvectors
pca=PCA(svd_solver='randomized',whiten=True)
pca.fit(train_images_std)


# The PCA extracted all possible eigenvectors (since the parameter 'n_components' was unset) .  
# We now need to come up with an ideal number of dimensions wherein maximum variance of the data is retained.   For that we use the *'explained_variance_ratio_' *parameter. which for the ${i}$-th eigenvector is the quantity ${\dfrac{\lambda_{i}}{\sum_{k} \lambda_{k}}}$

# In[ ]:


#print(pca.n_components_)
#print(pca.explained_variance_ratio_)
def cummulative(ll):
    nl=np.empty(len(ll))
    for i in range(len(ll)):
        if i==0:
            nl[i]=ll[i]
        else:
            nl[i]=nl[i-1]+ll[i]
    #print(nl)
    return nl/10


fig=tools.make_subplots(rows=1,cols=1)
bardata=go.Bar(
    x=[xx for xx in range(pca.n_components_)],
    y=[xx for xx in pca.explained_variance_ratio_],
    opacity=1.0,
    name='explained variance ratio'
)

cummulativeData=go.Bar(
    x=[xx for xx in range(pca.n_components_)],
    y=cummulative(pca.explained_variance_ratio_),
    opacity=0.4,
    name='cumulative sum (divided by 10)'
    )


#data=[bardata,cummulativeData]
fig.add_trace(bardata,1,1)
fig.add_trace(cummulativeData,1,1)
py.iplot(fig)


# From the above graph it looks like 94% of the variance in the data is retained by the first 144 eigenvectors. I am choosing 144 here as i intend to later resize the images to 12x12 for display

# In[ ]:


pca=PCA(n_components=144,svd_solver='randomized',whiten=True)
train_images_pca=pca.fit_transform(train_images_std)
test_images_pca=pca.transform(test_images_std)


# In[ ]:


train_images_pca=pd.DataFrame(train_images_pca)
train_images_pca.head()


# ok. now a side by side comparison between the original 784-Dimensional data and the reduced set of 144 dimensions.  **TO DO display MINST in reduced dim**
# 

# In[ ]:


import matplotlib.pyplot as plt


pd_train_images_std=pd.DataFrame(train_images_std)

fig,axes=plt.subplots(figsize=(10,10),ncols=2,nrows=2)
axes=axes.flatten()
for i in range(0,4):
    jj=np.random.randint(0,train_images_std.shape[0])          #pick a random image
    if i%2==0 :
        IMG_HEIGHT=12
        IMG_WIDTH=12
        axes[i].imshow(train_images_pca.iloc[[jj]].values.reshape(IMG_HEIGHT,IMG_WIDTH))
    else:
        IMG_HEIGHT=28
        IMG_WIDTH=28
        axes[i].imshow(pd_train_images_std.iloc[[jj]].values.reshape(IMG_HEIGHT,IMG_WIDTH))
    


# Now to train a classifier on the reduced dimensional dataset. In one of my [earlier kernels](https://www.kaggle.com/sharathnair/digit-recognizer-using-random-forest) I had used the RandomForestClassifier with a train and test scores of 99.9 : 93.65 resp . The input to the training was the 784-dim dataset . I will be using a similar classifier but in this case offer the reduced 441-dim dataset as an input.

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
forest=RandomForestClassifier(criterion='gini',random_state=1)
train_labels.shape
forest.fit(train_images_pca,train_labels.values.ravel())


# In[ ]:


forest.score(train_images_pca,train_labels.values.ravel())


# In[ ]:


forest.score(test_images_pca,test_labels.values.ravel())


# In[ ]:


subTest=pd.read_csv('../input/test.csv')
subTest_sc=stdsc.transform(subTest)
pred=forest.predict(pca.transform(subTest_sc))
submissions=pd.DataFrame({'ImageId':list(range(1,len(pred)+1)), 'Label':pred})
submissions.head()


# In[ ]:


submissions.to_csv("mnist_pca_randForests_submit.csv",index=False,header=True)


# In[ ]:


get_ipython().system('ls')

