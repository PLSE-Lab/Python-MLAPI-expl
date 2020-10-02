#!/usr/bin/env python
# coding: utf-8

# **If you like my notebook, please upvote my work!**
# 
# **If you use parts of this notebook in your scripts/notebooks, giving some kind of credit for instance link back to this notebook would be very much appreciated. Thanks in advance! :)**
# 
# Thankyou! :) Hope you like my work!

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # Importing important libraries.

# In[ ]:


import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA


# # What is PCA

# PCA is one of the most popular methods for dimensionality reduction and a method of lossy data compression.
# 
# For more information about PCA works under the hood and the mathematics behind it, read the medium article [here](https://towardsdatascience.com/a-one-stop-shop-for-principal-component-analysis-5582fb7e0a9c). Here we are will not dive into the mathematics and only discuss the advantages and how PCA transforms datasets.

# # Importance of PCA

# The reasons why one we we want to use PCA are:
# 1. It reduces the dimensionality in the data which significantly speeds up algorithms on the dataset.
# 2. It **sometimes** makes the data set more interpretable as lower dimensionality data are easier to understand and handle.
# 3. Significantly reduces the memory required for the storage of the dataset.
# 4. Ensures that the resulting components are independent of each other thus making it easier to understand the depedence between the variables and brings out the true dimensionality of the data.
# 5. It brings out the true underlying dimensions in the data.
# 
# The disadvantage of using PCA is that when the data has distinctly labelled columns, PCA reduces interpretability of the data.

# # Loading the Dataset.

# In[ ]:


df_train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
df_train.describe()


# **About the dataset: **Our dataset which has been derived from the famous MNIST dataset consists of 784 data fields which contain pixel values from 0-255 inclusive and a target field "label".Thus in our data set, each image is a handwritten digit converted to greyscale image which is 28 pixels in height and 28 pixels in width, for a total of 784 pixels in total given as pixel values from 0-783. Each pixel has a single pixel-value associated with it, indicating the lightness or darkness of that pixel, with higher numbers meaning darker. The taget field contains the digit in the handwritten image.

# # Splitting the target and predictor variables.

# In[ ]:


df_train_x = df_train.drop('label',axis =1)
df_train_y = df_train[['label']]


# # Visualising percentage variance loss.

# ## Fetching the variance ratios for PCA over the given dataset.

# In[ ]:


pca = PCA().fit(df_train_x)
pca.explained_variance_ratio_


# The variance ratio matrix denotes the variance due the eigen vectors in decreasing order(first ratio denotes the variance due to the top eigen vector)

# ## Plotting the loss in variance as we reduce number of components

# In[ ]:


a = []
s = 0
a.append([0,(1-s)*100,'Percentage varience lost is :'+str((1-s)*100)+'%'])
for i in range(len(pca.explained_variance_ratio_)):
    s+=pca.explained_variance_ratio_[i]
    a.append([i+1,(1-s)*100,
              'Percentage varience lost is : '+
              str((((1-s)*100)//0.0001)/10000)+'%'])
arr = pd.DataFrame(a)
arr = arr.rename(columns = {0:'No of components used:',
                            1:'Total varience lost (in percentage)'} )
px.line(data_frame = arr,x = 'No of components used:',
        y = 'Total varience lost (in percentage)',
        range_x = [0,784],range_y = [0,100],hover_name = 2,
        title='Graph depicting the loss in varience as we reduce the number of components.')


# This graph depicts how the loss in variance decreases as we increase the number of components.
# 1. We can see that using only 100 components we can retain almost 92% varaiance in the data
# 2. As we increase the number of components the variance retained increases rapidly at first and then slowly afterwords.
# 3. If we keep increasing the number of components, eventually the variance loss becomes 0 at 784 components.

# # Visualising the effect of PCA over input images.

# ## Training different PCA Models.

# In[ ]:


pca = []
pca784 = PCA(n_components = 784).fit(df_train_x)
pca.append(pca784)
pca10 = PCA(n_components = 10).fit(df_train_x)
pca.append(pca10)
pca20 = PCA(n_components = 20).fit(df_train_x)
pca.append(pca20)
pca50 = PCA(n_components = 50).fit(df_train_x)
pca.append(pca50)
pca100 = PCA(n_components = 100).fit(df_train_x)
pca.append(pca100)
pca200 = PCA(n_components = 200).fit(df_train_x)
pca.append(pca200)
pca300 = PCA(n_components = 300).fit(df_train_x)
pca.append(pca300)
pca500 = PCA(n_components = 500).fit(df_train_x)
pca.append(pca300)
pca


# ## Creating list of labels for the plot.

# In[ ]:


a = df_train_y['label'][0:20].to_numpy()
label = []
for i in range(len(a)):
    label.append("The Label for the digit is: "+str(a[i])+
                 "<br>The Sample no. is: Sample_"+str(i))
a = []
a.append('Border')
a.append('Border')
for i in label:
    a.append('Border')
    for j in range (28):
        a.append(i)
    a.append('Border')
a.append('Border')
a.append('Border')
label = a
border = ['Border']*604
a = []
a.append(border)
a.append(border)
for i in range (8):
    a.append(border)
    for j in range(28):
        a.append(label)
    a.append(border)
a.append(border)
a.append(border)
label = a


# ## Creating the image matrix for the dataset.

# In[ ]:


numpy_train_x = df_train_x.to_numpy()
for i in range(8):
    pca_trans = pca[i].transform(numpy_train_x)
    pca_invtrans = pca[i].inverse_transform(pca_trans)
    for j in range(20):
        if j ==0:
            if i==0:
                a = numpy_train_x[0].reshape(28,28)
                a = np.pad(a, pad_width=1,mode='constant',constant_values=400)
                stack = a
            else:
                b = pca_invtrans[0].reshape(28,28)
                b = np.pad(b, pad_width=1,mode='constant',constant_values=450)
                stack = b
        else:
            if i==0:
                a = numpy_train_x[j].reshape(28,28)
                a = np.pad(a, pad_width=1,mode='constant',constant_values=400)
                stack = np.hstack((stack,a))
            else:
                b = pca_invtrans[j].reshape(28,28)
                b = np.pad(b, pad_width=1,mode='constant',constant_values=450)
                stack = np.hstack((stack,b))
    if i ==0:
        final = stack
    else:
        final = np.vstack((final,stack))
final = np.pad(final,pad_width=2, mode='constant', constant_values=500)


# ## Plotting the image matrix.

# In[ ]:


fig = go.Figure(data = go.Heatmap(z = final,colorbar = None,text = label,
                                  colorscale = [[0,'white'],[0.51,'black'],
                                                [0.7,'black'],[0.8,'red'],
                                                [0.9,'blue'],
                                                [1.0,'rgb(255,0,255)']],
                                  zmin = 0,zmax = 500,zauto = False,
                                  hovertemplate='The value for z: %{z}<br>'+
                                                '%{text}<extra></extra>',
                                  hoverlabel_bgcolor = 'red'))
fig['layout']['yaxis']['autorange'] = "reversed"
fig.update_layout(title = 'The Distortion induced due to PCA while using different number of components.',
                  height= 600,width = 1200,xaxis_dtick = 30,xaxis_tick0=15,
                  yaxis_tickvals = [45,75,105,135,165,195,225],
                  yaxis_ticktext =['10','20','50','100','200','300','500'],
                  yaxis_title = 'No. of Components used for PCA: ',
                  xaxis_tickvals = [ 16,  46,  76, 106, 136, 166, 196, 226,
                                    256, 286, 316,346, 376, 406, 436, 466,
                                    496, 526, 556, 586],
                  xaxis_ticktext = ['Sample_1','Sample_2','Sample_3',
                                    'Sample_4','Sample_5','Sample_6',
                                    'Sample_7','Sample_8','Sample_9',
                                    'Sample_10','Sample_11','Sample_12',
                                    'Sample_13','Sample_14','Sample_15',
                                    'Sample_16','Sample_17','Sample_18',
                                    'Sample_19','Sample_20'],
                  xaxis_title = 'Samples (in red boxes are originals while in blue are their PCA transforms.)')
fig.update_traces(showscale = False)
fig.show()


# 1. In the matrix above the first row consists of the original image where as the rows below contain the image when using the top k eigen vectors or the top k components. 
# 2. Thus we can see that as the number of components increases the images also get more and more similar to the original.
# 3. For construction of any model we can take the optimal value of n_components = 300 as we can see that the difference between the original and the transformed image is very very subtle.
# 4. If we take the value of n as 300,we can reduce the number of dimensions from 784 to 300 which can speed up our machine learning algorithms by anywhere from 3 to 10 times without creating any significant difference in results obtained.
# 5. The images obtained using 500 components seem same as the original images but if we hover over the image to check the z values for the pixels we can see that they are quite different.
# 6. If you want to see an example of building a model after applying PCA over this dataset, check out my other notebook [here](https://www.kaggle.com/sidagar/digit-recogniser-using-pca-and-knn) where I have used PCA to build a model.

# # Creating animated plot:

# ## Creating PCA models:

# In[ ]:


pca = []
a = []
multiplier = 1.1
i = 1
while i < 784:
    a.append(i)
    pca.append(PCA(n_components = i).fit(df_train_x))
    if (i==784):
        break
    i = i*multiplier
    i = (int)(i+.99)//1
a.append(784)


# ## Creating the image matrix:

# In[ ]:


imgs = []
sample= 3
for i in pca:
    img = i.inverse_transform(i.transform(numpy_train_x))
    imgs.append(np.pad(img[sample].reshape(28,28),pad_width=1,mode='constant', 
             constant_values=400))
imgs.append(np.pad(numpy_train_x[sample].reshape(28,28),pad_width=1,
                   mode='constant',constant_values=400))


# ## Creating the labels matrix for the plot

# In[ ]:


labels = []
label = 'The label for the digit is: ' +str(df_train_y['label'][sample])
border = ['Border']* 30
labels.append(border)
for i in range(28):
    line = ['Border']
    for j in range(28):
        line.append(label)
    line.append('Border')
    labels.append(line)
labels.append(border)


# ## Plotting the animation

# In[ ]:


sliders_dict = {
    'active': 0,
    'yanchor': 'top',
    'xanchor': 'left',
    'currentvalue': {
        'font': {'size': 20},
        'prefix': 'No. of components:',
        'visible': True,
        'xanchor': 'right'
    },
    'transition': {'duration': 300, 'easing': 'linear'},
    'pad': {'b': 10, 't': 50},
    'len': 0.9,
    'x': 0.1,
    'y': 0,
    'steps': []
}
for i in range(len(a)):
    slider_step = {'args': [
        [a[i]],{
            'frame': {'duration': 300, 'redraw': True},
            'mode': 'immediate',
            'transition': {'duration': 300}
        }],
    'label': a[i],
    'method': 'animate'}
    sliders_dict['steps'].append(slider_step)
fig = go.Figure(
    data = go.Heatmap(z = imgs[0],colorbar = None,hoverlabel_bgcolor = 'red',
                      colorscale = [[0,'black'],[0.51,'white'],[0.75,'white'],
                                    [0.8,'red'],[0.9,'blue'],
                                    [1.0,'rgb(255,0,255)']],
                      zmin = 0,zmax = 500,zauto = False,text = labels,
                      hovertemplate='The value for z: %{z}<br>'+
                                    '%{text}<extra></extra>'),
    layout = go.Layout(updatemenus=[{
        'buttons': [{
            "args": [None,{"fromcurrent": True,
                           "transition": {"duration": 300,
                                          "easing": "linear"}}],
            'label': 'Play',
            'method': 'animate'
        },
        {
            'args': [[None],{'frame': {'duration': 0, 'redraw': False},
                             'mode': 'immediate',
                             'transition': {'duration': 0}}],
            'label': 'Pause',
            'method': 'animate'
        }],
        'direction': 'left',
        'pad': {'r': 10, 't': 87},
        'showactive': False,
        'type': 'buttons',
        'x': 0.1,
        'xanchor': 'right',
        'y': 0,
        'yanchor': 'top'
    }]),
    frames = [go.Frame(data = [go.Heatmap(z = imgs[i],name=str(i))],
                       name = a[i])for i in range(1,len(a))])
fig.update_traces(showscale = False)
fig['layout']['yaxis']['autorange'] = "reversed"
fig.update_layout(width=600,height=600)
fig['layout']['sliders'] = [sliders_dict]
fig.write_html('Animated plot showing the affect on dataset as ve cnage the number of components.html')
fig.show()


# I have used this PCA to over KNN in my other notebook [here](https://www.kaggle.com/sidagar/digit-recogniser-using-pca-and-knn).
# 
# **If you like my notebook please upvote my work!**

# In[ ]:




