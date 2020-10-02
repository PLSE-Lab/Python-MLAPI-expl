#!/usr/bin/env python
# coding: utf-8

# # Malaria Cells Identification
# 
# In this notebook I will explain the approach I took to create a parametric classifier to separate the infected cells from the healthy cells in the "**Malaria Cell Images Dataset**"

# In[2]:


get_ipython().run_line_magic('load_ext', 'line_profiler')


# ## Import libraries

# In[3]:


import time #This library helps to keep track of the time of compilation time that a function or loop is taking to execute
import matplotlib.pyplot as plt #This library is used to plot the figures shown below
import cv2 #This library is used to import the images from the dataset and change the color map.
from IPython.display import Image
from scipy.stats import multivariate_normal #This library is used to model the statistical distributions that emerge from the data

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from mpl_toolkits.mplot3d.axes3d import get_test_data
from matplotlib import colors

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os #Operating System
from multiprocessing import Pool
from scipy.stats import skewnorm


# ### Define the directory

# In[4]:



root = "../input/cell_images/cell_images" #This is the root directory 
root_uninfected = root + '/Uninfected' #Here root directory is appended to the "Uninfected" directory
root_infected = root + '/Parasitized' #Here root directory is appended to the "Infected or Parasitized" directory

files_uninfected = os.listdir(root_uninfected) #Let's assing the uninfected directory to a variable so it is easier to use it later
files_infected = os.listdir(root_infected)#Let's assing the infected directory to a variable so it is easier to use it later

#Let's print how many of each type of cells are there
print('{} uninfected files'.format(len(files_uninfected)))
print('{} infected files'.format(len(files_infected)))


#  ### Import the cell images 
#  
#  To do this, the **"CV2"** library will be used. Before importing the cells let's check what are the different color maps we could use to open them.\ By default the **"BGR"** (*Blue*, *Green*, *Red*) map is used but as it can be seen there are a lot of options in the list.
#  

# In[5]:



flags = [i for i in dir(cv2) if i.startswith('COLOR_')]
flags_1 = pd.DataFrame(flags)
flags


# #### Lets define a function to help us with the process of importation easier. This function returns the $k_{th}$ infected cell (y = 1) or not infected (y = 0)
# 
# Since the image is open by default in the "**BGR**" the visualization of the cells would give priority to the blues. At the beggining the cells are converted to the traditional "**RGB**" but later on I decided, after playing around a bit with the color maps that I wanted the cells in "**HSV**" map (*Hue, Saturation, Value*).

# In[6]:


def sample_rgb(y, k):
    if y == 0:
        return cv2.cvtColor(cv2.imread(os.path.join(root_uninfected, files_uninfected[k])), cv2.COLOR_BGR2RGB)
    elif y == 1:
        return cv2.cvtColor(cv2.imread(os.path.join(root_infected, files_infected[k])), cv2.COLOR_BGR2RGB)
    else:
        raise ValueError


# In[7]:


def sample_hsv(y, k):
    if y == 0:
        return cv2.cvtColor(cv2.imread(os.path.join(root_uninfected, files_uninfected[k])), cv2.COLOR_BGR2HSV)
    elif y == 1:
        return cv2.cvtColor(cv2.imread(os.path.join(root_infected, files_infected[k])), cv2.COLOR_BGR2HSV)
    else:
        raise ValueError


# #### Lets visualize in RGB some of the healthy cells (left) and infected ones (right) with the aim  of having ideas about the characterization of the cells for the extraction of "FEATURES".

# In[8]:


f, ax = plt.subplots(5, 2, figsize=(5, 10))
for k in range(5):
    for y in range(2):
        ax[k][y].imshow(sample_rgb(y, k))


# The two types cells are cleary different for us by just looking at them, but how we extract those characteristics that make the images different? Maybe by segregating the map by colors could give us an idea of the characteristics of the two images. Next, I define a function to make this segregation take from: (https://realpython.com/python-opencv-color-spaces/#simple-segmentation-using-color-spaces) and modified. 
# 

# In[9]:


def color_segmentation(figure, colormap):
    figure = figure[figure[:,:,0] > 0.01]#Here the black part of the image is taken out.
    a = np.reshape(figure,-1)
    aux = int(a.shape[0]/3)
    aa= a.reshape((aux,1,3))
    fig = plt.figure(figsize=plt.figaspect(0.5))
    axis = fig.add_subplot(1, 1, 1, projection="3d")
    pixel_colors = aa.reshape((np.shape(aa)[0]*np.shape(aa)[1], 3))
    r, g, b = cv2.split(aa)
    norm = colors.Normalize(vmin=-1.,vmax=1.)
    norm.autoscale(pixel_colors)
    pixel_colors = norm(pixel_colors).tolist()
    axis.scatter(r.flatten(), g.flatten(), b.flatten(), facecolors=pixel_colors, marker=".")
    if colormap == "rgb":
        axis.set_xlabel("Red")
        axis.set_ylabel("Green")
        axis.set_zlabel("Blue")
    elif colormap == "hsv":
        axis.set_xlabel("Hue")
        axis.set_ylabel("Saturation")
        axis.set_zlabel("Value")
        plt.show()
    return r,g,b
                           
        


# ### Plot colors uninfected
# 
# Here the colors are deployed to check how much red, green and blue are there in an uninfected cell image in the case of the **rgb color map**. This procedure is repeated for the **hsv color map**, in this case we can check how much hue, saturation and value(brigthness) are there in an uninfected cell image. The function created above return the color segregation. 

# In[10]:


figure_uninfected_sample = sample_rgb(0,0)  #RGB uninfected cell
r_uninfected, g_uninfected, b_uninfected = color_segmentation(figure_uninfected_sample, "rgb")
figure_uninfected_sample = sample_hsv(0,0) #HSV uninfected cell
h_uninfected, s_uninfected, v_uninfected = color_segmentation(figure_uninfected_sample, "hsv")
plt.show()


# ### Plot colors infected
# Here the colors are deployed to check how much red, green and blue are there in an infected cell image in the case of the **rgb color map**. This procedure is repeated for the **hsv color map**, in this case we can check how much hue, saturation and value(brigthness) are there in an infected cell image. The function created above return the color segregation. 

# In[11]:


figure_infected_sample = sample_rgb(1,0) #RGB infected cell
r_infected, g_infected, b_infected = color_segmentation(figure_infected_sample, "rgb")
figure_infected_sample = sample_hsv(1,0) #HSV infected cell 
h_infected, s_infected, v_infected = color_segmentation(figure_infected_sample, "hsv")
plt.show()


# In order to check if any of the values obtained from the images(**R,G,B or H,S,V**) constitutes a good feature, basic operations are done to check for differences between infected and non-infected cells.

# In[12]:


#Difference between the max and min from each color for RGB infected and uninfected.
dif_r_uninfected = max(r_uninfected) - min(r_uninfected)
dif_g_uninfected = max(g_uninfected) - min(g_uninfected)
dif_b_uninfected = max(b_uninfected) - min(b_uninfected)

dif_r_infected = max(r_infected) - min(r_infected)
dif_g_infected = max(g_infected) - min(g_infected)
dif_b_infected = max(b_infected) - min(b_infected)

table1 = pd.DataFrame({
    "Difference R Uninfected": dif_r_uninfected,
    "Difference R Infected": dif_r_infected,
    "Difference G Uninfected": dif_g_uninfected,
    "Difference G Infected": dif_g_infected,
    "Difference B Uninfected": dif_b_uninfected,
    "Difference B Infected": dif_b_infected
})
table1


# In[13]:


#Difference between the max and min from each color for HSV infected and uninfected.
dif_h_uninfected = max(h_uninfected) - min(h_uninfected)
dif_s_uninfected = max(s_uninfected) - min(s_uninfected)
dif_v_uninfected = max(v_uninfected) - min(v_uninfected)

dif_h_infected = max(h_infected) - min(h_infected)
dif_s_infected = max(s_infected) - min(s_infected)
dif_v_infected = max(v_infected) - min(v_infected)

table2 = pd.DataFrame({
    "Difference H Uninfected": dif_h_uninfected,
    "Difference H Infected": dif_h_infected,
    "Difference S Uninfected": dif_s_uninfected,
    "Difference S Infected": dif_s_infected,
    "Difference V Uninfected": dif_v_uninfected,
    "Difference V Infected": dif_v_infected
})
table2


# From this differences we can extract some features to establish our classifier. **For the RGB I am going to take the difference in green in the cells to separate those healthy from those infected.**
# 

# In[14]:


def X1_3(image):
    #image = image.reshape(int(image.shape[1]),int(image.shape[2]),3)
    std_g = []
    max_g = []
    min_g = []
    image_nb = []
    for i in range(len(image)):
       
        image_nb = image[i][image[i][:,:,0] > 0.01]
        image_nbr = np.reshape(image_nb.tolist(),-1)
        aux = int(image_nbr.shape[0]/3)
        image_final= image_nb.reshape((aux,1,3))
        #image_hsv= cv2.cvtColor(image_final, cv2.COLOR_RGB2HSV)
        h1, s1, v1 = cv2.split(image_final)
        std_aux = np.std(s1)
        std_g.append(std_aux)
        max_g_aux = np.max(s1)
        max_g.append(max_g_aux)
        min_g_aux = np.min(s1)
        min_g.append(min_g_aux)
    return np.asarray(max_g)-np.asarray(min_g)


# **For the HSV I am going to take the difference in saturation in the cells to separate those healthy from those infected.**

# In[15]:


def X2_3(image):
    #image = image.reshape(int(image.shape[1]),int(image.shape[2]),3)
    std_s = []
    max_s = []
    min_s = []
    image_nb = []
    for i in range(len(image)):
       
        image_nb = image[i][image[i][:,:,0] > 0.01]
        image_nbr = np.reshape(image_nb.tolist(),-1)
        aux = int(image_nbr.shape[0]/3)
        image_final= image_nb.reshape((aux,1,3))
        image_hsv= cv2.cvtColor(image_final, cv2.COLOR_RGB2HSV)
        h1, s1, v1 = cv2.split(image_hsv)
        std_aux = np.std(s1)
        std_s.append(std_aux)
        max_s_aux = np.max(s1)
        max_s.append(max_s_aux)
        min_s_aux = np.min(s1)
        min_s.append(min_s_aux)
    return np.asarray(max_s)-np.asarray(min_s)


# ### Let's prove with a 500 cells sample

# In[16]:


samples_uninfected =  np.asarray([sample_rgb(0, i) for i in range(500)])
samples_infected =  np.asarray([sample_rgb(1, i) for i in range(500)])
X1_uninfected =X1_3(samples_uninfected) # RGB Uninfected
X1_infected = X1_3(samples_infected) # RGB Infected
X2_uninfected = X2_3(samples_uninfected) # HSV Uninfected
X2_infected = X2_3(samples_infected) # HSV Infected


# ## Univariate plot for X1 (Difference in green RGB)

# In[17]:


f, ax = plt.subplots(1, 1, figsize=(12, 3))
ax.hist(X1_uninfected, color='b', density=True, alpha=0.4, label='uninfected')
ax.hist(X1_infected, color='y', density=True, alpha=0.4, label='infected')
plt.legend()
plt.show()


# ## Univariate plot for X2 (Difference in saturation HSV)

# In[18]:


f, ax = plt.subplots(1, 1, figsize=(12, 3))
ax.hist(X2_uninfected, color='b', density=True, alpha=0.4, label='uninfected')
ax.hist(X2_infected, color='y', density=True, alpha=0.4, label='infected')
plt.legend()
plt.show()


# ## Bivariate plot for X1,X2

# In[19]:


f, ax = plt.subplots(1, 1, figsize=(12, 3))
ax.scatter(X1_uninfected, X2_uninfected, color='b', alpha=0.4, label='uninfected')
ax.scatter(X1_infected, X2_infected, color='r', alpha=0.4, label='infected')
plt.legend()
plt.show()


# # Parametric Modeling of the First Feature

# In[44]:


N_samples = 1000
samples_uninfected = [sample_rgb(0, i) for i in range(N_samples)]
samples_infected = [sample_rgb(1, i) for i in range(N_samples)]

X_uninfected = X1_3(samples_uninfected)
X_infected = X1_3(samples_infected)


# Let's assume a **LogNormal distribution** for the uninfected 

# In[45]:


def lognormal(mu, sigma):
    def fun(x):
        return np.exp(- (np.log(x) - mu) ** 2 /(2 * sigma ** 2)) / (x * sigma * (2 * np.pi) ** 0.5) 
    return fun


# In[46]:


log_Xuninfected = np.log(X_uninfected)
mu0 = np.mean(log_Xuninfected)
sigma0 = np.std(log_Xuninfected)
print('mu: {:.3f}, sigma: {:.3f}'.format(mu0, sigma0))


# In[47]:


pdfX1_uninfected = lognormal(mu0, sigma0)
pdfX1_infected = sknorm(a,loc,space)


# Let's assume a **Skew Normal Distribution** for the infected cells

# In[48]:


def sknorm(a,loc,space):
    def fun(x):
        return skewnorm.pdf(x, a ,loc,space)
    return fun


# In[49]:


#Fitting INFECTED cells to a skew normal distribution
a, loc, space = skewnorm.fit(X_infected)
a,loc,space #Paramter of skew normal distribution function


# In[50]:


f, ax = plt.subplots(1, 1, figsize=(12, 3))
dom = np.linspace(10, 200, 100)
ax.hist(X_uninfected, bins=30, color='b', density=True, alpha=0.4, label='uninfected')
ax.hist(X_infected, bins=30, color='y', density=True, alpha=0.4, label='infected')
plt.plot(dom, list(map(pdfX1_uninfected, dom)), label='uninfected')
ax.plot(dom, list(map(pdfX1_infected, dom)), label='infected')
plt.legend()
plt.show()


# # Bivariate normal distribution
# We will assume a bivariate normal distribution for **X1** and **X2**

# In[51]:


N_samples = 500
samples_uninfected = [sample_rgb(0, i) for i in range(N_samples)]
samples_infected = [sample_rgb(1, i) for i in range(N_samples)]
X1_uninfected = X1_3(samples_uninfected)
X1_infected = X1_3(samples_infected)
X2_uninfected = X2_3(samples_uninfected)
X2_infected = X2_3(samples_infected)
X_uninfected = np.array(list(zip(X1_uninfected, X2_uninfected)))
X_infected = np.array(list(zip(X1_infected, X2_infected)))


# In[52]:


mean_uninfected = np.mean(X_uninfected, axis=0)
mean_infected = np.mean(X_infected, axis=0)
print('Mean of uninfected: {}'.format(mean_uninfected))
print('Mean of infected: {}'.format(mean_infected))


# In[53]:


cov_uninfected = np.cov(X_uninfected.T)
cov_infected = np.cov(X_infected.T)
print('Covariance of uninfected: \n{}'.format(cov_uninfected))
print('Covariance of infected: \n{}'.format(cov_infected))


# In[54]:


pdfX_uninfected = multivariate_normal(mean_uninfected, cov_uninfected).pdf
pdfX_infected = multivariate_normal(mean_infected, cov_infected).pdf


# In[55]:


def plot(x0, x1, y0, y1, mean, cov, ax):
    x, y = np.mgrid[x0:x1:.1, y0:y1:.1]
    pos = np.dstack((x, y))
    rv = multivariate_normal(mean, cov)
    ax.contourf(x, y, rv.pdf(pos), alpha=0.2, levels=10)
    
f, ax = plt.subplots(1, 1, figsize=(12, 3))
plot(20.0, 210.0, 0, 250, mean_uninfected, cov_uninfected, ax)
plot(20.0, 210.0, 0, 250, mean_infected, cov_infected, ax)
ax.scatter(X1_uninfected, X2_uninfected, color='b', alpha=0.4, label='uninfected')
ax.scatter(X1_infected, X2_infected, color='r', alpha=0.4, label='infected')
plt.legend()
plt.show()


# # Classifier
# 
# We can finally define our classifier as follows:
# 
# * r is the relative cost of a flase negative vs a negative positive
# * p1 is the proportion of positives in the population
# * pdfX_uninfected is the density function of X for the uninfected
# * pdfX_infectedis the density function of X for the infected

# In[56]:


def clasificador(r, p1, pdfX_uninfected, pdfX_infected):
    def phi(x):
        q0 = r * p1 * pdfX_infected(x)
        q1 = (1 - p1) * pdfX_uninfected(x)
        if q0 < q1:
            return 0
        else:
            return 1
    return phi


# Let's suppose that the incidence of malary in Colombia is 0.5 in 10000 people.

# In[57]:


p1 = 0.5 / 10000


# In[58]:


costs = [1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6]


# In[59]:


range_samples = range(10000, 11000) 
samples_uninfected_test = [sample_rgb(0, i) for i in range_samples]
samples_infected_test = [sample_rgb(1, i) for i in range_samples]
X1_uninfected_test = X1_3(samples_uninfected_test)
X1_infected_test = X1_3 (samples_infected_test)
X2_uninfected_test = X2_3(samples_uninfected_test)
X2_infected_test = X2_3(samples_infected_test)
X_uninfected_test = list(zip(X1_uninfected_test, X2_uninfected_test))
X_infected_test = list(zip(X1_infected_test, X2_infected_test))


# In[60]:


def fp_fn(phi, p1, X_uninfected, X_infected):
    fpr = sum(map(phi, X_uninfected)) / len(X_uninfected)
    fp = fpr * (1 - p1)
    fnr = 1.0 - sum(map(phi, X_infected)) / len(X_infected)
    fn = fnr * p1
    return fp, fn

def fpr_tpr(phi, X_uninfected, X_infected):
    fpr = sum(map(phi, X_uninfected)) / len(X_uninfected)
    tpr = sum(map(phi, X_infected)) / len(X_infected)
    return fpr, tpr


# # False Positive and False Negative Rates

# In[62]:


fn1s = []
fns = []
fp1s = []
fps = []
fpr1s = []
fprs = []
tpr1s = []
tprs = []
for r in costs:
    phi1 = clasificador(r, p1, pdfX1_uninfected, pdfX1_infected)
    phi = clasificador(r, p1, pdfX_uninfected, pdfX_infected)
    fp1, fn1 = fp_fn(phi1, p1, X1_uninfected_test, X1_infected_test)
    fp, fn = fp_fn(phi, p1, X_uninfected_test, X_infected_test)
    fpr1, tpr1 = fpr_tpr(phi1, X1_uninfected_test, X1_infected_test)
    fpr, tpr = fpr_tpr(phi, X_uninfected_test, X_infected_test)
    fn1s.append(fn1)
    fns.append(fn)
    fp1s.append(fp1)
    fps.append(fp)
    fpr1s.append(fpr1)
    fprs.append(fpr)
    tpr1s.append(tpr1)
    tprs.append(tpr)


# # ROC Curve

# In[63]:


f, ax = plt.subplots(1, 1, figsize=(12, 3))
ax.plot(fpr1s, tpr1s, label='One feature')
ax.plot(fprs, tprs, label='Both features')
ax.set_xlim(0.0, 1.0)
ax.set_ylim(0.0, 1.0)
ax.set_xlabel('False positive rate')
ax.set_ylabel('True positive rate')
ax.grid()
ax.legend()
plt.show()


# In[ ]:




