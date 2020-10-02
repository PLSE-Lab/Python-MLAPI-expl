#!/usr/bin/env python
# coding: utf-8

# 

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
get_ipython().run_line_magic('matplotlib', 'inline')

import os
os.listdir('../input')


# In[ ]:


years=[1,1000,1500,1600,1700,1750,1800,1850,1900,1950,1955,1960,1965,1970,1980,1985,1990,
       1995,2000,2005,2010,2015]
pops=[200,400,458,580,682,791,1000,1262,1650,2525,2758,3018,3322,3682,
      4061,4440,4853,5310,5735,6127,6520,7349]
plt.plot(years,pops)
plt.show


# ###  Adding labels and custom line color

# In[ ]:


years=[1950,1955,1960,1965,1970,1980,1985,1990,1995,2000,2005,2010,2015]
pops=[2.5,2.7,3.0,3.3,3.6,4.0,4.4,4.8,5.3,5.7,6.1,6.5,7.3]
plt.plot(years,pops,color=(255/255,100/255,100/255))
plt.ylabel("Population in Billions")
plt.xlabel("Population growth by year")
plt.title("Population Growth")
plt.show


# ### Legends, Titles, and Labels

# In[ ]:


x = [1,6,3]
y = [5,9,4]

x2 = [1,2,3]
y2 = [10,14,12]

plt.plot(x, y, label='First Line')
plt.plot(x2, y2, label='Second Line')
plt.xlabel('Plot Number')
plt.ylabel('Important var')
plt.title('Interesting Graph Check it out')
plt.legend()
plt.show()


# ### Multiple lines and line styling

# In[ ]:


years=[1950,1955,1960,1965,1970,1980,1985,1990,1995,2000,2005,2010,2015]
pops=[2.5,2.7,3.0,3.3,3.6,4.0,4.4,4.8,5.3,5.7,6.1,6.5,7.3]
death=[1.2,1.1,1.2,2.1,2.0,2.3,1.8,1.9,2.6,1.6,2.4,2.4,4.0]
plt.plot(years,pops,'--',color=(255/255,100/255,100/255))
plt.plot(years,death,color=(.6,.6,1))
plt.ylabel("Population in Billions")
plt.xlabel("Population growth by year")
plt.title("Population Growth")
plt.show


# ### Configuring the graph

# In[ ]:


years=[1950,1955,1960,1965,1970,1980,1985,1990,1995,2000,2005,2010,2015]
pops=[2.5,2.7,3.0,3.3,3.6,4.0,4.4,4.8,5.3,5.7,6.1,6.5,7.3]
death=[1.2,1.1,1.2,2.1,2.0,2.3,1.8,1.9,2.6,1.6,2.4,2.4,4.0]
lines=plt.plot(years,pops,years,death)
plt.grid(True)
plt.setp(lines,color=(1,.4,.5),marker='*')
plt.show


# ### Let's make pie (charts)

# In[ ]:


labels=['Python','C','C++','PHP','Java','Ruby']
sizes=[33,52,12,17,42,48]
separated=(.1,0,0,0,0,0)
plt.pie(sizes,labels=labels,autopct='%1.1f%%',explode=separated)
plt.axis('equal')
plt.show()


# ### Letting Pandas make data simpler

# In[ ]:


raw_data={'names':['Nick','Sani','John','Rubi','Maya'],
         'jan_ir':[123,124,125,126,128],
         'feb_ir':[23,24,25,27,29],
         'march_ir':[3,5,7,6,9]}

df=pd.DataFrame(raw_data,columns=['names','jan_ir','feb_ir','march_ir'])
df


# In[ ]:


raw_data={'names':['Nick','Sani','John','Rubi','Maya'],
         'jan_ir':[123,124,125,126,128],
         'feb_ir':[23,24,25,27,29],
         'march_ir':[3,5,7,6,9]}

df=pd.DataFrame(raw_data,columns=['names','jan_ir','feb_ir','march_ir'])
df['total_ir']=df['jan_ir']+df['feb_ir']+df['march_ir']
df


# ### Using Panda's data for pie charts

# In[ ]:


color=[(1,.4,.4),(1,.6,1),(.5,.3,1),(.3,1,.5),(.7,.7,.2)]
plt.pie(df['total_ir'],labels=df['names'],colors=color,autopct='%1.1f%%')
plt.axis('equal')
plt.show()


# ### Bar charts

# In[ ]:


korea_scores=(554,536,538)
canada_scores=(518,523,525)
china_scores=(413,570,580)
franch_scores=(495,505,499)
index=np.arange(3)
bar_width=.2
k1=plt.bar(index,korea_scores,bar_width,alpha=.9,label="Korea")
c1=plt.bar(index+bar_width,canada_scores,bar_width,alpha=.9,label="Canada")
ch1=plt.bar(index+bar_width*2,china_scores,bar_width,alpha=.9,label="China")
f1=plt.bar(index+bar_width*3,franch_scores,bar_width,alpha=.9,label="Franch")
plt.xticks(index+.6/2,('Mathematics','Reading','Science'))
plt.ylabel('Mean score in PISA in 2012')
plt.xlabel('Subjects')
plt.title('Test scores by Country')
plt.grid(True)
plt.legend()
plt.show()


# ### Bar Charts and Histograms
#  let's cover a bar chart.

# In[ ]:


plt.bar([1,3,5,7,9],[5,2,7,8,2], label="Example one")
plt.bar([2,4,6,8,10],[8,6,2,5,6], label="Example two", color='g')
plt.legend()
plt.xlabel('bar number')
plt.ylabel('bar height')

plt.title('Epic Graph\nAnother Line! Whoa')

plt.show()


# Now we cover histograms. Very much like a bar chart, histograms tend to show distribution by grouping segments together. Examples of this might be age groups, or scores on a test. Rather than showing every single age a group might be, maybe you just show people from 20-25, 25-30... and so on. 

# In[ ]:


population_ages = [22,55,62,45,21,22,34,42,42,4,99,102,110,120,121,122,130,111,115,112,80,75,65,54,44,43,42,48]

bins = [0,10,20,30,40,50,60,70,80,90,100,110,120,130]

plt.hist(population_ages, bins, histtype='bar', rwidth=0.8)

plt.xlabel('x')
plt.ylabel('y')
plt.title('Interesting Graph\nCheck it out')
plt.show()


# ### Scatter Plots

# In[ ]:


x = [1,2,3,4,5,6,7,8]
y = [5,2,4,2,1,4,5,2]

plt.scatter(x,y, label='skitscat Raggedy', color='k', s=25, marker="o")

plt.xlabel('x')
plt.ylabel('y')
plt.title('Interesting Graph\nCheck it out')
plt.legend()
plt.show()


# ### Stack Plots

# In[ ]:


days = [1,2,3,4,5]

sleeping = [7,8,6,11,7]
eating =   [2,3,4,3,2]
working =  [7,8,7,2,2]
playing =  [8,5,7,8,13]
plt.stackplot(days, sleeping,eating,working,playing, colors=['m','c','r','k'])

plt.xlabel('x')
plt.ylabel('y')
plt.title('Interesting Graph\nStack Plots')
plt.show()


# In[ ]:


days = [1,2,3,4,5]

sleeping = [7,8,6,11,7]
eating =   [2,3,4,3,2]
working =  [7,8,7,2,2]
playing =  [8,5,7,8,13]


plt.plot([],[],color='m', label='Sleeping', linewidth=5)
plt.plot([],[],color='c', label='Eating', linewidth=5)
plt.plot([],[],color='r', label='Working', linewidth=5)
plt.plot([],[],color='k', label='Playing', linewidth=5)

plt.stackplot(days, sleeping,eating,working,playing, colors=['m','c','r','k'])

plt.xlabel('x')
plt.ylabel('y')
plt.title('Interesting Graph\nCheck it out')
plt.legend()
plt.show()


# ### 3D graphs

# In[ ]:


from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import style

style.use('fivethirtyeight')

fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')

x = [1,2,3,4,5,6,7,8,9,10]
y = [5,6,7,8,2,5,6,3,7,2]
z = [1,2,6,3,2,7,3,3,7,2]

ax1.plot(x,y,z)

ax1.set_xlabel('x axis')
ax1.set_ylabel('y axis')
ax1.set_zlabel('z axis')

plt.show()


# ### 3D Scatter Plot

# In[ ]:


from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')

x = [1,2,3,4,5,6,7,8,9,10]
y = [5,6,7,8,2,5,6,3,7,2]
z = [1,2,6,3,2,7,3,3,7,2]

x2 = [-1,-2,-3,-4,-5,-6,-7,-8,-9,-10]
y2 = [-5,-6,-7,-8,-2,-5,-6,-3,-7,-2]
z2 = [1,2,6,3,2,7,3,3,7,2]

ax1.scatter(x, y, z, c='g', marker='o')
ax1.scatter(x2, y2, z2, c ='r', marker='o')

ax1.set_xlabel('x axis')
ax1.set_ylabel('y axis')
ax1.set_zlabel('z axis')

plt.show()


# ### 3D Bar Chart

# In[ ]:


from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style
style.use('ggplot')

fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')

x3 = [1,2,3,4,5,6,7,8,9,10]
y3 = [5,6,7,8,2,5,6,3,7,2]
z3 = np.zeros(10)

dx = np.ones(10)
dy = np.ones(10)
dz = [1,2,3,4,5,6,7,8,9,10]

ax1.bar3d(x3, y3, z3, dx, dy, dz)


ax1.set_xlabel('x axis')
ax1.set_ylabel('y axis')
ax1.set_zlabel('z axis')

plt.show()


# In[ ]:


from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style
style.use('ggplot')

fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')

x, y, z = axes3d.get_test_data()

ax1.plot_wireframe(x,y,z, rstride = 3, cstride = 3)

ax1.set_xlabel('x axis')
ax1.set_ylabel('y axis')
ax1.set_zlabel('z axis')

plt.show()


# ## Plotting Images
# ### Importing image data into Numpy arrays
# 

# In[ ]:


img = mpimg.imread('../input/unlabeled_images/unlabeled_image_png_30319.png')
print(img)


# ### Plotting numpy arrays as images

# In[ ]:


plt.figure(figsize=(10,6))
imgplot = plt.imshow(img)


# ### Applying pseudocolor schemes to image plots
# 
# Pseudocolor is a useful tool for enhancing contrast and visualizing data more easily. This is especially useful when making presentations of data using projectors - their contrast is typically quite poor.
# 
# Pseudocolor is only relevant to single-channel, grayscale, luminosity images. We currently have an RGB image. Since R, G, and B are all similar, we can just pick one channel of our data:

# In[ ]:


plt.figure(figsize=(10,6))
lum_img = img[:, :, 0]
plt.imshow(lum_img)


# In[ ]:


plt.figure(figsize=(10,6))
plt.imshow(lum_img, cmap="hot")


# Change colormaps on existing plot objects using the set_cmap() method:

# In[ ]:


plt.figure(figsize=(10,6))
imgplot = plt.imshow(lum_img)
imgplot.set_cmap('nipy_spectral')


# ### Color scale
# 
# It's helpful to have an idea of what value a color represents. We can do that by adding color bars.

# In[ ]:


plt.figure(figsize=(10,6))
imgplot = plt.imshow(lum_img)
plt.colorbar()


# ### Examining a specific data range
# To enhance the contrast in image, or expand the contrast in a particular region while sacrificing the detail in colors that don't vary much, or don't matter. A good tool to find interesting regions is the histogram. To create a histogram of image data, we use the `hist()` function.

# In[ ]:


plt.figure(figsize=(10,6))
plt.hist(lum_img.ravel(), bins=256, range=(0.0, 1.0), fc='k', ec='k')


# In[ ]:


plt.figure(figsize=(10,6))
imgplot = plt.imshow(lum_img, clim=(0.0, 0.7))


# Specify the clim using the returned object

# In[ ]:


fig = plt.figure(figsize=(10,6))
a = fig.add_subplot(1, 2, 1)
imgplot = plt.imshow(lum_img)
a.set_title('Before')
plt.colorbar(ticks=[0.1, 0.3, 0.5, 0.7], orientation='horizontal')
a = fig.add_subplot(1, 2, 2)
imgplot = plt.imshow(lum_img)
imgplot.set_clim(0.0, 0.7)
a.set_title('After')
plt.colorbar(ticks=[0.1, 0.3, 0.5, 0.7], orientation='horizontal')


# ### Array Interpolation schemes
# 
# Interpolation calculates what the color or value of a pixel "should" be, according to different mathematical schemes. One common place that this happens is when you resize an image. The number of pixels change, but you want the same information. Since pixels are discrete, there's missing space. Interpolation is how you fill that space. This is why your images sometimes come out looking pixelated when you blow them up. The effect is more pronounced when the difference between the original image and the expanded image is greater. Let's take our image and shrink it. We're effectively discarding pixels, only keeping a select few. Now when we plot it, that data gets blown up to the size on your screen. The old pixels aren't there anymore, and the computer has to draw in pixels to fill that space.

# In[ ]:


from PIL import Image

img = Image.open('../input/unlabeled_images/unlabeled_image_png_51042.png')
img.thumbnail((64, 64), Image.ANTIALIAS)  # resizes image in-place
imgplot = plt.imshow(img)


# In[ ]:


imgplot = plt.imshow(img, interpolation="nearest")


# In[ ]:


imgplot = plt.imshow(img, interpolation="bicubic")


# In[ ]:




