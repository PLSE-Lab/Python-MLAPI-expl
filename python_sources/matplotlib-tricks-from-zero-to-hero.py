#!/usr/bin/env python
# coding: utf-8

# # Matplotlib

# Matplotlib is a Python 2D plotting library which produces publication-quality figures in a variety of hardcopy formats and interactive environments across platforms.

# The basic steps or rules to creating plots with matplotlib are:
# 1. Prepare data 
# 2. Create plot 
# 3. Plot 
# 4. Customize plot 
# 5. Save plot 
# 6. Show plot

# In[ ]:


import matplotlib.pyplot as plt
x = [1,2,3,4]       #Step1
y = [10,20,25,30]   #Step1
fig = plt.figure()  #Step2
ax = fig.add_subplot(111)  #Step3
ax.plot(x, y, color='lightblue', linewidth=3)    #Step4
ax.scatter([2,4,6],[5,15,25],color='darkgreen',marker='^')   #Step4
ax.set_xlim(1, 6.5)          #Step4
plt.savefig('foo.png')       #Step5
plt.show()                   #Step6


# ## 1. Prepare Data

# ### 1-D Data

# In[ ]:


import numpy as np
x = np.linspace(0, 10, 100)
y = np.cos(x)
z = np.sin(x)


# ### 2-D Data or Images

# In[ ]:


data1 = 2 * np.random.random((10, 10))
data2 = 3 * np.random.random((10, 10))


# In[ ]:


Y, X = np.mgrid[-3:3:100j, -3:3:100j]


# In[ ]:


U = -1 - X**2 + Y
V = 1 + X - Y**2


# In[ ]:


from matplotlib.cbook import get_sample_data
img = np.load(get_sample_data('axes_grid/bivariate_normal.npy'))


# ## 2. Create Plot

# In[ ]:


import matplotlib.pyplot as plt


# ### Figure

# In[ ]:


fig = plt.figure()


# In[ ]:


fig2 = plt.figure(figsize=plt.figaspect(2.0))


# ### Axes
# All plotting is done with respect to an Axes. In most cases, a subplot will fit your needs. A subplot is an axes on a grid system

# In[ ]:


fig.add_axes()


# In[ ]:


ax1 = fig.add_subplot(221) # row-col-num


# In[ ]:


ax3 = fig.add_subplot(212)


# In[ ]:


fig3, axes = plt.subplots(nrows=2,ncols=2)


# In[ ]:


fig4, axes2 = plt.subplots(ncols=3)


# ## 3. Plotting Routines

# ### 1-D Data

# In[ ]:


lines = ax.plot(x,y)  #Draw points with lines or markers connecting them


# In[ ]:


ax.scatter(x,y)       #Draw unconnected points, scaled or colored


# In[ ]:


axes[0,0].bar([1,2,3],[3,4,5])  #Plot vertical rectangles (constant width)


# In[ ]:


axes[1,0].barh([0.5,1,2.5],[0,1,2])  #Plot horiontal rectangles (constant height)


# In[ ]:


axes[1,1].axhline(0.45)   #Draw a horizontal line across axes


# In[ ]:


axes[0,1].axvline(0.65)   #Draw a vertical line across axes


# In[ ]:


ax.fill(x,y,color='blue') #Draw filled polygons


# In[ ]:


ax.fill_between(x,y,color='yellow')  #Fill between y-values and 0


# ### 2-D Data or Images

# In[ ]:


fig, ax = plt.subplots()


# In[ ]:


im = ax.imshow(img,cmap='gist_earth', interpolation='nearest', vmin=-2,vmax=2)  #Colormapped or RGB arrays


# In[ ]:


axes2[0].pcolor(data2)   #Pseudocolor plot of 2D array


# In[ ]:


axes2[0].pcolormesh(data) #Pseudocolor plot of 2D array


# In[ ]:


CS = plt.contour(Y,X,U) #Plot contours


# In[ ]:


axes2[2].contourf(data1) #Plot filled contours


# In[ ]:


axes2[2]= ax.clabel(CS) #Label a contour plot


# ### Vector Fields

# In[ ]:


axes[0,1].arrow(0,0,0.5,0.5)    #Add an arrow to the axes


# In[ ]:


axes[1,1].quiver(y,z)      #Plot a 2D field of arrows


# In[ ]:


axes[0,1].streamplot(X,Y,U,V)  #Plot 2D vector fields


# ### Data Distributions

# In[ ]:


ax1.hist(y)   #Plot a histogram


# In[ ]:


ax3.boxplot(y) #Make a box and whisker plot


# In[ ]:


ax3.violinplot(z) #Make a violin plot


# ## 4. Customize Plot

# ### Colors, Color Bars & Color Maps

# In[ ]:


plt.plot(x, x, x, x**2, x, x**3)


# In[ ]:


ax.plot(x, y, alpha = 0.4)


# In[ ]:


ax.plot(x, y, c='k')


# In[ ]:


fig.colorbar(im, orientation='horizontal')


# In[ ]:


im = ax.imshow(img,cmap='seismic')


# ### Markers

# In[ ]:


fig, ax = plt.subplots()


# In[ ]:


ax.scatter(x,y,marker=".")


# In[ ]:


ax.plot(x,y,marker="o")


# ### LineStyles

# In[ ]:


plt.plot(x,y,linewidth=4.0)


# In[ ]:


plt.plot(x,y,ls='solid')


# In[ ]:


plt.plot(x,y,ls='--')


# In[ ]:


plt.plot(x,y,'--',x**2,y**2,'-.')


# In[ ]:


plt.setp(lines,color='r',linewidth=4.0)


# ### Text & Annotations

# In[ ]:


ax.text(1,-2.1, 'Example Graph', style='italic')


# In[ ]:


ax.annotate("Sine", xy=(8, 0),xycoords='data', xytext=(10.5, 0),
            textcoords='data', arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),)


# ### Mathtext

# In[ ]:


plt.title(r'$sigma_i=15$', fontsize=20)


# ### Limits, Legends & Layouts

# In[ ]:


# Limits & Autoscaling
ax.margins(x=0.0,y=0.1)   #Add padding to a plot
ax.axis('equal')         #Set the aspect ratio of the plot to 1
ax.set(xlim=[0,10.5],ylim=[-1.5,1.5])  #Set limits for x-and y-axis
ax.set_xlim(0,10.5) #Set limits for x-axis


# In[ ]:


# Legends
ax.set(title='An Example Axes',ylabel='Y-Axis', xlabel='X-Axis')  # Set a title and x-and y-axis labels
ax.legend(loc='best')   #No overlapping plot elements


# In[ ]:


# Ticks
ax.xaxis.set(ticks=range(1,5),ticklabels=[3,100,-12,"foo"])  #Manually set x-ticks
ax.tick_params(axis='y',direction='inout', length=10)    # Make y-ticks longer and go in and out


# In[ ]:


# Subplot Spacing
fig3.subplots_adjust(wspace=0.5,hspace=0.3,left=0.125,right=0.9,top=0.9,bottom=0.1)  # Adjust the spacing between subplots
fig.tight_layout()  #Fit subplot(s) in to the figure area


# In[ ]:


# Axis Spines
ax1.spines['top'].set_visible(False)  #Make the top axis line for a plot invisible
ax1.spines['bottom'].set_position(('outward',10))   #Move the bottom axis line outward


# ## 5. Save Plot

# In[ ]:


#Save figures
plt.savefig('foo.png')


# In[ ]:


#Save transparent figures
plt.savefig('foo.png', transparent=True)


# ## 6. Show Plot
# 

# In[ ]:


plt.show()


# ### Close & Clear

# In[ ]:


plt.cla()  #Clear an axis
plt.clf()  #Clear the entire figure
plt.close() #Close a window


# In[ ]:




