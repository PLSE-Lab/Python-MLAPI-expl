#!/usr/bin/env python
# coding: utf-8

# # Line Plots in Matplotlib  
# ##### Code snippets from codecademy
# #### Content:
# 1. Creating a line graph from data
# 2. Changing the appearance of the line
# 3. Zooming in on different parts of the axis
# 4. Putting labels on titles and axes
# 5. Creating a more complex figure layout
# 6. Adding legends to graphs
# 7. Changing tick labels and positions
# 8. Saving what you've made

# In[ ]:


from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ## 1. Changing the appearance of the line

# In[ ]:


time = [0, 1, 2, 3, 4] 
revenue = [200, 400, 650, 800, 850] 
costs = [150, 500, 550, 550, 560] 
plt.plot(time, revenue)
plt.plot(time, costs)
plt.show() 


# ## 2. Changing the appearance of the line

# In[ ]:


time = [0, 1, 2, 3, 4] 
revenue = [200, 400, 650, 800, 850] 
costs = [150, 500, 550, 550, 560] 
plt.plot(time, revenue, color='purple', linestyle='--') 
plt.plot(time, costs, color='#82edc9', marker='s') 
plt.show() 


# ## 3. Zooming in on different parts of the axis

# In[ ]:


time = [0, 1, 2, 3, 4] 
revenue = [200, 400, 650, 800, 850] 
costs = [150, 500, 550, 550, 560] 
plt.plot(time, costs)
plt.axis([1, 4, 500, 560]) 
plt.show() 


# ## 4. Putting labels on titles and axes

# In[ ]:


x = range(12)
y = [3000, 3005, 3010, 2900, 2950, 3050, 3000, 3100, 2980, 2980, 2920, 3010]
plt.plot(x, y)
plt.axis([0, 11, 2900, 3100])

plt.xlabel('Time')
plt.ylabel('Dollars spent on coffee')
plt.title('My Last Twelve Years of Coffee Drinking')
plt.show()


# ## 5. Creating a more complex figure layout
# 
# When we have multiple axes in the same picture, we call each set of axes a **subplot**.  
# The picture or object that contains all of the subplots is called a **figure**.

# In[ ]:


x = [1, 2, 3, 4]
y = [1, 2, 3, 4]
z = [3, 2, 1, 1]

plt.subplot(1, 2, 1)
plt.plot(x, y, color='green')
plt.ylabel('Y Label 1')
plt.title('First Subplot')

plt.subplot(1, 2, 2)
plt.plot(x, z, color='steelblue')
plt.ylabel('Y Label 2')
plt.title('Second Subplot')

plt.show()


# #### Subplots spacing

# In[ ]:


plt.subplot(1, 2, 1)
plt.plot(x, y, color='green')
plt.ylabel('Y Label 1')

plt.subplot(1, 2, 2)
plt.plot(x, z, color='steelblue')
plt.ylabel('Y Label 2')

plt.subplots_adjust(wspace=0.4)
plt.show()


# #### Subplots arrangement

# In[ ]:


x = range(7)
straight_line = [0, 1, 2, 3, 4, 5, 6]
parabola = [0, 1, 4, 9, 16, 25, 36]
cubic = [0, 1, 8, 27, 64, 125, 216]

plt.subplot(2, 1, 1)
plt.plot(x, straight_line) 

plt.subplot(2, 2, 3)
plt.plot(x, parabola)

plt.subplot(2, 2, 4)
plt.plot(x, cubic)

plt.subplots_adjust(wspace=0.35, bottom=0.2)
plt.show()


# ## 6. Adding legends to graphs

# In[ ]:


months = range(12)
hyrule = [63, 65, 68, 70, 72, 72, 73, 74, 71, 70, 68, 64]
kakariko = [52, 52, 53, 68, 73, 74, 74, 76, 71, 62, 58, 54]
gerudo = [98, 99, 99, 100, 99, 100, 98, 101, 101, 97, 98, 99]

plt.plot(months, hyrule)
plt.plot(months, kakariko)
plt.plot(months, gerudo)

# Alternative
## plt.plot(months, hyrule, legend='hyrule')
## plt.legend()

legend_labels = ['Hyrule', 'Kakariko', 'Gerudo Valley']
plt.legend(legend_labels, loc=8)

plt.show()


# ## 7. Changing tick labels and positions

# In[ ]:


month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep","Oct", "Nov", "Dec"]
months = range(12)
conversion = [0.05, 0.08, 0.18, 0.28, 0.4, 0.66, 0.74, 0.78, 0.8, 0.81, 0.85, 0.85]

plt.xlabel("Months")
plt.ylabel("Conversion")

plt.plot(months, conversion)

ax = plt.subplot()
ax.set_xticks(months)
ax.set_xticklabels(month_names)
ax.set_yticks([0.10, 0.25, 0.5, 0.75]) 
ax.set_yticklabels(['10%', '25%', '50%', '75%'])
plt.show()


# ## 8. Saving what you've made

# In[ ]:


word_length = [8, 11, 12, 11, 13, 12, 9, 9, 7, 9]
power_generated = [753.9, 768.8, 780.1, 763.7, 788.5, 782, 787.2, 806.4, 806.2, 798.9]
years = [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009]

plt.close('all')

plt.figure()
plt.plot(years, word_length)
plt.savefig('winning_word_lengths.png')

plt.figure(figsize=(7,3))
plt.plot(years, power_generated)
plt.savefig('power_generated.png')


# ### That is all for now. Hope it helped you!
