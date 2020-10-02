#!/usr/bin/env python
# coding: utf-8

# ## A Central Limit Theorem exploration
# 
# #### An interesting theorum that states "in some situations, when independent random variables are added, their properly normalized sum tends toward a normal distribution (informally a "bell curve") even if the original variables themselves are not normally distributed."  
# 
# #### See: https://en.wikipedia.org/wiki/Central_limit_theorem
# 
# ##### I still find this s not really intuative and a bit hard to grasp. But it seems to be a big concept in how maths applies across diverse problems.  So for fun here is a dynamic animation demonstration. 
# 
# ###### Thanks to Les Hatton for getting me into this!  https://www.leshatton.org/

# # Edit this page and then run cells individually. 'Run All' does not show the graphs properly

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'notebook')

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import random
from ipywidgets import interact, widgets
print("Packages Imported")


# ### Get some user inputs. 
# ### 1. How many simulations to run, 500 seems tt produce a good approximation in most cases. Experiment with more or less. 

# In[ ]:


s=widgets.IntSlider(value=500, min = 10, max= 1000, step=10, description = 'Simulations:') 
display(s)


# ### 2. Now pick a distribution to generate (others could be added)

# In[ ]:


d=widgets.RadioButtons(
    options=['binomial','exponential', 'gamma','geometric', 'gumbel'],
    description='Distribution:',
    value='exponential',
    disabled=False
)
display(d)


# In[ ]:


# Retrieve the two parameters and set up the appropriate random generator code
sim=s.value
dist=d.value

if dist == 'exponential':
    def draw_function():
        return np.random.exponential(1.0, i)
elif dist == 'gamma': 
    def draw_function():
        return np.random.gamma(2.0, 2.0, i)
elif dist == 'binomial': 
    def draw_function():
        return np.random.binomial(10, 0.5, i)   
elif dist == 'geometric': 
    def draw_function():
        return np.random.geometric(0.35, i)       
elif dist == 'gumbel': 
    def draw_function():
        return np.random.gumbel(0, 0.1, i)

print("Set up to run", sim, "simulations on a", dist, "distribution")


# In[ ]:


# Do multiple draws (from 2 to the simulation limit) on the chosen distribution and store all the results and the averages 
#    of each simulation draw

avg = []  # the averages of each draw
all_draws = []   # also record all the draws to check the original distribution

draws = sim+1  # a little fix for the way python counts

# Starting with two draws, add an extra draw each time
for i in range(2,draws):
    draw = draw_function()
    avg.append(np.average(draw)) 
    all_draws.append(draw.tolist()) # convert np.arrays to lists 


# In[ ]:


# Flatten the original draw data (lists of lists) into a single list ready for plotting
flat_draws = [val for sublist in all_draws for val in sublist]
print("Ten entries from the original simulation draws")
print(flat_draws[:10])


# In[ ]:


# Define a function to generate the animation of the generated distribution
def clt_input(i_current):
       
    plt.cla()
    plt.hist(flat_draws[0:i_current], bins='auto')  
    plt.gca().set_xlabel('Draw values', labelpad=5)
    plt.gca().set_ylabel('Frequency')
    plt.annotate('Generation run = {}'.format(i_current), xycoords='axes fraction', xy=(0.5,0.9))   


# In[ ]:


# Show the overall shape of the distribution that was generated
fig_i = plt.figure(figsize=(6.4, 4.8), dpi=96)

title_i='An ' + str(dist) + ' distribution drawn from ' + str(sim) + ' samples'

fig_i.suptitle(title_i, fontsize=14)

a_i = animation.FuncAnimation(fig_i, clt_input, interval=1, repeat=False, frames=draws) 


# ### Now see how the Central Limit Theorem plays out on the multiple draw averages we kept for this distribution and creates a regular 'bell curve' distribution.

# In[ ]:


# Move the current title into subtitle for the next plot
subtitle=title_i


# In[ ]:


# Define a new function to generate the main CLT animation
def clt(current):
    global subtitle
    #clear current axes
    plt.cla()
    plt.hist(avg[0:current], bins='auto')
    
    plt.gca().set_title(subtitle,fontsize=10)
    plt.gca().set_xlabel('Averages from draws', labelpad=5)
    plt.gca().set_ylabel('Frequency')
    plt.annotate('Simulation run = {}'.format(current), xycoords='axes fraction', xy=(0,0.9))


# In[ ]:


# Generate the animation of the successive draws
fig_clt = plt.figure(figsize=(6.4, 4.8), dpi=96)
fig_clt.suptitle('Central Limit Theorum in action', fontsize=14)

# Pass the parameters to the animation to itterate over the clt averages 
a_clt = animation.FuncAnimation(fig_clt, clt, interval=1, repeat=False, frames=draws)


# ### Try again with a different distribution / simulation size
