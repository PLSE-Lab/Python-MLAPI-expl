#!/usr/bin/env python
# coding: utf-8

# # IPython Magic Functions 
# 
# **Built-in magic functions** can support your programming **in Kaggle-Kernels and IPython Notebooks.**    
# You can use those functions to **optimize your run-time, to combine different programming languages** in a single notebook or to **execute bash-scripts.**
# 
#  - Lines starting with an **'%'** will be interpreted as magic function.    
#  - A single **'%'** will only evaluate the attached **line of code.**    
#  - A double **'%%'** will operate on the **whole notebook-cell.**    
# 
# **P.S:**   
# Since **some outputs cannot be saved in a kernel**, fork this notebook and run the cells on your own to see all outputs.
# 
# ## Which Functions Are Available? 

# In[ ]:


get_ipython().run_line_magic('lsmagic', '')
# You can add more on your local machine 


# ## How To Get The Function Description? 

# In[ ]:


get_ipython().run_line_magic('pinfo', '%lsmagic')
# Use a questionmark to get a short description 


# ## How To Quantify Execution Time? 

# In[ ]:


# Get random numbers to measure sorting time 
import numpy as np
n = 100000
random_numbers = np.random.random(size=n)

get_ipython().run_line_magic('time', 'random_numbers_sorted = np.sort(random_numbers)')
# Get execution time of a single line 


# In[ ]:


get_ipython().run_cell_magic('time', '', '# Get execution time of a whole cell (Has to be the first command in the cell) \n\nrandom_numbers = np.random.random(size=n)\nrandom_numbers_sorted = np.sort(random_numbers)')


# Measuring a short statement onetime can be inaccurate.<br>
# That is why the **'%timeit'** function executes the statement several times and returns a summary of all durations.

# In[ ]:


get_ipython().run_line_magic('timeit', '-n 100 -r 5 random_numbers_sorted = np.sort(random_numbers)')
# n - execute the statement n times 
# r - repeat each loop r times and return the best 


# In[ ]:


get_ipython().run_cell_magic('timeit', '-n 100 -r 5', '# n - execute the statement n times \n# r - repeat each loop r times and return the best \n\nrandom_numbers = np.random.random(size=n)\nrandom_numbers_sorted = np.sort(random_numbers)')


# ## How To Measure The Time Of Each Function In A Cell? 

# In[ ]:


get_ipython().run_cell_magic('prun', '', '# Returns a duration ranking of all called functions in the cell as well as a count for all funcition calls (Can only be seen by running it on your own) \n\nfor _ in range(5):\n    random_numbers = np.random.random(size=n)\n    random_numbers_sorted = np.sort(random_numbers)')


# The output can be a little bit cryptic. Therefore using the **%lprun** ([Install separatly](http://pynash.org/2013/03/06/timing-and-profiling/)) can be more helpful since it measures the execution-time for each line in the cell. 
# 
# ## How To Measure Memory Usage In A Cell? 
# 
# Use **%mprun** (Memory line-by-line in a function)  and **%memit** (Memory iteratively several times) just as the previous example.  ([Install separatly](http://pynash.org/2013/03/06/timing-and-profiling/))
# 
# ## How To Execute Bash Commands? 
# 
# Here are a few standard bash commands to use them right in the cells.

# In[ ]:


# Return working directory 
get_ipython().run_line_magic('pwd', '')


# In[ ]:


# Create a new folder 
get_ipython().run_line_magic('mkdir', "'test_folder'")


# In[ ]:


# Save to new .py file 
text = 'I am going into a new file'
get_ipython().run_line_magic('save', "'new_file' text")


# In[ ]:


# Copy files to a new location 
get_ipython().run_line_magic('cp', 'new_file.py new_file_2.py')


# In[ ]:


# List of elements in current directory 
get_ipython().run_line_magic('ls', '')


# In[ ]:


# Read7show files 
get_ipython().run_line_magic('cat', 'new_file.py')


# In[ ]:


# Remove folder 
get_ipython().run_line_magic('rmdir', 'test_folder/')

# Remove files 
get_ipython().run_line_magic('rm', 'new_file.py')


# In[ ]:


# Rename files 
get_ipython().run_line_magic('mv', 'new_file_2.py renamed_file.py')


# In[ ]:


get_ipython().run_line_magic('ls', '')


# In[ ]:


# Using a '!' neables arbitrary single-line bash-commands 
get_ipython().system('ls | grep .py')


# In[ ]:


get_ipython().run_cell_magic('!', '', '# This executes the whole cell in a bash and returns a list \npwd\nls')


# In[ ]:


get_ipython().run_cell_magic('bash', '', '# This executes the whole cell in a bash and returns single elements \npwd\nls')


# In[ ]:


# The returned values can be stored in variables 
working_directory = get_ipython().getoutput('pwd')
working_directory


# ## How To Style Your Notebooks? 

# In[ ]:


# Compile latex in cells 


# In[ ]:


get_ipython().run_cell_magic('latex', '', '$\\frac{awe}{some}$')


# In[ ]:


# Html in cells 


# In[ ]:


get_ipython().run_cell_magic('HTML', '', '\n<h1>Awesome</h1>')


# ## How To Use Different Programming Languages? 

# In[ ]:


get_ipython().run_cell_magic('javascript', '', "\nwindow.alert('Here you can learn how to use magic functions inside notebooks.\\nHave a good day!')")


# ## More Tutorials 
# A few more tipps can be found here:    
# [Timing and Profiling in IPython](http://pynash.org/2013/03/06/timing-and-profiling/)    
# [Advanced Jupyter Notebook](https://blog.dominodatalab.com/lesser-known-ways-of-using-notebooks/)    
# [Make Jupyter/IPython Notebook even more magical with cell magic extensions!](https://www.youtube.com/watch?v=zxkdO07L29Q)
# 
# **Have a good day!**

# In[ ]:




