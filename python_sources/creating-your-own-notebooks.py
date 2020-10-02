#!/usr/bin/env python
# coding: utf-8

# **[Data Visualization Home Page](https://www.kaggle.com/learn/data-visualization)**
# 
# ---
# 

# Congratulations for making it to the end of the micro-course!
# 
# In this final tutorial, you'll learn an efficient workflow that you can use to continue creating your own stunning data visualizations on the Kaggle website.

# ## Workflow
# 
# Begin by navigating to the site for Kaggle Notebooks:
# > https://www.kaggle.com/kernels
# 
# Then, in the top right corner, click on **[New Notebook]**.
# 
# ![tut7_new_kernel](https://i.imgur.com/qND102B.png)
# 
# This opens a pop-up window.
# 
# ![tut7_notebook](https://i.imgur.com/1QRsU30.png)
# 
# Then, click on **[Create]**.  (Don't change the default settings: so, **"Python"** should appear under "Select language", and you should have **"Notebook"** selected under "Select type".)
# 
# This opens a notebook with some default code.  **_Please erase this code, and replace it with the code in the cell below._**  (_This is the same code that you used in all of the exercises to set up your Python environment._)

# In[ ]:


import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
print("Setup Complete")


# The next step is to attach a dataset, before writing code to visualize it.  (_You learned how to do that in the previous tutorial._) 
# 
# Then, once you have generated a figure, you need only save it as an image file that you can easily add to your presentations!

# ---
# **[Data Visualization Home Page](https://www.kaggle.com/learn/data-visualization)**
# 
# 
# 
# 
# 
# *Have questions or comments? Visit the [Learn Discussion forum](https://www.kaggle.com/learn-forum/161291) to chat with other Learners.*
