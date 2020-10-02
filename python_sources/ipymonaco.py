#!/usr/bin/env python
# coding: utf-8

# ipymonaco
# =========
# 
# This Jupyter notebook renders Microsoft's Monaco text editor as a ipywidget widget.

# ## Installation
# For those who are using Kaggle Kernel or a Classic Notebook, run the following cell to install and enable the ipywidget as a Jupyter notebook extension.*

# In[ ]:


get_ipython().system('pip install ipymonaco=="0.0.21a"')

# if notebook < 5.3
get_ipython().system('jupyter nbextension enable --py --sys-prefix ipymonaco')


# ## Render Microsoft Monaco as a Jupyter ipywidget
# Before you can run the following cell, run the prior installation steps and **refresh this page with the browser's cache disabled**.

# In[ ]:


from ipymonaco import *
hello = Monaco(value="SELECT * FROM table;", theme="vs-dark", language="sql", readOnly=False)
hello

