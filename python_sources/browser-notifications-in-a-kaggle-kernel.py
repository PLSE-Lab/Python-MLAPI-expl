#!/usr/bin/env python
# coding: utf-8

# # Browser Notifications

# You can display a browser notification when a cell is completed with [jupyter-notify](https://github.com/shoprunner/jupyter-notify).
# 
# I added the package directly to the kernel (settings -> packages -> custom packages -> jupyernotify). This approach is necessary when working on competition kernels where internet is not allowed. However if internet is allowed, it can be installed with *!pip install jupyernotify*
# 
# Note that the notifications will only popup if you are directly working in a kernel. Notifications will not show when you *commit* a kernel.
# 

# In[ ]:


# install if kernel has internet access
# !pip install jupyternotify


# In[ ]:


# Load the extension!
get_ipython().run_line_magic('load_ext', 'jupyternotify')


# Lets first try a simple notification. We need to add the *%%notify* magic command to the cell's first line
# 
# Should get a browser notification that says "Cell execution has finished!" after 10 seconds

# In[ ]:


get_ipython().run_cell_magic('notify', '', 'import time\ntime.sleep(10)')


# We can modify the text on the notification with the -m flag
# 
# Should now get a browser notification that says 'finished sleeping' after 25 seconds

# In[ ]:


get_ipython().run_cell_magic('notify', "-m 'finished sleeping'", 'import time\ntime.sleep(25)')


# More customization can be found on https://github.com/shoprunner/jupyter-notify
