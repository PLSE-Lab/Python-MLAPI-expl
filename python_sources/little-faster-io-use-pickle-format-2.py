#!/usr/bin/env python
# coding: utf-8

# ## Little Faster IO : Use pickle format (2)
# 
# >** Use only before last submission. Because this is Kernel Only Competion.**
# 
# **Previous Content**
# 
# - [Little Faster IO : Use pickle format](https://www.kaggle.com/subinium/little-faster-io-use-pickle-format)
# 
# ---
# 
# We made the pickle file an output file from the previous kernel.
# We will use **+ Add Data** next to **commit Button**.
# 
# ![Add Data](https://i.imgur.com/npbYvAx.png)
# 
# Click to get the file from the kernel. In my case I loaded the pickle file created by the previous kernel. (click add please)
# 
# ![Kernel Output only](https://i.imgur.com/WwzAh6h.png)
# 

# 
# Then you can see the import is successful in about 30 seconds.
# 
# ![Time](https://i.imgur.com/ZMJmBP8.png)
# 
# Now you can check if the file you loaded is in data.

# You can use the Python code provided by the Kaggle kernel to get the path.

# In[ ]:


import os
import pandas as pd
for dirname, _, filenames in os.walk('/kaggle/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# You can easily import it by replacing `read_csv` with `read_pickle` as follows:

# In[ ]:


get_ipython().run_cell_magic('time', '', "train_pickle = pd.read_pickle('/kaggle/input/little-faster-io-use-pickle-format/train.pkl')")


# You can see that the import, which previously required about 1 minute and 15 seconds, now takes near 20 seconds.
# 
# Now you can do competition with pandas.

# In[ ]:


get_ipython().run_cell_magic('time', '', 'train_pickle.head()')


# In[ ]:




