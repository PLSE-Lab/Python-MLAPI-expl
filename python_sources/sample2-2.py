#!/usr/bin/env python
# coding: utf-8

# ### Step 1. Import the numpy libraries

# In[ ]:


import numpy as np


# * ### Step 2. Assign [1.73, 1.68, 1.71, 1.89, 1.79] of np.array to np_height

# In[ ]:


np_height = np.array([1.73, 1.68, 1.71, 1.89, 1.79])


# ### Step 3. Assign [65.4, 59.2, 63.6, 88.4, 68.7] of np.array to np_weight

# In[ ]:


np_wieght = np.array([65.4, 59.2, 63.6, 88.4, 68.7])


# # Type of NumpPy Array

# ### Step 4. Type of NumpPy Array np_height

# In[ ]:


type(np_height)


# ### Step 5. Type of NumpPy Array np_weight

# In[ ]:


type(np_height)


# # 2D NumPy Arrays

# ### Step 6. Assign [[1.73, 1.68, 1.71, 1.89, 1.79], [65.4, 59.2, 63.6, 88.4, 68.7]] of np.array to np_2d

# In[ ]:


np_2d = np.array([[1.73, 1.68, 1.71, 1.89, 1.79], [65.4, 59.2, 63.6, 88.4, 68.7]])


# ### Step 7. Display np_2d

# In[ ]:


np_2d


# ### Step 8. Display the shape of np_2d

# In[ ]:


np_2d.shape


# ### Step 9. Add a new value "68.7" to np.array

# In[ ]:





# # Subsetting

# ### Step 10. Display row 0 of np_2d

# In[ ]:


np_2d[0]


# ### Step 11. Display row 0 and column 2 of np_2d

# In[ ]:


np_2d[0][2]


# ### Step 12. Using other methods from the above cell such as "," to display row 0 and column 2 of np_2d

# In[ ]:


np_2d[0,2]


# ### Step 13. Display all rows and column 1 to 3 of np_2d

# In[ ]:


np_2d[:, 1:3]


# ### Step 14. Display row from 1 to end and all columns of np_2d

# In[ ]:


np_2d[1:, :]

