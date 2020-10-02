#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ## In a secondary school in Nigeria, a census taken regarding the various ethnic groups gives the following result.
# 
# | Group       | Number of people     |
# | :------------- | :----------: |
# |  Yoruba | 200   |
# | Ibo   | 50 |
# | Hausa | 90 |
# | Tiv | 100 |
# | Ijaw | 60 |
# | Igala | 80 |
# | Others | 140 |

# In[ ]:



# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = 'Yoruba', 'Ibo', 'Hausa', 'Tiv', 'Ijaw', 'Igala', 'Others'
sizes = [200, 50, 90, 100, 60, 80, 140]
explode = (0, 0, 0, 0, 0, 0, 0.1)  # only "explode" the last slice (i.e. 'Others')

fig1, ax1 = plt.subplots(figsize=(15,10))
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=False, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()


# In[ ]:




