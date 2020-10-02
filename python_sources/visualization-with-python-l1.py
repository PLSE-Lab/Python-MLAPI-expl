#!/usr/bin/env python
# coding: utf-8

# # First Example with marplotlib: barplot

# Loading librairies

# In[ ]:


import matplotlib.pyplot as plt
plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
plt.rcdefaults()
fig, ax = plt.subplots()


# Data

# In[ ]:


people = ('Tom', 'Dick', 'Harry', 'Slim', 'Jim')
y_pos = np.arange(len(people))
performance = 3 + 10 * np.random.rand(len(people))
error = np.random.rand(len(people))


# In[ ]:


people


# In[ ]:


y_pos


# length of the bars

# In[ ]:


performance


# Error bars

# In[ ]:


error


# In[ ]:


ax.barh(y_pos, performance, xerr=error, align='center',
        color='green', ecolor='black')
ax.set_yticks(y_pos)
ax.set_yticklabels(people)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Performance')
ax.set_title('How fast do you want to go today?')

plt.show()


# **Exercice**

# Download the IDR data from my Kaggle account: https://www.kaggle.com/dhafer/regional-development-index-tunisia and draw the barplot of the IDR of the gouvernorats of Tunisia
# 

# **Solution**

# Laoding the data 

# In[ ]:


import pandas as pd
import glob


# In[ ]:


print(glob.glob("*.csv"))


# In[ ]:


idr=pd.read_csv('../input/regional-development-index-tunisia/idr_gouv.csv')


# In[ ]:


idr


# In[ ]:


idr.head()


# In[ ]:


idr.describe()


# In[ ]:


idr['IDR']


# In[ ]:


idr['gouvernorat']


# In[ ]:


import numpy as np


# In[ ]:


gouv=np.asarray(idr['gouvernorat'])


# In[ ]:


gouv


# In[ ]:


y_pos=np.arange(len(gouv))


# In[ ]:


y_pos


# In[ ]:


idrvar=np.asarray(idr['IDR'])


# In[ ]:


plt.rcdefaults()
fig, ax = plt.subplots()
ax.barh(y_pos, idrvar, align='center',
        color='red', ecolor='black')
ax.set_yticks(y_pos)
ax.set_yticklabels(gouv)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('IDR')
ax.set_title('Regional Development Index (2010)')

plt.show()


# # A scatter plot

# **Example on Decathlon Data**

# let us import the decathlon data 

# In[ ]:


decat=pd.read_csv('../input/decathlon/decathlon.csv')


# In[ ]:


decat.columns


# In[ ]:


x=np.asarray(decat['100m'])
y=np.asarray(decat['Long.jump'])


# In[ ]:


plt.scatter(x, y,  alpha=0.5)
plt.xlabel("100m")
plt.ylabel("Long Jump")
plt.show()


# # A scatter plot with ggplot2

# ### Loading packages 

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# In[ ]:


decat.plot(kind='scatter', x='Discus', y='Long.jump',  s=decat['100m']*10)
plt.show()


# ### With ggplot2 

# You need first to install ggplot 
$ pip install ggplot
# and followed with
$ conda install ggplot
# In[ ]:


from ggplot import *


# In[ ]:


ggplot(aes(x='Discus', y='Long.jump', color='Competition'), data=decat) +    geom_point() +    theme_bw() +    xlab("Discus") +    ylab("Long Jump") +    ggtitle("Discus x Long Jump")


# In[ ]:




