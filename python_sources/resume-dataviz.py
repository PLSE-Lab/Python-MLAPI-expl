#!/usr/bin/env python
# coding: utf-8

# Those are some code I used to draw figures and export them as SVG before using them as models in Affinity Designer to aggrement my resume.

# In[ ]:


# libraries
import matplotlib.pyplot as plt
import squarify    # pip install squarify (algorithm for treemap)
 
# Change color
squarify.plot(sizes=[50,22,5, 15, 25], label=["Numpy", "Panda", "PyTorch", "Matplotlib", "Jupyter"], alpha=.4 )
plt.axis('off')
plt.title('Python')
plt.show()


# In[ ]:


plt.savefig('treeplot.png')


# In[ ]:


import matplotlib.pyplot as plt
from matplotlib import cm

labels = ["English", "French", "German"]
data = [95, 100, 40]
#number of data points
n = len(data)
#find max value for full ring
m=100
#radius of donut chart
r = 1.5
#calculate width of each ring
w = r / n 

#create colors along a chosen colormap
colors = [cm.terrain(i / n) for i in range(n)]

#create figure, axis
fig, ax = plt.subplots()
ax.axis("equal")

#create rings of donut chart
for i in range(n):
    #hide labels in segments with textprops: alpha = 0 - transparent, alpha = 1 - visible
    innerring, _ = ax.pie([m - data[i], data[i]], radius = r - i * w, startangle = 90, labels = ["", labels[i]], labeldistance = 1 - 1 / (1.5 * (n - i)), textprops = {"alpha": 0}, colors = ["white", colors[i]])
    plt.setp(innerring, width = w, edgecolor = "white")

plt.legend()
plt.show()


# In[ ]:


plt.savefig('radial_languages.svg')


# In[ ]:


import plotly.express as px
fig = px.treemap(
    names = ["Python","SQL", "C", "GIT", "Numpy", "Panda", "Pytorch", "Matplotlib"],
    parents = ["", "", "", "", "Python", "Python", "Python", "Python"]
)
fig.show()


# In[ ]:


import plotly.express as px
import pandas as pd
libraries = ["Numpy", "Matplotlib", "Panda", "Pytorch", "Jupyter", None, None, None]
languages = ["Python", "Python", "Python", "Python", "Python","SQL", "C", "GIT"]
scores = [4, 4, 5, 7, 2, 2, 2, 2]
df = pd.DataFrame(
    dict(libraries=libraries, languages=languages, scores=scores)
)
#df["all"] = "all" # in order to have a single root node
print(df)
fig = px.treemap(df, path=['languages', 'libraries'], values='scores')
fig.show()


# In[ ]:


conda install -c plotly plotly-orca


# In[ ]:


fig.write_image("prog_lang_treemap.svg")


# In[ ]:




