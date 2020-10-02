#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


df=pd.read_csv('/kaggle/input/data.csv')
df


# # line width

# In[ ]:



import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
# line 1 points
x1 = df['time']
y1 = df['temp_1']
# plotting the line 1 points 
plt.plot(x1, y1, label = "line 1",linewidth=15)
# line 2 points
x2 = df['time']
y2 = df['temp_2']
# plotting the line 2 points 
plt.plot(x2, y2, label = "line 2",linewidth=10)
plt.xlabel('x - axis')
# Set the y axis label of the current axis.
plt.ylabel('y - axis')
# Set a title of the current axes.
plt.title('line width demo ')
# show a legend on the plot
plt.legend()
# Display a figure.
plt.show()


# # colour - you can use css also

# In[ ]:


import matplotlib.pyplot as plt
# line 1 points
x1 = df['time']
y1 = df['temp_1']
# plotting the line 1 points 
plt.plot(x1, y1, label = "line 1",linewidth=10,color='pink')
# line 2 points
x2 = df['time']
y2 = df['temp_2']
# plotting the line 2 points 
plt.plot(x2, y2, label = "line 2",linewidth=10,color='red')
plt.xlabel('x - axis')
# Set the y axis label of the current axis.
plt.ylabel('y - axis')
# Set a title of the current axes.
plt.title('line color demo ')
# show a legend on the plot
plt.legend()
# Display a figure.
plt.show()


# # line styles - '-', '--', '-.', ':', '', (offset, on-off-seq), ...

# https://matplotlib.org/api/_as_gen/matplotlib.lines.Line2D.html?highlight=line2d#matplotlib.lines.Line2D.set_linestyle
# 

# In[ ]:


import matplotlib.pyplot as plt
# line 1 points
x1 = df['time']
y1 = df['temp_1']
# plotting the line 1 points 
plt.plot(x1, y1, label = "line 1",linewidth=1,color='green',linestyle='--')
# line 2 points
x2 = df['time']
y2 = df['temp_2']
# plotting the line 2 points 
plt.plot(x2, y2, label = "line 2",linewidth=1,color='red',linestyle='-')
plt.xlabel('x - axis')
# Set the y axis label of the current axis.
plt.ylabel('y - axis')
# Set a title of the current axes.
plt.title('linestyles ')
# show a legend on the plot
plt.legend()
# Display a figure.
plt.show()


# # Style line ends and joins

# # try -solid_capstyle="round", solid_joinstyle="round")

# In[ ]:


import matplotlib.pyplot as plt

# line 1 points
x1 = df['time']
y1 = df['temp_1']
# plotting the line 1 points 
plt.plot(x1, y1, label = "line 1",linewidth=10,color='green',solid_capstyle="butt", solid_joinstyle="miter")
# line 2 points
x2 = df['time']
y2 = df['temp_2']
# plotting the line 2 points 
plt.plot(x2, y2, label = "line 2",linewidth=10,color='red',solid_capstyle="projecting", solid_joinstyle="bevel")
plt.xlabel('x - axis')
# Set the y axis label of the current axis.
plt.ylabel('y - axis')
# Set a title of the current axes.
plt.title('styles ')
# show a legend on the plot
plt.legend()
# Display a figure.
plt.show()


# # scatter sizes

# In[ ]:


plt.scatter(x1,y1,s=5)


# # every marker different size

# In[ ]:


sizes = (np.random.sample(size=x1.size) * 10) ** 2
plt.scatter(x1,y1,s=sizes)


# # marker types
# 
#     ".": point
#     "o": circle
#     "s": square
#     "^": triangle
#     "v": upside down triangle
#     "+": plus
#     "x": X
# 

# In[ ]:


plt.scatter(x1,y1,marker='v')


# # change marker colour

# In[ ]:


plt.scatter(x1,y1,c='orange')


# # change colour of each marker

# In[ ]:


plt.scatter(x1,y1,c=x1-y1)


# # make markers transparent

# In[ ]:


plt.scatter(x1, y1, marker=".", alpha=.5, edgecolors="none")


# # getting layout right

# In[ ]:


import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
fig = plt.figure()
plt = fig.gca()
plt.plot(x1, y1)


# # Hide axis spines 

# In[ ]:


import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
fig = plt.figure()
plt = fig.gca()
plt.spines["top"].set_visible(False)
plt.spines["right"].set_visible(False)
plt.spines["left"].set_visible(False)
plt.plot(x1, y1)


# # moving spines around

# In[ ]:


import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
fig = plt.figure()
plt = fig.gca()
plt.spines["top"].set_position(("outward", 15))
plt.spines["bottom"].set_position(("data", 80))
plt.spines["left"].set_position(("axes", .3))
plt.spines["right"].set_position(("outward", -40))

plt.plot(x1, y1)


# # modifying spines

# In[ ]:


import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
fig = plt.figure()
plt = fig.gca()
plt.spines["top"].set_color("orange")

plt.spines["right"].set_linestyle("--")
plt.spines["bottom"].set_linewidth(6)
plt.spines["bottom"].set_capstyle("round")

plt.plot(x1, y1)


# # change axes background colour

# In[ ]:


import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
fig = plt.figure()
plt = fig.gca()
plt.set_facecolor("#ff0000")

plt.plot(x1, y1)


# # changing figure background

# In[ ]:


import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
fig = plt.figure()

fig = plt.figure(facecolor="#e1ddbf")
fig.savefig("image_filename.png", facecolor=fig.get_facecolor())

plt.plot(x1, y1)


# # framing the figure

# In[ ]:


import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
fig = plt.figure()

fig = plt.figure(linewidth=10, edgecolor="#04253a")
fig.savefig("image_filename.png", edgecolor=fig.get_edgecolor())
plt.plot(x1, y1)


# # XKCD Function

# In[ ]:


import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
with plt.xkcd():
    fig = plt.figure()
    fig = plt.figure(linewidth=10, edgecolor="#04253a")
    fig.savefig("image_filename.png", edgecolor=fig.get_edgecolor())
    plt.plot(x1,y1)


# # Set axis style - plt.style.use('default')

# In[ ]:


fig = plt.figure(dpi=100, figsize=(10, 20), tight_layout=True)
available = ['default'] + plt.style.available
for i, style in enumerate(available):
    with plt.style.context(style):
        ax = fig.add_subplot(10, 3, i + 1)
        ax.plot(x1, y1)
    ax.set_title(style)


# # Upvote if you like
