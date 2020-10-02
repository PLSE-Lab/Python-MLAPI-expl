#!/usr/bin/env python
# coding: utf-8

# [-> Learn Data Science by Coding](https://www.kaggle.com/andyxie/learn-data-science-by-coding/)

# # Notes
# To draw several several lines on one plot is as easy as repeating `plt.plot`:

# In[ ]:


# RUN ALL THE CODE BEFORE YOU START
import numpy as np
from matplotlib.pylab import plt #load plot library
# indicate the output of plotting function is printed to the notebook
get_ipython().run_line_magic('matplotlib', 'inline')


def create_random_walk():
    x = np.random.choice([-1,1],size=100, replace=True) # Sample with replacement from (-1, 1)
    return np.cumsum(x) # Return the cumulative sum of the elements
X = create_random_walk()
Y = create_random_walk()
Z = create_random_walk()

# Plotting functionality starts here
plt.plot(X)
plt.plot(Y)
plt.plot(Z)
plt.show()


# In addition to the line style like `'r'`, you can gain more detailed control over color configuration by specifying `color` parameter. More colors can be found in [http://matplotlib.org/users/colors.html](http://matplotlib.org/users/colors.html).

# In[ ]:


plt.plot(X, '-.', color="#333333")
plt.plot(Y, '-.', color="chocolate")
plt.plot(Z, '-.', color="green")
plt.show()


# To add legend, you will first need to specify a label when plot `plt.plot(X, label="X")`.  Then call `plt.legend()`. If you are not happy with the location of the legend, pass `loc="Location String"` to `plt.legend()`. Here are some `Location String` to choose from:
# ```
# ===============   =============
# Location String   Location Code
# ===============   =============
# 'best'            0
# 'upper right'     1
# 'upper left'      2
# 'lower left'      3
# 'lower right'     4
# 'right'           5
# 'center left'     6
# 'center right'    7
# 'lower center'    8
# 'upper center'    9
# 'center'          10
# ===============   =============
# ```

# In[ ]:


plt.plot(X, label="X")
plt.plot(Y, label="Y")
plt.plot(Z, label="Z")
# Add legend
plt.legend(loc='lower left')
# Add title and x, y labels
plt.title("Random Walk Example", fontsize=16, fontweight='bold')
plt.suptitle("Random Walk Suptitle", fontsize=10)
plt.xlabel("Number of Steps")
plt.ylabel("Accumulative Sum")
plt.show()


# `plt.plot` will use default setting to make a plot, `plt.show()` will then render the plot:

# # Task 1: Plot Multiple Lines
# Plot the following functions:
# $y=5x$, $y=x^2$, $y=e^{\frac{3}{2}x}, x \in [0, 6]$

# In[ ]:


# RUN ALL THE CODE BEFORE YOU START
import numpy as np
from matplotlib.pylab import plt #load plot library
# indicate the output of plotting function is printed to the notebook
get_ipython().run_line_magic('matplotlib', 'inline')

x = np.linspace(0, 6, 100)
y_1 = 5*x
y_2 = np.power(x, 2)
y_3 = np.exp(x/1.5)
plt.plot(x, y_1)
plt.plot(x, y_2)
plt.plot(x, y_3)
plt.show()


# # Task 2: Change Line Style
# Change the lines to different level of gray, style to line, dotted-line and dash line.

# In[ ]:


plt.plot(x, y_1, '-.', color="#333333")
plt.plot(x, y_2, '--', color="#999999")
plt.plot(x, y_3, '-', color="#aaaaaa")
plt.show()


# # Task 3: Plot Line with Error
# Add legend. To use latex in label string, here is an exmpale:
# `label="$y=x$"`

# In[ ]:


plt.plot(x, y_1, '-.', color="#333333", label="$y=5 * x$")
plt.plot(x, y_2, '--', color="#999999", label="$y=x^2$")
plt.plot(x, y_3, '-', color="#aaaaaa", label="$y=e^{3/2x}$") # TODO: Fix the notation here
plt.legend(loc="upper left")

plt.title("Several Functions", fontsize=16, fontweight='bold')
plt.xlabel("X Values")
plt.ylabel("Y Values")

plt.show()


# [-> Learn Data Science by Coding](https://www.kaggle.com/andyxie/learn-data-science-by-coding/)
