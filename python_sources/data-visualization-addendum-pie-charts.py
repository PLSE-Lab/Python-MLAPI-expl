#!/usr/bin/env python
# coding: utf-8

# # Addendum: on Pie Charts
# 
# One common form of data visualization notably absent so far in the pie chart. This, too, is easy to make in `pandas`:

# In[ ]:


import pandas as pd
reviews = pd.read_csv("../input/winemag-data_first150k.csv", index_col=0)


# In[ ]:


reviews['province'].value_counts().head(10).plot.pie()

# Unsquish the pie.
import matplotlib.pyplot as plt
plt.gca().set_aspect('equal')


# But you shouldn't use it. The reason why is simple: can you tell me, looking at this chart, which providence produces more wine: Veneto, or Burgundy?
# 
# Research has shown that pie charts work well for quantities that are near common fractional values: one-half, one-third, and one-quarter. However, once you start to drill down into tenths, and twelvths, and so on, our ability to visually compare two pie slices, especially ones not immediately adjacent to one another, breaks down.
# 
# Pie charts are like bar charts, but wrapped around a circle. You should just use a bar chart instead.
