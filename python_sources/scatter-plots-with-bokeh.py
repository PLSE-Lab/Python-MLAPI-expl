#!/usr/bin/env python
# coding: utf-8

# # Scatter Plots  with  Bokeh 

# I will showing in this notebook how can you make a scatter plot using Bokeh with Python. I will be using also the `decathlon` dataset that can be found the FactoMineR R package 

# Let's first import the data using Panda Library

# In[ ]:


import pandas as pd


# In[ ]:


decathlon = pd.read_csv("../input/decathlon.csv")


# In[ ]:


decathlon.shape


# In[ ]:


decathlon.columns[0]


# In[ ]:


decathlon.head()


# We import the necessarly libraries 

# In[ ]:


decathlon.Javeline.head()


# In[ ]:


decathlon.Discus.head()


# In[ ]:


from bokeh.io import push_notebook, show, output_notebook
from bokeh.layouts import row 
from bokeh.plotting import figure

output_notebook()


# In[ ]:


p = figure(title = "Decathlon: Discus x Javeline")
p.circle('Discus','Javeline',source=decathlon,fill_alpha=0.2, size=10)


# In[ ]:


show(p)


# I will  now customize the scatter. I  will be coloring the bubbles according to the categorical variable `Competition`. We need to import `factor_cmap`. It will be used to map the  colors according to the levels of `Competition`.   

# In[ ]:


from bokeh.transform import factor_cmap


# In[ ]:


decathlon.Competition.unique()


# In[ ]:


index_cmap = factor_cmap('Competition', palette=['red', 'blue'], 
                         factors=sorted(decathlon.Competition.unique()))



# In[ ]:


p = figure(plot_width=600, plot_height=450, title = "Decathlon: Discus x Javeline")
p.scatter('Discus','Javeline',source=decathlon,fill_alpha=0.6, fill_color=index_cmap,size=10,legend='Competition')
p.xaxis.axis_label = 'Discus'
p.yaxis.axis_label = 'Javeline'
p.legend.location = "top_left"


# In[ ]:


show(p)


# I will be adding an Hoover to the bubbles. It means that a pop-up will show off when we click on the bulble. I will make the name of the 

# In[ ]:


decathlon['Athlets'].head()


# In[ ]:


p = figure(plot_width=600, plot_height=450, 
           title = "Decathlon: Discus x Javeline",
           toolbar_location=None,
          tools="hover", 
           tooltips="@Athlets: (@Discus,@Javeline)")
p.scatter('Discus','Javeline',source=decathlon,fill_alpha=0.6, fill_color=index_cmap,size=10,legend='Competition')
p.xaxis.axis_label = 'Discus'
p.yaxis.axis_label = 'Javeline'
p.legend.location = "top_left"
show(p)


# We will be adding now the names of the Athlets in the graph. We need first to import some functions and transform `decathlon` data to a `ColumnDataSource` object.  

# In[ ]:


from bokeh.models import  ColumnDataSource,Range1d, LabelSet, Label

decath=ColumnDataSource(data=decathlon)


# We draw then the scatter plot 

# In[ ]:


p = figure(plot_width=700, plot_height=450, title = "Decathlon: Discus x Javeline")
p.scatter('Discus','Javeline',source=decath,fill_alpha=0.6, fill_color=index_cmap,size=10,legend='Competition')
p.xaxis.axis_label = 'Discus'
p.yaxis.axis_label = 'Javeline'
p.legend.location = "top_left"


# We add then the labels 

# In[ ]:


labels = LabelSet(x='Discus', y='Javeline', text='Athlets', level='glyph',text_font_size='9pt',
              text_color=index_cmap,x_offset=5, y_offset=5, source=decath, render_mode='canvas')

p.add_layout(labels)


# In[ ]:


show(p)


# We will be adding Box Annotations from  top to  bottom or left to right 

# In[ ]:


from bokeh.models import BoxAnnotation

p = figure(plot_width=700, plot_height=450, title = "Decathlon: Discus x Javeline")
p.scatter('Discus','Javeline',source=decath,fill_alpha=0.6, fill_color=index_cmap,size=10,legend='Competition')
p.xaxis.axis_label = 'Discus'
p.yaxis.axis_label = 'Javeline'
p.legend.location = "top_left"


# In[ ]:


low_box = BoxAnnotation(top=55, fill_alpha=0.1, fill_color='red')
mid_box = BoxAnnotation(bottom=55, top=65, fill_alpha=0.1, fill_color='green')
high_box = BoxAnnotation(bottom=65, fill_alpha=0.1, fill_color='red')


# In[ ]:


p.add_layout(low_box)
p.add_layout(mid_box)
p.add_layout(high_box)


# In[ ]:


p.xgrid[0].grid_line_color=None
p.ygrid[0].grid_line_alpha=0.5
show(p)


# We can make an interactive legend too 

# In[ ]:


p = figure(plot_width=600, plot_height=450, title = "Decathlon: Discus x Javeline")
p.title.text = 'Click on legend entries to hide the corresponding lines'


# In[ ]:


decathlon.loc[(decathlon.Competition=='OlympicG')].head()


# In[ ]:


x=['OlympicG','Decastar']


# In[ ]:


x


# In[ ]:


for i in x:
    df=decathlon.loc[(decathlon.Competition==i)]
    p.scatter('Discus','Javeline',source=df,fill_alpha=0.6, fill_color=index_cmap,size=10,legend='Competition')

p.legend.location = "top_left"
p.legend.click_policy="hide"
show(p)

    


# We will now color the circles according to a continuous variable. Let's say we will the `Points` variable
# 

# In[ ]:


from bokeh.models import ColumnDataSource, ColorBar
from bokeh.palettes import Spectral6
from bokeh.transform import linear_cmap


# In[ ]:


Spectral6


# In[ ]:


mapper = linear_cmap(field_name='Points', palette=Spectral6 ,low=min(decathlon['Points']) ,high=max(decathlon['Points']))


# In[ ]:


p = figure(plot_width=700, plot_height=450, title = "Decathlon: Discus x Javeline")
p.scatter('Discus','Javeline',source=decath,fill_alpha=0.6, line_color=mapper,color=mapper,size=10)
p.xaxis.axis_label = 'Discus'
p.yaxis.axis_label = 'Javeline'


# Adding the legend 

# In[ ]:


color_bar = ColorBar(color_mapper=mapper['transform'], width=8,  location=(0,0),title="Points")

p.add_layout(color_bar, 'right')


# In[ ]:


show(p)


# In[ ]:


p = figure(plot_width=700, plot_height=450, title = "Decathlon: Discus x Javeline")
p.scatter('Discus','Javeline',source=decath,fill_alpha=0.6, line_color=mapper,color=mapper,size='Points')
p.xaxis.axis_label = 'Discus'
p.yaxis.axis_label = 'Javeline'


# In[ ]:


show(p)


# We need then to do this in order to control the size of the bubbles in the scatter plot 

# In[ ]:


from bokeh.models import LinearInterpolator


# In[ ]:


decath


# In[ ]:


size_mapper=LinearInterpolator(
    x=[decathlon.Points.min(),decathlon.Points.max()],
    y=[5,50]
)


# In[ ]:


p = figure(plot_width=700, plot_height=450, title = "Decathlon: Discus x Javeline",
          toolbar_location=None,
          tools="hover", tooltips="@Athlets: @Points")
p.scatter('Discus','Javeline',
          source=decathlon,
          fill_alpha=0.6, 
          fill_color=index_cmap,
          size={'field':'Points','transform': size_mapper},
          legend='Competition'
         )
p.xaxis.axis_label = 'Discus'
p.yaxis.axis_label = 'Javeline'
p.legend.location = "top_left"


# In[ ]:


show(p)

