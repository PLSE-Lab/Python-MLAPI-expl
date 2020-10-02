#!/usr/bin/env python
# coding: utf-8

# 

# In[ ]:


# Importing Libraries

# pandas
import pandas as pd
from pandas import Series,DataFrame

#numpy, matplotlib, seaborn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')

#bokeh
from bokeh.plotting import figure
from bokeh.charts import output_file, show, output_notebook
from bokeh.models import HoverTool, ColumnDataSource
output_notebook()

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# machine Learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


# In[ ]:


# import train and test data frames
train_df = pd.read_csv("../input/train.csv", dtype={"Age": np.float64},)
test_df  = pd.read_csv("../input/test.csv", dtype={"Age": np.float64},)

# preview the data
train_df.head()




# In[ ]:


# view summary of the data
train_df.info()


# In[ ]:


#Because I can't get CategoricalColorMapper to import correctly, I'm using the below code to add a column of color to the dataframe
train_df['color'] = np.where(train_df['Survived']==1, 'SpringGreen', 'Crimson')
#If CategoricalColorMapper I would use the following code:
#color_mapper=CategoricalColorMapper(factors=[0, 1],
#                                    palette=['Crimson', 'SpringGreen'])
#This element would be added to p.circle:  color=dict(field='Survived', transform=color_mapper)

#Define the source of data for our graphing
source = ColumnDataSource(train_df)

#Define the axis labels for the figure
p=figure(x_axis_label='Age',y_axis_label='Fare')

#x and y axis display for scatter plot, size 5 circles and color is the additional color column
p.circle('Age','Fare',source=source,size=5,alpha=0.8,color='color')

#adding the hover tool to show name, age, and fare
hover = HoverTool(tooltips=[('Name: ','@Name'),
                            ('Age: ','@Age'),
                            ('Fare: ','@Fare')
                           ])
p.add_tools(hover)

show(p)


# In[ ]:


#Because I can't get CategoricalColorMapper to import correctly, I'm using the below code to add a column of color to the dataframe
train_df['color'] = np.where(train_df['Survived']==1, 'SpringGreen', 'Crimson')
#If CategoricalColorMapper I would use the following code:
#color_mapper=CategoricalColorMapper(factors=[0, 1],
#                                    palette=['Crimson', 'SpringGreen'])
#This element would be added to p.circle:  color=dict(field='Survived', transform=color_mapper)

#Define the source of data for our graphing
source = ColumnDataSource(train_df)

#Define the axis labels for the figure
p=figure(x_axis_label='Age',y_axis_label='Fare')

#x and y axis display for scatter plot, size 5 circles and color is the additional color column
p.circle('Age','Fare',source=source,size=5,alpha=0.8,color='color')

#adding the hover tool to show name, age, and fare
hover = HoverTool(tooltips=[('Name: ','@Name'),
                            ('Age: ','@Age'),
                            ('Fare: ','@Fare')
                           ])
p.add_tools(hover)

show(p)

