#!/usr/bin/env python
# coding: utf-8

# ## Overview and Outline
# 
# In this notebook we will cover exploratory data analysis with python. We will focus on the **plotly** library for all of the visualizations. This library makes some of the most aesthetic plots, while providing a wide variety of options. For more extensive information and examples see [plotly](https://plotly.com/python/).
# 
# 
# ### Objectives
# 
# - Create univariate plots such as boxplot and histogram
# - Facet plots by categorical variables
# - Create multivariate plots
# - Create interactive visualizations
# 
# 
# ### Libraries Used
# 
# - numpy: To compute summary statistics of the data
# - pandas: To read and query the data
# - plotly: To make visually aesthetic and interactive data visualizations

# In[ ]:


import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# ## Getting Started
# 
# ### Getting the data
# 
# The data for this tutorial is available on kaggle at: [pokemon data](https://www.kaggle.com/abcsds/pokemon)
# 
# ### Overview of the data
# 
# The dataset contains pokemon ability values. These include HP(Health), Attack, Defense, Sp. Atk(Special Attack), Sp. Def(Special Defense), and Speed. Other information includes the primary type and secondary type. These types include fire, water, psychic, dark, dragon, and many more. The generation is also included, which are numbers from 1 to 6. This tells us which generation or set of games that the pokemon are from. Lastly, there is a boolean variable that tells us whether a pokemon is legendary or not.
# 
# ### Reading the data
# 
# We use `read_csv` from the **pandas** library to read the data.

# In[ ]:


pokemon = pd.read_csv('../input/pokemon/Pokemon.csv')


# ### Data Summaries and Types
# 
# We can check the data types of each variable in the data frame with `dtype` from **numpy**

# In[ ]:


pokemon.dtypes


# With `.describe()` from **numpy**, we can get summary statistics of each numeric variable. The statistics provided are count, mean, standard deviation(std), minimum, 25th percentile, 50th percentile or median, 75th percentile, and maximum.

# In[ ]:


pokemon.describe()


# ## Univariate visualizations
# 
# Univariate distributions are good for an initial exploration of the data. With histograms and density plots, we can examine the distribution of the data. This is good for detecting skew, and assessing normality of the data. 
# 
# 
# ### Histogram
# 
# With `histogram` we can make a histogram of a variable by entering the data frame **pokemon**, and setting `x = "Attack"`. This allows us to see the distribution of Attack stats for all pokemon within the data set.

# In[ ]:


px.histogram(pokemon, x = "Attack")


# With the `color =` parameter, we can see where the legendary pokemon fit within the attack value distribution.

# In[ ]:


px.histogram(pokemon, x = "Attack",color = "Legendary")


# Here we add the `facet_col =` parameter to make histograms faceted by what generation the pokemon are from. We keep the color parameter to see where the legendary pokemon fit within each generation's attack value distribution. 

# In[ ]:


px.histogram(pokemon, x="Attack", color="Legendary",facet_col = "Generation",facet_col_wrap = 3)


# ### Boxplots
# 
# Boxplots are another useful univariate visualization. They show the spread of the data and any outliers. With `box` from **plotly.express**, we can easily make one. As with most **plotly** visualizations hovering the mouse over the
# plot gives useful information. In this case, it is the min, q1, median, q3, lower fence, upper fence, and the values 
# of any outliers.

# In[ ]:


px.box(pokemon, y = 'Defense')


# By adding the `y = ` parameter, we can makes faceted boxplots. From the plot, we can see that generation 1 and 2 have the biggest outliers in HP(Health) values.

# In[ ]:


px.box(pokemon,x= 'Generation',y = 'HP')


# We can do something similar to the facets, by adding the `color =` parameter. This makes our facets different colors, and more visually aesthetic.

# In[ ]:


px.box(pokemon, y="Attack", color = "Type 1")


# ## Multivariate Visualizations
# 
# 
# ### Scatterplot
# 
# With scatterplots, we can see linear  and nonlinear relationships between two variables. To make one with **plotly**, we use `scatter` and set `x= 'Defense'`, and `y = 'Sp. Def'`.

# In[ ]:


px.scatter(pokemon, x= 'Defense', y = 'Sp. Def')


# #### Adding a regression line
# 
# ##### Linear
# 
# To add a regression line to the scatterplot, we set `trendline = "ols`. OLS stands for ordinary least squares, which corresponds to a fitted line in the case of simple bivariate regression. Hovering the mouse over the trend line gives
# us the R-Squared value, slope, and intercept.

# In[ ]:


px.scatter(pokemon,x= "Defense", y = "Sp. Def",trendline = "ols")


# We can separate the data on whether or not the pokemon is legendary, by setting `color = Legendary`. This will give us two regression lines, one for ordinary pokemon, and the other for legendary pokemon. 

# In[ ]:


px.scatter(pokemon,x= "Defense", y = "Sp. Def",trendline = "ols",color = "Legendary")


# ##### Loess
# 
# We do not have to stick to a fitted line, but can use a lowess smoother instead. This is done by setting `trendline = "lowess"`

# In[ ]:


px.scatter(pokemon, x= "Defense", y = "Sp. Def", color = "Legendary", trendline = "lowess")


# ### Bubbleplot
# 
# For the bubble plot, we use `scatter`, and add the `size =` parameter. This gives use bubbles with size corresponding to **HP** in our case.

# In[ ]:


px.scatter(pokemon,x= 'Attack',y = 'Defense', size = 'HP', color = 'Legendary')


# ### Bivariate Histograms and Density Visualizations
# 
# With `density_contour` from **plotly**, we can make a density contour plot. We add the univariate histograms to the 
# x and y axis

# In[ ]:


px.density_contour(pokemon,x="Attack",y="Defense",marginal_x="histogram",marginal_y="histogram")


# Using **plotly.graph_objects**, we can make more additions to the plot. We start by assigning **Attack** to be **x**, and **defense** to be **y** for convenience. We set **fig** to be `go.Figure()`. Then we use `add_trace()` to add 
# a trace to the figure. This will allow us to overlay more layers later on. Then we use `Histogram2dContour()` to add
# the the density contours. We set the colorscale to be Purples. `reversescale = True` gives darker colors for less areas and lighter shades for more dense areas.

# In[ ]:


fig = go.Figure()
x = pokemon["Attack"]
y = pokemon["Defense"]
fig.add_trace(go.Histogram2dContour(
        x = x,
        y = y,
        colorscale = 'Purples',
        reversescale = True,
        xaxis = 'x',
        yaxis = 'y'
    ))


# Here we will add two traces to the figure. We start with the contour histogram as with above, but use the green colorscale. We add a second trace with `add_trace()`. This time we use `Scatter()` from **plotly.graph_objects**. The function parameters are similar here as with the contour histogram, but we set `mode = 'markers'`, and use `marker =` to format the scatterplot points.

# In[ ]:


fig = go.Figure()
fig.add_trace(go.Histogram2dContour(
        x = x,
        y = y,
        colorscale = 'Greens',
        xaxis = 'x',
        yaxis = 'y'
    ))
fig.add_trace(go.Scatter(
        x = x,
        y = y,
        xaxis = 'x',
        yaxis = 'y',
        mode = 'markers',
        marker = dict(
            color = 'black',
            size = 2
        )
    ))


# ### Parallel Coordinates Plot
# 
# The parallel coordinates plot is useful to examine relationships between multiple variables at once. To make one, we use `parallel_coordinates`. We use this on all six variables that compose a pokemon's stats. We set `color = 'Generation'`.

# In[ ]:


px.parallel_coordinates(pokemon, color = 'Generation',
                       dimensions = ['Attack', 'Defense', 'HP',
                                    'Speed', 'Sp. Atk', 'Sp. Def'])


# The above plot is very busy, so we will reduce the number of pokemon in the plot by querying down to only first generation pokemon. This easily down with `query` from the **pandas** library. We use the conditional expression `Generation == 1` to filter out observations or pokemon that do not have True values for the expression. Then we use the same code as above, but with **gen1** instead of **pokemon**.

# In[ ]:


gen1 = pokemon.query("Generation == 1")
px.parallel_coordinates(gen1,
                       dimensions = ['Attack', 'Defense', 'HP',
                                    'Speed', 'Sp. Atk', 'Sp. Def'])


# ### Parallel Categories Diagram
# 
# The parallel categories diagram is similar to the parallel coordinates plot, but it is for categorical variables. We will use **Type 1** and **Type 2** for this plot. We use `parellel_categories` to make the plot, and it follows a 
# similar format to `parallel_coordinates`

# In[ ]:


px.parallel_categories(pokemon, dimensions=['Type 1', 'Type 2'])


# ### Scatterplot Matrix

# A scatterplot matrix is useful to examine the relationships between multiple sets of variables. The diagonal plots are not useful, as they show a variable vs itself. We use `scatter_matrix` to make the plot. `dimension =` is where we 
# specify the variables we want to examine. We use `color = 'Legendary'` as with the earlier scatterplots to see the differences between ordinary and legendary pokemon with the data.

# In[ ]:


px.scatter_matrix(pokemon,dimensions = ['Attack','Defense','HP','Speed'],
                 color = 'Legendary')


# ### 3D Scatterplot
# 
# 3D scatterplots are a cool way to visualize three variables. We use `scatter_3d` to make this plot. This function takes an **x**, **y**, and **z** variable. We also set `color = "Legendary"` to visualize the difference between ordinary and legendary pokemon. 

# In[ ]:


px.scatter_3d(pokemon,x='Attack',y='Defense',z='Speed', color = 'Legendary',opacity = .5)


# We can make the 3d scatterplot a bubble plot by adding the `size =` parameter.

# In[ ]:


px.scatter_3d(pokemon,x='Attack',y='Defense',z='Speed', size = 'HP', color ='Legendary')

