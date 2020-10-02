#!/usr/bin/env python
# coding: utf-8

# # My first step in EDA using Plotly
# <br>
# In this Kernel I am going to share with you my first step of EDA only on  training set:
# > - Plot target variable distribution
# > - Plot distribution for all features using histogram subplots
# > - Compare distribution of target variable for all features using histogram subplots
# 
# The main purpose of this kernel is not to extract insights which are very welcome in comments,
# instead it was to understand how to use Plotly to create visualizations.
# 
# I hope this can be helpful to you as well.

# ## I) Setup the environment

# In[ ]:


import pandas as pd
import numpy as np
from math import ceil

import plotly
import plotly.plotly as py

# Allows to plot offline
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly.graph_objs as go

# Allows to plot into the notebook
init_notebook_mode(connected=True)


# To make code more readable, I need to use a module with some functions I created.<br>
# The code is hidden in the next code cell. Since it is quite long, I suggest you to keep it hidden.

# In[ ]:


def hist_sub_plots(start, stop, overlayed = False):

    # Set the default layout
    layout = reset_layout()

    # Create subplot grid
    cols = 5
        # this was to automatically define the number of nrows to plot all hist in one figure
        # however, the plot got too much squeezed and is better
#     nrows = ceil(len(train_df.columns[2 : ]) / 2)
    rows = 5
    from plotly import tools
    fig = tools.make_subplots(rows=rows, cols=cols, shared_yaxes=True, print_grid = False,
                          #subplot_titles=(train_df.columns[2 : 2 + (rows * cols)]))
                            subplot_titles=(train_df.columns[1 + start : 1 + stop]))


    if overlayed == False:
        all_traces = {start + trace : create_hist(col) for trace, col in enumerate(train_df.columns[1 + start : 1 + stop])}

        plot = 0
        for i in range(rows):
            for j in range(cols):
                try:
                    fig.append_trace(all_traces[start + plot], i + 1, j + 1)
                    plot += 1
#               if plot == len(train_df.columns[2 : ]):
                except:
                    fig['layout'].update(height=1000, width=1000,
                                         title='Multiple Subplots with Shared Y-Axes',
                                        showlegend = False)
                    iplot(fig, filename='multiple-subplots-shared-yaxes')
                    return

        fig['layout'].update(height=1000, width=1000,
                             title='Multiple Subplots with Shared Y-Axes',
                             showlegend = False)
        iplot(fig, filename='multiple-subplots-shared-yaxes')

        return


    elif overlayed == True:

        traces_detection = {start + trace: create_hist(col, overlayed = True, target = 1) for trace, col in enumerate(train_df.columns[1 + start : 1 + stop])}
        traces_no_detection = {start + trace: create_hist(col, overlayed = True, target = 0) for trace, col in enumerate(train_df.columns[1 + start : 1 + stop])}

        plot = 0
        for i in range(rows):
            for j in range(cols):
                try:
                    fig.append_trace(traces_no_detection[start + plot], i + 1, j + 1)
                    fig.append_trace(traces_detection[start + plot], i + 1, j + 1)
                    plot += 1
                except:
                #if plot == len(train_df.columns[2 : ]):
                    fig['layout'].update(height=1000, width=1000,
                             title='Multiple Subplots with Shared Y-Axes',
                             barmode = "overlay",
                             showlegend = False)
                    iplot(fig, filename='multiple-subplots-shared-yaxes')
                    return

        fig['layout'].update(height=1000, width=1000,
                             title='Multiple Subplots with Shared Y-Axes',
                             barmode = "overlay",
                             showlegend = False)
        iplot(fig, filename='multiple-subplots-shared-yaxes')

        return

################################################################################

def create_hist(col, overlayed = False, target = None):
    if overlayed == False:
        trace = go.Histogram(x = train_df[col],
                    histnorm = "probability",
                    marker = dict(color = 'rgb(191, 190, 190)'),
                    name = col)
    elif overlayed == True:
        if target == 0:
            trace = go.Histogram(x =  train_df[train_df["target"] == 0][col],
                      histnorm = "probability",
                      marker = dict(color = 'rgb(191, 190, 190)'),
                      opacity = 1,
                    name = "No Detection"
                     )
        elif target == 1:
            trace = go.Histogram(x = train_df[train_df["target"] == 1][col],
                      histnorm = "probability",
                      marker = dict(color = "rgba(255,255,255,0)",
                                    line = dict(width = 1, color = 'rgb(195, 81, 78)')),
                      name = "Detection"
                     )
    return trace

################################################################################

def reset_layout():
    layout = go.Layout(
        width=600,
        height=800,
        yaxis=dict(
                        range = [0, 0.6],
                        showline = True,
                        tickmode = "array",
                        tickvals = [val/100 for val in range(0,101,20)],
                        ticktext = [str(text/100) + "%" for text in range(0,101,20)],
                        color = "grey",
                        tickfont = dict(
                                        size = 15
                                        ),
                        zeroline = False
        )
    )

    return layout

################################################################################

def desc_stats(df, var_type):
    '''
    type df: pandas.core.frame.DataFrame
    type var_type: string
    type r: pandas.core.frame.DataFrame
    '''
    import sys
    import pandas as pd
    
    if var_type not in ["category", "number"]:
        if var_type.find("categ") > -1:
            raise ValueError("Maybe you meant category and not {}".format(var_type))
        elif var_type.find("num") > -1:
            raise ValueError("Maybe you meant number and not {}".format(var_type))
        raise ValueError("Summary statistics available only for {} variables".                         format(" or ".join(["categorical", "numerical"])))


    if var_type == "category":

        # Count number of variables to describe
        n = len(df.select_dtypes("category").columns)

        # Create structure of the dataframe for categoricl descriptive statistics
        desc_stats = pd.DataFrame(columns = ["Feature", "Type", "% missing", "Unique values", "Mode", "% mode"])

        for i, colname in enumerate(df.select_dtypes("category").columns):

            # Compute freq in order to find biggest category
            freq = df[colname].                                value_counts(normalize = True, dropna = False).                                to_frame().reset_index().                                iloc[0, : ]

            # Create a temporary dictionary to append the desc_stats for categorical features
            temp = {"Feature": colname,
                   "Type" : str(df[colname].dtype),
                   "% missing": round(df[colname].isnull().sum() * 100 / df.shape[0], 2),
                   "Unique values": df[colname].nunique(),
                    "Mode": freq[0],
                    "% mode": round(freq[1] * 100, 2)
                   }

            # Append to desc_stats
            desc_stats = desc_stats.append(temp, ignore_index = True)


            # Print out progress
            sys.stdout.write("\rProgress: {:2.2%}".format( (i + 1) / n) )
            sys.stdout.flush()

    elif var_type == "number":

        # Import relevant packages for calculations
        from  statistics import median, variance, stdev, mean
        from scipy.stats import kurtosis, skew

        # Count number of variables to describe
        n = len(df.select_dtypes("number").columns)

        # Create structure of the dataframe for categoricl descriptive statistics
        desc_stats = pd.DataFrame(columns = ["Feature", "Type", "% missing",
                                             "Mean", "Median",
                                             "Min", "Max", "Range", "StDev", "Variance",
                                             "Skewness", "Kurtosis"])

        for i, colname in enumerate(df.select_dtypes("number").columns):

            #Create a temporary dictionary to append the desc_stats for numeric features
            temp = {"Feature": colname,
                    "Type": str(df[colname].dtype),
                    "% missing": round(df[colname].isnull().sum() * 100 / df.shape[0], 2),
                    "Mean": round(df[colname].dropna().mean(), 2),
                    "Median": median(df[colname].dropna()),
                    "Min": min(df[colname].dropna()),
                    "Max": max(df[colname].dropna()),
                    "Range": max(df[colname].dropna()) -  min(df[colname].dropna()),
                    "StDev": stdev(df[colname].dropna()),
                    "Variance": variance(df[colname].dropna()),
                    "Skewness": round(skew(df[colname].dropna()), 2),
                    "Kurtosis": round(kurtosis(df[colname].dropna()), 2)
                   }

            # Append to desc_stats
            desc_stats = desc_stats.append(temp, ignore_index = True)

            # Print out progress
            sys.stdout.write("\rProgress: {:2.2%}".format( (i + 1) / n) )
            sys.stdout.flush()


    return desc_stats.sort_values(by = "% missing", ascending = False)

################################################################################

def write_csv(file, outdir : str):
    filename = outdir.split("/")[-1]
    if outdir.find("/"):
        import os
        directory = "/".join(outdir.split("/")[ : -1])
        if not os.path.exists(directory):
                os.makedirs(directory)
    file.to_csv(outdir, index = False)

################################################################################

def reduce_mem_usage(df, verbose=True):
    """
    type df: pandas.core.frame.DataFrame
    type r: pandas.core.frame.DataFrame
    """
    # This function takes df as input and return a df for which numeric dtypes have been optimized
    import numpy as np
    import pandas as pd
    import re

    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]

    # Memory occupied by df before dtypes transformation
    start_mem_usg = df.memory_usage(deep=True).sum() / 1024**2
    print("Memory usage of input dataframe is: {:5.2f} MB".
          format(start_mem_usg))

    for col in df.columns:

        # Extract dtypes as string and remove numbers
        col_type = re.findall("([a-z]+)", str(df[col].dtypes))[0]

        if col_type == "object":
            df[col] = df[col].astype("category")

        # use extract only root type from numerics list
        elif col_type in [re.findall("([a-z]+)", dtype)[0] for dtype in numerics]:
            # Print current column type
#             print("******************************")
#             print("Column: ", col)
#             print("dtype before: ", df[col].dtype)

            # Obtain minimum and maximum values for df[col]
            # Used later to understand which is the most suitable dtype
            has_nan = False if df[col].isnull().any() == False else True
            c_Min, c_Max = df[col].                                        agg(["min", "max"]).                                        tolist()


            # Since col_type already contains type str, no need of further transfomations
            if col_type == "int":

                # If df[col] is only positive, use np.uint classes
                if c_Min >= 0:
                    if c_Max < np.iinfo(np.uint8).max:
                        df[col] = df[col].astype(np.uint8)
                    elif c_Max < np.iinfo(np.uint16).max:
                        df[col] = df[col].astype(np.uint16)
                    elif c_Max < np.iinfo(np.uint32).max:
                        df[col] = df[col].astype(np.uint32)
                    elif c_Max < np.iinfo(np.uint64).max:
                        df[col] = df[col].astype(np.uint64)

                else:

                  # Check the minimum int size that can store the column
                    if c_Min > np.iinfo(np.int8).min and c_Max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_Min > np.iinfo(np.int16).min and c_Max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_Min > np.iinfo(np.int32).min and c_Max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_Min > np.iinfo(np.int64).min and c_Max < np.iinfo(np.int64).max:  #not clear why not use else:
                        df[col] = df[col].astype(np.int64)

            elif col_type == "float":

                # Check the minimum float size that can store the column
                if c_Min > np.finfo(np.float16).min and c_Max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_Min > np.finfo(np.float32).min and c_Max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)


        # Print new column dtype
#         print("dtype after: ", df[col].dtype)
#         print("******************************")

    # Compute memory of the resulting df
    end_mem_usg = df.memory_usage(deep = True).sum() / 1024**2

    if verbose:
        print("Mem. usage decreased to {:5.2f} MB ({:.1f}% reduction)".
             format(end_mem_usg, 100 * (start_mem_usg - end_mem_usg) / start_mem_usg))

    write_csv(df.dtypes, "dtypes/dtypes.csv")

################################################################################


# I want to see all rows and all columns.

# In[ ]:


pd.options.display.max_rows = 350
print("Display max rows set to 100")
pd.set_option('display.max_columns', None)
print("Display all columns")


# ## II) Import the training dataset

# In[ ]:


train_df = pd.read_csv('../input/train.csv', index_col = 0)


# In[ ]:


train_df.head(50)


# ## III) First steps toward EDA

# ### 0) Summary Statistics

# I decided to extract summary statistics describing each variable.

# In[ ]:


try:
    num_features = pd.read_csv("desc_stats/numerical.csv")
except:
    num_features = desc_stats(train_df, "number")
    write_csv(num_features, "desc_stats/numerical.csv")


# In[ ]:


num_features


# ### 1) Target variable distribution

# In[ ]:


temp =  (train_df["target"].                    value_counts(normalize = True) * 100).                    to_frame().                    reset_index().                    sort_values("index", ascending = True)

trace0 = go.Bar(
        x = ["HasDetections(0)", "HasDetections(1)"],
        y = temp["target"],
        width = [0.55, 0.55],
        marker = dict(
                        color=['rgb(191, 190, 190)', 'rgb(195, 81, 78)']
                    )
        )

data = [trace0]

layout = go.Layout(
    width=500,
    height=500,
    title = dict(
                    text = "target",
                    x = 0.07,
                    y = 0.90,
                    font = dict(
                                size = 25
                                )
                ),

    xaxis = dict(
                    showline = False,
                    showgrid = False,
                    zeroline = True,
                    tickfont = dict(
                                    size = 20
                                    )
                ),
    yaxis=dict(
                    range = [0, 100],
                    showline = True,
                    tickmode = "array",
                    tickvals = [val for val in range(0,101,20)],
                    ticktext = [str(text) + "%" for text in range(0,101,20)],
                    color = "grey",
                    tickfont = dict(
                                    size = 15
                                    ),
                    zeroline = False
                )
    )


annotations = []
for i in range(0, temp.shape[0]):
    annotations.append(dict(x=temp["index"][i], y=temp.target[i], text=str(round(temp.target[i], 2)) + "%",
                            font=dict(family='Arial', size=25,
                                      color='white'),
                            showarrow=False,
                            yshift = -15
                                       ))
    layout['annotations'] = annotations

fig = go.Figure(data=data, layout=layout)
iplot(fig, filename='target')


# ### 2) Histogram subplots
# y-axis represents the relative frequency of the bin.

# In[ ]:


i = 0
n_plots = 25
while i < (len(train_df.columns[1 : ])):
    hist_sub_plots(start = i + 0, stop = i + n_plots, overlayed = False)
    i += n_plots


# ### 3) Overlayed Histogram subplots
# <br>
# In the next visualization, for each features, I am comparing the distribution of target 0 (gray histogram) to the distribution of target 1 (red line)<br>
# y-axis represents the relative frequency of the bin.

# In[ ]:


i = 0
n_plots = 25
while i < (len(train_df.columns[1 : ])):
    hist_sub_plots(start = i + 0, stop = i + n_plots, overlayed = True)
    i += n_plots


# ### IV) Next steps
# <br>
# The next steps I want to follow are:
# 1. plot correlation matrix (Pearson & Spearman)
# 2. measure association of each variable to target variable
# 3. try to cluster features based on Summary Statistics
# 4. compute Summary Statistics by splitting each feature distribution depending on target value
# 
# 5. calculate results also on test set and understand how similar/dissimilar are the two datasets

# In[ ]:




