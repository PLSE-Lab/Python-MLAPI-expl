# %% [code]
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def plot_series(series=None, annotate=False, annotate_distance=100, annotate_rotation='horizontal',xticks_rotation='horizontal', custom_xticks=[], xticks_freq=False, figsize=(10,5)):
    
    series = series.sort_values(ascending=False)

    # Plot
    plt.figure(figsize=(figsize))
    bars = plt.bar(x=series.index, height=series.values,width=0.3, )
    plt.suptitle('Frequency')
    # plt.title(f'Feature: {feature_name}')
    plt.xlabel('Values')
    plt.ylabel('Values Counts')

    # x_ticks configuration
    plt.xticks(series.index.values) if xticks_freq else 0
    plt.xticks(custom_xticks) if len(custom_xticks)>0 else 0
    0 if xticks_rotation == 'horizontal' else plt.xticks(rotation=90)

    # Annotate
    annotate_rotation_val = 90 if annotate_rotation == 'vertical' else 0

    if annotate:
        for bar in bars:
            height = bar.get_height()
            plt.annotate(s=f'{height}', xy=(bar.get_x()+bar.get_width()/2, height+annotate_distance), ha='center', rotation=annotate_rotation_val)

def plot_frequency(data=None, feature_name:str=None, annotate=False, annotate_distance=100, annotate_rotation='horizontal',xticks_rotation='horizontal', custom_xticks=[], xticks_freq=False, figsize=(10,5)):

    '''
    Plots value counts for the discrete column in a bar chart
    
    Parameters
    --------------------------------
    feature_name      :  (string)                Name of the feature in the dataframe named 'data'
    annotate          :  (boolean)               Provide annotations to the graph bars/bins
    annotate_distance :  (int/float)             Adjust the distance between the annotation and bin
    xticks_rotation   :  (horizontal/vertical)   Align x axis labels vertically or horizontally
    custom_xticks     :  (array)                 Provide xticks labels seperately passed as an array
    xticks_freq       :  (boolean)               Let xticks labels pick up value from the frequency of feature_name occurances
    figsize           :  (tuple)                 Pass figsize as tuple with (x,y) as in (width,height) in as a multiple of 1000
    --------------------------------
    '''
    
    # Get data
    value_count = data[feature_name].value_counts().sort_values(ascending=False)

    # Plot
    plt.figure(figsize=(figsize))
    bars = plt.bar(x=value_count.index, height=value_count.values,width=0.3, )
    plt.suptitle('Frequency')
    plt.title(f'Feature: {feature_name}')
    plt.xlabel('Values')
    plt.ylabel('Values Counts')

    # x_ticks configuration
    plt.xticks(value_count.index.values) if xticks_freq else 0
    plt.xticks(custom_xticks) if len(custom_xticks)>0 else 0
    0 if xticks_rotation == 'horizontal' else plt.xticks(rotation=90)

    # Annotate
    annotate_rotation_val = 90 if annotate_rotation == 'vertical' else 0

    if annotate:
        for bar in bars:
            height = bar.get_height()
            plt.annotate(s=f'{height}', xy=(bar.get_x()+bar.get_width()/2, height+annotate_distance), ha='center', rotation=annotate_rotation_val)

def plot_frequency_sns(data:pd.DataFrame=None, feature_name:str=None,hue:str=None , annotate=False, annotate_distance=100, annotate_rotation='horizontal',xticks_rotation='horizontal', custom_xticks=[], xticks_freq=False, figsize=(10,5), palette:str=None):

    '''
    Plots value counts for the discrete column in a bar chart
    
    Parameters
    --------------------------------
    feature_name      :  (string)                Name of the feature in the dataframe named 'data'
    annotate          :  (boolean)               Provide annotations to the graph bars/bins
    annotate_distance :  (int/float)             Adjust the distance between the annotation and bin
    xticks_rotation   :  (horizontal/vertical)   Align x axis labels vertically or horizontally
    custom_xticks     :  (array)                 Provide xticks labels seperately passed as an array
    xticks_freq       :  (boolean)               Let xticks labels pick up value from the frequency of feature_name occurances
    figsize           :  (tuple)                 Pass figsize as tuple with (x,y) as in (width,height) in as a multiple of 1000
    --------------------------------
    '''
    
    # Get data
    value_count = data[feature_name].value_counts().sort_values(ascending=False)

    # Plot
#     plt.figure(figsize=(figsize))
    # bars = plt.bar(x=value_count.index, height=value_count.values,width=0.3, )
    splot = sns.countplot(x=feature_name, hue=hue, data=data, palette=palette)
    plt.suptitle('Frequency')
    plt.title(f'Feature: {feature_name}')
    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.legend()

    # x_ticks configuration
    splot.xticks(value_count.index.values) if xticks_freq else 0
    splot.xticks(custom_xticks) if len(custom_xticks)>0 else 0
    0 if xticks_rotation == 'horizontal' else plt.xticks(rotation=90)

    # Annotate
    annotate_rotation_val = 90 if annotate_rotation == 'vertical' else 0


    if annotate:
        for patch in splot.patches:                         # patches->list of rectangles
            height = patch.get_height()
            splot.annotate(s=f'{height}', xy=(patch.get_x()+patch.get_width()/2, height+annotate_distance), ha='center', rotation=annotate_rotation)

def plot_distribution(data=None, feature_name: str=None, bins=10, annotate=False, annotate_distance=100, xticks_rotation='horizontal', figsize=(10,5)):

    '''
    Plots data distribution for the contineous column in a histogram
    
    Parameters
    --------------------------------
    feature_name      :  (string)                Name of the feature in the dataframe named 'data'
    bins              :  (int)                   Number of bins to distribute the data into
    annotate          :  (boolean)               Provide annotations to the graph bars/bins
    annotate_distance :  (int/float)             Adjust the distance between the annotation and bin
    xticks_rotation   :  (horizontal/vertical)   Align x axis labels vertically or horizontally 
    --------------------------------
    '''

    # Plot
    plt.figure(figsize=(figsize))
    n, bins, patches = plt.hist(x=data[feature_name], bins=bins)
    plt.suptitle('Distribution')
    plt.title(f'Feature: {feature_name}')
    plt.xlabel('Value Bins')
    plt.ylabel('')

    # x_ticks rotation
    0 if xticks_rotation=='horizontal' else plt.xticks(rotation=90)

    # Annotate
    if annotate:
        for index, current_bin, bin_height in zip(np.arange(len(bins)),bins,n):
            # print(current_bin)
            bin_avg = (current_bin+bins[index+1])/2
            plt.annotate(s=f'{bin_height}',xy=(bin_avg,bin_height+annotate_distance), rotation=90)

