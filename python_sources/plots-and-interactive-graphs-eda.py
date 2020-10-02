#!/usr/bin/env python
# coding: utf-8

# ## Contents
# * [Loading and Cleaning Data](#Loading-and-Cleaning-Data)
# * [Plots](#Plots)
# * [Interactive Graphs](#Interactive-Graphs)

# In[ ]:


import gc
import numpy as np
import pandas as pd
import copy
import os
import json
import igraph
import plotly.plotly as py
import plotly.graph_objs as go
import networkx as nx
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
get_ipython().run_line_magic('matplotlib', 'inline')
np.warnings.filterwarnings('ignore')

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)

nrows = None


# ### Loading and Cleaning Data

# In[ ]:


# just a list of trainable label codes
classes_trainable = pd.read_csv('../input/classes-trainable.csv', dtype=str)
classes_trainable = set(classes_trainable.label_code)
print(f'{len(classes_trainable)} trainable labels')


# In[ ]:


class_descriptions = pd.read_csv('../input/class-descriptions.csv', dtype=str)
label_code_to_description = dict(zip(class_descriptions.label_code, class_descriptions.description))
print(f'{len(label_code_to_description)} labels with descriptions')
class_descriptions.head()


# A description does not always uniquely identify a label_code.

# In[ ]:


class_description_counts = class_descriptions.groupby('description').count().rename(columns={'label_code': 'count'})
repeated_descriptions = class_description_counts[(class_description_counts > 1).values].sort_values('count', ascending=False)
print(f'{len(repeated_descriptions)} descriptions have more than one associated label code')
repeated_descriptions.head(15)


# ### Train Human labels
# These are the image-level labels (no bounding box) annotated by humans with Confidence=1. Most of these labels are from annotators hired by Google ('verification') and the rest are from crowd-sourcing ('crowdsource-verification'). Read more details at [Overview of Open Images V4](https://storage.googleapis.com/openimages/web/factsfigures.html). One important note from that page is that false positives (class incorrectly labelled) are all but eliminated, but false negatives (class not labelled when it actually is in the image) are present.

# In[ ]:


train_human_labels = pd.read_csv('../input/train_human_labels.csv', dtype={'ImageID': str, 'Source': str, 'LabelName': str, 'Confidence': np.float64}, nrows=nrows)
print(f'{len(train_human_labels)} human-labeled labels on training images')
print(f'Sources: {set(train_human_labels.Source)}')
print(f'Number unique labels: {len(train_human_labels.LabelName.unique())}')
train_human_labels['type'] = 'human'
train_human_labels.head()


# ### Train Machine Labels
# These are the image-level labels annotated by a model with varying confidence levels. More details at [Overview of Open Images V4](https://storage.googleapis.com/openimages/web/factsfigures.html).
# 

# In[ ]:


train_machine_labels = pd.read_csv('../input/train_machine_labels.csv', dtype={'ImageID': str, 'Source': 'category', 'LabelName': 'category', 'Confidence': np.float64}, nrows=nrows)
print(f'{len(train_machine_labels)} machine-labeled labels on training images')
print(f'Sources: {set(train_machine_labels.Source)}')
print(f'Number unique labels: {len(train_machine_labels.LabelName.unique())}')
train_machine_labels['type'] = 'machine'
train_machine_labels.head()


# ### Bounding Boxes
# 
# These are the labels with bounding boxes with Confidence=1. There are only 599 label codes represented in the bounding boxes. Most of these have been added by professional annotators. Again, more details at [Overview of Open Images V4](https://storage.googleapis.com/openimages/web/factsfigures.html). I believe the 'xclick' boxes are "manually drawn by  professional annotators at Google using the efficient extreme clicking interface" (from provided link) and the 'activemil' is done somewhat automatically with human verification (from provided link).
# 
#  Attribute definitions from the [Open Images Website](https://storage.googleapis.com/openimages/web/download.html#attributes):
# > * IsOccluded: Indicates that the object is occluded by another object in the image.
# > * IsTruncated: Indicates that the object extends beyond the boundary of the image.
# > * IsGroupOf: Indicates that the box spans a group of objects (e.g., a bed of flowers or a crowd of people). We asked annotators to use this tag for cases with more than 5 instances which are heavily occluding each other and are physically touching.
# > * IsDepiction: Indicates that the object is a depiction (e.g., a cartoon or drawing of the object, not a real physical instance).
# > * IsInside: Indicates a picture taken from the inside of the object (e.g., a car interior or inside of a building).
# >
# > For each of them, value 1 indicates present, 0 not present, and -1 unknown.

# In[ ]:


train_bounding_boxes = pd.read_csv('../input/train_bounding_boxes.csv', 
                                   dtype={
                                       'ImageID': str,
                                       'Source': 'category',
                                       'LabelName': 'category',
                                       'Confidence': np.int8,
                                       'XMin': np.float32,
                                       'XMax': np.float32,
                                       'YMin': np.float32,
                                       'YMax': np.float32,
                                       'IsOccluded': np.int8,
                                       'IsTruncated': np.int8,
                                       'IsGroupOf': np.int8,
                                       'IsDepiction': np.int8,
                                       'IsInside': np.int8,
                                   }, 
                                   nrows=nrows)

# replace unknown placeholder -1 with nan
attr_cols = ['IsOccluded', 'IsTruncated', 'IsGroupOf', 'IsDepiction', 'IsInside']
train_bounding_boxes[attr_cols] = train_bounding_boxes[attr_cols].replace(to_replace=-1, value=np.nan)

print(f'{len(train_bounding_boxes)} bounding boxes on {len(set(train_bounding_boxes.ImageID))} training images')
print(f'Sources: {set(train_bounding_boxes.Source)}')
print(f'Number unique labels: {len(train_bounding_boxes.LabelName.unique())}')
train_bounding_boxes['type'] = 'bbox'
train_bounding_boxes.head()


# In[ ]:


train_labels = pd.concat([train_human_labels, train_machine_labels, train_bounding_boxes], axis='rows', ignore_index=True, sort=False).reset_index(drop=True)
del train_human_labels
del train_machine_labels
del train_bounding_boxes

# get human readable label description
train_labels['LabelDescription'] = train_labels['LabelName'].apply(lambda x: label_code_to_description.get(x, ''))

# make a new col that has both label and description for easy identification while maintaining uniqueness
train_labels['LabelAndDescription'] = train_labels['LabelName'].str.cat(train_labels['LabelDescription'], sep=' - ')

dtypes = {
    'ImageID': str,
    'Source': str,
    'LabelName': str,
    'LabelDescription': str,
    'LabelAndDescription': str,
    'Confidence': np.float32,
    'IsOccluded': np.float32,
    'IsTruncated': np.float32,
    'IsGroupOf': np.float32,
    'IsDepiction': np.float32,
    'IsInside': np.float32,
}
train_labels = train_labels.astype(dtypes)

print(f'{len(train_labels)} total labels on {len(train_labels.ImageID.unique())} training images')


# In[ ]:


train_labels = train_labels.drop(['XMin', 'XMax', 'YMin', 'YMax'], axis='columns')


# In[ ]:


gc.collect()
train_labels.head(1)


# In[ ]:


tuning_labels = pd.read_csv('../input/tuning_labels.csv', header=None, names=['ImageID', 'LabelNames'], dtype=str)

# transforming list of label names into a row for each label name
tuning_labels.index = tuning_labels['ImageID']
labels = tuning_labels.LabelNames.str.split(expand=True).stack()
labels.index = labels.index.droplevel(-1)
labels.name = "LabelName"
tuning_labels = pd.concat([tuning_labels, labels], axis=1, join='inner')

tuning_labels = tuning_labels.drop('LabelNames', axis='columns')
tuning_labels = tuning_labels.reset_index(drop=True)

print(f'{len(tuning_labels)} tuning labels on {len(set(tuning_labels.ImageID))} stage 1 test images')

# get human readable label description
tuning_labels['LabelDescription'] = tuning_labels['LabelName'].apply(lambda x: label_code_to_description.get(x, ''))

# make a new col that has both label and description for easy identification while maintaining uniqueness
tuning_labels['LabelAndDescription'] = tuning_labels['LabelName'].str.cat(tuning_labels['LabelDescription'], sep=' - ')

tuning_labels.head()


# In[ ]:


labels_without_description = set(train_labels[train_labels.LabelDescription == '']['LabelName'])
print(f'Labels in train dataset with no description: {labels_without_description}')
print(f'Trainable classes without description: {labels_without_description.intersection(classes_trainable)}')


# In[ ]:


human_train_label_set = set(train_labels[train_labels['type'] == 'human'].LabelName)
machine_train_label_set = set(train_labels[train_labels['type'] == 'machine'].LabelName)
bbox_train_label_set = set(train_labels[train_labels['type'] == 'machine'].LabelName)
stage_1_tuning_label_set = set(tuning_labels.LabelName)

print(f'Human train labels: {len(human_train_label_set.intersection(classes_trainable))}/{len(human_train_label_set)} are "trainable"')
print(f'Machine train labels: {len(machine_train_label_set.intersection(classes_trainable))}/{len(machine_train_label_set)} are "trainable"')
print(f'Bbox train labels: {len(bbox_train_label_set.intersection(classes_trainable))}/{len(bbox_train_label_set)} are "trainable"')
print(f'Stage 1 tuning labels: {len(stage_1_tuning_label_set.intersection(classes_trainable))}/{len(stage_1_tuning_label_set)} are "trainable"')


# ### Plots

# In[ ]:


def plot_label_per_image_dist(df, title, bins=[300, 60], xmax=100):

    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    mean = df['LabelName_count'].mean()
    median = df['LabelName_count'].median()
    p = sns.distplot(df['LabelName_count'], bins=bins[0], kde=False, ax=axes[0])
    _ = p.set_xlabel('Number of labels per ImageID')
    _ = p.set_xlim((0,xmax))
    _ = p.text(1, 1, f'mean: {mean:.2f}\nmedian: {median:.2f}',  horizontalalignment='right', verticalalignment='top', transform=p.axes.transAxes)
    
    mean = df['LabelName_nunique'].mean()
    median = df['LabelName_nunique'].median()
    p = sns.distplot(df['LabelName_nunique'], bins=bins[1], kde=False, ax=axes[1])
    _ = p.set_xlabel('Number of unique labels per ImageID')
    _ = p.set_xlim((0,xmax))
    _ = p.text(1, 1, f'mean: {mean:.2f}\nmedian: {median:.2f}',  horizontalalignment='right', verticalalignment='top', transform=p.axes.transAxes)
    
    fig.suptitle(title, x=.5, y=1.01, fontsize=16)
    fig.tight_layout()


# In[ ]:


aggs = {
    'LabelName': ['count', 'nunique']
}
counts = train_labels.groupby(['ImageID']).agg(aggs)
counts.columns = counts.columns.map('_'.join)

plot_label_per_image_dist(counts, 'Labels and Unique Labels per Image in Entire Train Dataset')


# In[ ]:


aggs = {
    'LabelName': ['count', 'nunique']
}
counts = tuning_labels.groupby(['ImageID']).agg(aggs)
counts.columns = counts.columns.map('_'.join)

plot_label_per_image_dist(counts, 'Labels and Unique Labels per Image in Tuning Dataset', bins=[10, 10], xmax=10)


# In[ ]:


def plot_label_frequency_grid(df, title):
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))

    count_mean = df['ImageID_count'].mean()
    count_median = df['ImageID_count'].median()
    p = sns.barplot(y='LabelAndDescription', x='ImageID_count', data=df.sort_values('ImageID_count', ascending=False)[0:30], alpha=.7, ax=axes[0][0])
    _ = p.set_xlabel('Number of instances of Label in Train Dataset')
    _ = p.set_ylabel('Label And Description')
    _ = p.text(1, 1, f'mean: {count_mean:.2f}\nmedian: {count_median:.2f}',  horizontalalignment='right', verticalalignment='top', transform=p.axes.transAxes)
    
    nunique_mean = df['ImageID_nunique'].mean()
    nunique_median = df['ImageID_nunique'].median()
    p = sns.barplot(y='LabelAndDescription', x='ImageID_nunique', data=df.sort_values('ImageID_nunique', ascending=False)[0:30], alpha=.7, ax=axes[0][1])
    _ = p.set_xlabel('Number of unique images with Label in Train Dataset')
    _ = p.set_ylabel('Label And Description')
    _ = p.text(1, 1, f'mean: {nunique_mean:.2f}\nmedian: {nunique_median:.2f}',  horizontalalignment='right', verticalalignment='top', transform=p.axes.transAxes)

    p = sns.distplot(df['ImageID_count'], bins=20, kde=False, ax=axes[1][0])
    _ = p.set_yscale('log')
    _ = p.set_ylabel('Count')
    _ = p.set_xlabel('Number of instances of Label in Train Dataset')
    _ = p.text(1, 1, f'mean: {count_mean:.2f}\nmedian: {count_median:.2f}',  horizontalalignment='right', verticalalignment='top', transform=p.axes.transAxes)

    p = sns.distplot(df['ImageID_nunique'], bins=20, kde=False, ax=axes[1][1])
    _ = p.set_yscale('log')
    _ = p.set_ylabel('Count')
    _ = p.set_xlabel('Number of unique images with Label in Train Dataset')
    _ = p.text(1, 1, f'mean: {nunique_mean:.2f}\nmedian: {nunique_median:.2f}',  horizontalalignment='right', verticalalignment='top', transform=p.axes.transAxes)
    
    fig.suptitle(title, x=.5, y=1.01, fontsize=16)
    fig.tight_layout()


# In[ ]:


aggs = {
    'ImageID': ['count', 'nunique']
}
counts = train_labels.groupby(['LabelAndDescription']).agg(aggs)
counts.columns = counts.columns.map('_'.join)
counts = counts.reset_index(drop=False)
counts['LabelAndDescription'] = counts['LabelAndDescription'].astype(str)


# In[ ]:


plot_label_frequency_grid(counts, 'Frequencies of Labels in Entire Train Dataset')


# In[ ]:


aggs = {
    'ImageID': ['count', 'nunique']
}
trainable_counts = train_labels[train_labels.LabelName.isin(classes_trainable)].groupby(['LabelAndDescription']).agg(aggs)
trainable_counts.columns = trainable_counts.columns.map('_'.join)
trainable_counts = trainable_counts.reset_index(drop=False)
trainable_counts['LabelAndDescription'] = trainable_counts['LabelAndDescription'].astype(str)


# In[ ]:


plot_label_frequency_grid(trainable_counts, 'Frequencies of Labels in "Trainable" Portion of Dataset')


# In[ ]:


del counts
del trainable_counts
gc.collect()


# In[ ]:


bbox_labels = train_labels[train_labels['type'] == 'bbox']

aggs = {
    'ImageID': ['count', 'nunique'],
    'IsOccluded': ['mean'],
    'IsTruncated': ['mean'],
    'IsGroupOf': ['mean'],
    'IsDepiction': ['mean'],
    'IsInside': ['mean'],
}
bbox_counts = bbox_labels.groupby(['LabelAndDescription']).agg(aggs)
bbox_counts.columns = bbox_counts.columns.map('_'.join)
bbox_counts = bbox_counts.reset_index(drop=False)
bbox_counts['LabelAndDescription'] = bbox_counts['LabelAndDescription'].astype(str)


# In[ ]:


plot_label_frequency_grid(bbox_counts, 'Frequencies of Labels in Bbox Train Dataset')


# In[ ]:


aggs = {
    'ImageID': ['count', 'nunique'],
}
tuning_counts = tuning_labels.groupby(['LabelAndDescription']).agg(aggs)
tuning_counts.columns = tuning_counts.columns.map('_'.join)
tuning_counts = tuning_counts.reset_index(drop=False)
tuning_counts['LabelAndDescription'] = tuning_counts['LabelAndDescription'].astype(str)


# In[ ]:


plot_label_frequency_grid(bbox_counts, 'Frequencies of Labels in Tuning Dataset')


# In[ ]:


def plot_attributes(df, title, attribute):
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    count_mean = df[attribute].mean()
    count_median = df[attribute].median()
    p = sns.barplot(y='LabelAndDescription', x=attribute, data=df.sort_values(attribute, ascending=False)[0:30], alpha=.7, ax=axes[0])
    _ = p.set_xlabel('Number of instances of Label in Train Dataset')
    _ = p.set_ylabel('Label And Description')
    _ = p.text(1, 1, f'mean: {count_mean:.2f}\nmedian: {count_median:.2f}',  horizontalalignment='right', verticalalignment='top', transform=p.axes.transAxes)

    p = sns.distplot(df[attribute], bins=20, kde=False, ax=axes[1])
    _ = p.set_yscale('log')
    _ = p.set_ylabel('Count')
    _ = p.text(1, 1, f'mean: {count_mean:.2f}\nmedian: {count_median:.2f}',  horizontalalignment='right', verticalalignment='top', transform=p.axes.transAxes)
    
    fig.suptitle(title, x=.5, y=1.01, fontsize=16)
    fig.tight_layout()


# In[ ]:


attributes = ['IsOccluded', 'IsTruncated', 'IsGroupOf', 'IsDepiction', 'IsInside']
attributes = [f'{x}_mean' for x in attributes]
for attribute in attributes:
    plot_attributes(df=bbox_counts, title=attribute, attribute=attribute)


# In[ ]:


# downloading the class hierarchy for the bbox classes from https://storage.googleapis.com/openimages/web/factsfigures.html
get_ipython().system('wget https://storage.googleapis.com/openimages/2018_04/bbox_labels_600_hierarchy.json')


# In[ ]:


with open('./bbox_labels_600_hierarchy.json', 'r') as f_in:
    class_hierarchy = json.loads(f_in.read())


# In[ ]:


class_hierarchy


# In[ ]:


bbox_counts['LabelName'] = bbox_counts['LabelAndDescription'].apply(lambda x: x.split(' - ')[0])
bbox_counts.index = bbox_counts['LabelName']
bbox_counts['ImageID_count_log10'] = np.log10(bbox_counts['ImageID_count'])
bbox_counts['ImageID_nunique_log10'] = np.log10(bbox_counts['ImageID_nunique'])


# In[ ]:


bbox_counts.head()


# In[ ]:


def get_label_features(df):
    features = {}
    record_dicts = df.to_dict(orient='records')
    for record_dict in record_dicts:
        features[record_dict.pop('LabelName')] = record_dict
    return features


# In[ ]:


label_features = get_label_features(bbox_counts.drop(['LabelAndDescription'], axis='columns'))


# In[ ]:


# example of one item in the created dict
next(iter(label_features.items()))


# In[ ]:


def make_graph(d, graph, prev_node_added=None, level=1, label_code_to_description=label_code_to_description, label_features=label_features):
    if graph is None:
        graph = nx.DiGraph()
    for key, value in d.items():
        if key == 'LabelName':
            class_desc = label_code_to_description.get(value, None)
            label_feats = label_features.get(value, None)
            if label_feats is None:
                sample_feats_dict = next(iter(label_features.values()))
                label_feats = {k: 0 for k in sample_feats_dict.keys()}
            graph.add_node(value, Description=label_code_to_description.get(value), Level=level, **label_feats)
            if prev_node_added:
                graph.add_edge(prev_node_added, value)
            prev_node_added = value
        elif key == 'Subcategory':# or key == 'Part':
            for subcat in value:
                graph = make_graph(subcat, graph, prev_node_added=prev_node_added, level=level + 1, label_code_to_description=label_code_to_description, label_features=label_features)
    return graph

def add_adjacency_to_nodes(g):
    # adjacent nodes will be a dict of nodes names of all children / outgoing edge nodes
    for node_name, adjacent_nodes in g.adjacency():
        g.node[node_name].update({'num_adjacent_nodes': len(adjacent_nodes)})
    return g

def get_node_coords(g, layout_type):
    ig = igraph.Graph(directed=True)
    ig.add_vertices(list(g.nodes))
    ig.add_edges(list(g.edges))
    if layout_type == 'layout_reingold_tilford':
        ig_layout = ig.layout_reingold_tilford(mode='OUT', root=[0])
    elif layout_type == 'layout_fruchterman_reingold':
        ig_layout = ig.layout_fruchterman_reingold()
    elif layout_type == 'layout_kamada_kawai':
        ig_layout = ig.layout_kamada_kawai()
    else:
        ig_layout = ig.layout_reingold_tilford_circular(mode='OUT', root=[0])
    root_at_top_coords = [[coord[0], -coord[1]] for coord in ig_layout.coords]
    coords = {vertex['name']: coord for vertex, coord in zip(igraph.VertexSeq(ig), root_at_top_coords)}
    return coords


# In[ ]:


def build_node_traces(keys):
    node_traces = {}
    for key in keys:
        trace = go.Scatter(
            x=[],
            y=[],
            text=[],
            mode='markers',
            hoverinfo='text',
            marker=dict(
                showscale=True,
                colorscale='Viridis',
                color=[],
                size=10,
                colorbar=dict(thickness=15, title=key, xanchor='left', titleside='right'),
                line=dict(width=2)
            ),
            visible=False,
        )
        node_traces[key] = trace
    return node_traces


# In[ ]:


def plot_tree(graph, node_coords, title):
    edge_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        line=dict(width=0.5,color='#888'),
        hoverinfo='text',
        mode='lines')

    for edge in graph.edges():
        x0, y0 = node_coords[edge[0]]
        x1, y1 = node_coords[edge[1]]
        edge_trace['x'] += tuple([x0, x1, None])
        edge_trace['y'] += tuple([y0, y1, None])

    sample_node_name = next(iter(g.nodes()))
    sample_node = graph.node[sample_node_name]
    nonlabel_keys = [x for x in sample_node.keys() if x != 'Description']
    
    # construct dict of traces for each attribute
    node_traces = build_node_traces(nonlabel_keys)
    
    # add the position and data of each node to the node traces
    for node in graph.nodes():
        x, y = node_coords[node]
        node_data = graph.node[node]
        trace_txt = f'<br>Label: {node}<br>'
        str_list = [f'{k}: {v}' if not isinstance(v, float) else f'{k}: {v:.3f}' for k,v in node_data.items()]
        trace_txt += '<br>'.join(str_list)
        
        for trace_key, trace in node_traces.items():
            # position
            trace['x'] += tuple([x])
            trace['y'] += tuple([y])
            
            # text and marker
            trace['text']+=tuple([trace_txt])
            trace['marker']['color']+=tuple([node_data[trace_key]])
            
    # construct buttons
    buttons = {}
    for idx, key in enumerate(nonlabel_keys):
        # first visible trace will be the edge_trace which will constantly be visible
        trace_visibilities = [True]
        
        # set all other node traces to not visible besides the current one
        trace_visibilities += [False] * len(node_traces)
        trace_visibilities[idx + 1] = True
        
        buttons[key] = dict(args=[{'visible': trace_visibilities}], label=key, method='update')
    
    # construct menu
    active_button = 1
    updatemenus = list([
        dict(
            active=active_button, 
            buttons=list(list(buttons.values())),
            direction='down',
            showactive=True,
            pad={'r': 10, 't': 10},
            x=0.1,
            xanchor='left',
            y=1.05,
            yanchor='top' 
        ), 
    ])
    node_traces[nonlabel_keys[active_button]]['visible'] = True
    
    layout = go.Layout(
        title=title,
        titlefont=dict(size=14),
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20,l=5,r=5,t=40),
        annotations=[
            dict(
                text="Colormap Features",
                showarrow=False,
                xref="paper", 
                yref="paper",
                x=0.1,
                y=1.08
            )],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        updatemenus=updatemenus,
    )
    
    fig = go.Figure(data=[edge_trace] + [node_traces[x] for x in nonlabel_keys], layout=layout)

    iplot(fig, filename='class_hierarchy_graph')


# In[ ]:


label_code_to_description.update({'/m/0bl9f': 'Entity'})
g = make_graph(class_hierarchy, graph=None, prev_node_added=None, label_code_to_description=label_code_to_description, label_features=label_features)
g = add_adjacency_to_nodes(g)


# ### Interactive Graphs

# In[ ]:


p = get_node_coords(g, layout_type='layout_fruchterman_reingold')
plot_tree(graph=g, node_coords=p, title='Bbox Class Hierarchy with Features from Bbox Train Dataset')


# In[ ]:


# tree-like layout
p = get_node_coords(g, layout_type='layout_reingold_tilford')
plot_tree(graph=g, node_coords=p, title='Bbox Class Hierarchy with Features from Bbox Train Dataset')


# In[ ]:


p = get_node_coords(g, layout_type='layout_kamada_kawai')
plot_tree(graph=g, node_coords=p, title='Bbox Class Hierarchy with Features from Bbox Train Dataset')


# In[ ]:





# In[ ]:


tuning_counts['LabelName'] = tuning_counts['LabelAndDescription'].apply(lambda x: x.split(' - ')[0])
tuning_counts.index = tuning_counts['LabelName']
tuning_counts['ImageID_count_log10'] = np.log10(tuning_counts['ImageID_count'])
tuning_counts['ImageID_nunique_log10'] = np.log10(tuning_counts['ImageID_nunique'])


# In[ ]:


label_features = get_label_features(tuning_counts.drop(['LabelAndDescription'], axis='columns'))


# In[ ]:


label_code_to_description.update({'/m/0bl9f': 'Entity'})
g = make_graph(class_hierarchy, graph=None, prev_node_added=None, label_code_to_description=label_code_to_description, label_features=label_features)
g = add_adjacency_to_nodes(g)


# In[ ]:


p = get_node_coords(g, layout_type='layout_fruchterman_reingold')
plot_tree(graph=g, node_coords=p, title='Bbox Class Hierarchy with Features from Tuning Dataset')


# In[ ]:


p = get_node_coords(g, layout_type='layout_kamada_kawai')
plot_tree(graph=g, node_coords=p, title='Bbox Class Hierarchy with Features from Tuning Dataset')


# In[ ]:





# In[ ]:


stage_1_attribs = pd.read_csv('../input/stage_1_attributions.csv', dtype=str)
stage_1_attribs.head()
print(f"{stage_1_attribs.image_id} image_ids with {} unique sources")


# In[ ]:




