#!/usr/bin/env python
# coding: utf-8

# **Kernel SHAP values with Plotly interactive visualization for the UCI Heart Disease data set**

# # Content
# 1. Calculation of SHAP values for a Tensorflow model with:
#     - KernelExplainer from the SHAP package 
#     - Toy implementation of the Shapley Value algorithm
# 2. Interactive visualization with Plotly library 
# 
# **Reference:** For information on interpreting models and Plotly visualization, you can refer to the documentation of the [SHAP package](https://github.com/slundberg/shap) and [Plotly](https://plot.ly/python/) library, the [Interpretable Machine Learning](https://christophm.github.io/interpretable-ml-book/) website or check out kernels: [Intermediate visualization tutorial using Plotly ](https://www.kaggle.com/thebrownviking20/intermediate-visualization-tutorial-using-plotly) and 
# [What Causes Heart Disease? Explaining the Model](https://www.kaggle.com/tentotheminus9/what-causes-heart-disease-explaining-the-model).

# In[ ]:


import tensorflow as tf
import numpy as np
import pandas as pd
import math
from sklearn.model_selection import train_test_split
import itertools
from itertools import chain, combinations
import scipy.special

import plotly
from plotly.graph_objs import Scatter, Layout, Heatmap, Bar, Scene, XAxis, YAxis, ZAxis
import plotly.figure_factory as ff
import plotly.graph_objs as go
from plotly import tools
   
import shap    

import warnings  
warnings.filterwarnings('ignore')

np.random.seed(10)

plotly.offline.init_notebook_mode(connected=True)


# # 1. Data

# In[ ]:


dt = pd.read_csv('../input/heart.csv')
dt.head()


# # 2. Tensorflow model
# # 2.1 Build model

# In[ ]:


class ModelUCI1:
    def __init__(self, data):
        self.data = data
        self.X_oryg = self.data[[x for x in dt.columns if x != 'target']].values
        #normalized input, no-one-hot version
        self.X = np.apply_along_axis(lambda x: (x-x.min())/(x.max()-x.min()), 0, self.X_oryg)
        self.y = self.data.target.values
        self.X_train, self.X_test, self.y_train, self.y_test, self.X_train_oryg, self.X_test_oryg             = train_test_split(self.X, self.y, self.X_oryg, test_size=0.2, random_state=42)
        self.build_model()
    
    def __del__(self):
        try:
            self.sess.close()
        except:
            pass

    def build_model(self):
        def dense(input_layer, units, name):
            return tf.layers.dense(input_layer, units=units, activation='relu', name=name, 
                                   kernel_initializer=tf.initializers.random_normal(), 
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1))
    
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(10)
            
            self.x_input = tf.placeholder(tf.float32, shape=[None, self.X_train.shape[1]], name='x_input')
            self.dense_1 = dense(self.x_input, 64, 'dense_1')
            self.dense_2 = dense(self.dense_1, 32, 'dense_2')
            self.dense_3 = dense(self.dense_2, 16, 'dense_3')
            self.output = tf.layers.dense(self.dense_3, units=2, activation=None, name='output') 
            
            self.y_target = tf.placeholder(tf.int64, shape=[None], name='target') 
            self.learning_rate = 0.01            
            self.loss_fn = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.output, labels=self.y_target))
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_fn) 
            
    def train(self):
        n_samples = self.X_train.shape[0]
        n_epochs = 200
        batch_size = 16
        n_batches = math.ceil(n_samples/batch_size)

        with self.graph.as_default():
            dataset = tf.data.Dataset.from_tensor_slices({'X':self.X_train,'y': self.y_train})
            dataset = dataset.shuffle(300).batch(batch_size).repeat(n_epochs)
            iter = dataset.make_one_shot_iterator()
            next_elem = iter.get_next() 
            init_op = tf.global_variables_initializer()
            self.sess = tf.Session()
            self.sess.run(init_op)
            for epoch in range(n_epochs):
                for i in range(n_batches):
                    batch = self.sess.run(next_elem)
                    feed_dict = {self.x_input: batch['X'], self.y_target: batch['y']}
                    loss, _ = self.sess.run([self.loss_fn, self.train_op], feed_dict)

    def predict(self, X_test):
        with model.graph.as_default():
            predictions = self.sess.run(tf.argmax(tf.nn.softmax(self.output), axis=1), {self.x_input: X_test})
            return predictions
        
    def acc(self, y_pred, y_true):
        return len([x for x in abs(y_true - y_pred) if x == 0])/len(y_pred)


# # 2.2 Train and predict 

# In[ ]:


model = ModelUCI1(dt)
model.train()


# In[ ]:


X_test_preds = model.predict(model.X_test)
print('test accuracy: ', model.acc(X_test_preds, model.y_test))


# # 3. SHAP values
# # 3.1 KernelExplainer from the SHAP package
# Model's prediction is being treated like a black-box function.

# In[ ]:


e_kernel = shap.KernelExplainer(model.predict, model.X_train, link='identity')
shap_values_kernel = e_kernel.shap_values(model.X_test)


# In[ ]:


#Expected value is based on the class distribution in the background dataset
print('Kernel SHAP expected value:', e_kernel.expected_value)
print('Class distribution (positive to all ratio) in the background set (X_train):', 
      len([x for x in model.y_train if x==1])/len(model.y_train))
print('Kernel SHAP shape:', shap_values_kernel.shape, ': No of features:', model.X_test.shape[1],
      ' for each test sample:', model.X_test.shape[0])


# For each sample, the model's expected value and shap values sum up to the model's prediction:
# 

# In[ ]:


round(max(abs(np.sum(shap_values_kernel, axis=1) + e_kernel.expected_value - X_test_preds)),6)


# # 3.2 Toy implementation of SHAP values
# Kernel SHAP is model-agnostic and only requires a function returning the score for a given sample.
# This is a naive straightforward implementation of the Shapley Value algorithm for one-actor coalitions.
# 
# __[Wiki Shapley value](https://en.wikipedia.org/wiki/Shapley_value)__
# 
# <center>
# $\varphi _{i}(v)={\frac {1}{N}}\sum _{S\subseteq N\setminus \{i\}}{\binom {N-1}{|S|}}^{-1}(v(S\cup \{i\})-v(S))$
# </center>

# In[ ]:


class SimpleExplainer:
    '''   
    Feature values are treated as players forming coalitions and a reference value 
    is used in place of a missing player. Model's prediction is used as score function. 
    
    https://en.wikipedia.org/wiki/Shapley_value
    '''
    def __init__(self, reference_set=model.X_train, model_fun=model.predict):        
        self.reference_set = reference_set
        self.N = reference_set.shape[-1]
        self.model_fun = model_fun
        self.all_subsets = [self.tpl2indx(xx, self.N-1) for xx in self.powerset(range(self.N))]
        #reference values (here: mean value for each column)        
        self.reference = np.apply_along_axis(lambda x: x.mean(), 0, self.reference_set)
        #expected value: model output for the empty (reference) subset
        self.expected_value = self.model_fun([self.reference])
        self.x_samples = None
        self.shap_values = np.empty
        
    def powerset(self, range_s):
        return chain.from_iterable(combinations(range_s, r) for r in range(len(range_s)+1))   

    def tpl2indx(self, tpl, max_len):
        return tuple(1 if i in tpl else 0 for i in range(max_len+1)) 

    def subset2args(self, x, reference, subset):     
        return [ref if pres==0 else xx for xx, ref, pres in 
                [(xx,ref,pres) for xx,ref,pres in zip(x, reference, subset)]]            

    def calculateSHAP(self, x_samples):
        self.shap_values = np.asarray([self.calculateSHAP_one(x_samples[i:(i+1),:]) 
                                       for i in range(x_samples.shape[0])])
    
    def calculateSHAP_one(self, x_sample):
        #calculate model output for all subsets
        def calc_model_output4subsets(x_sample):
            model_input_all_subsets = [self.subset2args(x_sample[0], self.reference, s) for s in self.all_subsets]
            return dict(zip(self.all_subsets, self.model_fun(model_input_all_subsets)))
    
        model_output4subsets = calc_model_output4subsets(x_sample)
        N = x_sample.shape[1]
        binomS = dict(zip(range(N+1), [int(scipy.special.binom(N-1,s)) for s in range(N+1)]))
        s = range(N)
        sw = [0]*N 
        subsetsI_g = list(filter(lambda x: sum(xx == 0 for xx in x)>=1, list(model_output4subsets.keys())))

        for s in subsetsI_g:
            not_in_set = list([(i,x) for i,x in enumerate(s) if x == 0])
            group2update = []
            indx_not_in_set = tuple(zip(*not_in_set))[0]
            group2update = [x[0] for x in itertools.combinations(indx_not_in_set, 1)]
            model_output_s = model_output4subsets[s] 
            for g in group2update:
                s_ij = tuple(1 if i in {g} else x for i,x in enumerate(s))
                model_output_sij = model_output4subsets[s_ij]
                sLen = len([x for x in s if x == 1])
                sw[g] =  sw[g] + (model_output_sij-model_output_s)/binomS[sLen]  
        return [x/N for x in sw]      


# # 3.2.1 SHAP values for the UCI heart disease data set

# In[ ]:


simpleExplainer = SimpleExplainer()
simpleExplainer.calculateSHAP(model.X_test)
shap_values_simple = simpleExplainer.shap_values
print(shap_values_simple.shape)


# For each sample, the model's expected value and shap values add up to the model's prediction:

# In[ ]:


round(max(abs(np.sum(shap_values_simple, axis=1) + simpleExplainer.expected_value - X_test_preds)),6)


# # 3.2.2 Simple example: SHAP for the Glove game
# The Wikipedia article on Shapley value uses an example of the __[Glove game](https://en.wikipedia.org/wiki/Shapley_value#Glove_game)__.
# Simple implementation of SHAP can be used to calculate the contribution of individual players to the result of the game.

# In[ ]:


#Glove game score function:
def score4Glove_game_fun(x):
    return [1*(any(xx==1 for xx in row[0:2]) and row[2]==1) for row in x]


# In[ ]:


simpleExplainer_glove = SimpleExplainer(reference_set=np.asarray([[0,0,0]]), model_fun=score4Glove_game_fun)        
simpleExplainer_glove.calculateSHAP(np.asarray([[1,1,1]]))
{i: simpleExplainer_glove.shap_values[0][i] for i in range(3)}


# In the Wiki example:
# 
# $\varphi _{1}(v)=\!\left({\frac {1}{6}}\right)(1)={\frac {1}{6}}$
# 
# $ \varphi _{2}(v)=\varphi _{1}(v)={\frac {1}{6}}$
# 
# $\varphi _{3}(v)={\frac {4}{6}}={\frac {2}{3}}$

# # 4. Interactive plots
# # 4.1 Scatter plot

# In[ ]:


def plot1(data_shap, data_features, preds, columns):
    def x_title(feature_indx):
        return 'SHAP value for feature: ' + columns[feature_indx]

    def y_title(feature_indx):
        return 'Feature value for: ' + columns[feature_indx]

    def plot_title(feature_indx):
        return 'SHAP value vs feature value for: ' + columns[feature_indx]

    def get_plot_data(feature_indx):
        f = data_shap.swapaxes(0,1)[feature_indx]
        v = data_features.swapaxes(0,1)[feature_indx]
        return {'x1': [x for x,y in zip(f, preds) if y == 1],
                'y1': [x for x,y in zip(v, preds) if y == 1],
                'x0': [x for x,y in zip(f, preds) if y == 0],
                'y0': [x for x,y in zip(v, preds) if y == 0]}    
    
    feature_indx_default = 0
    d0 = get_plot_data(feature_indx_default)
    traces = [go.Scatter(x = d0['x0'], y = d0['y0'], mode='markers', name='negative'),
              go.Scatter(x = d0['x1'], y = d0['y1'], mode='markers', name='positive')]
    
    options = []
    feature_cnt = len(columns)
    for feature_indx in range(feature_cnt):
        feature_name = model.data.columns[feature_indx]
        options.append(
            dict(label = feature_name,
                method = 'update',
                args = [{'x': \
                         [get_plot_data(feature_indx)['x0'],
                          get_plot_data(feature_indx)['x1']],
                         'y': \
                         [get_plot_data(feature_indx)['y0'],
                          get_plot_data(feature_indx)['y1']]},
                        {'title': plot_title(feature_indx), 'xaxis': {'title': x_title(feature_indx)}, 
                        'yaxis': {'title':y_title(feature_indx)}}]))
    return {'traces': traces, 'options': options, 'plot_title': plot_title(feature_indx_default),
            'xtitle': x_title(feature_indx_default), 'ytitle': y_title(feature_indx_default)}   

plot = plot1(shap_values_kernel, model.X_test_oryg, X_test_preds, model.data.columns[:-1])

updatemenus = list([
    dict(active=0,
         buttons=plot['options'],
         y=1.4
    )
])

layout = Layout(title = plot['plot_title'], xaxis = {'title': plot['xtitle']}, yaxis = {'title': plot['ytitle']},
               updatemenus=updatemenus)

plotly.offline.iplot({'data': plot['traces'], 'layout': layout})


# # 4.2 Summary plot from the SHAP library
# Summary plot from the shap library for comparison

# In[ ]:


shap.summary_plot(shap_values_kernel, pd.DataFrame(model.X_test_oryg, 
    columns=[x for x in model.data.columns if x != 'target']))


# # 4.3 Scatter3d SHAP

# In[ ]:


f1 = 0
f2 = 1
f3 = 2
data4f = shap_values_kernel.swapaxes(0,1)
feature_cnt = len(model.data.columns[:-1])

def get_data4feature(feature_indx):
    return data4f[feature_indx]

def get_data4axis(axis):
    options = []
    for feature_indx in range(feature_cnt):
        feature_name = model.data.columns[feature_indx]
        options.append(dict(args = [{axis: [get_data4feature(feature_indx)]}],
                label = feature_name,
                method = 'update' ))
    return list(options)    

updatemenus=list([
    dict( 
        active = f1,
        buttons = get_data4axis('x'),
        x = 0.1,
        y = 1.2
    ),
    dict(
        active = f2,
        buttons = get_data4axis('y'),
        x = 0.3,
        y = 1.2
    ),   
    dict(
        active = f3,
        buttons = get_data4axis('z'),
        x = 0.5,
        y = 1.2
    )    
])

cols = ['green' if x == 0 else 'orange' for x in X_test_preds]
data = [go.Scatter3d(
    x = get_data4feature(f1),
    y = get_data4feature(f2),
    z = get_data4feature(f3),
    mode = 'markers',
    marker = dict(color = cols))]

annotations = list([
    dict(text='X-axis', x=-0.05, y=1.27, yref='paper', align='left', showarrow=False ),
    dict(text='Y-axis', x=0.15, y=1.27, yref='paper', showarrow=False ),
    dict(text='Z-axis', x=0.38, y=1.27, yref='paper', showarrow=False),
    dict(text='Negative', x=1, y=1, bgcolor='green', align='left', showarrow=False),
    dict(text='Positive', x=1, y=0.9, bgcolor='orange', align='left', showarrow=False)
])

layout = Layout(updatemenus = updatemenus, annotations = annotations, 
                title = {'text': 'SHAP values for features', 'y': 0},
                scene=Scene(aspectratio=dict(x = 1, y = 1, z = 1)))

plotly.offline.iplot({'data': data, 'layout': layout})


# # 4.4 Violin plot
# # 4.4.1 Features

# In[ ]:


def plot_violin(data_shap, data_features, preds, columns):
    traces = []
    data_shap_f = data_shap.swapaxes(0,1)
    data_features_f = data_features.swapaxes(0,1)
    for feature_indx in range(len(columns)):
        f = data_shap_f[feature_indx]
        v = data_features_f[feature_indx]
        feature_name = columns[feature_indx]
        traces.append({'meanline': {'visible': True},
                'name': feature_name, 'type': 'violin', 
                 'x': pd.Series([feature_name]*len(v)), 
                 'y': pd.Series(f)})
    return traces
     
plotly.offline.iplot({'data': plot_violin(shap_values_kernel, model.X_test_oryg, X_test_preds, model.data.columns[:-1]),
                    'layout': Layout(title = 'SHAP values for features')})


# # 4.4.2 Features and output - split violin

# In[ ]:


data_shap_0 = [x for x,y in zip(shap_values_kernel, X_test_preds) if y == 0]
data_shap_1 = [x for x,y in zip(shap_values_kernel, X_test_preds) if y == 1]

y0 = [xx for x in data_shap_0 for xx in x]    
y1 = [xx for x in data_shap_1 for xx in x]
x0 = list(model.data.columns[:-1])*len([x for x,y in zip(shap_values_kernel, X_test_preds) if y == 0])
x1 = list(model.data.columns[:-1])*len([x for x,y in zip(shap_values_kernel, X_test_preds) if y == 1])

def data_violin(x, y, caption, side, color):
    return  {"type": 'violin', "x": x, "y": y, "legendgroup": caption, "scalegroup": caption,
            "name": caption, "side": side, "meanline": {"visible": True},
            "line": {"color": color}}  

fig = {
    "data": [data_violin(x0, y0, 'Negative', 'negative', 'green'), data_violin(x1, y1, 'Positive', 'positive', 'blue')],
    "layout" : {
        "yaxis": {"zeroline": True,},
        "violingap": 0,
        "violinmode": "overlay",
        "title": "Split violin plot for positive and negative outputs"
    }
}

plotly.offline.iplot(fig)


# # 4.5 Line scatter plot 
# # 4.5.1 All features
# The plot shows SHAP values against ordered values of the respective feature.

# In[ ]:


def plot_rank_all():
    shap4features = shap_values_kernel.swapaxes(0,1)    
    values4features = model.X_test_oryg.swapaxes(0,1)
    traces = []
    for feature_indx in range(len(model.data.columns[:-1])):
        feature_name = model.data.columns[feature_indx]
        f = shap4features[feature_indx]
        v = values4features[feature_indx]
        #rank vector of the value list
        v_rank = [sorted(v).index(x) for x in v]
        f_sorted_by_v = [y for x,y in sorted(zip(v_rank, f))]
        traces.append(go.Scatter(y = f_sorted_by_v,
            mode='lines+markers', name = feature_name ) )       
    return traces

layout = Layout(title = 'SHAP value for feature rank', 
                xaxis = {'title': 'Feature rank'}, 
                yaxis = {'title': 'SHAP value'})

plotly.offline.iplot({'data': plot_rank_all(), 'layout': layout})


# # 4.5.2 By feature
# The plot shows SHAP values against ordered values of the respective feature.

# In[ ]:


shap4features = shap_values_kernel.swapaxes(0,1)    
values4features = model.X_test_oryg.swapaxes(0,1)

def plot_lines_get_data(feature_indx):
    traces = []
    f = shap4features[feature_indx]
    v = values4features[feature_indx]
    feature_name = model.data.columns[feature_indx]
    v_rank = [sorted(v).index(x) for x in v]
    f_sorted_by_v = [y for x,y in sorted(zip(v_rank, f))]
    return {'y': f_sorted_by_v, 
            'title': feature_name + ": SHAP value vs rank", 
            'xtitle': 'Feature rank for ' + feature_name, 
            'ytitle': 'SHAP value for ' + feature_name}

def plot_lines_get_menus():
    options = []
    for feature_indx in range(len(model.data.columns[:-1])):
        feature_name = model.data.columns[feature_indx]
        d = plot_lines_get_data(feature_indx)
        options.append(
                  dict(label = feature_name,
                      method = 'update',
                      args = [{'y': [d['y']]},
                              {'title': d['title'], 'xaxis': {'title': d['xtitle']}, 
                              'yaxis': {'title': d['ytitle']}}]
                      ))
    return options

feature_indx_default = 0
plot_data = plot_lines_get_data(feature_indx_default)
data = [go.Scatter(y = plot_data['y'], mode='lines+markers')]

updatemenus=list([dict(active = feature_indx_default, y = 1.2, buttons = plot_lines_get_menus())])   

layout = Layout(updatemenus = updatemenus, title = plot_data['title'],
               xaxis = {'title': plot_data['xtitle']}, yaxis = {'title': plot_data['ytitle']})

plotly.offline.iplot({'data': data, 'layout': layout})

