#!/usr/bin/env python
# coding: utf-8

# quote from module site:
# > Auto-Keras is an open source software library for automated machine learning (AutoML). It is developed by DATA Lab at Texas A&M University and community contributors. The ultimate goal of AutoML is to provide easily accessible deep learning tools to domain experts with limited data science or machine learning background. Auto-Keras provides functions to automatically search for architecture and hyperparameters of deep learning models.
# 
# 
# "domain experts with limited data science or machine learning background" aren't the main public here on kaggle but the finding of automated search on different datasets may give some insights
# 
# [https://autokeras.com/](https://autokeras.com/)

# In[ ]:


import  datetime
date_depart=datetime.datetime.now()
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[ ]:


get_ipython().system('pip install git+https://github.com/onnx/onnx-tensorflow.git')
get_ipython().system('pip install keras2onnx ')

get_ipython().system('apt install -y graphviz ')

get_ipython().system('pip install -q -U --user imageio')
get_ipython().system('pip install -q  keras torch torchvision graphviz autokeras  onnxmltools')
get_ipython().system('pip install -q    onnxmltools onnx-tf')
#!pip install --user autokeras 
get_ipython().system('pip install git+https://github.com/jhfjhfj1/autokeras.git')
get_ipython().system('pip install  -U --user tensorflow-gpu')
import keras
import autokeras as ak
import onnxmltools


# 

# In[ ]:


get_ipython().system('ls -Rlh ../input')


# In[ ]:


(x_train, y_train), (x_test, y_test)=keras.datasets.mnist.load_data()


# In[ ]:


x_train= x_train[...,None]
x_test=x_test[...,None]


# In[ ]:


train = pd.read_csv('../input/emnist/emnist-balanced-train.csv', header=None)
test = pd.read_csv('../input/emnist/emnist-balanced-test.csv', header=None)
train.head()


# In[ ]:



train_data = train.iloc[:, 1:]
train_labels = train.iloc[:, 0]
test_data = test.iloc[:, 1:]
test_labels = test.iloc[:, 0]


# In[ ]:



x_train=train_data.values.reshape((-1,28,28))
y_train=train_labels.values
x_test=test_data.values.reshape((-1,28,28))
y_test=test_labels.values


# In[ ]:





clf = ak.ImageClassifier(verbose=True, augment=False)


# In[ ]:


duree_max=datetime.timedelta(hours=5,minutes=30)
date_limite= date_depart+duree_max
duree_max.total_seconds()


# In[ ]:



delai=date_limite-datetime.datetime.now()

time_limit=delai.total_seconds()

print(delai)
time_limit


# In[ ]:


clf.fit(x_train, y_train, time_limit=time_limit)


# In[ ]:


clf.final_fit(x_train, y_train, x_test, y_test)


# In[ ]:


clf.evaluate(x_test, y_test)


# In[ ]:


results = clf.predict(x_test)


# In[ ]:


model=clf.cnn.best_model #keras.models.load_model("model.h5")


# In[ ]:


import IPython
from graphviz import Digraph
dot = Digraph(comment='autokeras model')
graph=clf.cnn.best_model
for index, node in enumerate(graph.node_list):
    dot.node(str(index), str(node.shape))

for u in range(graph.n_nodes):
    for v, layer_id in graph.adj_list[u]:
      dot.edge(str(u), str(v), str(graph.layer_list[layer_id]))
dot.render(filename='model.png',format='png')
dot.render(filename='model.svg',format='svg')
IPython.display.Image(filename='model.png')


# In[ ]:





# In[ ]:


import IPython

keras_model=clf.cnn.best_model.produce_keras_model()

keras.utils.plot_model(keras_model, show_shapes=True, to_file='model_keras_mnist.png')
IPython.display.Image(filename='model_keras_mnist.png')


# In[ ]:


keras_model.summary()


# In[ ]:


keras_model.compile("adam","mse")
keras_model.save("model.h5")


# In[ ]:


#clf.export_keras_model("model.h5")


# In[ ]:





# In[ ]:



keras_model=keras.models.load_model("model.h5")
keras_model.compile("adam","mse")
onnx_model = onnxmltools.convert_keras(keras_model, target_opset=7) 


# In[ ]:



# Save as text
onnxmltools.utils.save_text(onnx_model, 'model.json')

# Save as protobuf
onnxmltools.utils.save_model(onnx_model, 'model.onnx')


# In[ ]:


import keras2onnx
onnx_model = keras2onnx.convert_keras(keras_model,"autokeras nmist")
# Save as text
onnxmltools.utils.save_text(onnx_model, 'model_keras2onnx.json')

# Save as protobuf
onnxmltools.utils.save_model(onnx_model, 'model_keras2onnx.onnx

