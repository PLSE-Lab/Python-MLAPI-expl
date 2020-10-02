#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import  datetime
date_depart=datetime.datetime.now()
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[ ]:


get_ipython().system('pip install git+https://github.com/onnx/onnx-tensorflow.git')
get_ipython().system('pip install keras2onnx ')
get_ipython().system('apt install -y graphviz ')


get_ipython().system('pip install -q  keras torch torchvision graphviz autokeras  onnxmltools')
get_ipython().system('pip install -q    onnxmltools onnx-tf')
get_ipython().system('pip install --user autokeras ')
#!pip install git+https://github.com/jhfjhfj1/autokeras.git
import keras
import autokeras as ak
import onnxmltools


# In[ ]:


get_ipython().system('ls -Rlh ../input')


# In[ ]:


(x_train, y_train), (x_test, y_test)=keras.datasets.mnist.load_data()


# In[ ]:


y_train


# In[ ]:


duree_max=datetime.timedelta(hours=4,minutes=30)
date_limite= date_depart+duree_max
duree_max.total_seconds()


# In[ ]:


x_train= x_train[...,None]
x_test=x_test[...,None]


# In[ ]:





clf = ak.ImageClassifier(verbose=True, augment=False)


# In[ ]:



delai=date_limite-datetime.datetime.now()

time_limit=delai.total_seconds()

print(delai)
time_limit


# In[ ]:


print(datetime.datetime.now()-date_depart)


# In[ ]:


clf.fit(x_train, y_train, time_limit=time_limit)


# In[ ]:


print(datetime.datetime.now()-date_depart)


# In[ ]:


clf.final_fit(x_train, y_train, x_test, y_test)


# In[ ]:


print(datetime.datetime.now()-date_depart)


# In[ ]:


clf.evaluate(x_test, y_test)


# In[ ]:


print(datetime.datetime.now()-date_depart)


# In[ ]:


results = clf.predict(x_test)


# In[ ]:


print(datetime.datetime.now()-date_depart)


# 

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


print(datetime.datetime.now()-date_depart)


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


print(datetime.datetime.now()-date_depart)


# In[ ]:


import keras2onnx
onnx_model = keras2onnx.convert_keras?(keras_model,"autokeras nmist")
# Save as text
onnxmltools.utils.save_text(onnx_model, 'model_keras2onnx.json')

# Save as protobuf
onnxmltools.utils.save_model(onnx_model, 'model_keras2onnx.onnx')


# In[ ]:


print(datetime.datetime.now()-date_depart)

