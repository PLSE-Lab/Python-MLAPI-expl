#!/usr/bin/env python
# coding: utf-8

# # Perceptron Neural Netework
# ## Project: Process analysis of petroleum fractional distillation  

# ### First we implement the algorithm of Perceptron

# In[ ]:


import numpy as np

def thresholdF(u):
    if(u >= 0):
        return 1
    else:
        return -1
    

class Perceptron(object):
    def __init__(self, samples, classes, step, seed):
        self.samples = samples
        self.classes = np.array(classes)
        self.step = step
        self.seed = seed
        self.age = 0
        
        np.random.seed(seed)
        self.w0 = np.random.random((len(self.samples[0])))
        self.w = self.w0
        
    def traning(self):
        erro = True

        while(erro):
            erro = False
            cont = 0
            for i, x in enumerate(self.samples):
                u = np.multiply(self.w,x).sum()
                y = thresholdF(u)

                if(y != self.classes[i]):
                    self.w =  self.w + self.step*(self.classes[i] - y)*x
                    erro = True
                else:
                    cont += 1
            self.age += 1    
            if(cont/len(self.samples)>= 0.9 and self.age >= 1000):
                erro = False
        return self.w
    
    def classifier(self, samples):
        d = []
        
        for i, x in enumerate(samples):
            u = np.multiply(self.w,np.array(x)).sum()
            d.append(thresholdF(u))
        return d


# In[ ]:


import matplotlib.pyplot as plt
import pandas as pd


# In[ ]:


data = pd.read_csv("../input/sample.csv")
data.insert(0, "const", [-1]*len(data), True) 

test = pd.read_csv("../input/test.csv")
test.insert(0, "const", [-1]*len(test), True) 


# In[ ]:


data.plot.scatter(x='x1', y='x2', c='d', colormap='viridis')
data.plot.scatter(x='x1', y='x3', c='d', colormap='viridis')
data.plot.scatter(x='x2', y='x3', c='d', colormap='viridis')


# # Traning the Perceptron
# 
# ## Traning Paramatrers
# * ### Numpy Seed: $10$
# * ### $W_0$: [0.77132064 0.02075195 0.63364823 0.74880388]
# * ### Learning Rate: $0.01$
# 

# In[ ]:


first = Perceptron(data[["const","x1","x2","x3"]].values,data[["d"]].values,0.01,10)
print(first.traning())
first.age


# ## Classifier Paramatrers
# * ### Age: $447$
# * ### $W$: [-3.12867936,  1.61669995,  2.52421423, -0.74714412]
# 

# In[ ]:


d_tf = first.classifier(test.values)
test.plot.scatter(x='x1', y='x2', c = d_tf, colormap='viridis')


# ## Traning Paramatrers
# * ### Numpy Seed: $1043$
# * ### $W_0$: [0.55128538 0.95755756 0.03540917 0.93788529]
# * ### Learning Rate: $0.01$

# In[ ]:


second = Perceptron(data[["const","x1","x2","x3"]].values,data[["d"]].values,0.01,1043)
print(second.w0)
print(second.traning())
second.age


# ## Classifier Paramatrers
# * ### Age: $406$
# * ### $W$: [-3.06871462  1.56173756  2.46559517 -0.73229471

# In[ ]:


d_ts = second.classifier(test.values)
test.plot.scatter(x='x1', y='x2', c = d_ts, colormap='viridis')


# ## Traning Paramatrers
# * ### Numpy Seed: $104311$
# * ### $W_0$: [0.75270862 0.78774319 0.84155995 0.63831767]
# * ### Learning Rate: $0.01$
# 

# In[ ]:


third = Perceptron(data[["const","x1","x2","x3"]].values,data[["d"]].values,0.01,104311)
print(third.w0)
print(third.traning())
third.age


# ## Classifier Paramatrers
# * ### Age: $415$
# * ### $W$: [-3.04729138  1.54980319  2.47179395 -0.72895233]
# 

# In[ ]:


d_tt = third.classifier(test.values)
test.plot.scatter(x='x1', y='x2', c = d_tt, colormap='viridis')


# ## Traning Paramatrers
# * ### Numpy Seed: $10431153$
# * ### $W_0$: [0.96175189 0.13144601 0.30382036 0.72794536]
# * ### Learning Rate: $0.01$

# In[ ]:


fourth = Perceptron(data[["const","x1","x2","x3"]].values,data[["d"]].values,0.01,10431153)
print(fourth.w0)
print(fourth.traning())
fourth.age


# ## Classifier Paramatrers
# * ### Age: $389$
# * ### $W$: [-2.93824811  1.41988401  2.43192836 -0.70336864]
# 

# In[ ]:


d_tfo = fourth.classifier(test.values)
test.plot.scatter(x='x1', y='x2', c = d_tfo, colormap='viridis')


# ## Traning Paramatrers
# * ### Numpy Seed: $1043115310$
# * ### $W_0$: [0.43233505 0.36987213 0.8559367  0.09424125]
# * ### Learning Rate: $0.01$

# In[ ]:


fifth = Perceptron(data[["const","x1","x2","x3"]].values,data[["d"]].values,0.01,1043115310)
print(fifth.w0)
print(fifth.traning())
fifth.age


# ## Classifier Paramatrers
# * ### Age: $409$
# * ### $W$: [-3.06766495  1.55463213  2.4751427  -0.73269275]

# In[ ]:


d_tfi = fifth.classifier(test.values)
test.plot.scatter(x='x1', y='x2', c = d_tfi, colormap='viridis')


# In[ ]:




