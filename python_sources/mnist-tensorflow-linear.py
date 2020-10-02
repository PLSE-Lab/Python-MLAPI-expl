import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix


train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")

print(tf.__version__)

#from tensorflow.examples.tutorials.mnist import input_data
#data = input_data.read_data_sets("data/MNIST/", one_hot=True)

x = 1000000000
for i in range(1000000):
    x = x + 0.000001
    
x = x - 1000000000
print(x)