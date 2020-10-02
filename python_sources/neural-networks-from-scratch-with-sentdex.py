import sys
import numpy as np
import matplotlib

## intro and Neuron Code

print("Python:" , sys.version)
print("Numpy:" , np.__version__)
print("Matplotlib:" , matplotlib.__version__)

## feed forward Multy leyare perceptron
inputs = [1.2,5.1,2.1]
weights = [3.1,2.1,8.7]
bias = 3 

output = inputs[0]*weights[0] +inputs[1]*weights[1]+inputs[2]*weights[2] +bias

print(output)
 
 
