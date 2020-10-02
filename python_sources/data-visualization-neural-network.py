#!/usr/bin/env python
# coding: utf-8

# In[5]:


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd

import sklearn 

import seaborn as sns
#import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt


dataset = pd.read_csv('../input/t1train/T1_simplified.csv')

x = dataset.iloc[:,2:6].values
y= dataset.iloc[:,6].values

print(dataset.isnull().any().any())
print(dataset.isnull().sum())

#dataset["date"] = dataset["date"].dt.strftime("%m-%d-%Y")
dataset['date'] = pd.to_datetime(dataset['date'], errors='coerce')



from pandas.plotting import parallel_coordinates
#parallel_coordinates(dataset.drop("date", axis=1), "energy")

#dataset.plot(kind="scatter", x="air_speed", y="energy")

dataset.hist()

dataset['date'].describe()

#dataset.drop('tstamp', axis=1)
#dataset.drop('date', axis=1)


#f, ax = plt.subplots(4, figsize=(12,24))
#sns.distplot(dataset.energy ='c',ax=ax[0])

#ax[0].set_title('Diamond carat distribution')


#df = pd.DataFrame([[0.529, 5.0, 7.0, 4.0], [0, 3.0, 4.0, 2.0], [0.774, 10.0, 7.0, 6.0], [0.774, 10.0, 8.0, 5.0], [1, 3.0, 0.0, 2.0]])
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.scatter(dataset.iloc[:,2].values, dataset.iloc[:,3].values, dataset.iloc[:,6].values, c=dataset.iloc[:,6].values, cmap='gray', s=50, vmin=0.,vmax=1)

#ax.set_xlabel('X Label')
#ax.set_ylabel('Y Label')
#ax.set_zlabel('Z Label')

#plt.show()


#plt.subplots(figsize=(17,14))
#sns.heatmap(dataset.corr(),annot=True,linewidths=0.5,linecolor="Black",fmt="1.1f")
#plt.title("Data Correlation",fontsize=50)
#plt.show()


#dataset.plot(kind="scatter",x="air_speed", y="energy", alpha=0.5, color="red")
#plt.xlabel("air_speed")
#plt.ylabel("energy")
#plt.title("air_speed impact")

#dataset.plot(kind="scatter",x="temp", y="energy", alpha=0.5, color="red")
#plt.xlabel("temp")
#plt.ylabel("energy")
#plt.title("temp impact")










df_norm = dataset[['air_pressure', 'air_speed', 'air_dir', 'temp', 'energy']].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
df_norm.sample(n=5)

df_norm.describe()









#sns.pairplot( data=df_norm, vars=('air_pressure','air_speed','air_dir','temp'), hue='energy' )




# In[ ]:


train_test_per = 90/100.0
df_norm['train'] = np.random.rand(len(df_norm)) < train_test_per
df_norm.sample(n=5)




train = df_norm[df_norm.train == 1]
train = train.drop('train', axis=1).sample(frac=1)
train.sample(n=5)

test = df_norm[df_norm.train == 0]
test = test.drop('train', axis=1)
test.sample(n=5)

X = train.values[:,:4]
X[:5]

targets = [[1,0,0],[0,1,0],[0,0,1]]
y = np.array([targets[int(x)] for x in train.values[:,4:5]])
#y[:5]

num_inputs = len(X[0])
hidden_layer_neurons = 5
np.random.seed(4)
w1 = 2*np.random.random((num_inputs, hidden_layer_neurons)) - 1
#w1

num_outputs = len(y[0])
w2 = 2*np.random.random((hidden_layer_neurons, num_outputs)) - 1
#w2


                
                
                
#fig = plt.figure(figsize=(12, 12))
#ax = fig.gca()
#ax.axis('off')
#draw_neural_net(ax, .1, .9, .1, .9, [4, 5, 3])

# sigmoid function representation
_x = np.linspace( -5, 5, 50 )
_y = 1 / ( 1 + np.exp( -_x ) )
plt.plot( _x, _y )



learning_rate = 0.2 # slowly update the network
for epoch in range(100000):
    l1 = 1/(1 + np.exp(-(np.dot(X, w1)))) # sigmoid function
    l2 = 1/(1 + np.exp(-(np.dot(l1, w2))))
    er = (abs(y - l2)).mean()
    l2_delta = (y - l2)*(l2 * (1-l2))
    l1_delta = l2_delta.dot(w2.T) * (l1 * (1-l1))
    w2 += l1.T.dot(l2_delta) * learning_rate
    w1 += X.T.dot(l1_delta) * learning_rate
print('Error:', er)




X = test.values[:,:4]
y = np.array([targets[int(x)] for x in test.values[:,4:5]])

l1 = 1/(1 + np.exp(-(np.dot(X, w1))))
l2 = 1/(1 + np.exp(-(np.dot(l1, w2))))

np.round(l2,3)


yp = np.argmax(l2, axis=1) # prediction
res = yp == np.argmax(y, axis=1)
correct = np.sum(res)/len(res)

testres = test

testres['Prediction'] = yp

print(testres)
print('Correct:',sum(res),'/',len(res), ':', (correct*100),'%')


# In[3]:


# Kannca deneme

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd

import sklearn 

import seaborn as sns
#import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt


dataset = pd.read_csv('../input/t1-100/T1_100.csv')

x = dataset.iloc[:,1:6].values
y = dataset.iloc[:,6].values

df_norm = dataset[['air_pressure', 'air_speed', 'air_dir', 'temp', 'energy']].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
df_norm.sample(n=5)


train_test_per = 90/100.0


df_norm['train'] = np.random.rand(len(df_norm)) < train_test_per
df_norm.sample(n=5)


train = df_norm[df_norm.train == 1]
train = train.drop('train', axis=1).sample(frac=1)
train.sample(n=5)

test = df_norm[df_norm.train == 0]
test = test.drop('train', axis=1)
test.sample(n=5)

X = train.values[:1,:4]
X[:5]

Y = train.values[:,4]

x_train = X
y_train = Y

print(train)

print(X)





# In[4]:




def initialize_parameters_and_layer_sizes_NN(x_train, y_train):
    parameters = {"weight1": np.random.randn(3,x_train.shape[0]) * 0.1,
                  "bias1": np.zeros((3,1)),
                  "weight2": np.random.randn(y_train.shape[0],3) * 0.1,
                  "bias2": np.zeros((y_train.shape[0],1))}
    return parameters



def forward_propagation_NN(x_train, parameters):

    Z1 = np.dot(parameters["weight1"],x_train) +parameters["bias1"]
    A1 = np.tanh(Z1)
    Z2 = np.dot(parameters["weight2"],A1) + parameters["bias2"]
    A2 = sigmoid(Z2)

    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
    
    return A2, cache



def compute_cost_NN(A2, Y, parameters):
    logprobs = np.multiply(np.log(A2),Y)
    cost = -np.sum(logprobs)/Y.shape[1]
    return cost


# Backward Propagation
def backward_propagation_NN(parameters, cache, X, Y):

    dZ2 = cache["A2"]-Y
    dW2 = np.dot(dZ2,cache["A1"].T)/X.shape[1]
    db2 = np.sum(dZ2,axis =1,keepdims=True)/X.shape[1]
    dZ1 = np.dot(parameters["weight2"].T,dZ2)*(1 - np.power(cache["A1"], 2))
    dW1 = np.dot(dZ1,X.T)/X.shape[1]
    db1 = np.sum(dZ1,axis =1,keepdims=True)/X.shape[1]
    grads = {"dweight1": dW1,
             "dbias1": db1,
             "dweight2": dW2,
             "dbias2": db2}
    return grads


# update parameters
def update_parameters_NN(parameters, grads, learning_rate = 0.01):
    parameters = {"weight1": parameters["weight1"]-learning_rate*grads["dweight1"],
                  "bias1": parameters["bias1"]-learning_rate*grads["dbias1"],
                  "weight2": parameters["weight2"]-learning_rate*grads["dweight2"],
                  "bias2": parameters["bias2"]-learning_rate*grads["dbias2"]}
    
    return parameters


# prediction
def predict_NN(parameters,x_test):
    # x_test is a input for forward propagation
    A2, cache = forward_propagation_NN(x_test,parameters)
    Y_prediction = np.zeros((1,x_test.shape[1]))
    # if z is bigger than 0.5, our prediction is sign one (y_head=1),
    # if z is smaller than 0.5, our prediction is sign zero (y_head=0),
    for i in range(A2.shape[1]):
        if A2[0,i]<= 0.5:
            Y_prediction[0,i] = 0
        else:
            Y_prediction[0,i] = 1

    return Y_prediction


# 2 - Layer neural network
def two_layer_neural_network(x_train, y_train,x_test,y_test, num_iterations):
    cost_list = []
    index_list = []
    #initialize parameters and layer sizes
    parameters = initialize_parameters_and_layer_sizes_NN(x_train, y_train)

    for i in range(0, num_iterations):
         # forward propagation
        A2, cache = forward_propagation_NN(x_train,parameters)
        # compute cost
        cost = compute_cost_NN(A2, y_train, parameters)
         # backward propagation
        grads = backward_propagation_NN(parameters, cache, x_train, y_train)
         # update parameters
        parameters = update_parameters_NN(parameters, grads)
        
        if i % 100 == 0:
            cost_list.append(cost)
            index_list.append(i)
            print ("Cost after iteration %i: %f" %(i, cost))
    plt.plot(index_list,cost_list)
    plt.xticks(index_list,rotation='vertical')
    plt.xlabel("Number of Iterarion")
    plt.ylabel("Cost")
    plt.show()
    
    # predict
    y_prediction_test = predict_NN(parameters,x_test)
    y_prediction_train = predict_NN(parameters,x_train)

    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_train - y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))
    return parameters

parameters = two_layer_neural_network(x_train, y_train,x_test,y_test, num_iterations=2500)


# In[ ]:




