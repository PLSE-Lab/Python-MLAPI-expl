# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
# Load Data
print("Loading Raw Data")
train_raw = pd.read_csv("../input/train.csv")
test_raw = pd.read_csv("../input/test.csv")

# Helper Functions 
def get_features(raw_data):
    cols = []
    # Get data of each row from pixel0 to pixel783 
    for px in range(784):
        cols.append("pixel"+str(px))   
    return raw_data.as_matrix(cols)/255

def cross_validated(X, n_samples):
    kf = KFold(n_samples, shuffle = True)
    result = [group for group in kf.split(X)]
    return result        
    
# Deep Neural Net
# Initialize Parameters 
def init_dnn_parameters(n, activations, epsilons, filter=None):
    L = len(n)
    params = {}
    for l in range(1,L):
        W = np.random.randn(n[l],n[l-1]) * epsilons[l] 
        # Experiment, multiply filter in case of input layer weights 
        if filter1 is not None and l == 1:
            W = np.dot(W, filter) 
        b = np.zeros((n[l],1))
        params["W"+str(l)] = W
        params["b"+str(l)] = b                        
        params["act"+str(l)] = activations[l]
    params["n"] = n
    return params

# Activation Functions 
def gdnn(X, activation_function):
    leak_factor = 1/100
    if activation_function == 'tanh':
        return np.tanh(X)
    if activation_function == 'lReLU':
        return ((X > 0) * X) + ((X <= 0)* X * leak_factor)
    else: 
        return 1 / (1 +np.exp(-X))

def gdnn_prime(X, activation_function):
    leak_factor = 1/100
    if activation_function == 'tanh':
        return 1-np.power(X,2)
    if activation_function == 'lReLU':
        return ((X > 0) * 1) + ((X <= 0)* leak_factor)
    else: 
        return (1 / (1 +np.exp(-X)))*(1-(1 / (1 +np.exp(-X))))

# Cost 
def get_dnn_cost(Y_hat, Y):
    #print(Y.shape)
    m = Y.shape[1]
    logprobs = np.multiply(np.log(Y_hat),Y) + np.multiply(np.log(1-Y_hat),1-Y)
    cost = - np.sum(logprobs) /m
    return cost
    
# Forward Propagation 
def forward_dnn_propagation(X, params):
    # Get Network Parameters 
    n = params["n"]
    L = len(n)
    
    A_prev = X
    cache = {}
    cache["A"+str(0)] = X
    for l in range(1,L):
        W = params["W"+str(l)]
        b = params["b"+str(l)]
        Z = np.dot(W,A_prev)+b
        A = gdnn(Z,params['act'+str(l)])
        cache["Z"+str(l)] = Z
        cache["A"+str(l)] = A
        
        A_prev = A
    return A, cache 

# Backward Propagation
def back_dnn_propagation(X, Y, params, cache, alpha = 0.01, _lambda=0, keep_prob=1):
    n = params["n"]
    L = len(n) -1
    
    m = X.shape[1]
    W_limit = 5
    A = cache["A"+str(L)]
    A1 = cache["A"+str(L-1)]
    grads = {}
    
    # Outer Layer 
    dZ = A - Y#gdnn_prime(A - Y, params["act"+str(L)])
    dW = 1/m * np.dot(dZ, A1.T)
    db = 1/m * np.sum(dZ, axis=1, keepdims=True)
    grads["dZ"+str(L)] = dZ
    grads["dW"+str(L)] = dW + _lambda/m * params["W"+str(L)]
    grads["db"+str(L)] = db
    
    # Update Outer Layer
    params["W"+str(L)] -= alpha * dW
    params["b"+str(L)] -= alpha * db
    for l in reversed(range(1,L)):
        dZ2 = dZ
        W2 = params["W"+str(l+1)]
        b = params["b"+str(l)]
        A2 = cache["A"+str(l)]
        A1 = cache["A"+str(l-1)]
        d = np.random.randn(A1.shape[0],A1.shape[1]) > keep_prob
        A1 = A1 * d/keep_prob
        dZ = np.dot(W2.T, dZ2)*gdnn_prime(A2, params["act"+str(l)])
        dW = 1/m * np.dot(dZ, A1.T) + _lambda/m * params["W"+str(l)]
        db = 1/m * np.sum(dZ, axis=1, keepdims=True)
        grads["dZ"+str(l)] = dZ
        grads["dW"+str(l)] = dW
        grads["db"+str(l)] = db
        params["W"+str(l)] -= alpha *dW
        params["b"+str(l)] -= alpha *db
    
    return grads, params    

def batch_back_propagation(X, Y, params, cache, alpha = 0.01, _lambda=0, keep_prob=1,chunk_size=128):
    # slice input and output data into smaller chunks 
    m = X.shape[1]
    include_probability = keep_prob
    idx_from = 0
    batch_size = chunk_size * np.random.randint(1,10)
    idx_to = min(batch_size, m)
    X_train = X[:,idx_from:idx_to]
    y_train = Y[:,idx_from:idx_to]
    while idx_to < m:
        batch_size = chunk_size * np.random.randint(1,10)
        #if np.random.random() < include_probability:
        for looper in range(1+np.random.randint(3)):
            A, cache = forward_dnn_propagation(X_train, params)
            grads, params= back_dnn_propagation(X_train, y_train, params, cache, alpha ,_lambda, keep_prob) 
            print("Mini-Batch Size [{}] @ {:4.2f} % [{}] Training Score {:3.2f}".format(
                batch_size, 
                100*idx_to/m,
                looper,
                np.mean(get_dnn_cost(A, y_train))))
        idx_from += batch_size
        idx_from = min(m, idx_from)
        idx_to += batch_size
        idx_to = min(m, idx_to)
        if idx_from < idx_to:
            X_train = X[:,idx_from:idx_to]
            y_train = Y[:,idx_from:idx_to]
    return grads, params
    
# Train Model 
print("Loading Training and Dev Data ")
X2 = get_features(train_raw)

labels = np.array(train_raw['label'])
m = labels.shape[0]
y = np.zeros((m,10))
for j in range(10):
    y[:,j]=(labels==j)*1

k = 38
folds = 5
oinst = 1
h_layers = 6
np.random.seed(1)
print("Cross Validation using {} folds".format(folds))
print("Building Deep Network of {} Hidden Layer Groups".format(h_layers))
print("Cross Validation ..")
cv_groups = cross_validated(X2, folds)
print("Done")
alphas = np.linspace(0.0567, 0.0567, oinst)
epsilons = np.linspace(0.76,0.78,oinst)
gammas =  np.linspace(0.01,0.01,oinst)
lambdas=  np.linspace(25.91,25.91,oinst)
keep_probs=  np.linspace(0.99,0.99,oinst)
alph_decays = np.linspace(0.9,0.9,oinst) 
iterations = 1
n_1 = []
break_tol = 0.00000001
etscost = []
etrcost= []
seeds = []
layers = []
for j in range(oinst):
    batch_processing = True
    batch_size = 512 # min size

    print("Building Network")
    X = X2 # Direct Map
    n = [X.shape[1]]
    acts = ['input']
    gamma = [0]
    for layer in range(h_layers):
        n.append((28)**2) #((28-layer*3))**2)
        acts.append('lReLU') #tanh')
        gamma.append(np.sqrt(2/n[layer-1]))
        print("Hidden Layer[{:03d}] n = {}, Activation Fn [{}], Weight init Factor = {:3.2f}".format(
            len(n)-1, n[-1], acts[-1], gamma[-1]))
    #for layer in range(h_layers):
        n.append((28)**2) #((28-layer*3))**2)
        acts.append('tanh') #tanh')
        gamma.append(np.sqrt(2/n[layer-1]))
        print("Hidden Layer[{:03d}] n = {}, Activation Fn [{}], Weight init Factor = {:3.2f}".format(
            len(n)-1, n[-1], acts[-1], gamma[-1]))
    layers.append(j+1)    
    n.append(y.shape[1])
    acts.append('sigmoid')
    gamma.append(np.sqrt(1/n[layer-1]))
    print("Output Layer n = {}, Activation Function [{}], Weight init Factor = {:3.2f}".format(
            n[-1], acts[-1], gamma[-1]))
    n_1.append(j+4)
    np.random.seed(1)
   
    alpha = alphas[j]#0.166# 
    _lambda = lambdas[j] # 0.5#
    keep_prob = keep_probs[j]
    epsilon = 0.76#epsilons[j] #0.02 
    print("Hyper-parameters")
    print("alpha = {:4.2f}, iterations = {}, lambda = {:3.2f}, keep probability = {:3.2f}%".format(
        alpha, iterations, _lambda, keep_prob*100))

    L = len(n) - 1

    # Prepare Training and testing sets 
    X_train = X[cv_groups[0][0],:].T 
    y_train = y[cv_groups[0][0],:].T 
    labels_train = labels[cv_groups[0][0]]
    # Experiment - Filter based on linear correlation
    
    depth = 1024
    print("Building Input Layer Initialization Filter, Depth = {}".format(depth))
    filter1 = np.zeros((n[0],n[0]))
    for dim in range(10):
        for monomial in range(1,min(4, h_layers)):
            X_sample = X_train[:,:depth].T**monomial
            X_mean = np.reshape(np.mean(X_sample,axis=0),(1,-1))
            y_sample = np.reshape(y_train[dim, :depth],(-1,1))

            y_mean = np.mean(y_sample)
            y_var = (y_sample - y_mean)*X_sample**0
            numer = (np.dot((X_sample-X_mean).T,y_var))
            denom = np.sqrt(np.sum(np.dot((X_sample-X_mean).T,(X_sample-X_mean))))*np.sqrt(np.dot((y_sample - y_mean).T,(y_sample - y_mean)))
            filter1 += np.abs(np.diag((numer/denom)[:,0]))
    filter1 /= np.linalg.norm(filter1)
    filter2 = 1*(np.abs(filter1) > 0.0001 )
    params = init_dnn_parameters(n, acts,gamma, np.abs(filter1))

    # Experiment 
    
    X_test = X[cv_groups[0][1],:].T 
    y_test = y[cv_groups[0][1],:].T
    print("Experiment [{}] - Eps = {}, Alph = {:3.2f}, Decay = {:3.2f}, lambda={:3.2f}".format(j, epsilon, alpha,alph_decays[j], _lambda))
    print("k = {}, |X| = {}, max(i) = {}".format( k, X_test.shape[0], iterations))
    #print("Keep Prob = {}%, gamma = {}".format(keep_prob*100, gamma))
    print("Network {} {}".format(n,acts))
    cost = []
    tcost=[]
    print("Batch Processing [{}], Batch Size [{}]".format(batch_processing, batch_size))
    A, cache = forward_dnn_propagation(X_train, params)
    for i in range(iterations):
        if batch_processing:
            grads, params = batch_back_propagation(X_train, 
                                                   y_train, 
                                                   params, 
                                                   cache, 
                                                   alpha,
                                                   _lambda, 
                                                   keep_prob,                                                  
                                                   batch_size)
            A, cache = forward_dnn_propagation(X_train, params)
            cost.append(np.mean(get_dnn_cost(A, y_train)))
            A2, vectors = forward_dnn_propagation(X_test, params)
            tcost.append(get_dnn_cost(A2, y_test))
        else:
            A, cache = forward_dnn_propagation(X_train, params)
            cost.append(get_dnn_cost(A, y_train))
            grads, params= back_dnn_propagation(X_train, 
                                                y_train, 
                                                params, 
                                                cache, 
                                                alpha,
                                                _lambda, 
                                                keep_prob)
            A2, vectors = forward_dnn_propagation(X_test, params)
            tcost.append(get_dnn_cost(A2, y_test))
        
        if alpha*np.abs(np.linalg.norm(grads["dW"+str(L)])) < break_tol:
            break
        if i % 1 == 0:
            alpha *= alph_decays[j]
            print("---------------------------------------------------------------")
            print("i = {:3d}, trc = {:3.2f}, tsc={:3.2f}, alph.dWL = {:3.2f}".format(i,cost[-1],
                                                                   tcost[-1], 
                                                                   alpha*np.abs(np.linalg.norm(grads["dW"+str(L)]))))
            print(" active alph = {:3.2f}".format(alpha))
            if 1==1:
                print("Number Matching")
                for num in range(10):
                    y_hat = A2[num,:] > 0.5
                    y_star = y_test[num,:]
                    matched = np.sum((1-np.abs(y_star-y_hat))*y_star)
                    distance = np.linalg.norm((y_star - A2[num,:])*y_star)
                    m_test = sum(y_test[num,:]==1)
                    y_size = y_test.shape[1]
                    pct = matched/m_test
                    print("[{}] Matched {} {:3.2f}% m_pos={}, Distance {:3.2f}".format(num, 
                                                                            matched,
                                                                            pct*100, 
                                                                            m_test,distance ))
                print("---------------------------------------------------------------")
    etscost.append(tcost[-1])
    etrcost.append(cost[-1])

    
    # Prepare Data For submission
print("Preparing Data for submission")
X_test = get_features(test_raw)
print("Running Test Data On Model")
A2, vectors = forward_dnn_propagation(X_test.T, params)
print("Output Vector Shape {}".format(A2.shape))
# use A2 to construct 

#y_hat = A2 > 0.5
#y_hat = y_hat*1

#data = np.zeros((y_hat.shape[1],1))
# To Do, pick maximum index 
#for idx in range(y_hat.shape[0]):
#    data += idx * np.reshape(y_hat[idx,:],(-1,1))
#data = np.clip(data,0,9).astype(int)   

data = np.clip(A2.T, 0,1)
data = data.argmax(axis=1)
#data = np.reshape(data,(-1,1))
print(data.shape)
#data[len(data),0] = 0 # Add missing entry 
data = np.reshape(data,(-1,1))

print("Data Dump")
print(data)
print("Prepared Output Vector Shape {}".format(data.shape))
index = np.reshape(np.arange(1, data.shape[0]+1),(-1,1))
data = np.concatenate((index, data), axis=1)

#s1 = pd.Series(data)
print(data[0])
print(data.shape)

s0 = pd.Series(index)
df = pd.DataFrame(data = data[:,1], index=data[:,0])
df.index.name = 'ImageId'
df.columns = ['Label']
df.replace([np.inf, -np.inf, np.nan], 0)
print("Dumping Dataframe")
print(df)
df = df.astype(int)
file_name = "deep_nn.csv"
print("Saving Data to [{}]".format(file_name))
df.to_csv(file_name, sep=',')
print("========= End ===========")