## Digit Recognizer - Neural Network

import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import MinMaxScaler
from scipy import optimize

def fitNN(X, y, lambdareg, hidden_layer_size, method='L-BFGS-B', maxiter=100):
    # Input layer size (number of features)
    input_layer_size  = X.shape[1] 
    # Number of labels (K)
    num_labels = np.unique(y).size          
    
    # Initializing Pameters
    initial_Theta1 = randInitializeWeights(input_layer_size,
                                                hidden_layer_size)
    initial_Theta2 = randInitializeWeights(hidden_layer_size,
                                                num_labels)
    # Unroll parameters
    initial_nn_params = np.concatenate((initial_Theta1.flatten(),
                                        initial_Theta2.flatten())) 
    
    # Train Neural Network
    # Short hand for grad function
    costFunc = lambda p : nnCostFunction(p, input_layer_size,
                    hidden_layer_size, num_labels, X, y, lambdareg)
    gradFunc = lambda p : nnGradient(p, input_layer_size, 
                    hidden_layer_size, num_labels, X, y, lambdareg)
    # Optimization
    nn_params = trainNN(costFunc, gradFunc, initial_nn_params, maxiter, method)
    # Obtain Theta1 and Theta2 back from nn_params
    Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
                 (hidden_layer_size, input_layer_size+1)) # s2 n+1
    Theta2 = np.reshape(nn_params[hidden_layer_size * (input_layer_size+1):],
                 (num_labels, hidden_layer_size + 1)) # K s2+1

    return [Theta1, Theta2]
    
def trainNN(costFunc, gradFunc, initial_nn_params, maxiter=50, 
                                                method='L-BFGS-B'):
    opts = {'maxiter': maxiter}
    # 'L-BFGS-B' #'CG' # 'BFGS' # 'Newton-CG'  # 'TNC'
    print('Running Optimization (' + str(method) + ' method, ' + 
            str(maxiter) + ' max number iterations)...')
    #start_time = timeit.default_timer()
    ret = optimize.minimize(costFunc, initial_nn_params, method=method,
                            jac=gradFunc, options=opts)
    nn_params = ret['x']
    cost = ret['fun']
    #print('Optimization by ' + method + '. Cost: ' + str(cost))
    print(ret['message'])
    print('Cost: ' + str(cost))
    #print(str(timeit.default_timer() - start_time) + ' seconds')

    return nn_params
    
def scoreNN(Xtest, ytest, Thetas):
    Theta1 = Thetas[0]
    Theta2 = Thetas[1]
    # Compute accuracy on our training set
    p = predictNN(Theta1, Theta2, Xtest)
    score = 100*np.mean((p==ytest).astype(float))
    return score

def nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels,
                    X, y, lambdareg):
    # Reshape nn_params back into the parameters Theta1 and Theta2
    Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
                 (hidden_layer_size, input_layer_size+1)) # s2 n+1
    Theta2 = np.reshape(nn_params[hidden_layer_size * (input_layer_size+1):],
                 (num_labels, hidden_layer_size + 1)) # K s2+1
    #print('sumX ' + str(np.sum(X)))
    #print('Theta1 ' + str(Theta1[:5,:5]))
    #print('Theta2 ' + str(Theta2[:5,:5]))
    m = y.size  # number of training examples
    # Add a column of ones to X
    X = np.hstack((np.ones((m, 1)), X)) # m n+1
    # Positions of 1s in yK, by columns
    pos = y + num_labels * np.arange(m) # m 1    
    #[y1, K + y2, 2*K + y3, 3*K + y4,..., (m-1)*K + ym]
    yK = np.zeros((m * num_labels)) # m*K
    # A one in every position of yK set by "pos"
    yK[pos] = np.ones(m)
    # yK: y transformed to boolean K-vectors,  m K
    yK = np.reshape(yK, (m, num_labels))  # m K
    # print('yK '+str(yK.shape))
    #print('yK ' + str(yK[::150]))
    #print('y: ' + str(y[::150]))

    # X: m n+1       Theta1: s2 n+1       Theta2: K s2+1
    
    # FORWARD PROPAGATION: a1=X -> z2 -> a2 -> z3 -> a3=h(theta)
    # First activation unit. Input data
    a1 = X  # m n+1
    z2 = np.dot(a1, Theta1.T) # m s2
    # Second activation units. Hidden layer
    a2 = sigmoid(z2)  # m s2
    # Add ones to the units (bias)
    a2 = np.hstack((np.ones((m, 1)), a2))  # m s2+1
    z3 = np.dot(a2, Theta2.T)  # m K
    # Third activation units. Output units, predictions
    a3 = sigmoid(z3)  # m K 
    predK = a3  # m K 
    #print('predK ' + str(predK[::150]))
    
    # Cost function
    costik = -yK * np.log(predK) - (1.-yK) * np.log(1.-predK)  # m K
    #print('costik ' + str(costik[::150]))
    #print('sumcostik ' + str(np.sum(costik)))
    J = (1./m) * np.sum(costik) # 1 
    
    # Regularize cost function (thetas without 1st column)
    jreg1 = np.sum(Theta1[:,1:]**2)  # 1 
    jreg2 = np.sum(Theta2[:,1:]**2)  # 1 
    J = J + (lambdareg/(2.*m))*(jreg1 + jreg2) # 1

    return J
    
def nnGradient(nn_params, input_layer_size, hidden_layer_size, num_labels,
                    X, y, lambdareg):
    # Reshape nn_params back into the parameters Theta1 and Theta2
    Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
                 (hidden_layer_size, input_layer_size+1)) # s2 n+1
    Theta2 = np.reshape(nn_params[hidden_layer_size * (input_layer_size+1):],
                 (num_labels, hidden_layer_size + 1)) # K s2+1
    m = y.size  # number of training examples
    # Add a column of ones to X
    X = np.hstack((np.ones((m, 1)), X)) # m n+1
    # Positions of 1s in yK, by columns
    pos = y + num_labels * np.arange(m) # m 1    
    #[y1, K + y2, 2*K + y3, 3*K + y4,..., (m-1)*K + ym]
    yK = np.zeros((m * num_labels)) # m*K
    # A one in every position of yK set by "pos"
    yK[pos] = np.ones(m)
    # yK: y transformed to boolean K-vectors,  m K
    yK = np.reshape(yK, (m, num_labels))  # m K

    # X: m n+1       Theta1: s2 n+1       Theta2: K s2+1
    
    # FORWARD PROPAGATION: a1=X -> z2 -> a2 -> z3 -> a3=h(theta)
    # First activation unit. Input data
    a1 = X  # m n+1
    z2 = np.dot(a1, Theta1.T) # m s2
    # Second activation units. Hidden layer
    a2 = sigmoid(z2)  # m s2
    # Add ones to the units (bias)
    a2 = np.hstack((np.ones((m, 1)), a2))  # m s2+1
    z3 = np.dot(a2, Theta2.T)  # m K
    # Third activation units. Output units, predictions
    a3 = sigmoid(z3)  # m K 
    
    # BACKPROPAGATION: a3 -> delta3 -> delta2 -> Deriv
    # Errors in prediction
    delta3 = a3 - yK  # m K
    # BP to delta2, removing delta2,0
    delta2 = np.dot(delta3, Theta2)[:, 1:] * sigmoidGradient(z2)  # m s2
    #                (m K) (K s2+1)  (,s2)      (m s2)
    # Accumulate deltas
    acDelta1 = np.dot(delta2.T, a1) # s2 n+1
    acDelta2 = np.dot(delta3.T, a2) # K s2+1
    # Set gradients
    Theta1_grad = (1./m) * acDelta1  # s2 n+1
    Theta2_grad = (1./m) * acDelta2  # K s2+1
    
    # Regularize gradients (except 1st column)
    g1reg = np.hstack((np.zeros((hidden_layer_size,1)), Theta1[:, 1:])) # s2 n+1
    g2reg = np.hstack((np.zeros((num_labels,1)), Theta2[:, 1:])) # K s2+1
    Theta1_grad = Theta1_grad + (lambdareg/m) * g1reg  # s2 n+1
    Theta2_grad = Theta2_grad + (lambdareg/m) * g2reg  # K s2+1

    # Unroll gradients
    grad = np.concatenate((Theta1_grad.flatten(),
                           Theta2_grad.flatten())) # s2*(n+1)*K*(s2+1)  1

    return grad
    
def randInitializeWeights(L_in, L_out):
    W = np.zeros((L_out, L_in + 1))
    epsilon_init = 0.12
    # epsilon_init = np.sqrt(6./(L_in+L_out))
    W = (2. * np.random.rand(L_out, L_in + 1) - 1.) * epsilon_init
    return W
    
def predictNN(Theta1, Theta2, X):
    m = X.shape[0] # Number of data
    X = np.hstack((np.ones((m, 1)), X)) # Add a column of ones to X
    p = np.zeros(m) # Prediction
    # X: m n+1       Theta1: s2 s1+1(n+1)       Theta2: s3(K) s2+1
    h1 = sigmoid(np.dot(X, Theta1.T)) # m s2
    h1 = np.hstack((np.ones((m, 1)), h1)) # m s2+1,  Add a column of ones to h1

    h2 = sigmoid(np.dot(h1, Theta2.T)) # m K 
    p = np.argmax(h2, axis=1) # m 1
    return p
    
def sigmoid(z):
    g = np.zeros(z.shape)
    g = 1./(1. + np.exp(-z))
    #print('g shape '+str(g.shape))
    return g
    
def sigmoidGradient(z):
    g = np.zeros(z.shape)
    g = np.exp(-z)/(1. + np.exp(-z))**2
    return g

print('Loading data...')
#dataset = np.genfromtxt(open("../input/train.csv",'r'), delimiter=',', skip_header=1, dtype=int)
dataset = np.loadtxt(open("../input/train.csv",'r'), delimiter=',', skiprows=1, dtype=int)
X = dataset[:, 1:]
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
y = dataset[:, 0]

# Create random train and validation sets out of 20% samples
Xtrain, Xval, ytrain, yval = train_test_split(X, y, test_size=0.2,
                                    stratify=y, random_state=11)
print('Xtrain, ytrain shapes ' + str((Xtrain.shape, ytrain.shape)))
print('Xval, yval shapes ' + str((Xval.shape, yval.shape)))
# number of training examples and features 
m, n = X.shape

# Train neural Network   0.1, 50: 99.6, 96.2
# 0.01, 25: 97.5, 94.5      # 0.01, 50: 99.4, 96.3      # 0.01, 100: 99.9, 96.8
# 0.1, 25: 97.8, 94.9       # 0.1, 50: 99.4, 96.1       # 0.1, 100: 99.9, 96.9
# 1, 25: 97.6, 94.8         # 1, 50: 99.2, 96.1         # 1, 100: 99.7, 96.9
# 10, 25: 96.0, 94.1        # 10, 50: 97.4, 95.3        # 10, 100: 98.2, 96.7

# 0.1, 200: 99.7, 97.4 
# 1, 200: 99.5, 97.3
# 10, 200: 98.2, 96.9
lambdareg = 0.1
hidden_layer_size = 200 # 100 #25
filename = 'DigitRecogSubmissionMyNN_%0.3f_%d.csv' % ((lambdareg, hidden_layer_size))
print(filename)
print('\nTraining Neural Network (lambda=%0.3f, hidden units=%d)...' % 
                    (lambdareg, hidden_layer_size))
Thetas = fitNN(Xtrain, ytrain, lambdareg, hidden_layer_size)
# Accuracies
trainAccur = scoreNN(Xtrain, ytrain, Thetas)
valAccur = scoreNN(Xval, yval, Thetas)
print('Train Accuracy: %0.2f' % (trainAccur))
print('Validation Accuracy: %0.2f' % (valAccur))


print('\nValidation of lambda and hidden size...')
lamb_vec = [0.01] #[0.1, 1, 10] #[0.01, 0.1, 1, 10]
hidsize_vec = [200] #[25, 50, 100]
for lam in lamb_vec:
    for hidsize in hidsize_vec:
        print('\nValidating Neural Network (lambda=%0.3f, hidden units=%d)...' % 
                    (lam, hidsize))
        nThetas = fitNN(Xtrain, ytrain, lam, hidsize)
        trainAccur = scoreNN(Xtrain, ytrain, nThetas)
        valAccur = scoreNN(Xval, yval, nThetas)
        print('Train Accuracy: %0.2f. ' % (trainAccur))
        print('Validation Accuracy: %0.2f' % (valAccur))

#Load test set
print('\nLoading test set..')
test = np.loadtxt(open("../input/test.csv",'r'), delimiter=',', skiprows=1, dtype=int)
Xtest = scaler.transform(test)
print('Test shape: ' + str(Xtest.shape))
Theta1 = Thetas[0]
Theta2 = Thetas[1]
pred = predictNN(Theta1, Theta2, Xtest)
print('pred ' + str(pred[:5]))
# Saving submission file
print('\nSaving submission file...')
pred_digits = np.vstack((np.arange(1, pred.size+1), pred)).T
filename = 'DigitRecogSubmissionMyNN_%0.3f_%d.csv' % ((lambdareg, hidden_layer_size))
print(filename)
np.savetxt(filename, pred_digits, fmt='%d,%d', delimiter=',', header='ImageId,Label', comments = '')
