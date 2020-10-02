import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys as sys

def calc_accuracy(x, y, output):
    
    m = x.shape[0]
    print ("all data: " + str(y.shape[0]))
    print ("total 1: " + str((y[y[:, -1] == 1]).shape[0]))
    print ("predicted 1: " + str((output[output[:,-1] == 1]).shape[0]))

    not_matched = np.count_nonzero(output - y)
    print ("Test sample count: " + str(m))
    print ("Not_matched sample count: " + str(not_matched))
    model_accuracy = ((m - not_matched) * 100) / m
    print ("Model accuracy: " + str(model_accuracy) + "%")

def calc_precision_recall(y, output):

    output_for_1 = output[y == 1]
    output_for_0 = output[y == 0]

    true_positive = output_for_1[output_for_1 == 1].shape[0]
    true_negative = output_for_0[output_for_0 == 0].shape[0]
    false_positive = output_for_0[output_for_0 == 1].shape[0]
    false_negative = output_for_1[output_for_1 == 0].shape[0]

    confusion_matrix = np.matrix(([true_positive, false_positive],[false_negative, true_negative]), dtype=int)
    print ("Confusion Matrix")
    print (confusion_matrix)

    model_precision = true_positive/(true_positive+false_positive)
    model_recall = true_positive/(true_positive+false_negative)
    print ("Model precision")
    print (model_precision)
    print ("Model recall")
    print (model_recall)

    model_f_score = (2.0 * model_precision * model_recall) / (model_precision + model_recall)
    print ("Model F score")
    print (model_f_score)

def load_data(path, header):
    pulsar_stars_df = pd.read_csv(path, header=header)
    return pulsar_stars_df
    
def sigmoid(x):
    return (1.0 / (1.0 + np.exp(-1.0 * x.astype(float))))

def sigmoid_derivative(x):
    return x * (1.0 - x)

# assume a neural network with 1 hidden layer comprising of 4 units

class NeuralNetworkUsingBP:
    def __init__(self, eta=0.0012, lbd=0.001, n_iterations=11000):
        self.eta = eta
        self.lbd = lbd
        self.n_iterations = n_iterations

    def feedforward(self):

        self.layer1reg = np.dot(self.input, self.weights1) + self.bias1
        self.layer1 = sigmoid(self.layer1reg)

        self.outputreg = np.dot(self.layer1, self.weights2) + self.bias2
        self.pred_activation = sigmoid(self.outputreg)
        
        #print ("intermediate output")
        #print (self.pred_activation)

    def backprop(self):
        diff_error2 = (self.pred_activation - self.y) * sigmoid_derivative(self.pred_activation)
        diff_error1 = (np.dot(diff_error2, self.weights2.T)) * sigmoid_derivative(self.layer1)

        # calculate diff in weights and biases from errors calculated above

        diff_weights2 = np.dot(self.layer1.T, diff_error2)
        diff_bias2 = np.sum(diff_error2, axis=0)
        diff_weights1 = np.dot(self.input.T, diff_error1)
        diff_bias1 = np.sum(diff_error1, axis=0)

        # adding regularization terms to weights (not to bias)
        diff_weights2 += self.weights2 * self.lbd
        diff_weights1 += self.weights1 * self.lbd

        self.weights2 -= diff_weights2 * self.eta
        self.bias2 -= diff_bias2 * self.eta
        self.weights1 -= diff_weights1 * self.eta
        self.bias1 -= diff_bias1 * self.eta

        #print (diff_weights1)
        #print (diff_weights2)
        #print ("weights1")
        #print (self.weights1)
        #print ("bias1")
        #print (self.bias1)
        #print ("weights2")
        #print (self.weights2)
        #print ("bias2")
        #print (self.bias2)

    def fit(self, x, y):

        self.input      = x
        self.weights1   = np.random.rand(self.input.shape[1],12) # 12 is the number of units in hidden layer 1 
        self.bias1      = np.zeros((1, 12))
        self.weights2   = np.random.rand(12, 1) # 1 is the number of units in output layer
        self.bias2      = np.zeros((1, 1))
        self.y          = y
        self.pred_activation     = np.zeros(self.y.shape) # vector of m elements

        print ("Initial weights")
        print (self.weights1)
        print (self.weights2)

        m = self.input.shape[0]
        self.cost_ = []

        for i in range(self.n_iterations):
            self.feedforward()
            self.backprop()
            
            # cost 
            cost = (- (1 / m) * (np.sum(self.y * np.log(self.pred_activation) + (1 - self.y) * np.log(1 - self.pred_activation)))) + ( (self.lbd / (2 * m))  * ((np.sum(np.square(self.weights1))) + (np.sum(np.square(self.weights2)))))
            
            print ("cost after iteration: " + str(i) + " : " + str(cost))

            self.cost_.append(cost)

        # total cost
        #print ("total cost of nn")
        #print (self.cost_)
        
        #plt.plot(np.arange(1, self.n_iterations + 1), self.cost_)
        #plt.show()

    def predict(self, x, y):

        # Final output
        print ("Final weights")
        print (self.weights1)
        print (self.bias1)
        print (self.weights2)
        print (self.bias2)

        self.input = x
        self.y = y
        self.feedforward()

        #print ("regression")
        #print (self.pred_activation)
        
        m = self.input.shape[0]

        return self.pred_activation

if __name__ == "__main__":

    #np.set_printoptions(threshold=sys.maxsize)

    #load data from file
    data = load_data("../input/pulsar_stars.csv", None)

    all_data = data.iloc[:, :]
    all_data = all_data.values
    all_data = all_data[1:,:].astype(np.float64) # removing labels and assigning type

    print ("all_data")
    print (all_data)

    positive_data = all_data[all_data[:,-1] == 1]
    #print ("positive_data")
    #print (positive_data.shape)
    positive_cloned_data = np.tile(positive_data, (8,1))
    #print ("positive_cloned_data")
    #print (positive_cloned_data.shape)
    oversampled_data = np.concatenate((all_data, positive_cloned_data), 0)
    #print ("oversampled_data")
    #print (oversampled_data.shape)
    oversampled_randomized_data = np.take(oversampled_data,np.random.permutation(oversampled_data.shape[0]),axis=0,out=oversampled_data)
    #print ("oversampled_randomized_data")
    #print (oversampled_randomized_data.shape)
    # final positive/negative samples
    #f1 = oversampled_randomized_data[oversampled_randomized_data[:,-1] == 1]
    #f0 = oversampled_randomized_data[oversampled_randomized_data[:,-1] == 0]

    # x = feature values
    #x = data.iloc[:, :-1]
    x = all_data[:,:-1]
    #x = oversampled_data[:,:-1]
    #x = oversampled_randomized_data[:,:-1]

    # y = class values
    #y = data.iloc[:, -1]
    y = all_data[:,-1]
    #y = oversampled_data[:,-1]
    #y = oversampled_randomized_data[:,-1]
    y = y[:, np.newaxis]

    print ("final shape of matrices")
    print (x.shape)
    print (y.shape)

    # final set of x, y, m
    #ones = np.ones((x.shape[0], 1))
    #x = np.concatenate((ones, x), 1)
    #x = x.values
    #y = y[:,np.newaxis]
    m = x.shape[0]
    n = x.shape[1]
    
    # standardizing (optional, but mostly necessary for gaussian distribution)

    mean = np.mean(x, axis=0)
    #print ("mean")
    #print (mean)
    sd = np.std(x, axis=0)
    #print ("standard deviation")
    #print (sd)

    for j in range(n):
        x[:,j] = (x[:,j] - mean[j])/ (sd[j])

    #print ("normalized/standardized and extended input")
    #print (x)

    # initialize sets
    percent_train = 0.7
    percent_cv = 0.2
    percent_test = 0.1
    classification_threshold = 0.5
    
    train_index = int(np.floor(m * percent_train))
    cv_index = int(train_index + np.floor(m * percent_cv))
    test_index = int(cv_index + np.floor(m * percent_test) + 1)

    # initialize model
    neuralnetwork_model = NeuralNetworkUsingBP()

    # Fit the data (train the model)
    x_train = x[0:train_index+1,:]
    y_train = y[0:train_index+1,:]
    neuralnetwork_model.fit(x_train, y_train)

    # Predict for training set
    print ("Predictions for training set")
    y_predicted = neuralnetwork_model.predict(x[0:train_index+1,:], y[0:train_index+1,:])
    # Output classes on training set
    y_predicted_class = np.where(y_predicted < classification_threshold, 0, 1)
    # Calculate accuracy
    calc_accuracy(x[0:train_index+1,:], y[0:train_index+1,:], y_predicted_class)
    # Calculate precision and recall
    calc_precision_recall(y[0:train_index+1,:], y_predicted_class)
    # Plot regressed y output
    #plt.scatter(np.arange(1, y_predicted.shape[0] + 1), y_predicted.flatten())


    # Predict for cross validation set
    print ("Predictions for cross validation set")
    y_predicted = neuralnetwork_model.predict(x[train_index+1:cv_index,:], y[train_index+1:cv_index,:])
    # Output classes on cross validation set
    y_predicted_class = np.where(y_predicted < classification_threshold, 0, 1)
    # Calculate accuracy
    calc_accuracy(x[train_index+1:cv_index,:], y[train_index+1:cv_index,:], y_predicted_class)
    # Calculate precision and recall
    calc_precision_recall(y[train_index+1:cv_index,:], y_predicted_class)
    # Plot regressed y output
    #plt.scatter(np.arange(1, y_predicted.shape[0] + 1), y_predicted.flatten())

    #plt.show()
    
    