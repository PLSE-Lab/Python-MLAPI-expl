# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 16:22:32 2019

@author: admin
"""
import numpy as np

class SimpleNeuralNetwork:
    
    
    
    # initialise the neural network
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        
        # set number of nodes in each input, hidden, output layer
        self.i_nodes = input_nodes
        self.h_nodes = hidden_nodes
        self.o_nodes = output_nodes
        
        # link weight matrices, wih and who
        # weights inside the arrays are w_i_j, where link is from node i to node j in the next layer
        # w11 w21
        # w12 w22 etc 
        self.wih = np.random.normal(0.0, pow(self.h_nodes, -0.5), (self.h_nodes, self.i_nodes))
        self.who = np.random.normal(0.0, pow(self.o_nodes, -0.5), (self.o_nodes, self.h_nodes))

        # learning rate
        self.lr = learning_rate
        
        # activation function - sigmoid function
        self.activation_function = lambda x: 1 / ( 1+ np.exp(-x))
        
    
    # train simple neural network
    def train(self, inputs_list, targets_list):
        
        # convert inputs list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T
        
        # calculate signals into hidden layer
        hidden_inputs = np.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # calculate signals into final output layer
        final_inputs = np.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)
        
        # output layer error is the (target - actual)
        output_errors = targets - final_outputs
        # hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden_errors = np.dot(self.who.T, output_errors)
        
        # update the weights for the links between the hidden and output layers
        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))
        
        # update the weights for the links between the input and hidden layers
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))
        
        pass
    
    # run the simple neural network
    def run(self, inputs_list):
        
        # convert inputs list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T
        
        # calculate signals into hidden layer
        hidden_inputs = np.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # calculate signals into final output layer
        final_inputs = np.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs



def start_func():
    
    # number of input, hidden and output nodes
    input_nodes = 784
    hidden_nodes = 500
    output_nodes = 10

    # learning rate
    learning_rate = 0.01

    # create instance of neural network
    n = SimpleNeuralNetwork(input_nodes,hidden_nodes,output_nodes, learning_rate)

    # Now, let's get the training data and Train the simple neural network with mnist data

    # load the mnist small 100 rows only training data CSV file into a list
    training_data_file = open("../input/mnist_train_small.csv", 'r')

    # Unzip (gunzip -c mnist_train.gz) on local before training on main mnist data set
    # training_data_file = open("mnist_dataset/mnist_train.csv", 'r')
	
    training_data_list = training_data_file.readlines()
    print ("mnist train data loaded successfully")
    training_data_file.close()
    
    # epochs is the number of times the same training data set is used to train the neural network
    epochs = 4
        
    for e in range(epochs):
        print (("Training started on {0} Epoch").format(e+1))
	    # we will go through all records in the training data set
        for record in training_data_list:
            all_values = record.split(',')

            # then scale and shift the inputs
            inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
            # let's create the target output values (all 0.01, except the desired label which is 0.99)
            targets = np.zeros(output_nodes) + 0.01

            # Do note, all_values[0] is the target label for this record
            targets[int(all_values[0])] = 0.99
            n.train(inputs, targets)
        print (("Training completed on {0} Epoch").format(e+1))

	
    # Now, lets get the test data and Test the simple neueral network with mnist data

    # load the mnist test data CSV file into a list
    test_data_file = open("../input/mnist_test.csv", 'r')
    test_data_list = test_data_file.readlines()
    test_data_file.close()


    # let's keep a scorecard, to calculate how well the network performs
    score = []

    # we will go through all the records in the test data set
    for record in test_data_list:
        # then split the record by the ',' commas
        all_values = record.split(',')

        # do note correct answer is first value
        correct_label = int(all_values[0])

        # then scale and shift the inputs
        inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01

        # then run the simple neural network on test data 
        outputs = n.run(inputs)

        # do note, the index of the highest value corresponds to the label
        label = np.argmax(outputs)

        # now, append correct or incorrect to scorecard list
        if (label == correct_label):   
            # simple neural network's answer matches the correct answer, add 1 to scorecard
            score.append(1)
        else:
            # simple neural network's answer doesn't match correct answer, add 0 to scorecard
	        score.append(0)
	        

	 
    # now, let's calculate the performance score which is the ratio of correct answers
    
    print ("performance = ", sum(score) / len(score))

    pass

if __name__ == '__main__':
    start_func()