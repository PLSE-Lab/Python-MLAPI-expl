# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
# Any results you write to the current directory are saved as output.

'''
	ASSIGNMENT 6

	Team : THE DATA DIRT DEVILS
	Members : Tanya Mehrotra (01FB15ECS323)
			  Varun Ranganathan (01FB15ECS339)

'''

# Importing packages for scikit-learn

# cross_validation helps to split the samples into training and testing samples
from sklearn import cross_validation

# svm class contains the SVM classifier
from sklearn import svm

# naive_bayes class for the naive bayes classifier
from sklearn import naive_bayes

# neighbors class contains the k-nearest neighbors classifier
from sklearn import neighbors

# preprocessing helps us to normalize the vectors using the scale function
from sklearn import preprocessing

# pandas for processing the csv file
import pandas as pd

# numpy is used for manipulating with arrays
import numpy as np

outputFileObject = open("TheDataDirtDevils_Prediction.txt", "w")
outputFileObject.write("""

	ASSIGNMENT 6

	Team : THE DATA DIRT DEVILS
	Members : Tanya Mehrotra (01FB15ECS323)
			  Varun Ranganathan (01FB15ECS339)

""")

outputFileObject.write("Reading CSV Using Pandas.\n\n")

# df is dataframe from the required csv file
df = pd.read_csv('../input/Indian Liver Patient Dataset (ILPD).csv')

outputFileObject.write("Filling up null values using 'ffill' method.\n\n")

# 'alkphos' has 1 null value, fill it up using pandas 'ffill' method
df = df.fillna(method='ffill')

outputFileObject.write("Creating X and y vectors.\n\n") 

# x will contain values of independent attributes
# 	therefore we drop 'is_patient' (y attribute) from the dataframe
x = np.array(df.drop(['is_patient'], 1))

# y contains values of the class to be predicted
y = np.array(df['is_patient'])

inputX = []

# Female <- 0; Male <- 1; 
# Assignments are done as numeric values are only allowed in the input vector
for sample in x:
	temp = []
	for value in sample:
		if (value == "Female"):
			temp.append(0)
		elif (value == "Male"):
			temp.append(1)
		else:
			temp.append(value)
	inputX.append(temp)

inputY = [] 

# class 1 (no) <- 0; class 2 (yes) <- 1 
# This changes in class numbers will improve accuracy
for sample in y:
	if (sample == 1):
		inputY.append(0)
	else:
		inputY.append(1)

outputFileObject.write("Class Numbers Assigned\n")
outputFileObject.write("Female: 0,  Male: 1\n")
outputFileObject.write("No (No Liver Disease): 0,  Yes (Liver Disease Exists): 1\n\n")

outputFileObject.write("Splitting samples for training and testing\n\n")

# 10% of the samples are used as test samples
# cross_validation.train_test_split will also shuffle the samples
inputX_Train, inputX_Test, inputY_Train, inputY_Test = cross_validation.train_test_split(inputX, inputY, test_size = 0.1)

outputFileObject.write("Creating Naives Bayes classifier\n\n")

# svm classifier object is created with kernel 'rbf'
classifierNaiveBayes = naive_bayes.GaussianNB()

outputFileObject.write("Training the naive bayes classifier\n\n")

#training the naive bayes classifier
classifierNaiveBayes.fit(inputX_Train, inputY_Train)

outputFileObject.write("Testing the naive bayes classifier\n")

# score function gives the accuracy using testing samples
accuracy = classifierNaiveBayes.score(inputX_Test, inputY_Test)

outputFileObject.write("Naive Bayes Classifier: Accuracy using test samples = "+ str(accuracy) + "\n\n\n")

outputFileObject.write("Creating Naives Bayes classifier\n\n")

# creating the K Nearest Neighbors classifier with k = 13 
classifierKNN = neighbors.KNeighborsClassifier(n_neighbors = 7)

outputFileObject.write("Training the K Nearest Neighbors classifier\n\n")

#training the naive bayes classifier
classifierKNN.fit(inputX_Train, inputY_Train)

outputFileObject.write("Testing the naive bayes classifier\n")

# score function gives the accuracy using testing samples
accuracy = classifierKNN.score(inputX_Test, inputY_Test)

outputFileObject.write("K Nearest Neighbors Classifier: Accuracy using test samples = "+ str(accuracy) + "\n\n\n")

outputFileObject.write("Creating SVM classifier with 'rbf' kernel\n\n")

# svm classifier object is created with kernel 'rbf'
classifierSVM = svm.SVC(kernel = "rbf")

outputFileObject.write("Training SVM classifier\n\n")

# training the svm classifier
classifierSVM.fit(inputX_Train, inputY_Train)

outputFileObject.write("Testing SVM classifier\n")

# score function gives the accuracy using testing samples
accuracy = classifierSVM.score(inputX_Test, inputY_Test)

outputFileObject.write("SVM Classifier: Accuracy using test samples = "+ str(accuracy) + "\n\n\n")


# Using Deep Neural Networks to classify and predict

import random
import numpy

#############################################################################
#	 				  Neural Network Classes Definition.					#
#					  	Author: Varun Ranganathan							#
#																			#
#			   Implementation of a new backpropagation algorithm.			#
#############################################################################


class Neuron():

	"""
		This class defines 1 neuron
		Contains weights, and biases
		Defines functions SetWeightsAndBiases, ForwardPropagate, and BackwardPropagate
	"""

	# sets number of inputs coming into the neuron
	# sets random weights and biases to the neurons,
	# 	equal to the number of inputs.
	def __init__(self, numberOfIncoming):

		# print("Neuron.__init__ :\nnumberOfIncoming", numberOfIncoming, "\n")

		self.numberOfIncoming = numberOfIncoming
		self.weights = []
		self.biases = []

		self.__SetWeightsAndBiases__()

		return

	# called by __init__()
	def __SetWeightsAndBiases__(self):

		# print("Neuron.SetWeightsAndBiases\n")

		# _ is used when variable count won't matter
		# appends random weights to the weight list
		for _ in range(self.numberOfIncoming):
			self.weights.append(random.random())

		# appends random biases to the bias list
		for _ in range(self.numberOfIncoming):
			self.biases.append(random.random())

		return

	# for printing on the console
	def __repr__(self):
		print("Weights", "Biases")

		for i in range(self.numberOfIncoming):
			print(self.weights[i], self.biases[i])

	# for printing will using a print statement
	# returns a string to print function
	def __str__(self):
		string = "Weights\t\t\tBiases\n"

		for i in range(self.numberOfIncoming):
			string += str(self.weights[i]) + "\t" + str(self.biases[i]) + "\n"

		return string

	# feed-forward propagation
	# returns a single output
	def ForwardPropagate(self, incomingVector):

		# print("Neuron.ForwardPropagate :\nincomingVector", incomingVector, "\n")

		# check if the shape of the vector matches the number of weights and biases
		# 	raises Exception if condition is not met
		if (len(incomingVector) != self.numberOfIncoming):
			raise Exception("Incoming Vector size is not equal to required vector size.\n \
							 Incoming Vector:" + len(incomingVector) + "\n \
							 Required Vector Size:" + self.numberOfIncoming)

		output = 0

		# output is the linear combination of weights and biases.
		for i in range(self.numberOfIncoming):
			output += (incomingVector[i] * self.weights[i]) + self.biases[i]

		return output

	"""
		New Backpropagation algorithm.

		Derivation : 

			1 Input C --> Neuron (Weight w, Bias b) --> Output x but required output is x'
			Cw + b = x
			Cw' + b' = x'

			Let, w' = w - deltaW
				 b' = b - deltaB

			C(w - deltaW) + (b - deltaB) = x'
			Cw - C*deltaW + b - deltaB = x'
			Cw + b - (C*deltaW + deltaB) = x'
			C*deltaW + deltaB = x - x'

			[C 1][deltaW deltaB]' = [(x - x')]
		
			Let, A = [C 1]
				 X = [deltaW deltaB]'
				 B = [(x - x')]

			A * X = B

			A'A * X = A'B

			A'A = [C^2 C; C 1]
			A'B = [C(x - x'); (x - x')]

			X = ((A'A)^-1)(A'B)

			but A'A is not invertible

			Therefore, take the Moore-Penrose Pseudo Inverse.

			X = (A'A)+ (A'B)

			deltaW = X[0]
			deltaB = X[1]

			Now use these errors to calculate the new weight and bias.

			w' = w - (learningRate) * (deltaW)
			b' = b - (learningRate) * (deltaB)
	"""

	# numpy.linalg.pinv helps to find the Moore-Penrose Pseudo Inverse.
	# "/self.numberOfIncoming"
	# 	this means that equal weightage is given to every input coming from previous layer neurons or input.
	#	weights are randomized, therefore absolute weightage is also randomized.
	#	but relative weightage between all inputs is the same. Contributes the same amount to the output.
	def BackwardPropagate(self, outputRequired, inputGiven, learningRate):

		# print("Neuron.BackwardPropagate :\noutputRequired", outputRequired, "\ninputGiven", inputGiven, "\nlearningRate", learningRate, "\n")

		for i in range(self.numberOfIncoming):
			C = inputGiven[i]
			outputGiven = (C * self.weights[i]) + self.biases[i]
			A = [[C, 1]]
			B = [[outputGiven - (outputRequired / self.numberOfIncoming)]]
			X = numpy.dot((numpy.linalg.pinv(numpy.dot(numpy.transpose(A), A))), (numpy.dot(numpy.transpose(A), B)))
			deltaWeight = X[0][0]
			deltaBias = X[1][0]

			self.weights[i] = self.weights[i] - (deltaWeight * learningRate)
			self.biases[i] = self.biases[i] - (deltaBias * learningRate)

		return

	"""
		What should be the input vector so that the output vector may be correct?

		That is,
			If Cw + b = x
			Then what C' would satisfy the equation: C'w + b = x'

			C' = (x' - w) / b
	"""

	# This function returns a vector which will act as the required output for
	# the previous layers during backpropagation	
	def ModifiedInputVector(self, outputRequired, inputGiven):
		modifiedInputVector = []

		for i in range(len(inputGiven)):
			modifiedInputVector.append(((outputRequired - self.biases[i]) / self.weights[i]))

		return modifiedInputVector


class Layer():

	"""
		Class for a layer.
		Defines the type of layer the object would be.
		Defines the number of neurons in the layer, and initializes the neurons too.
	"""

	# Sets the type of the layer, and the number of neurons in the layer.
	# activation is not really required.
	def __init__(self, numberOfNeurons = 1, typeOfLayer = "hidden", numberOfIncoming = None, inputSize = None, activation = None):
	
		# print("Layer.__init__ :\nnumberOfNeurons", numberOfNeurons, "\ntypeOfLayer", typeOfLayer, "\nnumberOfIncoming", numberOfIncoming, "\ninputSize", inputSize, "\nactivation", activation, "\n")

		self.numberOfNeurons = numberOfNeurons
		self.activation = activation
		self.neurons = []

		self.typeOfLayer = typeOfLayer

		# if the layer is an hidden layer (default) 
		if (self.typeOfLayer == "hidden"):
			self.numberOfIncoming = numberOfIncoming
		
			# initialize the neurons
			for _ in range(self.numberOfNeurons):
				self.neurons.append(Neuron(self.numberOfIncoming))
		
		# for a input layer
		# "input" layer works like a hidden layer itself
		# 		not the conventional input layer, where each element of the vector goes through a separate neuron.
		# 		here the whole vector goes through all the neurons
		elif (self.typeOfLayer == "input"):
			self.inputSize = inputSize
			
			# initialize the neurons
			for _ in range(self.numberOfNeurons):
				self.neurons.append(Neuron(inputSize))
		
		# for a output layer
		elif (self.typeOfLayer == "output"):
			self.numberOfIncoming = numberOfIncoming
			
			# initialize the neurons
			for _ in range(self.numberOfNeurons):
				self.neurons.append(Neuron(self.numberOfIncoming))

		return

	# used to print information of the layer using print statement
	def __str__(self):
		string = "typeOfLayer: " + str(self.typeOfLayer) + "\n"
		string += "numberOfNeurons: " + str(self.numberOfNeurons) + "\n"
		if (self.typeOfLayer == "hidden"):
			string += "numberOfIncoming: " + str(self.numberOfIncoming) + "\n\n"
		elif (self.typeOfLayer == "input"):
			string += "inputSize: " + str(self.inputSize) + "\n\n"
		elif (self.typeOfLayer == "output"):
			string += "outputNeurons: " + str(self.numberOfNeurons) + "\n"
			string += "activation: " + str(self.activation) + "\n\n"

		i = 1
		for neuron in self.neurons:
			string += "Neuron " + str(i) + "\n"
			string += neuron.__str__() + "\n\n"
			i += 1

		return string

	# used to print information of the layer directly on the console
	def __repr__(self):
		print ("typeOfLayer: " + str(self.typeOfLayer) + "\n")
		print ("numberOfNeurons: " + str(self.numberOfNeurons) + "\n")
		if (self.typeOfLayer == "hidden"):
			print ("numberOfIncoming: " + str(self.numberOfIncoming) + "\n\n")
		elif (self.typeOfLayer == "input"):
			print ("inputSize: " + str(self.inputSize) + "\n\n")
		elif (self.typeOfLayer == "output"):
			print ("outputNeurons: " + str(self.numberOfNeurons) + "\n")
			print ("activation: " + str(self.activation) + "\n\n")

		i = 1
		for neuron in self.neurons:
			print ("Neuron " + str(i) + "\n")
			print (neuron.__str__() + "\n\n")
			i += 1

		return



class Network():

	"""
		This class defines the attributes and the methods for a network.
		Defines methods to train and test the network with the given dataset.
	"""

	# sets the number of layers and puts the layers passed to it in a object list.
	def __init__(self, listOfLayers):
		self.numberOfLayers = len(listOfLayers)

		# must always start with an input layer.
		# input layer has an inputSize
		if (listOfLayers[0].typeOfLayer != "input"):
			raise Exception("Always start with input layer")
		
		# checks through the layers is the shape matches.
		for i in range(1, self.numberOfLayers - 1):
			if (listOfLayers[i].numberOfIncoming != listOfLayers[i-1].numberOfNeurons):
				raise Exception("Number of Incoming values in " + i + " \
								  not equal to Number of Neurons in " + i-1)
		
		# must always end with an output layer.
		# output layer can have an activation function.
		if (listOfLayers[-1].typeOfLayer != "output"):
			raise Exception("Always end with output layer")

		self.layers = listOfLayers

		return

	# function that initializes the training process.
	# 	Calls pseudo-private function __PropagateEachLayerTrain__ for the actual training operation.
	# after one epoch cross-validation is done.
	def Train(self, trainX, trainY, validateX, validateY, learningRate = 0.0001, numberOfEpochs = 10, type = "classification"):
		self.trainErrorHistory = []
		self.testErrorHistory = []

		for epochNumber in range(numberOfEpochs):

			outputFileObject.write("For epoch "+ str(epochNumber+1) + "\n")

			for i in range(len(trainX)):
				self.__PropagateEachLayerTrain__(trainX[i], trainY[i], learningRate)

				# input("Trained for one example. Press enter key to continue\n")

			accuracy = self.Test(validateX, validateY, type = type)
			outputFileObject.write("\tValidation Accuracy = " + str(accuracy) + "\n")

			averageTrainError = self.AverageTrainError()
			outputFileObject.write("\tAverage Train Error = " + str(averageTrainError) + "\n")

			averageValidationError = self.AverageTestError()
			outputFileObject.write("\tAverage Validation Error = " + str(averageValidationError) + "\n\n")

			self.trainErrorHistory = []
			self.testErrorHistory = []

			# input("Trained for 1 epoch. Press enter key to continue\n\n")

		return

	# Does both the forward propagation and backward propagation.
	def __PropagateEachLayerTrain__(self, inputVector, outputNeeded, learningRate):
		
		# 1-off (numbering based on layer number)
		inputsToLayers = [inputVector]

		# 1-off (numbering based on layer number)
		outputsFromLayers = [None]

		for layerNumber in range(self.numberOfLayers):
			output = [self.layers[layerNumber].neurons[i].ForwardPropagate(inputsToLayers[-1]) for i in range(len(self.layers[layerNumber].neurons))]
			outputsFromLayers.append(output)
			inputsToLayers.append(output)

		outputsNeededFromLayers = [outputNeeded]

		# error is the euclidean distance between the required output vector point and the output vector point given
		self.trainErrorHistory.append(abs(numpy.linalg.norm([outputsFromLayers[-1], outputsNeededFromLayers[-1]])))

		for layerNumber in range(self.numberOfLayers-1, 0, -1):

			allModifiedInputVectors = [self.layers[layerNumber].neurons[i].ModifiedInputVector(outputsNeededFromLayers[0][i], inputsToLayers[layerNumber]) for i in range(len(self.layers[layerNumber].neurons))]

			modifiedInputVector = []

			for i in range(len(allModifiedInputVectors[0])):
				modifiedInputVector.append((sum([x[i] for x in allModifiedInputVectors]) / len(allModifiedInputVectors)))

			for i in range(len(outputsNeededFromLayers[0])):
				self.layers[layerNumber].neurons[i].BackwardPropagate(outputsNeededFromLayers[0][i], inputsToLayers[layerNumber], learningRate)

			outputsNeededFromLayers.insert(0, modifiedInputVector)

		return

	# Tests and returns accuracy of the model.
	def Test(self, testX, testY, type = "classification"):
		correct = 0
		wrong = 0

		totalErrorPercentage = 0

		for i in range(len(testX)):

			# copied portions from __PropagateEachLayerTrain__
			inputsToLayers = [testX[i]]
			outputsFromLayers = [None]

			for layerNumber in range(self.numberOfLayers):
				output = [self.layers[layerNumber].neurons[i].ForwardPropagate(inputsToLayers[-1]) for i in range(len(self.layers[layerNumber].neurons))]
				outputsFromLayers.append(output)
				inputsToLayers.append(output)

			# print("Output:", outputsFromLayers[-1])	
			# print("Required:", testY[i], "\n")

			self.testErrorHistory.append(abs(numpy.linalg.norm([outputsFromLayers[-1], testY[i]])))

			if (type == "classification"):

				# checks if the position of the max output is the equal to the position of the max output required.
				# 	i.e., if the classes match. 
				if (numpy.array(outputsFromLayers[-1]).argmax() == numpy.array(testY[i]).argmax()):
					correct += 1
				else:
					wrong += 1

			elif (type == "regression"):

				# sums up all the error percentages.

				totalErrorPercentage += self.AverageTestError()

		
		if (type == "classification"):
			return correct/(correct + wrong)
		elif (type == "regression"):
			# print(totalErrorPercentage/len(testX))
			return 1 - (totalErrorPercentage/len(testX))

	def AverageTrainError(self):
			return (sum(self.trainErrorHistory) / len(self.trainErrorHistory))

	def AverageTestError(self):
			return (sum(self.testErrorHistory) / len(self.testErrorHistory))



#############################################################################
#	 			  End of Neural Network Classes Definition.					#
#############################################################################


outputFileObject.write("Creating DNN classifier with 2 layers, with each containing 16 neurons.\nThis Neural network implements a new backpropagation algorithm.\n\n")

outputFileObject.write("Reshaping inputs to pass through the neural network\n\n")

# reshaping the inputs for the neural network
inputX = []

for sample in x:
	temp = []
	for value in sample:
		if (value == "Female"):
			temp.append(0)
		elif (value == "Male"):
			temp.append(1)
		else:
			temp.append(value)
	inputX.append(temp)

inputY = [] 

for sample in y:
	if (sample == 1):
		inputY.append([0])
	else:
		inputY.append([1])

inputX = preprocessing.scale(inputX)
inputY = preprocessing.scale(inputY)

inputX_Train, inputX_Test, inputY_Train, inputY_Test = cross_validation.train_test_split(inputX, inputY, test_size = 0.1)


layer0 = Layer(numberOfNeurons = 16, typeOfLayer = "input", inputSize = len(inputX_Train[0]))
layer1 = Layer(numberOfNeurons = 16, typeOfLayer = "hidden", numberOfIncoming = layer0.numberOfNeurons)
layer2 = Layer(numberOfNeurons = 1, typeOfLayer = "output", numberOfIncoming = layer1.numberOfNeurons, activation = "step")

network = Network([layer0, layer1, layer2])

outputFileObject.write("Training DNN for 30 epochs and 0.0001 as learning rate\n\n")
network.Train(inputX_Train, inputY_Train, inputX_Test, inputY_Test, numberOfEpochs = 30, learningRate = 0.0001)

outputFileObject.write("Testing DNN\n\n")
accuracy = network.Test(inputX_Test, inputY_Test)

outputFileObject.write("DNN Classifier: Accuracy using test samples = "+ str(accuracy) + "\n\n\n")

outputFileObject.write("We would choose to use a neural network to train our classifier as it will give us the best fit possible various linear and non-linear functions.\n")
