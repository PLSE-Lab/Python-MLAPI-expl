#!/usr/bin/env python
# coding: utf-8

# <h1>guessing the color of a dot by using a given set of known dots</h1>
# the ideas that I tried out:
# <ol>
#     <li>solving the issue with a linear function</li>
#     <li>solving the issue by using the color of the closest known dot</li>
#     <li>solving the issue with a single Perceptron</li>
#     <li>solving the issue with the MLP of a library</li>
#     <li>just assuming the same color for all of the dots</li>
# </ol>

# reads in a given file and returns data sets<br>
# in this case the file that contains the training data<br>
# @return list containing all datasets in the file

# In[108]:


def read_data(filename):
    f = open(filename)
    data_line = False
    data = []
    for l in f:
        l = l.strip() # get rid of newline at the end of each input line
        if data_line:
            content = [float(x) for x in l.split(',')]
            if len(content) == 3:
                data.append(content)
        else:
            if l.startswith('@DATA'):
                data_line = True
    return data


# saves the training data set in a variable<br>
# training_data is a 2 dimensional list<br>
# first index is used to select the data<br>
# second index is used to select the value (0 = X, 1 = Y, 2 = Color)

# In[109]:


# import numpy as np
training_data = read_data("../input/train.arff")
# print(training_data)
# print(np.array(training_data).shape)
# import pandas as pd

# test = pd.read_csv("../input/test.csv")
# test = test.drop(labels=["Id"], axis=1)

test = read_data("../input/eval.arff")
new_test = []
correct_result = []
for data in test:
    new_test.append([data[0], data[1]])
    correct_result.append(data[2])
test = new_test
del new_test


# In[110]:


def compare_with_true_result(values):
    i = 0
    correct = 0
    while i < len(correct_result):
        if correct_result[i] == values[i]:
            correct += 1
        i += 1
    return correct / len(correct_result)


# # 1. solving the task with a linear function
# here I assume that the shown image of the example datas in the diagram are the here given training datas<br>
# I created a really simple linear function where all of the dots that are above get one color and the ones below get the other color

# In[111]:


#linear function
def f(x):
    return -x

def __get_guess(x, y):
    return 1 if y >= f(x) else -1

def get_guess(values):
    return __get_guess(values[0], values[1])


# In[112]:


# import csv

# with open('1-linear_function.csv', 'w', newline='') as test_submission:
#     writer = csv.writer(test_submission)
#     writer.writerow(['Id (String)', 'Category (String)'])
#     for i in range(400):
#         unknown_dot = test[i]
#         writer.writerow([i, get_guess(unknown_dot)])
        
# test_submission.close()
# print("done")
my_result = []
for i in range(400):
    unknown_dot = test[i]
    my_result.append(get_guess(unknown_dot))
print("1. solving the task with a linear function: %s" % compare_with_true_result(my_result))


# # 2. solving the issue by using the color of the closest known dot
# I didn't want to just go through all datas of the training set per unknown dot (it would have taken an exponantial amount of time) so I decide to go for an approximation<br><br>
# I am only checking all datas of the training set which are in a n by n square where the unknown dot is the center<br>
# if there is no known dot in that range then I try again with an increased range until I atleast find one dot<br>
# (I don't add the unknown dots to the known dots after assuming their color)

# <hr>quick sort (I copied the algorithm from https://stackoverflow.com/a/27461889 and changed it a little bit so that it works in this case)<br>
# pos = 0 => sort by X<br>
# pos = 1 => sort by Y

# In[113]:


def partition(array, pos, begin, end):
    pivot = begin
    for i in range(begin+1, end+1):
        if array[i][pos] <= array[begin][pos]:  # changed this line so that it sorts by X or Y value
            pivot += 1
            array[i], array[pivot] = array[pivot], array[i]
    array[pivot], array[begin] = array[begin], array[pivot]
    return pivot


def quicksort(array, pos, begin=0, end=None):
    if end is None:
        end = len(array) - 1
    def _quicksort(array, pos, begin, end):
        if begin >= end:
            return
        pivot = partition(array, pos, begin, end)
        _quicksort(array, pos, begin, pivot-1)
        _quicksort(array, pos, pivot+1, end)
    return _quicksort(array, pos, begin, end)


# In[114]:


# print(training_data)
# training_data_sorted_by_x = training_data
# quicksort(training_data_sorted_by_x, 0)
# print("---------------------------")
# print(training_data_sorted_by_x)


# In[126]:


def get_closest_color(unknown_dot, square_size):
    no_dots_in_range = True
    current_square_size = 0
    i = 0 #  adsadasdasdsadasdasdasd
    training_data_only_in_range = []
    while no_dots_in_range:
        current_square_size += square_size
        training_data_only_in_range = remove_unneeded_X_and_Y_data(unknown_dot, current_square_size)
        no_dots_in_range = not training_data_only_in_range  # checks if the list is empty
        if no_dots_in_range:
            print("no dots in range %s" % i)
            i += 1
    return training_data_only_in_range[index_of_closest_dot(unknown_dot, training_data_only_in_range)][2]
    
def remove_unneeded_X_and_Y_data(unknown_dot, square_size):
    # X
    training_data_sorted_by_x = training_data
    quicksort(training_data_sorted_by_x, 0)
    index = get_first_and_last_index_in_range(training_data_sorted_by_x, unknown_dot, square_size, 0)
    first_index_in_range = index[0]
    first_index_after_range = index[1]
    # print(first_index_in_range, first_index_in_range - first_index_after_range)
    training_data_with_x_in_range = training_data[first_index_in_range : first_index_after_range]
    # Y
    training_data_sorted_by_y = training_data
    quicksort(training_data_sorted_by_y, 1)
    index = get_first_and_last_index_in_range(training_data_sorted_by_y, unknown_dot, square_size, 1)
    first_index_in_range = index[0]
    first_index_after_range = index[1]
    training_data_with_x_and_y_in_range = training_data[first_index_in_range : first_index_after_range]
    # return train[first_index_in_range : first_index_after_range]
    return training_data_with_x_and_y_in_range

# pos == 0 => X
# pos == 1 => Y
def get_first_and_last_index_in_range(data, unknown_dot, square_size, pos):
    first_index_in_range = -1
    first_index_after_range = -1
    i = 0
    while i < len(training_data) and first_index_after_range == -1:
        if first_index_in_range == -1 and data[i][pos] >= unknown_dot[pos] - square_size:
            first_index_in_range = i
        elif first_index_after_range == -1 and data[i][pos] > unknown_dot[pos] + square_size:
            first_index_after_range = i
        i += 1
    if first_index_in_range == -1:
        print("first in range")
    if first_index_after_range == -1:
        print("first after range")
        first_index_after_range = len(data) - 1
    return [first_index_in_range, first_index_after_range]

def index_of_closest_dot(unknown_dot, training_data):
    distance = 999999
    index = -1
    for i in range(len(training_data)):
        # to get the actual distance you would need to take the square root of it but that doesn't matter
        # because we never take the square root so it is consistant
        new_distance = (training_data[i][0] - unknown_dot[0]) ** 2 + (training_data[i][1] - unknown_dot[1]) ** 2
        if new_distance < distance:
            # print(distance, new_distance)
            distance = new_distance
            index = i
    # if index == -1:  # should be impossible to be true but this is just to see it if there should be a mistake
    #     print("ERROR in index_of_closest_dot")
    return index


# In[116]:


# import numpy as np
# unknown_dot = [1, 2]
# limited_by_X = get_closest_color(unknown_dot, 1)
# print(np.asarray(limited_by_X).shape)


# In[127]:


# import csv

# with open('2-closest_known.csv', 'w', newline='') as test_submission:
#     writer = csv.writer(test_submission)
#     writer.writerow(['Id (String)', 'Category (String)'])
#     for i in range(400):
#         unknown_dot = test[i]
#         writer.writerow([i, get_closest_color(unknown_dot, 0.1)])  # 0.084651 is the minimal value that would work without a retry
        
# test_submission.close()
# print("done")
my_result = []
for i in range(400):
    unknown_dot = test[i]
    my_result.append(get_closest_color(unknown_dot, 0.1))  # 0.084651 is the minimal value that would work without a retry
print("2. assuming same color as closest dot: %s" % compare_with_true_result(my_result))


# # 3. solving the issue with a single Perceptron
# I used the perceptron that I created for the logical operators<br>
# it is poorly written because I somehow didn't think of the possibility to get a value for inputs which weren't part of the training data so I had to add that option in afterwards

# In[118]:


import random
def sign(value):
    if value >= 0:
        value = 1
    else:
        value = -1
    return value

class PerceptronIO:
    def __init__(self, inputs, correctValue):
        self.inputs = inputs + [1]  # +1 for the bias
        self.correctValue = correctValue

class Perceptron:
    def __init__(self, perceptronIOs, learningRate, amountOfTries):
        self.perceptronIOs = perceptronIOs
        self.weights = list()
        self.generateWeights()
        self.learningRate = learningRate
        self.amountOfTries = amountOfTries

    def generateWeights(self):  # 1 per input
        for i in range(len(self.perceptronIOs[0].inputs)):
            self.weights.append(random.randrange(-100, 100, 1) / 100)

    def sum(self, perceptronIO):  # 1 per weight
        sum = 0
        for i in range(len(self.weights)):
            sum += self.weights[i] * perceptronIO.inputs[i]
        return sum

    def sums(self):  # 1 per IO
        sums = list()
        for i in range(len(self.perceptronIOs)):
            sums.append(self.sum(self.perceptronIOs[i]))
        return sums
    
    def guesses(self):  # 1 per IO
        guesses = list()
        sums = self.sums()
        for i in range(len(sums)):
            guesses.append(sign(sums[i]))
        return guesses

   # def error(self):
   #     return correctValue - guess
    def errors(self):  # 1 per IO
        errors = list()
        guesses = self.guesses()
        for i in range(len(self.perceptronIOs)):
            errors.append(self.perceptronIOs[i].correctValue - guesses[i])
#            print("error(%s): correct value - %s , guess - %s" % (i, self.perceptronIOs[i].correctValue, guesses[i]))  # testing error calculation
        # print(errors)
        return errors  # [errors, guesses]

    def adjustWeights(self):  # returns whether the weightsChanged or not
        errors = self.errors()
        #create a copy of the oldWeights to adjust them without manipulating the values for the calculations
        newWeights = list()
        for i in range(len(self.weights)):
            newWeights.append(self.weights[i])

        for i in range(len(self.perceptronIOs)):  # weight += error * input * learning rate
            for c in range(len(self.weights)):
                newWeights[c] += errors[i] * self.perceptronIOs[i].inputs[c] * self.learningRate
        weightsChanged = self.weights != newWeights
        self.weights = newWeights
        return weightsChanged

    def learn(self):
        i = 0
        while self.adjustWeights() and i < self.amountOfTries:
    #        print(i)
            i += 1
        print("done training")
    
    # def learn(self):
    #     for i in range(self.amountOfTries):
    #         print(i + 1, end = ". - ")
    #         if not self.adjustWeights():
    #             break
                
    #output for a specific case
    def specificGuess(self, x, y):
        return sign(self.sum(PerceptronIO([x, y], -1)))


# training the perceptron

# In[119]:


perceptron_inputs = []
for data in training_data:
    perceptron_inputs.append(PerceptronIO([data[0], data[1]], data[2]))
# perceptronInputs = [PerceptronIO([1, 1], -1), PerceptronIO([1, 0], 1), PerceptronIO([0, 1], 1), PerceptronIO([0, 0], 1)]
learning_rate = 1 / len(perceptron_inputs)
amount_of_tries = 100
perceptron = Perceptron(perceptron_inputs, learning_rate, amount_of_tries)
perceptron.learn()


# In[120]:


# import csv

# with open('3-my_perceptron.csv', 'w', newline='') as test_submission:
#     writer = csv.writer(test_submission)
#     writer.writerow(['Id (String)', 'Category (String)'])
#     for i in range(400):
        # unknown_dot = test.loc[i, :].values
#         unknown_dot = test[i]
#         writer.writerow([i, perceptron.specificGuess(unknown_dot[0], unknown_dot[1])])
        
# test_submission.close()
# print("done")
my_result = []
for i in range(400):
    unknown_dot = test[i]
    my_result.append(perceptron.specificGuess(unknown_dot[0], unknown_dot[1]))
print("3. single perceptron: %s" % compare_with_true_result(my_result))


# # 4. solving the issue with the MLP of a library
# here I used the MLP of sklearn to solve the issue

# In[121]:


from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier()
training_data_inputs = []
training_data_outputs = []
for data in training_data:
    training_data_inputs.append([data[0], data[1]])
    training_data_outputs.append(data[2])
    
mlp.fit(training_data_inputs, training_data_outputs)
print("done")


# In[122]:


# import csv

# with open('4-library.csv', 'w', newline='') as test_submission:
#     writer = csv.writer(test_submission)
#     writer.writerow(['Id (String)', 'Category (String)'])
#     results = mlp.predict(test)  
#     for i in range(400):
#         writer.writerow([i, results[i]])
# test_submission.close()
# print("done")
my_result = []
results = mlp.predict(test)  
for i in range(400):
    my_result.append(results[i])
print("4. MLP of a library: %s" % compare_with_true_result(my_result))


# # 5. just assuming the same color for all of the dots

# In[123]:


# import csv

# with open('5-all_same_color.csv', 'w', newline='') as test_submission:
#     writer = csv.writer(test_submission)
#     writer.writerow(['Id (String)', 'Category (String)'])
#     for i in range(400):
#         writer.writerow([i, -1])
        
# test_submission.close()
# print("done")
my_result = []
for i in range(400):
    my_result.append(-1)
print("5. assuming the same color for all the dots: %s" % compare_with_true_result(my_result))

