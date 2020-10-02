import random
import numpy as np

from sklearn.decomposition import PCA

print("importing dataset...")

#import the dataset
f = open("../input/train.csv")
L = f.read().split('\n')[1:-1]
f.close()

L = list(map(lambda txt: txt.split(','), L))
L = [list(map(int, lst)) for lst in L]

#shuffle the dataset
random.shuffle(L)

print("splitting dataset...")

#spliting the dataset into a training and testing set
training_set = L[:int(.8 * len(L))]
testing_set  = L[int(.8 * len(L)):]

#separing data and labels...
training_set_data   = [L[1:] for L in training_set]
training_set_labels = [L[0]  for L in training_set]

testing_set_data   = [L[1:] for L in testing_set]
testing_set_labels = [L[0]  for L in testing_set]

print("one hot label encoding...")

def one_hot(v):
    L = [0] * 10
    L[v] = 1
    return L

#testing_set_labels  = map(one_hot, testing_set_labels)    
training_set_labels = list(map(one_hot, training_set_labels))


print("dimentionnality reduction...")

#dimentionality reduction
pca = PCA(n_components=.95)
pca.fit(training_set_data[:200])

print("transforming data...")
training_set_data = pca.transform(training_set_data)
testing_set_data  = pca.transform(testing_set_data)

#normalize...
training_set_data -= training_set_data.mean()
training_set_data /= training_set_data.std()

testing_set_data -= training_set_data.mean()
testing_set_data /= training_set_data.std()

#MLP
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dsigmoid(y):
    return y * (1. - y) #sigmoid(x) * (1. - sigmoid(x))

class NeuralNetwork:
    def __init__(self, nb_input, nb_hidden, nb_output):
        self.nb_input  = nb_input  + 1 #bias node
        self.nb_hidden = nb_hidden + 1 #bias node
        self.nb_output = nb_output
        self.activation_input  = [1.0] * self.nb_input
        self.activation_hidden = [1.0] * self.nb_hidden
        self.activation_output = [1.0] * self.nb_output
        self.input_weights  = np.random.randn(self.nb_input,  self.nb_hidden)
        self.hidden_weights = np.random.randn(self.nb_hidden, self.nb_output)
        self.change_input  = np.zeros((self.nb_input,  self.nb_hidden))
        self.change_hidden = np.zeros((self.nb_hidden, self.nb_output))

    def feed_forward(self, inputs):
        if len(inputs) != self.nb_input - 1:
            raise ValueError("bad nb of inputs ! :'(")

        self.activation_input[:-1] = inputs

        #hidden propagation
        for j in range(self.nb_hidden):
            s = 0.
            for i in range(self.nb_input):
                s += self.activation_input[i] * self.input_weights[i][j]
            self.activation_hidden[j] = sigmoid(s)

        #output propagation
        for i in range(self.nb_output):
            s = 0.
            for j in range(self.nb_hidden):
                s += self.activation_hidden[j] * self.hidden_weights[j][i]
            self.activation_output[i] = sigmoid(s)
        return self.activation_output[:]

    def back_propagate(self, targets, learning_rate):
        if len(targets) != self.nb_output:
            raise ValueError("bad nb of targets ! :'(")

        #output error
        output_delta = [0.] * self.nb_output
        for i in range(self.nb_output):
            error = -(targets[i] - self.activation_output[i])
            output_delta[i] = dsigmoid(self.activation_output[i]) * error

        #hidden error
        hidden_delta = [0.] * self.nb_hidden
        for j in range(self.nb_hidden):
            error = 0.
            for k in range(self.nb_output):
                error += output_delta[k] * self.hidden_weights[j][k]
            hidden_delta[j] = dsigmoid(self.activation_hidden[j]) * error

        #update hidden to output weights
        for j in range(self.nb_hidden):
            for k in range(self.nb_output):
                change = output_delta[k] * self.activation_hidden[j]
                self.hidden_weights[j][k] -= learning_rate * change + self.change_hidden[j][k]
                self.change_hidden[j][k] = change

        #update input to hidden weights
        for i in range(self.nb_input):
            for j in range(self.nb_hidden):
                change = hidden_delta[j] * self.activation_input[i]
                self.input_weights[i][j] -= learning_rate * change + self.change_input[i][j]
                self.change_input[i][j] = change

        #calculate error
        error = 0.
        for k in range(len(targets)):
            error += .5 * (targets[k] - self.activation_hidden[k]) ** 2
        return error

    def train(self, patterns, iterations=3000, learning_rate=.0005):
        for i in range(iterations):
            error = 0.
            
            for inputs, targets in patterns:
                self.feed_forward(inputs)
                error = self.back_propagate(targets, learning_rate)
            if i % 1 == 0:
                print('%d :\terror %-.5f' % (i, error))
        
    def predict(self, X):
        preds = []
        for p in X:
            preds.append(self.feed_forward(p))
        return preds

print("training nn...")
nn = NeuralNetwork(pca.n_components_, 10, 10)
nn.train(zip(training_set_data, training_set_labels), 50, .1)

print("testing nn...")
pred = nn.predict(testing_set_data)
pred = list(map(lambda v: v.index(max(v)), pred))
acc = sum(list(map(lambda c: c[0] == c[1], zip(testing_set_labels, pred)))) / float(len(pred))
print("accuracy : {}%".format(acc * 100))

