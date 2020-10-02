# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import matplotlib.pyplot as plt
import scipy
from sklearn.metrics import accuracy_score

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
# Reading the dataset
df = pd.read_csv('/kaggle/input/sms-spam-collection-dataset/spam.csv', encoding = "latin")
df.head()

# dropping unnecessary columns and renaming
df = df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis = 1)
df = df.rename(columns = {'v1' : 'Spam/Ham', 'v2' : 'message'})
df.groupby('Spam/Ham').describe()

# Replacing spam = 1 and ham = 0
df['Spam/Ham'].replace(('spam', 'ham'), (1, 0), inplace = True)

# Taking a copy of the messages column to apply text processing
X = df['message'].copy()
y = df['Spam/Ham'].copy()
y = np.array(y)
y = y.reshape(-1, 1)

# Function for removal of punctuation marks and stopwords
def text_preprocess(text):
    text = text.translate(str.maketrans('', '', string.punctuation))    # get rid of punctuation
    text = [word for word in text.split() if word.lower() not in stopwords.words('english')]
    return " ".join(text)

# Apply to the messages column of the dataset
X = X.apply(text_preprocess)

# Stemming and normalizing message lengths
#Stemming : Reducing words to root words
def stemmer(text):
    text = text.split()
    words = ""
    for i in text:
        stemmer = SnowballStemmer('english')
        words += (stemmer.stem(i)) + " "
        
    return words

X = X.apply(stemmer)    

# getting the vectorizer
vectorizer = TfidfVectorizer("english")
X = vectorizer.fit_transform(X)
print("Shape of the vectorized matrix is : ", X.shape, "\n")

# converting to numpy array
X = scipy.sparse.csr_matrix.toarray(X)

############################################## LOGISTIC REGRESSION IMPLEMENTATION FROM SCRATCH ############################################

# Logistic/Sigmoid Function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# The cost function
def compute_cost(X, y, params):
    m = len(y)
    h = sigmoid(X @ params)
    epsilon = 1e-5
    cost = (1/m) * (((-y).T @ np.log(h + epsilon)) - ((1-y).T @ np.log(1-h + epsilon)))
    return cost

# Stochastic Gradient Descent
def gradient_descent(X, y, params, learning_rate, iterations):
    m = len(y)
    J_hist = np.zeros((iterations,1))
    
    for i in range(iterations):
        params = params - (learning_rate/m) * (X.T @ (sigmoid(X @ params) - y))
        J_hist[i] = compute_cost(X, y, params)
        if i%10 == 0:
            print("The cost after {}th iteration is: ".format(i), compute_cost(X, y, params))
        
    return (J_hist, params)

# predicting the results
def predict(X, params):
    return np.round(sigmoid(X@params))

m = len(y)    # number of training examples

# concatenating a column of 1's at initial position
X = np.hstack((np.ones((m,1)),X))

# splitting the dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

n = np.size(X, axis = 1)    # number of features
params = np.zeros((n,1))    # initializing weights

# Defining the hyperparameters:
iterations = 10000
learning_rate = 0.03

# Computing the cost initially
initial_cost = compute_cost(X_train, y_train, params)
print("Initial Cost is: {} \n".format(initial_cost))

# getting the cost matrix along with optimised parameters
(cost_history, params_optimal) = gradient_descent(X_train, y_train, params, learning_rate, iterations)
print("Optimal Parameters are: \n", params_optimal, "\n")

# Plot the learning curve
plt.figure()
plt.plot(range(len(cost_history)), cost_history, 'r')
plt.title("Convergence Graph of Cost Function")
plt.xlabel("Number of Iterations")
plt.ylabel("Cost")
plt.show()

# Make the predictions
y_pred = predict(X_test, params_optimal)

# Determine the accuracy
acc = accuracy_score(y_test, y_pred)
print("The accuracy of predictions is: {}%".format(acc*100))