# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import csv

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# from subprocess import check_output
# print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

# help functions
def read_data(file):
  x = []
  y = []
  with open(file, "r") as f:
    data = csv.reader(f, delimiter=",")
    next(data, None) # skip the header
    for row in data:
      if len(row) < 2:
        continue
      x.append((row[0]))
      y.append((row[1]))
  return x, y

# return k, b
def step_gradient(current_k, current_b, x, y, learning_rate):
  delta_k = 0
  delta_b = 0
  N = len(y)
  for i in xrange(0, N):
    xi = float(x[i])
    yi = float(y[i])
    delta = yi - ((current_k * xi) + current_b)
    delta_k -= (2.0/N) * xi * delta
    delta_b -= (2.0/N) * delta

  k = current_k - (learning_rate * delta_k)
  b = current_b - (learning_rate * delta_b)
  return k, b

# return k, b
def gradient_descent(x, y, initial_k, initial_b, learning_rate, num_interations):
  k = initial_k
  b = initial_b
  for i in xrange(0, num_interations):
    k, b = step_gradient(k, b, x, y, learning_rate)
  return k, b

# Read train/test data
xtrain, ytrain = read_data("../input/train.csv")
xtest, ytest = read_data("../input/test.csv")

# Plot train data
#plt.scatter(xtrain, ytrain)
#plt.xlabel('x')
#plt.ylabel('y')
#plt.show()

# Get the coef and b
# learning_rate should be small.
learning_rate = 0.0001
initial_k = 0
initial_b = 0
num_interations = 3000
k, b = gradient_descent(xtrain, ytrain, initial_k, initial_b, learning_rate, num_interations)

# Pridict the test data
ypredict = []
for i in xrange(0, len(xtest)):
  yi = k * float(xtest[i]) + b
  ypredict.append(yi)

# Plot with test data and the predict linear line.
plt.scatter(xtest, ytest)
plt.plot(xtest, ypredict, color = "#ff7f0e")
plt.xlabel('x')
plt.ylabel('y')
plt.show()


