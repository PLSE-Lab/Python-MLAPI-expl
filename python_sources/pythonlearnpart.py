import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd  

data = [[1,2,3],[4,5,6]]  
index = ['d','e']  
columns=['a','b','c']  
df = pd.DataFrame(data=data, index=index, columns=columns)
df.drop(["a"], 1)
print(df)
#%matplotlib inline
'''
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

def load_data(data_dir):
    train_data = open(data_dir + "train.csv").read()
    train_data = train_data.split("\n")[1:-1]
    train_data = [i.split(",") for i in train_data]
    #transfer to array, x_train, y_train
    x_train = np.array([[int(i[j]) for j in range(1,len(i))] for i in train_data])
    y_train = np.array([int(i[0]) for i in train_data])
    
    test_data = open(data_dir + "test.csv").read()
    test_data = test_data.split("\n")[1:-1]
    test_data = [i.split(",") for i in test_data]
    #transfer
    x_test = np.array([[int(i[j]) for j in range(len(i))] for i in test_data])
    #return
    return x_train, y_train, x_test

data_dir = "../input/"
x_train, y_train, x_test = load_data(data_dir)
sum_square_x_test = np.square(x_test).sum(axis = 1).T
print(sum_square_x_test)

a = np.array([1,2,3])
b = np.array([1,3,1])
print(a,b)
c = np.array([[1,2,3],[4,5,6],[7,8,9]])
d = np.matrix(a).T + b + c
print(a[0])

class KNN_v0:
    def __init__(self):
        pass
    def train(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
    def comp_dist(self, x_test):
           dot = np.dot(x_test, self.x_train.T)
           sum_square_x_test = np.square(x_test).sum(axis = 1)
           sum_square_x_train = np.square(self.x_train).sum(axis = 1)
           dists = np.sqrt(-2*dot + np.matrix(sum_square_x_test) + sum_square_x_train.T)
           #return
           return dists
    def pred(self, x_test, k):
        dists = self.comp_dist(self, x_test)
        num_test = dists.shape[0]
        y_test = np.zeros(num_test)
        #predict
        for i in range(num_test):
            labels = self.y_train[np.argsort(dists[i,:])].flatten()
            y_test_closest_k = labels[:k]
            count = Counter(y_test_closest_k)
            y_test[i] = count.most_common(1)[0][0]
        #
        return y_test
#main
data_dir = "../input/"
x_train, y_train, x_test = load_data(data_dir)
print(x_train.shape, y_train.shape, x_test.shape)
print(x_test[1],reshape((28,28)))

'''
'''
classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
num_classes = len(classes)
samples = 8

y = 0
idxs = np.nonzero([i == y for i in y_train])
print(idxs)
idxs = np.random.choice(idxs[0], samples, replace=False)
print(idxs)
for i , idx in enumerate(idxs):
    plt_idx = i * num_classes + y + 1
    plt.subplot(samples, num_classes, plt_idx)
    plt.imshow(x_train[idx].reshape((28, 28)))
    plt.axis("off")
    if i == 0:
        plt.title("0")
plt.show()
'''

'''
a = "1,2,3\n4,5,6\n7,8,9"  
b = a.split("\n")
c = [i.split(",") for i in b]
print(c)

x = np.array([[int(i[j]) for j in range(0, len(i))] for i in c])
print(x)
y = int('128')
print(y)
m = np.square(x).sum(axis = 1)
print(m)
n = m.T
print(n)

a = np.array([1,2,3])
b = a.T
print(b)
dot = np.dot(a, a)
print(dot)
a = np.array([[1,2,3],[4,5,6]])
b = a.T
print(b)
# Write to the log:
a = np.array([1,1,1,2,3,4])
c = a[:2]
print(c)
count = Counter(a)
print(count)
b = count.most_common(1)[0][0]
print(b)
'''

# Any files you write to the current directory get shown as outputs

