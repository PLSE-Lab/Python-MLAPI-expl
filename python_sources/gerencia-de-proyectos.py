import numpy as np
import pandas as pd
import seaborn as sb
from sympy import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def graph(mat, pairgrid=True):
    if pairgrid:
        # PairGrid with heatmaps and kde plots in the diagonal graphs
        g = sb.PairGrid(mat, diag_sharey=False)
        g.map_lower(sb.kdeplot, cmap="Blues_d")
        g.map_upper(sb.kdeplot, cmap="Blues_d")
        g.map_diag(sb.kdeplot, lw=3)
    else:
        # bar chart 
        g = sb.factorplot(x="budget", data=mat, kind="count",
                           palette="BuPu", size=6, aspect=1.5)
    
def get_eigenvectors():
    # eigenvector finding
    mat = data[['budget','price','rating']]
    vect = Matrix(mat.cov()).eigenvects()
    vectors_formatted = [[0 for _ in range(3)] for _ in range(3)]
    for i in range(3):
        for j in range(3):
            vectors_formatted[i][j] = re(np.array(vect[i][2][0]).T[0][j])
    vectors_formatted = np.array(vectors_formatted).T
    return mat

def data_mapping(data):
    # default categories
    categories = [['excellent', 3], ['very good', 2], ['satisfactory', 1], ['?', 0], ['dislike', -1]]
    prices_cat = [['under_20', 20],['20_to_30', 30],['30_to_50', 50],['over_50', 80]]
    ages_cat = [['under_19', 20],['20_34', 35],['35_49', 50],['50_64', 65], ['65_and_over', 100]]
    
    # data cleansing
    data.rating = replace_lel(data.rating, categories)
    data.price = replace_lel(data.price, prices_cat)
    data.budget = replace_lel(data.budget, prices_cat)
    data.age = replace_lel(data.age, ages_cat)
    data.cuisine_type = replace_lel(data.cuisine_type, [[i, j] for i, j in zip(set(data.cuisine_type), range(len(set(data.cuisine_type))))])
    data.gender = np.array(data.gender == 'male').astype('int')

# maps each value in the input array to a category value
def replace_lel(arr, cat):
    temp_arr = arr
    for i in cat:
        temp_arr = [i[1] if j == i[0] else j for j in temp_arr]
    return np.nan_to_num(temp_arr).astype('int')
    
def NeuralNet(x, y):
    # Randomly initialize weights
    w1 = np.random.randn(15593, 5)
    w2 = np.random.randn(5, 15593)
    
    learning_rate = 1e-6
    for t in range(5):
        # Forward pass: compute predicted y
        h = x.dot(w1.T)
        h_relu = np.maximum(h, 0)
        y_pred = h_relu.dot(w2.T)
    
        # Compute and print loss
        loss = np.square(y_pred - y).sum()
        print(t, loss)
    
        # Backprop to compute gradients of w1 and w2 with respect to loss
        grad_y_pred = 2.0 * (y_pred - y)
        grad_w2 = h_relu.T.dot(grad_y_pred)
        grad_h_relu = grad_y_pred.dot(w2.T)
        grad_h = grad_h_relu.copy()
        grad_h[h < 0] = 0
        grad_w1 = x.T.dot(grad_h)
    
        # Update weights
        print("w1:{} - grad_w1:{} - rate:{}".format(w1.shape, grad_w1.shape, learning_rate))
        w1 -= learning_rate * np.array(grad_w1[0])
        w2 -= learning_rate * np.array(grad_w2[0])
    print('--------RESULTS-----------')
    print(w1)
    print(w2)
    print("execution {}".format(t))

# load data
data = pd.read_csv("../input/Restaurant Data.csv")
# data = data.drop(['Unnamed: 6', 'Unnamed: 7', 'Unnamed: 8', 'Unnamed: 9', 'Unnamed: 10'], axis=1)

data_mapping(data)
NeuralNet(data[['age', 'gender', 'budget', 'price', 'cuisine_type']], data.rating)
# mat = get_eigenvectors(data)
# graph(mat)
