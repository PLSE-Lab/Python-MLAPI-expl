# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

from sklearn.model_selection import train_test_split # split train and test data 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # data visualising 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

class LinearRegression():
    """ Apply Linear Regression
        
        Args: 
            lr : float. learning rate 
            iterations : int. Hom many iteration for training 
            
    """
    def __init__(self,lr=0.01, iterations=100):
        self.lr = lr 
        self.iterations = iterations
        
    def fit(self,x,y):
        """ Fit the our model
        
            Args: 
                x : np.array, shape = [n_samples, n_features]. Training Data 
                y : np.array, shape = [n_samples, n_conclusion]. Target Values
                
            Returns: 
                self : object
        """
        
        self.cost_list = []
        self.theta = np.zeros((x.shape[1],1))   
        self.theta_zero = np.ones((1,1))
        m = x.shape[0]   # samples in the data
        name_list=[]     # for plot x-axis name
        
        for i in range(self.iterations):  # Feed forward
            
            h_pred = np.dot(x,self.theta) + self.theta_zero    
            error = h_pred - y 
            cost = np.sum((error ** 2))
            gradient_vector = np.dot(x.T, error)
            self.theta -= (lr/m) * gradient_vector
            self.cost_list.append(cost)
            name_list.append(i)
        
        plt.plot(name_list,self.cost_list)
    
        return self
    
    def predict(self,x):
        """ Predicts the value after the model has been trained.
        
            Args: 
                x: np.array, shape = [n_samples, n_features]. Training Data
                
            Returns: 
                Predicted value 
        """
        
        return np.dot(x,self.theta) + self.theta_zero

def delete_comma_and_convert_float(df,column_name):
    """ Delete comma that in the coumn's values 
    
        Args:
            df : dataframe.
            column_name : string. 
    """
    index = df.columns.get_loc(column_name)
    for i in range(len(df[column_name])):
        value = df.iloc[i,index]
        value_list = value.split(',')
        if len(value_list) == 2:
            new_value = float(''.join(value_list)) / 10
            df.iloc[i,index] = new_value
        else:
            df.iloc[i,index] = float(value)

#----------------------------------------------------------------------------------------------------------------------------#

dataset = pd.read_csv("../input/car-consume/measurements.csv")   # read data from file 

# Prepared Data for Training #

dropped_data = dataset.drop(['temp_inside', 'specials', 'AC', 'refill liters', 'refill gas'], axis=1) # drop some features 

## Delete comma and float ##
delete_comma_and_convert_float(dropped_data, 'distance')
dropped_data['distance'] = dropped_data['distance'].astype(float)
delete_comma_and_convert_float(dropped_data, 'consume')
dropped_data['consume'] = dropped_data['consume'].astype(float)

dropped_data['gas_type'] = dropped_data['gas_type'].map({'SP98': 1, 'E10': 0})  # change 'gas_type' values to 1 and 0. String to int

dataset_x = dropped_data.drop(['consume'],axis=1)
dataset_y = dropped_data.values

# Normalization #
dataset_x = (dataset_x - np.min(dataset_x)) / (np.max(dataset_x) - np.min(dataset_x))

# Data Train and Test Split #
x_train, x_test, y_train, y_test = train_test_split(dataset_x, dataset_y, test_size= 0.2, random_state= 42)

# Train #
x_train_distance = x_train.distance.values
x_train_distance = x_train_distance.reshape((len(x_train_distance), 1))
y_train = y_train.reshape((len(y_train), 1))

LinearRegression = LinearRegression()

LinearRegression.fit(x_train_distance,y_train)









 


            
            
        
        
        
        
        
        
    

