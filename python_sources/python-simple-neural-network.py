#!/usr/bin/env python
# coding: utf-8

# # Digit Recognizer Kaggle competition
# 
# ## Introduction  
# In order to solve the problem proposed at Kaggle's Digit Recognizer competition, a simple neural network will be trained and used for predictions.  
# The objective is to correctly identify digits from a dataset of handwritten images.  
# The algorithm used belongs to the class MLPClassifier from sklearn.neural_network and trains the neural network for multiclass classification using backpropagation.  
# During the analysis, the influence of some parameters and configurations of the network in the predictions result will be tested.  

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit, cross_validate
from sklearn.pipeline import Pipeline
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#Configure plots
rc={'savefig.dpi': 75, 'figure.figsize': [12,8], 'lines.linewidth': 2.0, 'lines.markersize': 8, 'axes.labelsize': 18,   'axes.titlesize': 18, 'font.size': 18, 'legend.fontsize': 16}

sns.set(style='dark',rc=rc)


# ## Dataset  
# Two datasets were provided. One with training data and the other with test data.  
# They contain gray-scale images of hand-drawn digits, from zero through nine. Each image is 28 pixels in height and 28 pixels in width. That is a total of 784 features.
# 
# Loading the data:

# In[3]:


train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')


# In[4]:


train_data.keys()


# In[5]:


train_data.shape


# In[6]:


test_data.shape


# Checking for missing values

# In[7]:


train_data.columns[train_data.isnull().any()]


# In[8]:


test_data.columns[test_data.isnull().any()]


# There are no missing values in the train or test datasets.
# 
# Counting the number of examples of each digit in the training dataset:

# In[9]:


sns.countplot(x='label', data=train_data, color="#99c2ff")


# Split train dataset between labels and features:

# In[10]:


labels_train = train_data.iloc[:,0]
features_train = train_data.iloc[:,1:]


# In[11]:


labels_train.head()


# In[12]:


labels_train.shape


# In[13]:


features_train.shape


# ## Neural Network

# **Scaling**  
# Before building the model, the features will be scaled so that they have have mean 0 and variance 1.  
# 
# **Model and Optimization**  
# The Neural Network proposed for this problem has 1 hidden layer. The activation function is the logistic sigmoid function. And the optimization will be through the LBFGS solver.  
# These configurations will remain unchanged during this analysis.  
# The following settings will be modified and different options tested in order to verify how they influence the results:  
# The number of units in the hidden layer.  
# The regularization term (alpha).  
# The maximum number of iterations for the solver.  
# 
# **Validation**  
# The stratified shuffle split cross-validation method was applied to validate the results. 3 iterations were made. In each one, the training dataset was split in two: 70% used to train the model and 30% to test and to calculate the accuracy. The parameters alpha, max_iter and number of units in the hidden layer were selected according to the results of the cross-validation. 

# In[14]:


#cross-validation method
cv = StratifiedShuffleSplit(n_splits = 3, random_state = 0, 
                                train_size = 0.70)


# In[20]:


#Function to perform cross-validation
def nn_cross_validation(alpha, units, max_iter):
    mlp = MLPClassifier(solver='lbfgs', activation = 'logistic', random_state=0, 
                    max_iter = max_iter, alpha = alpha, hidden_layer_sizes = (units,))
    
    pipe = Pipeline([('scaling', StandardScaler()),
                 ('clf', mlp)])
    
    scores = cross_validate(pipe, features_train, labels_train, cv=cv, scoring = 'accuracy')

    return scores


# In[33]:


#Function to test the different configurations and plot the result
def test_options(alpha, units, max_iter, parameter):    
    mean_train = []
    std_train = []
    mean_test = []
    std_test = []
    option = 0
    
    if parameter == 'alpha':
        options = alpha
        var1, var2, var3 = option, units, max_iter
    elif parameter == 'max_iter':
        options = max_iter
        var1, var2, var3 = alpha, units, option
    else:
        options = units
        var1, var2, var3 = alpha, option, max_iter

    for option in options:
        
        if parameter == 'alpha':
            var1 = option
        elif parameter == 'max_iter':
            var3 = option
        else:
            var2 = option
            
        CV_scores = nn_cross_validation(var1, var2, var3)
        mean_train = np.append(mean_train, CV_scores['train_score'].mean())
        std_train = np.append(std_train, CV_scores['train_score'].std())
        mean_test = np.append(mean_test, CV_scores['test_score'].mean())
        std_test = np.append(std_test, CV_scores['test_score'].std())

    plt.figure()
    plt.errorbar(options, mean_train, std_train, linestyle='None', marker='*', c = 'b')
    plt.errorbar(options, mean_test, std_test, linestyle='None', marker='*', c = 'r') 

    plt.ylabel('Accuracy')
    plt.legend(['Train data', 'Test data'])


# The first test has different regularization terms. The following plot shows the main train accuracy, main test accuracy and the standard deviation of the three cross-validation iterations for each value of alpha.  
# The test data in this case is the 30% of the training dataset selected to test the data during each iteration of the cross-validation.

# In[32]:


alphas = [0.0001, 0.1, 1, 10, 100]

test_options(alphas, 50, 50, 'alpha')

plt.xscale('log', basex = 10)
plt.xlabel('Regularization term (alpha)')


# Adding regularization to the cost function helped to increase the test data accuracy and reduced overfitting. However a very large value for alpha results in underfitting. The value 10 will be selected for alpha for this neural network.  
# Next, different values of the maximum number of iterations will be tested.  

# In[35]:


max_iter = [50, 100, 150]

test_options(10, 50, max_iter, 'max_iter')

plt.xlabel('Maximum number of iterations')


# Increasing the maximum number of iterations makes the model prone to overfitting. In this way, one should be careful when increasing that number.  
# The parameter max_iter equal to 100 resulted in a higher accuracy of the test data. However the test for the value 150 didn't improve that accuracy very much. The value 100 will be selected.  
# 
# The next test tries different numbers of hidden layer units.  

# In[38]:


units = [50, 100, 150]

test_options(10, units, 100, 'units')

plt.xlabel('Number of units in the hidden layer')


# Increasing the number of units in the hidden layer to 100 improved the accuracy. In the case of 150, the gain wasn't very relevant. 100 units will be used in the model.

# ## Final Model and Predictions  
# 
# Based on the configurations selected through cross-validation, the final model will be trained using all the data from the training dataset. Then the test dataset will be used to make the predictions. 
# 
# Applying the standard scaler:

# In[39]:


scaler = StandardScaler()

features_transf = scaler.fit_transform(features_train)


# Training the neural network:

# In[40]:


mlp = MLPClassifier(solver='lbfgs', activation = 'logistic', random_state=0, 
                    max_iter = 100, alpha = 10, hidden_layer_sizes = (100,))

mlp.fit(features_transf, labels_train)


# In[41]:


mlp.score(features_transf, labels_train)


# The accuracy of the trained data is 0.99. That is very high and it is possible that the model is overfitted. However, that accuracy was high at all the cases tested. Improving the model with a more complex neural network or with another classification algorithm may avoid this issue and result in a better accuracy for the test dataset.  
# 
# Next, the predictions will be calculated for the test data and saved in an output file.

# In[4]:


test_data_transf = scaler.transform(test_data)
pred = mlp.predict(test_data_transf)


# Ploting the results:

# In[51]:


unique, counts = np.unique(pred, return_counts=True)
result = dict(zip(unique, counts))

plt.bar(result.keys(), result.values())


# In[77]:


output = {}

output['Label'] = pred


# In[79]:


output = pd.DataFrame(output)
output.index+=1


# In[80]:


output.to_csv('output.csv', index_label = 'ImageID')


# ## Conclusion
# 
# A simple neural network was created to recognize handwritten digits. The network configuration is described below:
# 
#  - Number of hidden layers = 1
#  - Number of hidden layer units = 100
#  - Activation function = logistic sigmoid function
#  - Solver = LBFGS
#  - Regularization term = 10
#  - Maximum number of iterations = 100
# 
# The algorithm used trains the neural network for multiclass classification using backpropagation.  
# The model built is prone to overfitting, since it presented train data accuracy above 0.99 and the test data accuracy wasn't that high during the cross-validation. In spite of that issue, the model presented very good results for the test data acurracy. The best result was above 0.96. Other methods could be applied to improve that accuracy, but depending on the application this method may offer a suitable result.  

# ## References  
# Links to references used in this project are listed below:  
# http://scikit-learn.org/stable/modules/neural_networks_supervised.html  
# http://ufldl.stanford.edu/wiki/index.php/Backpropagation_Algorithm  
# https://machinelearningmastery.com/train-final-machine-learning-model/  
# https://stackoverflow.com/questions/22481854/plot-mean-and-standard-deviation  
# https://stackoverflow.com/questions/15777951/how-to-suppress-pandas-future-warning  
# https://stackoverflow.com/questions/28663856/how-to-count-the-occurrence-of-certain-item-in-an-ndarray-in-python  
# https://stackoverflow.com/questions/20167930/start-index-at-1-when-writing-pandas-dataframe-to-csv  
