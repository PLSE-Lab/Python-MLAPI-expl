#!/usr/bin/env python
# coding: utf-8

# <h1><center> BANK CUSTOMER CHURN PREDICTION </center></h1><br>
# <b>Customer churn </b> can be defined as the customer terminates any relationship with a company that provides services either online or offline. Churn prediction can be referred to as the prediction of customers who are likely to cancel a subscription, product or service. <br>
# <br>
# <h3><b>:::::Importing following libraries:::::</b></h3>
# <ol>
#     <li><b>Pandas::</b> For the data manipulation and analysis.</li>
#     <li><b>Matplotlib::</b> For the data visualization.</li>
#     <li><b>Keras::</b> For building the neural network built on the Tensorflow backend.</li>
#     <li><b>Warning::</b> For dealing with the warnings coming while execution of the lines of code.</li>
# </ol>

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import os
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout


# In[ ]:


print(os.listdir("../input"))


# <h3><b>::::: Importing dataset:::::</b></h3><br>
# The data is imported using the pandas (alias name pd) pre-defined function read_csv() as our data file format is csv (comma-seprated values) in the dataset variable.

# In[ ]:


dataset = pd.read_csv('../input/churn-modelling/Churn_Modelling.csv')


# <h3><b>:::::The snapshot of the imported data:::::</b></h3><br>
# As the dataset variable of DATA_FRAME type so we can use the head() function to show the top 5 rows/tuples of the whole dataset.<br>
# <h3><b>:::::The featuers and Class_Label of the data:::::</b></h3><br>
# <ol>
#     <li><b>RowNumber:</b> The index number of the row.</li>
#     <li><b>CusomerId:</b> The customer ID. </li>
#     <li><b>Surname:</b> The last name of the customer.</li>
#     <li><b>CreditScore:</b> The credit score given by the bank.</li>
#     <li><b>Geography:</b> Country that customer belongs.\begin{equation}
# Geography \: \: \epsilon \: \:  R^{\{France,Germany,Spain\}}
# \end{equation}</li>
#     <li><b>Gender:</b> The gender of the customer. \begin{equation}
# Gender \: \: \epsilon \: \:  R^{\{Male\:,\:Female\}}
# \end{equation}</li>
#     <li><b>Age:</b>The age of the customer.</li>
#     <li><b>Tenure:</b>Number of years customer is with the bank.</li>
#     <li><b>Balance:</b>The current balance of the account.</li>
#     <li><b>NumOfProducts:</b>The number of the products taken by the customer.\begin{equation}
# NumOfProducts \: \: \epsilon \: \:  R^{\{1\:,\:2\:,\:3\:,\:4\:\}}
# \end{equation}</li>
#     <li><b>HasCrCard:</b> Is customer owing a credit card or not. \begin{equation}
# HasCrCard \: \: \epsilon \: \:  R^{\{\:0\: = \:No\:,\: 1\: =\: Yes\:\}}
# \end{equation}</li>
#     <li><b>IsActiveMember:</b>Is customer is active or not.\begin{equation}
# IsActiveMember \: \: \epsilon \: \:  R^{\{\:0\: = \:No\:,\: 1\: =\: Yes\:\}}
# \end{equation}</li>
#     <li><b>EstimatedSalary:</b>The annual salary of the customers.</li>
#     <li><b>Exited:</b>The <b>CLASS LABEL</b> whether customer still with bank or not.\begin{equation}
# Exited \: \: \epsilon \: \:  R^{\{\:0\: = \:No\:,\: 1\: =\: Yes\:\}}
# \end{equation}</li>
# </ol>

# In[ ]:


dataset.head()


# <h3><b>:::::Dropping the unsignificant featuers:::::</b></h3><br>
# The function dataset_name.drop(["LIST_OF_FEATUERS"], axis = 0/1) will drop the columns (when axis=1) and rows (when axis=0).<br><br>
# <b>The data divison into the dataset featuers and class label.</b>

# In[ ]:


X = dataset.iloc[:, 3:13].values


# In[ ]:


y = dataset.iloc[:, 13].values


# In[ ]:


print(X[1])
print(y)


# <h3><b>:::::The shape of the feature dataset:::::</b></h3>

# In[ ]:


X.shape


# <h3><b>:::::The shape of the class label dataset:::::</b></h3>

# In[ ]:


y.shape


# <h3><b>:::::The conversion of the categorical features into numerical using the One-Hot coding Technique.:::::</b></h3><br>
# The conversion of the categorical featuers into numerical featuers using the one-hot encoding technique in which each unique value in the feature will be converted into a seperate column.

# In[ ]:


labelencoder_X_1 = LabelEncoder()


# In[ ]:


X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])


# In[ ]:


labelencoder_X_2 = LabelEncoder()


# In[ ]:


X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])


# In[ ]:


onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]


# In[ ]:


X[1]


# <h3<<b>:::::The data division into training and testing:::::</b></h3><br>
# The data is divided into training and testing into 80-20 ratio.

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[ ]:


X_train[1:5]


# In[ ]:


y_train[1:5]


# In[ ]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# <h3><b>::::: The Keras Sequential Model:::::</b></h3><br>
# The Sequential model is a linear stack of layers.

# In[ ]:


classifier = Sequential()


# <h3><b>::::: Initializing the weights and bias for the neural network model:::::</b></h3><br>
# We have taken the initial vector values from normal/gaussian distribution (mean = 0 and standard deviation = 0.05) for the neural network weights and bias.

# In[ ]:


initializers = keras.initializers.RandomNormal(mean=0.0, stddev=1.0, seed=None)


# <h3><b>:::::The Input Layer:::::</b></h3>
# <ol>
#     <li><b>The Neurons:</b> 11</li>
#     <li><b>The weight matrix initial value:</b> Normal Distribution</li>
#     <li><b>The Bias initial values:</b> Normal Distribution</li>
#     <li><b>Activation Function:</b> Rectified Linear Unit (ReLU)</li>
# </ol>

# In[ ]:


classifier.add(Dense(units = 11, kernel_initializer=initializers ,bias_initializer=initializers, activation = 'relu'))


# In[ ]:


classifier.add(Dropout(0.5))


# <h3><b>:::::The Hidden Layer (L1):::::</b></h3>
# <ol>
#     <li><b>Neurons:</b> 8</li>
#     <li><b>The weight matrix initial value:</b> Normal Distribution</li>
#     <li><b>The Bias initial values:</b> Normal Distribution</li>
#     <li><b>Activation Function:</b> Rectified Linear Unit (ReLU)</li>
# </ol>

# In[ ]:


classifier.add(Dense( units = 8, kernel_initializer=initializers ,bias_initializer=initializers, activation = 'relu'))


# <h3><b>:::::The Hidden Layer (L2):::::</b></h3>
# <ol>
#     <li><b>Neurons:</b> 6</li>
#     <li><b>The weight matrix initial value:</b> Normal Distribution</li>
#     <li><b>The Bias initial values:</b> Normal Distribution</li>
#     <li><b>Activation Function:</b> Rectified Linear Unit (ReLU)</li>
# </ol>

# In[ ]:


classifier.add(Dense(units = 6, kernel_initializer=initializers ,bias_initializer=initializers, activation = 'relu'))


# <h3><b>:::::The Hidden Layer (L3):::::</b></h3>
# <ol>
#     <li><b>Neurons:</b> 4</li>
#     <li><b>The weight matrix initial value:</b> Normal Distribution</li>
#     <li><b>The Bias initial values:</b> Normal Distribution</li>
#     <li><b>Activation Function:</b> Rectified Linear Unit (ReLU)</li>
# </ol>

# In[ ]:


classifier.add(Dense(units = 4, kernel_initializer=initializers ,bias_initializer=initializers, activation = 'relu'))


# <h3><b>:::::The Output Layer:::::</b></h3>
# <ol>
#     <li><b>Neurons:</b> 1</li>
#     <li><b>The weight matrix initial value:</b> Normal Distribution</li>
#     <li><b>The Bias initial values:</b> Normal Distribution</li>
#     <li><b>Activation Function:</b> Sigmoid</li>
# </ol>

# In[ ]:


classifier.add(Dense(units = 1, kernel_initializer=initializers ,bias_initializer=initializers, activation = 'sigmoid'))


#  <h3><b>:::::The Weight's Optimizer:::::</b></h3><br>
#  Adam is an optimization algorithm that can used instead of the classical stochastic gradient descent procedure to update network weights iterative based in training data.<br>
#  The method computes individual adaptive learning rates for different parameters from estimates of first and second moments of the gradients.<br>
# <ol>
#     <li><b>Learning Rate (0.001):</b> The proportion that weights are updated (e.g. 0.001). Larger values (e.g. 0.3) results in faster initial learning before the rate is updated. Smaller values (e.g. 1.0E-5) slow learning right down during training</li>
#     <li><b>beta_1(0.9):</b> The exponential decay rate for the first moment estimates (e.g. 0.9).</li>
#     <li><b>beta_2(0.999):</b> The exponential decay rate for the second-moment estimates (e.g. 0.999). This value should be set close to 1.0 on problems with a sparse gradient (e.g. NLP and computer vision problems).</li>
# </ol>

# In[ ]:


opti = keras.optimizers.Adam(lr = 0.0001, beta_1 = 0.9, beta_2 = 0.999, amsgrad=False)


# <h3><b>:::::The Fiting Of The Neural Network:::::</b></h3><br>
# This method of the keras library will fit the all the layers of the neural network.
# <ol>
#     <li><b>Optimizer:</b> Adam Optimizer</li>
#     <li><b>Loss Function:</b> Binary Cross Entropy</li>
#     <li><b>Metrics:</b> Accuracy</li>
# </ol>

# In[ ]:


classifier.compile(optimizer = opti, loss = 'binary_crossentropy', metrics = ['accuracy'])


# <h3><b>:::::Training of the Neural Network</b></h3>
# <ol>
#     <li><b>The Batch Size:</b> 128</li>
#     <li><b>Number of Epochs:</b>10000</li>
# </ol>

# In[ ]:


history = classifier.fit(X_train, y_train, validation_split=0.10, batch_size = 32, epochs = 10000)


# In[ ]:


# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


# <h3><b>:::::The Testing of the Neural Network:::::</b></h3><br>
# We are testing the model with 2000 testing sample extracted randomly from the dataset during the spliting of dataset.

# In[ ]:


y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
print(y_pred)


# <h3><b>:::::The Evaluation of the Model:::::</b></h3><br>
# As, we are doing the binary classification so we have used the confusion matrix for the evaluation.<br>
# A <b>confusion matrix</b>, in predictive analytics, is a two-by-two table that tells us the rate of false positives, false negatives, true positives and true negatives for a test or predictor. 

# In[ ]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)


# <h3><b>::::Various Measures Calculated using the Confusion Matrix:::::</b></h3>
# <ol>
#     <li><b>Accuracy</b></li>
#     <li><b>Recall:</b> Recall can be defined as the ratio of the total number of correctly classified positive examples divide to the total number of positive examples. High Recall indicates the class is correctly recognized (small number of FN).</li>
#     <li><b>Precision:</b> Precision is calculated by the division of the total number of correctly classified positive examples by the total number of predicted positive examples. High Precision indicates an example labeled as positive is indeed positive (small number of FP).</li>
#     <li><b>F-Measure:</b> F-measure which uses Harmonic Mean in place of Arithmetic Mean as it punishes the extreme values more. The F-Measure will always be nearer to the smaller value of Precision or Recall.</li>
# </ol>

# In[ ]:


total_test_sample = 2000
Accuracy = ((cm[0][0]+cm[1][1])/total_test_sample)*100
Recall = (cm[0][0]/(cm[0][0]+cm[1][0]))*100
Precision = (cm[0][0]/(cm[0][0]+cm[0][1]))*100


# In[ ]:


Recall_1 = (cm[0][0]/(cm[0][0]+cm[1][0]))
Precision_1 = (cm[0][0]/(cm[0][0]+cm[0][1]))


# In[ ]:


F = (Recall_1+Precision_1)/(2*Recall_1*Precision_1)


# In[ ]:


print("********** CONFUSION MATRIX MEASURES**********")
print("The accuracy is:::::",Accuracy,"%")
print("\n")
print("**********************************************")
print("The Recall is:::::", Recall,"%")
print("\n")
print("**********************************************")
print("The Precision is:::::",Precision,"%")
print("\n")
print("**********************************************")
print("The F-Measure is:::::",F)

