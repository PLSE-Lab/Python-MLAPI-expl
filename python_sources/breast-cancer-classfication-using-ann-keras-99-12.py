#!/usr/bin/env python
# coding: utf-8

# ### Introduction:
# Using **ANN (Artificial Neural Network) using Keras** to classify **tumors into Malignant or Benign type**, when provided with the **tumor's dimensions**. In the output we will have probability of** tumor of belonging to either Malignant or Benign class.**   
# 
# **3 parts**  
# 1. Data pre-processing and quick analysis
# 2.  Building ANN
# 3.  Making Predictions So let's get started!

# # 1.Preprocessing Dataset

# In[25]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('../input/data.csv')


# In[26]:


def dataSetAnalysis(df):
    #view starting values of data set
    print("Dataset Head")
    print(df.head(3))
    print("=" * 100)
    
    # View features in data set
    print("Dataset Features")
    print(df.columns.values)
    print("=" * 100)
    
    # View How many samples and how many missing values for each feature
    print("Dataset Features Details")
    print(df.info())
    print("=" * 100)
    
    # view distribution of numerical features across the data set
    print("Dataset Numerical Features")
    print(df.describe())
    print("=" * 100)
    
    # view distribution of categorical features across the data set
    print("Dataset Categorical Features")
    print(df.describe(include=['O']))
    print("=" * 100)


# In[27]:


dataSetAnalysis(dataset)


# So we have a total of 33 columns, the last column "Unnamed: 32" contains all the null values, so we will exclude it. In addition to our diagnosis, we will not include the 'id' in our training set because it has no effect on the classification. So, we started with 30 features that are all types of float64 and do not include missing values. Cool! Now, let's separate the features and labels.

# In[28]:


X = dataset.iloc[:,2:32] # [all rows, col from index 2 to the last one excluding 'Unnamed: 32']
y = dataset.iloc[:,1] # [all rows, col one only which contains the classes of cancer]


# Notice that **'diagnosis'** contains **'M' or 'B'** to represent **Malignant or Benign tumor**. Let's encode them to **0** and **1.**

# In[29]:


from sklearn.preprocessing import LabelEncoder

print("Before encoding: ")
print(y[100:110])

labelencoder_Y = LabelEncoder()
y = labelencoder_Y.fit_transform(y)

print("\nAfter encoding: ")
print(y[100:110])


# **Splitting Dataset**  
# 
# Now let's split our data into training and testing datasets.

# In[30]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# **Features Scaling**  
# 
# Now let's apply features scaling. Scaling ensures that just because some features are big, the model won't lead to using them as a main predictor [(Read more)](https://stackoverflow.com/questions/26225344/why-feature-scaling)

# In[31]:


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# ### 2. Preparing ANN
# 
# ### Importing Keras and initialising ANN
# 
# 
# ### Building the layers
# Let's build the Layers. We can play around and change number of units but if we are not sure what number to initialize with then simply initialize the units of all layers except the last one with the (number of features + number of output nodes)/2 which equals to 15 in our case. My results were imporved by setting units = 16 for the first layer and decreasing the units in the hidden layers. Also we have to provide input dimension for the first layer only. 'relu' reffers to rectified linear unit and sigmoid reffers to sigmoid activation function. With the help of sigmoid activation function, we can get the probabilities of the classification which might be benificial in some cases to conduct further research.
# 
# 
# ### Tuning Hyper parameters
# Let's first find the hyper parameters using which model can give more accurate predictions. Here I'll tune batch_size, epochs and optimizer. This will take some time to run so sit back and relax.

# In[32]:


from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential() # Initialising the ANN

classifier.add(Dense(units = 16, kernel_initializer = 'uniform', activation = 'relu', input_dim = 30))
classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))


# ### Compiling ANN
# **Compiling classifier.** Using adam optimizer. Using **binary_crossentropy for loss function** since classification is binary, i.e. only two classes **'M' or 'B'.**

# In[33]:


classifier.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])


# ### Fitting the data
# Now let's fit the data. I trained it with batch size of 1 and 100 epochs and training accuracy was 99.12% and final predictions were 96.49% accurate.

# In[63]:


classifier.fit(X_train, y_train, batch_size = 128, epochs = 15, verbose = 2)


# ### Saving/Loading the model

# In[70]:


from keras.models import load_model

classifier.save('breast_cancer_model.h5') #Save trained ANN
#classifier = load_model('breast_cancer_model.h5')  #Load trained ANN


# ###  3. Making Predictions
# Now **y_pred** contains the **probability of tumor** being of type **Malignant or Benign**. We'll assign the results **true** or **false** based on their **probabilities (if probability >= 0.5 than true else false)**

# In[71]:


y_pred = classifier.predict(X_test)
y_pred = [ 1 if y>=0.5 else 0 for y in y_pred ]


# In[72]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

accuracy = (cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])
print("Accuracy: "+ str(accuracy*100)+"%")


# In[ ]:




