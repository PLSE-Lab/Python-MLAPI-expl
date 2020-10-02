#!/usr/bin/env python
# coding: utf-8

# # **Predicting Loan Sanction Status using Neural Network**
# 
# I am stepping into Deep Learning for the first time. I thought of building a classifier using Keras. In order to go through the full ML experience, I chose an unpopular dataset from Kaggle Datasets. Handled data viz and preprocessing, then built a Sequential NN model. It may seem unnecessary to use Neural Network for numerical prediction, but I would like to see if Deep Learning can perform well in these prediction tasks.

# Dataset source: Loan_data (https://www.kaggle.com/pallavi31/loan-data)
# 
# Dataset owner: Pallavi Vibhute (https://www.kaggle.com/pallavi31)
# 

# In[ ]:


# Importing necessary modules
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense
from matplotlib import pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

pd.options.mode.chained_assignment = None


# In[ ]:


train = pd.read_csv('../input/loan-data/Loan_Training_data.csv')
test = pd.read_csv('../input/loan-data/Loan_Test_Data.csv')


# In[ ]:


train.head()


# So, our training data has shape of (614, 13). The last column, Loan_Status, will be our target label.
# 
# It has two classes: 'Y' for sanctioned loans & 'N' for rejected loan applications
# 
# Most of the other columns are self-explanatory. 'Loan_Amount_Term' is the number of months within which the loan should be repaid; 'Credit_History' has values 1.0 if the applicant has credit history and 0.0 if they do not; 'Property_Area tells' us the category of location of the collateral property which is provided as security for the loan.

# # **Data Visualization**

# In[ ]:


train.Loan_Status.value_counts()


# The value counts of the label shows that we have an imbalanced dataset. Most ML models do not work well with imbalanced datasets. To tackle this, we can either collect more data or add synthetic data using popular tools like SMOTE or sklearn.utils.resample. 
# 

# In[ ]:


train.isnull().sum()


# We have a lot of null values in various columns. Since our training dataset is small, dropping rows is not a viable solution. So, I am gonna replace the null values of a feature with the most frequent value of that feature. We will see that in the preprocessing part.

# In[ ]:


sns.barplot(train.Gender, train.LoanAmount, hue=train.Loan_Status)
plt.legend(loc='upper right')
plt.title('Loan Amount vs Gender - grouped based on Loan Status')


# This plot tells us that male applicants are lent a larger amount compared to female applicants. But the sanction-to-rejection ratio is higher for female applicants than to that of male applicants.

# In[ ]:


# Changing categorical names for the sake of easier understanding
train.Married = train.Married.apply(lambda x: 'Married' if x == 'Yes' else 'Unmarried')

sns.barplot(train.Gender, train.LoanAmount, hue=train.Married, hue_order = ['Married', 'Unmarried'])
plt.legend(loc='upper left')
plt.title('Loan Amount vs Gender - grouped based on Marital Status')


# But when it comes to marital status, both men and women can get equal loans when they are married.

# In[ ]:


sns.scatterplot(train.ApplicantIncome, train.LoanAmount, hue=train.Loan_Status)
plt.xticks(rotation=45)


# There is no discernible correlation between Applicant's Income and the Loan Amount. But the scatter plot shows that most of the applicants are people with low-income and the chance of getting a loan does not depend on the applicant's income.

# In[ ]:


sns.countplot(train.Education, hue=train.Loan_Status)


# Not so surprisingly, people with graduate degree have a higher chance of getting a loan than the applicant's without one.

# In[ ]:


sns.countplot(train.Property_Area, hue=train.Loan_Status)


# Surprisingly, property in semi-urban areas render a higher chance of loan sanction than others. This might be due to the fact that semi-urban areas call for more reconstructional capabilities than the urban areas. A semi-urban area is a lot easier to remodel and build than an urban area. And as expected, rural areas score the least here.

# Next, for a full scale visualization, I am employing the pairplot from seaborn.

# In[ ]:


sns.pairplot(train, hue='Loan_Status', palette='Set2', diag_kind='kde')


# There are some interesting things going on here:
# 1. As already stated, Applicant's Income does not determine the loan sanction status (plot - row 1, col 1).
# 2. But, applicants having credit history have a higher chance of getting a loan (plot - row 5 col 5).
# 3. Applicants with higher income **and** credit history have a higher chance of getting a loan (plot row 1 col 5).
# 4. Loan Amount KDE (plot - row 3 col 3) peaks around 150 for both sanctioned and rejected loans, with sanctioned overpowering
#    rejected. Beyond the peak, more loan amount invites more rejected applications than sanctioned ones.

# Bottomline from these plots: You have a higher chance of getting a loan if you are a **Married Male Graduate having a Property in Semi-Urban area**.

# Now, the preprocessing part.

# # **Data Preprocessing**

# Step 1 is to split the dataset into features and target.

# In[ ]:


def target_split(train): 
    train_mod = train[~train['LoanAmount'].isnull()] # Loan Amount has few null values but they should not be imputed
    train_mod.drop('Loan_ID', axis = 1, inplace=True) # Dropping ID column as it is not relevant to the model

    y = train_mod.Loan_Status
    train_mod.drop('Loan_Status', axis = 1, inplace=True)

    y = y.apply(lambda x: 1 if x == 'Y' else 0) # Changing categories to numerical values
    
    return train_mod, y


# Next step is to fill the null values.

# In[ ]:


def impute(train):
    cols = train.columns
    nan_cols = []
    for col in cols:
        if(train[col].isnull().sum() > 0):
            nan_cols.append(col)
    # nan_cols contains the list of columns having null values
    
    argmax_in_nan = {}
    for col in nan_cols:
        argmax_in_nan[col] = None
        argmax_in_nan[col] = train[col].value_counts().idxmax() # Getting the most frequent value in the column
        
        train[col].fillna(argmax_in_nan[col], inplace=True)
            
    return train


# After filling null values, comes the normalization part. MinMaxScaler maps the values to values in range [0,1]. 

# In[ ]:


def scaler(train):
    num_cols = [col for col in train.select_dtypes(exclude='object').columns]
    scaler = MinMaxScaler()
    for col in num_cols:
        if (col != 'Credit_History'): # Credit_History belongs to int64 datatype but it is a categorical value. So it should not be scaled.
            train[col] = scaler.fit_transform(train[[col]])
            
    return train


# Now, to handle the categorical variables, I am using get_dummies function from pandas. This will create a column for each categorical variable under each feature

# In[ ]:


def cat_enc(train):
    cat_cols = [col for col in train.select_dtypes(include='object').columns]
    
    for col in cat_cols:
        dummies = pd.get_dummies(train[col], prefix=col)
        train = pd.concat([train,dummies], axis=1)
        train.drop([col],axis = 1 , inplace=True)
    
    return train


# Compiling all the above steps, this function renders a ML-ready dataset.

# In[ ]:


def preprocess(train):
    train, y = target_split(train)
    train = impute(train)
    train = scaler(train)
    train = cat_enc(train)
    
    return train, y


# In[ ]:


train_mod, y = preprocess(train)


# In[ ]:


train_mod.head()


# After dropping rows with null LoanAmount and preprocessing, the training data has shape of (592, 20)

# # **Model development**

# I'm using keras module to build the neural network. I'm going for a wider, shallow network rather than a narrow deeper one. Adam is currently the most popular optimizer. I'm printing the accuracy score of the model at every epoch.

# In[ ]:


np.random.seed(0)

model = Sequential()

model.add(Dense(48, kernel_initializer='normal',input_dim = train_mod.shape[1], activation='relu'))
model.add(Dense(96, kernel_initializer='normal',activation='relu'))
model.add(Dense(96, kernel_initializer='normal',activation='relu'))

model.add(Dense(1, kernel_initializer='normal',activation='linear'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()


# Here, I am using a checkpoint list to store the weights of the model when it performs the best. By this method, I can reuse the best weights for the test data.

# The filename stores the epoch number along with the validation loss at that epoch (it is useful in the coming part). I specified the model to use log_loss
# After training the model, the weights file with the least validation loss will be selected for predicting.

# In[ ]:


checkpoint_name = 'Weights-{epoch:02d}--{val_loss:.2f}.hdf5'
checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
callbacks_list = [checkpoint]


# In[ ]:


model.fit(train_mod, y, epochs=50, batch_size=37, validation_split = 0.2, callbacks=callbacks_list)


# In[ ]:


get_ipython().system('ls .')


# The ls command shows the checkpoints at which the model has the highest accuracy. The files contain the weights for that particular epoch (the two digit number after *Weights-*). 

# So, in this part, I had a problem. Each time I run this kernel, the name of weight file with best weights change. While committing, the filename changes and throws an error in final kernel. Hence, I could not specify one particular filename when loading weights. So, I wrote a piece of code where I check for the filename with lowest validation loss. I used that file in load_weights function.

# In[ ]:


import os

best_weight_file = str()
val_loss = 100
for filename in os.listdir():
    if(filename.startswith('W')):
        name, ext = os.path.splitext(filename)
        if(int(name[-2:]) < val_loss):
            val_loss = int(name[-2:])
            best_weight_file = filename
            
print(best_weight_file)


# In[ ]:


model.load_weights(best_weight_file)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

scores = model.evaluate(train_mod, y, verbose=0)
print("Accuracy of model: %.2f%%" % (scores[1]*100))


# The overall model gave 82.09% accuracy on training data, which is pretty decent for a shallow neural network.

# # Conclusion
# 
# In this kernel, I chose a random problem and built a decent classifier using a shallow neural network. I was able to tackle the problems I faced. If you have any questions or suggestions to improve the kernel, you are welcome to say it.
