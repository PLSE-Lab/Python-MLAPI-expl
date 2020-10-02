#!/usr/bin/env python
# coding: utf-8

# 1. Dataset was balanced with SMOTE
# 2. 25% Dropout was crucial to reduce overfitting

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
np.set_printoptions(suppress=True)


# In[ ]:


# Load the dataset
raw_data = pd.read_csv('../input/creditcardfraud/creditcard.csv')
raw_data.head()


# Since features V1 to V28 have resulted from PCA transformation, they are already standardized.

# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(raw_data.drop(['Class','Time'],axis=1), raw_data.Class, test_size=0.15, stratify=raw_data.Class)


# We need to standardize the Time, Amount columns because they have a much higher scale compared to rest of the features. We must use the same transformation for both train and test data.

# In[ ]:


amount_mean = x_train.Amount.mean()
amount_std = x_train.Amount.std()
amount_max = x_train.Amount.max()

x_train.Amount = x_train.Amount / amount_max
x_train.Amount = (x_train.Amount - amount_mean) / amount_std

x_test.Amount = x_test.Amount / amount_max
x_test.Amount = (x_test.Amount - amount_mean) / amount_std


# Let's see how imbalanced our data is.

# In[ ]:


num_fraud = len(y_train.loc[y_train == 1])
num_genuine = len(y_train.loc[y_train == 0])

print(y_train.value_counts()) 
plt.pie([num_genuine, num_fraud],labels=["Genuine", "Fraud"],autopct='%1.2f%%',startangle=45)
plt.show()


# We have only 0.17% positive samples. We use SMOTE to generate new positive samples.

# In[ ]:


smt = SMOTE()
x_train, y_train = smt.fit_sample(x_train, y_train) 


# In[ ]:


num_fraud = len(y_train[y_train == 1])
num_genuine = len(y_train[y_train == 0]) 
 
plt.pie([num_genuine, num_fraud],labels=["Genuine", "Fraud"],autopct='%1.2f%%',startangle=45)
plt.show()


# Now we have a balanced dataset.
# I'm going to build a neural network.

# In[ ]:


model = Sequential()
model.add(Dense(48, activation='tanh', input_shape=x_train.shape[1:]))
model.add(BatchNormalization()) 
model.add(Dropout(0.25))
model.add(Dense(48, activation='tanh'))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Dense(48, activation='tanh'))
model.add(BatchNormalization())
model.add(Dropout(0.25)) 
model.add(Dense(1, activation='sigmoid'))
model.summary()


# In[ ]:


model.compile(optimizer='adam', loss = 'binary_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, batch_size=32, epochs=10, shuffle=True)


# In[ ]:


y_predict = model.predict(x_test)


# In cases of anomaly detection, it is more valuable to use a precision-recall curve rather than ROC.

# In[ ]:


precision, recall, threshold = precision_recall_curve(y_test,y_predict)
plt.plot(threshold, recall[1:], 'b')
plt.plot(threshold, precision[1:], 'y')
plt.xlabel('Threshold')
plt.show()


# Although there is no best method to select a threshold, I will use the intersection of precision and recall curves.

# In[ ]:


intersection = np.where(precision == recall)[0][0]
optimal_cutoff = 0.95
print("Optimal Threshold =",optimal_cutoff)


# In[ ]:


genuine_idx = np.where(y_test == 0)[0]
fraud_idx = np.where(y_test == 1)[0]

plt.plot([y_predict[i] for i in fraud_idx], 'ro')
plt.axhline(y=optimal_cutoff, color='g', linestyle='-')
plt.title('Class 1 (Fraud)')
plt.show()

plt.plot([y_predict[i] for i in genuine_idx], 'bo')
plt.axhline(y=optimal_cutoff, color='g', linestyle='-')
plt.title('Class 0 (Genuine)')
plt.show()


# In[ ]:


print(classification_report(y_test,y_predict >= optimal_cutoff))

