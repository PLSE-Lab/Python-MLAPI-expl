#!/usr/bin/env python
# coding: utf-8

# # INSAID Hiring Exercise

# ## Important: Kindly go through the instructions mentioned below.
# 
# - The Sheet is structured in **4 steps**:
#     1. Understanding data and manipulation
#     2. Data visualization
#     3. Implementing Machine Learning models(Note: It should be more than 1 algorithm)
#     4. Model Evaluation and concluding with the best of the model.
#     
#     
#     
# 
# - Try to break the codes in the **simplest form** and use number of code block with **proper comments** to them
# - We are providing **h** different dataset to choose from(Note: You need to select any one of the dataset from this sample sheet only)
# - The **interview calls** will be made solely based on how good you apply the **concepts**.
# - Good Luck! Happy Coding!

# ### Importing the data

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.model_selection import train_test_split


# In[ ]:


data= pd.read_csv('../input/WA_Fn-UseC_-Telco-Customer-Churn.csv')


# In[ ]:


data.head()


# In[ ]:


print ("Rows     : " ,data.shape[0])
print ("Columns  : " ,data.shape[1])
print ("\nFeatures : \n" ,data.columns.tolist())
print ("\nMissing values :  ", data.isnull().sum().values.sum())
print ("\nUnique values :  \n",data.nunique())


# In[ ]:


#dropout customerID becouse it is not usefull for Churn
data=data.drop(['customerID'],axis=1)
data.head()


# ##Understanding the data

# In[ ]:


# Checking the data types of all the columns
data.info()


# In[ ]:


import seaborn as sns
sns.set(style="ticks", color_codes=True)
df=data
fig, axes = plt.subplots(nrows = 3,ncols = 5,figsize = (25,15))
sns.countplot(x = "gender", data = df, ax=axes[0][0])
sns.countplot(x = "Partner", data = df, ax=axes[0][1])
sns.countplot(x = "Dependents", data = df, ax=axes[0][2])
sns.countplot(x = "PhoneService", data = df, ax=axes[0][3])
sns.countplot(x = "MultipleLines", data = df, ax=axes[0][4])
sns.countplot(x = "InternetService", data = df, ax=axes[1][0])
sns.countplot(x = "OnlineSecurity", data = df, ax=axes[1][1])
sns.countplot(x = "OnlineBackup", data = df, ax=axes[1][2])
sns.countplot(x = "DeviceProtection", data = df, ax=axes[1][3])
sns.countplot(x = "TechSupport", data = df, ax=axes[1][4])
sns.countplot(x = "StreamingTV", data = df, ax=axes[2][0])
sns.countplot(x = "StreamingMovies", data = df, ax=axes[2][1])
sns.countplot(x = "Contract", data = df, ax=axes[2][2])
sns.countplot(x = "PaperlessBilling", data = df, ax=axes[2][3])
ax = sns.countplot(x = "PaymentMethod", data = df, ax=axes[2][4])
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
plt.show(fig)


# ##Data Manipulation

# In[ ]:


# Converting Total Charges to a numerical data type
data['TotalCharges'] = pd.to_numeric(data.TotalCharges, errors='coerce')
data.isnull().sum()


# In[ ]:


data = data.dropna()
data.isnull().sum()


# In[ ]:


fig, (ax1, ax2, ax3) = plt.subplots(3)
sns.kdeplot(data["tenure"], shade=True, color="b",ax = ax1)
sns.kdeplot(data["MonthlyCharges"], shade=True, color="r", ax = ax2)
sns.kdeplot(data["TotalCharges"], shade=True, color="g", ax = ax3)
fig.tight_layout()
plt.show(fig)


# In[ ]:


data.info()


# In[ ]:


data=pd.get_dummies(data,drop_first=True)
data.head()


# In[ ]:


data.info()


# In[ ]:


X = data.drop(['Churn_Yes'],axis=1)
Y = data['Churn_Yes']
print(X.shape,'\n',Y.shape)


# In[ ]:


X = X.astype('float32')
Y = Y.astype('float32')


# In[ ]:


from sklearn.model_selection import train_test_split
X_train , X_test , Y_train , Y_test = train_test_split(X ,Y , test_size =.10 ,random_state = 0)


# In[ ]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)


# ##Implement Machine Learning Models

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(random_state=0)
dtc.fit(X_train,Y_train)


# In[ ]:


y_pred = dtc.predict(X_test)
# Print the prediction accuracy
print (metrics.accuracy_score(Y_test, y_pred))
print(classification_report(Y_test, y_pred))


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rfc =RandomForestClassifier()
rfc.fit(X_train,Y_train)


# In[ ]:


y_pred = rfc.predict(X_test)
# Print the prediction accuracy
print (metrics.accuracy_score(Y_test, y_pred))
print(classification_report(Y_test, y_pred))


# In[ ]:


# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, Y_train)


# In[ ]:


y_pred = classifier.predict(X_test)
print (metrics.accuracy_score(Y_test, y_pred))
print(classification_report(Y_test, y_pred))


# #Using Keras

# In[ ]:


from keras.utils.np_utils import to_categorical
Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)


# In[ ]:


Nx=X_train.shape[1:][0]
Ny=Y_train.shape[1:][0]
print(Nx,Ny)


# In[ ]:


from tensorflow.keras.layers import Input, Dense
from keras.models import Sequential

input_layer = Input(shape = X_train.shape[1:])
hidden_layer = Dense(10, activation = 'relu',)(input_layer)
hidden_layer = Dense(10, activation = 'relu',)(hidden_layer)
output_layer = Dense(2, activation = 'sigmoid')(hidden_layer)


# In[ ]:


from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
model = Model(inputs=[input_layer], outputs=[output_layer])
model.compile(optimizer=optimizers.Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


# In[ ]:


history = model.fit(X_train, Y_train, epochs=50)


# In[ ]:


from sklearn.metrics import confusion_matrix
loss, accuracy = model.evaluate(X_test, Y_test,verbose=0)  # Evaluate the model
print('Accuracy :%0.3f'%accuracy)


# In[ ]:


history.history.keys()
import matplotlib.pyplot as plt
plt.plot(range(len(history.history['acc'])), history.history['acc'],c='blue')
plt.plot(range(len(history.history['loss'])), history.history['loss'],c='red')
plt.show()


# ##With LogisticRegression and Deep NN  I was able to increase the accuracy to upto 82%

# In[ ]:




