#!/usr/bin/env python
# coding: utf-8

# ## Titanic Disaster Survival Prediction with Keras and PCA   
# This is my first kernal submitted for the playground competition, after trying for more than 17 rounds to get a better score.   
# 
# You may found this kernal is bit different with others' as I did not do much EDA here, because I know you have read enough and got how to do the EDA for this dataset, but may be seeking a solution to reach a better score.  
# 
# I can tell you my lowest score is only 0.6, after that I tried find a better way to do the learning and prediction, now I got 0.81 after I applied Keras and PCA together.    
# 
# Please feel free to fork this and vote for me if you like this kernal, that will encourage me to countinue learning and sharing.  
# 

# In[ ]:


#Supporting libs
get_ipython().run_line_magic('matplotlib', 'notebook')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
sns.set()


# In[ ]:


#Load both trainand test datasets
train_data= pd.read_csv('../input/train.csv')
test_data=pd.read_csv('../input/test.csv')


# In[ ]:


#Add one column names 'dataType' to mark data types
train_data['dataType']='train'
test_data['dataType']='test'


# In[ ]:


#Combine train and test dataset to do feature engineering all together
all_data=pd.concat([train_data,test_data],axis=0)
all_data.set_index('PassengerId',inplace=True)
all_data.info()


# Process missing values first

# In[ ]:


# Get a summary of missing value for all fields
all_data.isnull().sum()


# As we found above, there are 263 missing in Age and 1014 missing in Cabin, only 2 missing in Embarked and 1 missing in Fare.

# In[ ]:


#Use 'U' to fill missing value in Cabin
all_data['Cabin'].fillna('U',inplace =True)
#Using mean value to fill missing value in Fare
all_data['Fare'].fillna(all_data['Fare'].mean(),inplace =True)


# Check value counts in Embarked, we found most of them are 'S'

# In[ ]:


all_data['Embarked'].value_counts()


# So we will use 'S' to fill missing value in Embarked

# In[ ]:


all_data['Embarked'].fillna('S',inplace = True)


# In[ ]:


#Process Age, while there are 263 missing value, this should impact the the probility of survival
#Let's use histogram to check how the Age distributed 
all_data['Age'].hist()


# Fill missing in Age, but make sure filling new data won't change the original data distribution, so we will use random integer between average value - standard value and average value + standard value

# In[ ]:


avg_age=all_data['Age'].mean()
std_age=all_data['Age'].std()
no_nan=all_data['Age'].isnull().sum()
rand=np.random.randint(avg_age-std_age,avg_age+std_age,size=no_nan)
all_data['Age'][all_data.Age.isnull()]=rand
all_data['Age'].hist()


# Now all missing value filled, moving to process other columns

# In[ ]:


all_data.info()


# Change type of Pclass from int64 to object to be prepared for one-hot encoding

# In[ ]:


all_data['Pclass']=all_data['Pclass'].astype(str)


# We will do one-hot encoding with Pclass, Sex, Cabin and Embared together

# In[ ]:


tobe_dummied_cols = ['Pclass', 'Sex', 'Cabin', 'Embarked']
obj_df = all_data[tobe_dummied_cols]
obj_df_dummy = pd.get_dummies(obj_df)


# In[ ]:


obj_df_dummy.shape


# In[ ]:


#Procced column Name, transform it into title
titles = set()
for name in all_data['Name']:
    titles.add(name.split(',')[1].split('.')[0].strip())

def status(feature):
    print('Processing', feature, ': Done')
    
Title_Dictionary = {
    "Capt": "Officer",
    "Col": "Officer",
    "Major": "Officer",
    "Jonkheer": "Royalty",
    "Don": "Royalty",
    "Sir" : "Royalty",
    "Dr": "Officer",
    "Rev": "Officer",
    "the Countess":"Royalty",
    "Mme": "Mrs",
    "Mlle": "Miss",
    "Ms": "Mrs",
    "Mr" : "Mr",
    "Mrs" : "Mrs",
    "Miss" : "Miss",
    "Master" : "Master",
    "Lady" : "Royalty"
}

# Extract the title from each name, and map names to titles
def get_titles(dataset):
    dataset['Title'] = dataset['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())
    dataset['Title'] = dataset.Title.map(Title_Dictionary)
    status('Title')
    return dataset


# In[ ]:


all_data = get_titles(all_data)


# Combine Parch and SibSp to a new column Family

# In[ ]:


all_data['Family']=all_data['Parch']+all_data['SibSp']
all_data.drop(['Parch','SibSp'],inplace = True, axis=1)


# In[ ]:


all_data.head()


# We drop Cabin, Embarked, Name and Pclass, Sex and Tickets

# In[ ]:


all_data.drop(['Cabin','Embarked','Name','Pclass','Sex','Ticket'],inplace=True,axis=1)


# One-hot encoding the new field Title and drop the Title field in all_data

# In[ ]:


dummy_title=pd.get_dummies(all_data['Title'],prefix='Title')
all_data.drop('Title',inplace=True,axis=1)


# Combine all data with encoded fields

# In[ ]:


all_data=pd.concat((all_data,dummy_title,obj_df_dummy),axis=1)


# Because the linear regression is sensitive to the number distribution and scale, we will use MinMaxScaler to scaler Age and Fare

# In[ ]:


from sklearn.preprocessing import MinMaxScaler
scaler= MinMaxScaler()
all_data['Age']=scaler.fit_transform(all_data.filter(['Age']))
all_data['Fare']=scaler.fit_transform(all_data.filter(['Fare']))


# Let check how the data set looks like, it seems ready to do learning

# In[ ]:


# all_data['Family']=scaler.fit_transform(all_data.filter(['Family']))


# In[ ]:


all_data.head()


# The reason being I don't remove any field here is because during my previous try outs, I actully used  some ways to find most important features and used them to do the training in RandomForest, XGBoost, and Tensorflow(Keras) as well, but the results were not good, so I decided not to drop any field but use a way to reduced the dimension while keep the most import components within this data sets, so I chose PCA to do this.

# In[ ]:


from sklearn.decomposition import PCA


# In[ ]:


new_train_data=all_data[all_data['dataType']=='train']
new_test_data=all_data[all_data['dataType']=='test']


# In[ ]:


new_train_data.drop(['dataType','Survived'],inplace = True,axis=1)
new_test_data.drop(['dataType','Survived'],inplace = True,axis=1)


# I tried to set n_components as 10,13, 15, 18 and 20, only with 15, I got the highest score among them

# In[ ]:


x_train_reduced = PCA(n_components=0.98).fit_transform(new_train_data)
x_test_reduced = PCA(n_components=66).fit_transform(new_test_data)


# In[ ]:


x_test_reduced.shape


# In[ ]:


y_label = train_data['Survived']


# In[ ]:


y_label.shape


# In[ ]:


import tensorflow as tf


# We use Adam with learning rate 0.0001 as optimizer and loss function "sparse_categorical_crossentropy" as this will a classsification task

# In[ ]:


optimizer = tf.keras.optimizers.Adam(0.0001)
loss_function = "sparse_categorical_crossentropy"


# Now, add set 1 input layer with 64 units and 1 hidden layer with 32 units and 2 units in output layer

# In[ ]:


model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=64, input_dim=66,
                          activation='relu'),  #input dims = number of fields
    tf.keras.layers.Dense(units=32, activation="relu"),
    tf.keras.layers.Dense(units=2, activation='softmax')
])
model.compile(optimizer = optimizer, loss = loss_function, metrics=['accuracy'])


# Check the summary of this model

# In[ ]:


model.summary()


# Train the model and save training metrics in history

# In[ ]:


history=model.fit(x_train_reduced,y_label,epochs=170,validation_split = 0.2)


# Define a function to plot the metrics

# In[ ]:


def v_train_history(trainhist, train_metrics, valid_metrics):
    plt.plot(trainhist.history[train_metrics])
    plt.plot(trainhist.history[valid_metrics])
    plt.title('Training metrics')
    plt.ylabel(train_metrics)
    plt.xlabel('Epochs')
    plt.legend(['train','validation'],loc='upper left')
    plt.show()


# Plotting train and validation loss curve

# In[ ]:


v_train_history(history,'loss','val_loss')


# Plotting train and validation accuray curve

# In[ ]:


v_train_history(history,'acc','val_acc')


# Based on the curve above, the result is good as there's no overfitting, so let's submit it

# In[ ]:


x_test_reduced.shape


# In[ ]:


y_pred=model.predict_classes(x_test_reduced)


# In[ ]:


pred_survied_pd = pd.DataFrame(y_pred,
                               index=new_test_data.index,
                               columns=['Survived'])
pred_survied_pd.reset_index()
pred_survied_pd.to_csv('submission.csv')

