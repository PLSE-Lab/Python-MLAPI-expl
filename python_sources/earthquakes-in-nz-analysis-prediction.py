#!/usr/bin/env python
# coding: utf-8

# # Earthquakes in NZ - Analysis & Prediction
# 
# In this notebook we will try to analysis and create a model to predict (hopefully!) earthquakes. Following will be the course of action
# 
# * Libraries
# * Data Load, Exploration & Preparation
# * Data Visualisation
# * Feature Engineering
# * Data Split (Training & Test)
# * Build, Train & Test Model (Simple)
# * Build Neural Network (Complex)

# ## Libraries

# In[ ]:


# Generic Libraries
import numpy as np
import pandas as pd

# Visualisation Libraries
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
from matplotlib import cm
from mpl_toolkits.basemap import Basemap
import folium
from folium.plugins import CirclePattern, FastMarkerCluster

#Data Formatting
pd.plotting.register_matplotlib_converters()
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('seaborn-whitegrid')
pd.set_option('display.max_columns', 500)
warnings.filterwarnings("ignore")
#pd.options.display.float_format = '{:.2f}'.format

#Garbage Collector
import gc

#Date-Time Libraries
import datetime
import time

#SK Learn Libraries
from sklearn import model_selection
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics

#Tabulate Library
from prettytable import PrettyTable


# ## Data Load, Exploration & Preparation

# In[ ]:


#Loading Data
url = '../input/earthquakes-data-nz/earthquakes_NZ.csv'
data = pd.read_csv(url, header='infer',parse_dates=True)


# In[ ]:


data.shape


# As we can observe, there are around 20k+ records

# In[ ]:


#Check for missing values
data.isna().sum()


# Good that we don't have any missing values.
# 
# Now, re-naming the column names & then formatting the data in 'depth' column to show upto 2 decimal points & the data in 'magnitude' column to show upto 1 decimal point.

# In[ ]:


#Checking the data-types for each column
data.info()


# In[ ]:


#Renaming Columns
data = data.rename(columns={"origintime": "Time", "longitude": "Long", " latitude": "Lat", " depth": "Depth", " magnitude": "Magnitude"})


# In[ ]:


#Formating Column Data
data[['Depth']] = data[['Depth']].applymap("{0:.2f}".format)
data[['Magnitude']] = data[['Magnitude']].applymap("{0:.1f}".format)


# In[ ]:


#Converting Depth & Magnitude columns to Float
for col in ['Depth', 'Magnitude']:
    data[col] = data[col].astype('float')

#Converting Time column to datetime
data['Time'] =  pd.to_datetime(data['Time'], format='%Y-%m-%d%H:%M:%S.%f')


# Adding a new column 'Desc' based on the value in the Magnitude column. Following is the categorization based on the information [here](https://www.gns.cri.nz/Home/Learning/Science-Topics/Earthquakes/Monitoring-Earthquakes/Other-earthquake-questions/What-is-the-difference-between-Magnitude-and-Intensity/The-Richter-Magnitude-Scale)
# 
# 0 - 2 = Micro
# 2 - 3.9 = Minor
# 4 - 4.9 = Light
# 5 - 5.9 = Moderate
# 6 - 6.9 = Strong
# 7 - 7.9 = Major
# 8 - 9.9 = Great
# 10+ = Epic
# 
# Let's create a function based on this new information

# In[ ]:


#function to categorize Magnitude
def desc(mag):
    if 0 <= mag <= 2.0:
        return 'Micro'
    elif 2.0 <= mag <= 3.9:
        return 'Minor'
    elif 4.0 <= mag <= 4.9:
        return 'Light'
    elif 5.0 <= mag <= 5.9:
        return 'Moderate'
    elif 6.0 <= mag <= 6.9:
        return 'Strong'
    elif 7.0 <= mag <= 7.9:
        return 'Major'
    elif 8.0 <= mag <= 9.9:
        return 'Great'
    else:
        return 'Epic'
    
#Applying the function to the Magnitude Column
data['Desc'] = data['Magnitude'].apply(lambda mag: desc(mag))


# Converting the UTC date-time data in the Time column into UNIX time for easy ingestion

# In[ ]:


#function to convert UTC time to Unix Time
def ConvertTime(UTCtime):
    dt = datetime.datetime.strptime(UTCtime, '%Y-%m-%d %H:%M:%S.%f')
    ut = time.mktime(dt.timetuple())
    return ut

#Converting to string type as the 'time' only accepts str
data['Time'] = data['Time'].astype('str')  

#Applying the function to the Magnitude Column
data['Time'] = data['Time'].apply(ConvertTime)
    


# In[ ]:


data.head()


# At this stage our data is all prep'd and ready for use. Before doing that we shall take a backup of this dataset

# In[ ]:


data_backup = data.copy()


# In[ ]:


#freeing Memory
gc.collect()


# # Data Visualisation

# In[ ]:


#Creating a Count Plot
sns.set(style="darkgrid")
fig, ax = plt.subplots(figsize=(8,8))
ax = sns.countplot(x="Desc", data=data, palette="Blues_d")

plt.title('Earthquake Count')
plt.ylabel('Count')
plt.xlabel('Earthquake Magnitude')

totals = []

for i in ax.patches:
    totals.append(i.get_height())

total = sum(totals)

for i in ax.patches:
   # ax.text(i.get_x()+0.25, i.get_height()+100,str(round((i.get_height()/total)*100, 1))+'%', fontsize=9,color='black')
   ax.text(i.get_x()+0.25, i.get_height()+100,str(i.get_height()), fontsize=9,color='black')



# **Conclusion:** The majority of the earthquake were between 0 & 3.9 magnitude.
# 

# In[ ]:


#Creating a plot of Magnitude vs Depth
plt.figure(figsize=(10,8))
sns.boxenplot(x='Desc', y='Depth', data=data, scale="linear")
plt.show();


# **Conclusion:**  
# 
# * Almost all of the earthquakes occured at the depth of 0 - 300 Kms. 
# * Some of the Light magnitude (4.0 - 4.9) earthquakes occured at a depth of beyond 300 Kms. 
# * All the Strong magnitude (6.0 - 6.9) earthquakes occured at a depth of less than 100 Kms.
# 
# 

# # Feature Engineering
# 
# Feature Engineering is process where the features are extracted from the given dataset. This is done by observing the correlation in the dataset.
# 

# In[ ]:


#Creating a seperate dataset for observing correlation
df = pd.DataFrame(data, columns=['Time','Long','Lat', 'Depth', 'Magnitude'])


# In[ ]:


#Visualizing the Correlation
corr = df.corr(method='pearson')
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,len(df.columns),1)
ax.set_xticks(ticks)
plt.xticks(rotation=90)
ax.set_yticks(ticks)
ax.set_xticklabels(df.columns)
ax.set_yticklabels(df.columns)
plt.show()


# **Conclusion:** As observed from the above plot, there is absolutely no correlation between the columns.
# 
# However, for the sake of this example we will define the following:
# 
# * Features = Time, Latitude, Longitude
# * Target = Magnitude, Depth
# 
# **Note**: Here we are going to use the dataframe created for observing correlations. This is because the original dataset contains categorical data and we will have to encode it before training the model. At this stage, I do not want to perform 'Encoding' and that is why using the dataframe with only numerical data.
# 

# In[ ]:


#Defining the feature & target
feature_col = ['Time','Long','Lat']
target_col = ['Depth','Magnitude']

#Applying to the current dataset
X = df[feature_col]
y = df[target_col]


# # Data Split (Traing & Test)
# 
# Splitting the data into sizeable chunks to train and then test the model. The training dataset will contain 90% of the records & the testing dataset will contain the remaining 10%.

# In[ ]:


size = 0.1
state = 0

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=size, random_state=state)


# In[ ]:


#Tabulate the Dataset & its size
t = PrettyTable(['Dataset','Size'])
t.add_row(['X_train', X_train.size])
t.add_row(['X_test', X_test.size])
t.add_row(['y_train', y_train.size])
t.add_row(['y_test', y_test.size])
print(t)


# # Build, Train & Test Model (Simple)
# 
# In the simple model, we are going to implement *Random Forest Regressor* model to train, test and predict the output. 

# In[ ]:


#Build the model
rfr = RandomForestRegressor(n_estimators=500, criterion='mse', min_samples_split = 4, verbose=0, random_state=0)

#Train the model
rfr.fit(X_train, y_train)


# In[ ]:


#Making a Prediction using the testing-data
y_pred = rfr.predict(X_test)

#Finding the accuracy & precision of the model
print("Simple Model Accuracy : ",'{:.1%}'.format(rfr.score(X_test, y_test)))


# The model has achieved an accuracy of 63.3%. This is strictly OK , however it can be increased by increasing the "n_estimators" in the model.
# 
# **Note:** Increasing the "n_estimators" will mean that training the model will increase the CPU usage so it is recommended to use GPU.

# In[ ]:


#garbage collection
gc.collect()


# # Build Neural Network (Complex)
# 
# The above model was a simple model that used a linear regression approach for prediction. This is probably why the model achieved an accuracy of 63.3%. So now, we will try to build a Neural Network using TensorFlow & Keras and then we shall train this model.

# In[ ]:


#TensorFlow & Keras Libraries
import tensorflow as tf    
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.metrics import categorical_accuracy
from keras.callbacks import ModelCheckpoint
from keras.optimizers import *
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import GridSearchCV


# In[ ]:


#Building the neural model
def build_model():
    model = Sequential()
    model.add(Dense(16, activation='relu', input_shape=(3,)))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    
    model.compile(loss='squared_hinge', metrics=['accuracy'],optimizer='adam')
    return model


# In[ ]:


#Instantiate the Model
model = build_model()

#Model Architecture Summary
model.summary()


# In[ ]:


#Train the Model
model.fit(X_train, y_train, batch_size=10, epochs=20, verbose=1, validation_data=(X_test, y_test))


# In[ ]:


#Evaluating the Model
[test_loss, test_acc] = model.evaluate(X_test, y_test)
print("Model Evaluation Results on Test Data : Loss = {:.1%}, Accuracy = {:.1%}".format(test_loss, test_acc))


# ![Wonder Why](https://waterfordwhispersnews.com/wp-content/uploads/2014/11/jpg)
# 
# 
# **Conclusion**: Hmm, wonder why the accuracy hasn't improved !!, probably because of the small test-data and/or no-correlation between the data.
