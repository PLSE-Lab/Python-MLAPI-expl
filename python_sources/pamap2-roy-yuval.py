#!/usr/bin/env python
# coding: utf-8

#  # Practical Deep Learning Workshop - Assignment 2
#  #### by: Roy Levy 313577611 & Yuval Sabag 205712151 
#  ---

# ### In this notebook we are going to fucus on the *PAMAP2* dataset.
# ### The dataset, provided in: https://archive.ics.uci.edu/ml/datasets/PAMAP2+Physical+Activity+Monitoring<br> consists on measurements that were monitored on 9 subject while performing certain phsical activities over time such as: running, standing or cycle.<br>The measurements were colected by various of wireless sensors that the subjects wore on their body. 
# ### Our goal for this report is to build a model that can predict the given task a subject is performing based on a given measurements.
# 
# ---

# # Loading the Data
# We will start first by loading and preparing our data
# ---

# In[ ]:


# import and all essentials for the report

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from matplotlib import pyplot as plt
from IPython.display import display


# In[ ]:


# this section contains the filenames of the data and columns mappings


list_of_files = ['PAMAP2_Dataset/Protocol/subject101.dat',
                 'PAMAP2_Dataset/Protocol/subject102.dat',
                 'PAMAP2_Dataset/Protocol/subject103.dat',
                 'PAMAP2_Dataset/Protocol/subject104.dat',
                 'PAMAP2_Dataset/Protocol/subject105.dat',
                 'PAMAP2_Dataset/Protocol/subject106.dat',
                 'PAMAP2_Dataset/Protocol/subject107.dat',
                 'PAMAP2_Dataset/Protocol/subject108.dat',
                 'PAMAP2_Dataset/Protocol/subject109.dat' ]

subjectID = [1,2,3,4,5,6,7,8,9]

activityIDdict = {0: 'transient',
              1: 'lying',
              2: 'sitting',
              3: 'standing',
              4: 'walking',
              5: 'running',
              6: 'cycling',
              7: 'Nordic_walking',
              9: 'watching_TV',
              10: 'computer_work',
              11: 'car driving',
              12: 'ascending_stairs',
              13: 'descending_stairs',
              16: 'vacuum_cleaning',
              17: 'ironing',
              18: 'folding_laundry',
              19: 'house_cleaning',
              20: 'playing_soccer',
              24: 'rope_jumping' }

colNames = ["timestamp", "activityID","heartrate"]

IMUhand = ['handTemperature', 
           'handAcc16_1', 'handAcc16_2', 'handAcc16_3', 
           'handAcc6_1', 'handAcc6_2', 'handAcc6_3', 
           'handGyro1', 'handGyro2', 'handGyro3', 
           'handMagne1', 'handMagne2', 'handMagne3',
           'handOrientation1', 'handOrientation2', 'handOrientation3', 'handOrientation4']

IMUchest = ['chestTemperature', 
           'chestAcc16_1', 'chestAcc16_2', 'chestAcc16_3', 
           'chestAcc6_1', 'chestAcc6_2', 'chestAcc6_3', 
           'chestGyro1', 'chestGyro2', 'chestGyro3', 
           'chestMagne1', 'chestMagne2', 'chestMagne3',
           'chestOrientation1', 'chestOrientation2', 'chestOrientation3', 'chestOrientation4']

IMUankle = ['ankleTemperature', 
           'ankleAcc16_1', 'ankleAcc16_2', 'ankleAcc16_3', 
           'ankleAcc6_1', 'ankleAcc6_2', 'ankleAcc6_3', 
           'ankleGyro1', 'ankleGyro2', 'ankleGyro3', 
           'ankleMagne1', 'ankleMagne2', 'ankleMagne3',
           'ankleOrientation1', 'ankleOrientation2', 'ankleOrientation3', 'ankleOrientation4']

columns = colNames + IMUhand + IMUchest + IMUankle


# In[ ]:


# loading the data into pandas DataFrame

data_pamap2 = pd.DataFrame()
inp_prefix = "/kaggle/input/"

for file in list_of_files:
    file = inp_prefix + file
    subject = pd.read_table(file, header=None, sep='\s+')
    subject.columns = columns
    subject['subject_id'] = int(file[-5])
    data_pamap2 = data_pamap2.append(subject, ignore_index=True)

data_pamap2.reset_index(drop=True, inplace=True)


# # Data Exploration

# Before we start to examine the loaded data we will the start by presanting an overview on the data.<br> if you want a full overview you can check out the provided PDF's.

# ![image.png](attachment:image.png)
# 
# We can see clearly from the picture above that the age of the subjects varies between 24-32.<br> also note that there are 8 males and only 1 female tested.

# ![image.png](attachment:image.png)
# 
# As it seen from the picture above, most of the data is going be balanced,<br>i.e. most of the task were performed in the same amount of time and with mostly even distribution across the range of subjects.<br><br>Note that **subject 109** performed the least amount of activities, this is a notion we need to refer while testing the model's, we don't want only him to represant our testing data.

# In[ ]:


display(data_pamap2.head())


# It can be seen from the sample of the loaded data that some **data cleaning** is required.
# For example the activityID 0 must be removed completely from the dataset because it is represanting a transient state(see the readme.pdf for more information).

# ## Data Cleaning
# ---

# From the given file **PerformedActivitiesSummary** we can see that few of the data is missing, due to wireless disconnections for example.<br>
# We chose also to remove the body orientation, because it seems to us irrelavnat and will take the model more time to train.<br><br> The easiest way to remove the NaN values is to use interpolation, we will start by removing the irrelavant values and then interpolate between the unkown values.
# 
# 
# 

# In[ ]:


def clean_data(data):
    # removing data with transient activity
    data = data.drop(data[data['activityID']==0].index)
    # remove non-numeric data cells
    data = data.apply(pd.to_numeric, errors = 'coerce')
    # removing NaN values using iterpolation
    data = data.interpolate()
    return data
    


# In[ ]:


data_clean = clean_data(data_pamap2)
data_clean.reset_index(drop=True,inplace=True)
display(data_clean.head(15))


# In[ ]:


print(data_clean.isnull().sum())


# As we can see there are still 4 more values of NaN for the heart rate for the first subject,<br>this is due the notion that the first 4 values were NaN so the interpolation uses these values.<br><br>As it can be seen the heart rate for this subject is 100.0 for at least 10 more timelapses, so we will assumes that the 4 values are also 100.0 since heart rate don't change rapidly, also note that the activity performed by the subject is 'lying' which makes our assumption more realistic.

# In[ ]:


for i in range(4):
    data_clean["heartrate"].iloc[i]=100
print(data_clean.isnull().sum())


# ### Data Distribution
# ---

# In[ ]:


data_clean_copy = data_clean
plt.figure()
# activity distribution
N = len(np.unique(data_clean_copy['activityID']))
xticks = np.arange(12)
xticks_lbl = [activityIDdict[x] for x in  np.unique(data_clean_copy['activityID']).tolist()]
data_clean_copy['activityID'].value_counts().plot(kind="bar", figsize=(20,10), color=plt.cm.Paired(np.arange(N)))
plt.xticks(ticks=xticks,labels=xticks_lbl)
plt.title("Activity Distribution")
plt.xlabel('Activity')
plt.ylabel("time interval for activity(0.01sec)")
plt.show()


# As we can see the data is mostly balanced so no need of data augmentation.<br>
# Next we will see some stats about the data.

# In[ ]:


display(data_clean.describe())


# We can see for example that max 'heartrate' is 202 and min is '57', this raises a question about how the activity impacts the heart rate.<br>
# The following graph will help us to answer that question.

# In[ ]:


data_clean['heartrate'].groupby(data_clean['activityID']).mean()


# In[ ]:


plt.figure()
df_heartrate = data_clean['heartrate'].groupby(data_clean['activityID']).mean()
df_heartrate.index = df_heartrate.index.map(activityIDdict)
plt.title("Heart Rate to Activity Mean")
plt.xlabel("Activity Name")
plt.ylabel("Heart Rate")
df_heartrate.plot(kind='bar',figsize=(20,10), color=plt.cm.Paired(np.arange(N)))
plt.show()


# We can clearly see that activities that requires more phisical endurance, such as running or rope jumping leads to a higher heart rate.<br>
# while activities likes sitting or lying leads to a lower heart rate.

# ## Additional preprocessing of the data
# ---
# 
# As it can be seen from the description above the data is vastly compromized on big numbers, for example heart rate of 202.<br> In order to reduce computation time we will use a scaler to scale the data, from various attemps we concluded that the best scaler is **RobustScaler**, probably due to the outliers which is the disconnections in the wireless sensors.<br><br>
# Also note that there is no need for the subjectID or the timestamps since we preprocessed our data to perform as a data for a supervised learning model, we are only interested in predicting the activityID

# In[ ]:


# data scaling helper functions

def scale_data(train_data,test_data,features):
    from sklearn.preprocessing import RobustScaler
    
    scaler = RobustScaler()
    train_data = train_data.copy()
    test_data = test_data.copy()
    
    scaler = scaler.fit(train_data[features])
    train_data.loc[:,features] = scaler.transform(train_data[features].to_numpy())
    test_data.loc[:,features] = scaler.transform(test_data[features].to_numpy())
    return train_data, test_data

def get_features_for_scale(test_list, remove_list):
    res = [i for i in test_list if i not in remove_list]
    return res


# ## Creating test and train data
# ---
# ### We decided to go with the following testing strategy: since subject 7 and 8 performed most of the tasks they will be a good subjects to use as our test data.<br> So our data will be splitted in the following manner:
# * #### train_data: subjects 1-6,9
# * #### test_data: subjects 7-8

# In[ ]:


def get_test_data(data_clean):
    sub_7 = data_clean[data_clean["subject_id"] == 7]
    sub_8 = data_clean[data_clean["subject_id"] == 8]
    return sub_7.append(sub_8)

def get_train_data(data_clean):
    not_sub_7 = data_clean[data_clean["subject_id"] != 7]
    not_sub_8 = not_sub_7[not_sub_7["subject_id"] != 8]
    return not_sub_8


# In[ ]:


print("data_clean shape:" + str(data_clean.shape))

test_data = get_test_data(data_clean)
print("test_data shape:" + str(test_data.shape))
train_data = get_train_data(data_clean)
print("train_data shape:" + str(train_data.shape))


# In[ ]:


test_data = test_data.drop(["subject_id","timestamp"], axis=1)
train_data = train_data.drop(["subject_id", "timestamp"], axis=1)


# In[ ]:


features_for_scale = get_features_for_scale(list(train_data.columns),['activityID'])


# In[ ]:


train_sc, test_sc = scale_data(train_data,test_data,features_for_scale)
print("scaled train data")
display(train_sc.head(5))
print("scaled test data")
display(test_sc.head(5))


# In[ ]:


# getting X,y values for train and test set

X_train = train_sc.drop('activityID', axis=1).values
y_train = train_sc['activityID'].values

X_test = test_sc.drop('activityID', axis=1).values
y_test = test_sc['activityID'].values


# # Building Our First Model
# ---

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense, LSTM, Input
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.linear_model import LogisticRegression


# In[ ]:


# visualizations helper functions
def plot_accuracy_vs_loss(history):
  # from lecture no.2

  fig, ax = plt.subplots(1,2,figsize=(12,4))
  ax[0].plot(history.history['accuracy'])
  ax[0].plot(history.history['val_accuracy'])
  ax[0].set_title('Model accuracy')
  ax[0].set_ylabel('Accuracy')
  ax[0].set_xlabel('Epoch')
  ax[0].legend(['Train', 'Test'], loc='upper left')

  # Plot training & validation loss values
  ax[1].plot(history.history['loss'])
  ax[1].plot(history.history['val_loss'])
  ax[1].set_title('Model loss')
  ax[1].set_ylabel('Loss')
  ax[1].set_xlabel('Epoch')
  ax[1].legend(['Train', 'Test'], loc='upper left')
  plt.show()

def plot_confusion_matrix(model,X_test,y_test,title='',is_ml=False, labels={}):
    
  ticks = list(map(lambda x : activityIDdict[x], np.unique(y_test).tolist())) if labels=={} else labels.values()
  preds = model.predict(X_test)
  pred_cat = preds if is_ml else np.argmax(preds,axis=1)
  print('model accuracy on test set is: {0:.2f}%'.format(accuracy_score(y_test,pred_cat)*100))
  plt.figure(figsize=(15,8),dpi=120)
  sns.heatmap(confusion_matrix(y_test,pred_cat),cmap='Blues',annot=True, fmt='d',xticklabels=ticks,yticklabels=ticks)
  plt.xlabel('Prediction')
  plt.ylabel('True label')
  plt.title(title)
  plt.show()


# ## building a naive model
# ---
# ### We will start by building a first naive model.<br>We will the standart deviation of the features and then fit a simple classic ML model.<br>We will then predict on the real train and test set in order to get a simple base results

# In[ ]:


train_act = train_sc.groupby(data_clean['activityID'])
X_train_base = train_act.std().drop('activityID', axis=1).values
y_train_base = np.array(train_act['activityID'].unique().explode().values).astype('float64')


# In[ ]:


# using logistic regression as our estimator for the naive model
logreg = LogisticRegression()
logreg.fit(X_train_base, y_train_base)


# In[ ]:


y_pred_train = logreg.predict(X_train)
print('model accuracy on train set is: {0:.2f}%'.format(accuracy_score(y_train,y_pred_train)*100))
y_pred_test = logreg.predict(X_test)
print('model accuracy on test set is: {0:.2f}%'.format(accuracy_score(y_test,y_pred_test)*100))


# ### As we can see the result are very low on both the training and test set, next we will try to better our model using another classic ML algorithm.

# # Using DecisionTreeClassifier as ML solid benchmark
# ---
# In this section we will use a DecisionTreeClassifier in order to get a better understanding of the results we are trying to improve.<br> The result given by this classifier will act as a benchmark with which his results we will try improve during our process.<br><br>
# After a few tries we concluded that it is best for us to use only the Accelerometer and the Gyroscope data as features for this model.

# In[ ]:


from sklearn.tree import DecisionTreeClassifier

features_used = ['handAcc16_1', 'handAcc16_2', 'handAcc16_3', 'handAcc6_1', 'handAcc6_2', 'handAcc6_3',
                'chestAcc16_1', 'chestAcc16_2', 'chestAcc16_3', 'chestAcc6_1', 'chestAcc6_2', 'chestAcc6_3',
                'ankleAcc16_1', 'ankleAcc16_2', 'ankleAcc16_3', 'ankleAcc6_1', 'ankleAcc6_2', 'ankleAcc6_3',
                'handGyro1', 'handGyro2', 'handGyro3',
                'chestGyro1', 'chestGyro2', 'chestGyro3',
                'ankleGyro1', 'ankleGyro2', 'ankleGyro3','activityID']

train_dt = train_sc.loc[:,features_used]
test_dt =  test_sc.loc[:, features_used]


# getting X,y values for train and test set

X_train_dt = train_dt.drop('activityID', axis=1).values
y_train_dt = train_dt['activityID'].values

X_test_dt = test_dt.drop('activityID', axis=1).values
y_test_dt = test_dt['activityID'].values


# In[ ]:


tclf = DecisionTreeClassifier()
tclf.fit(X_train_dt,y_train_dt)


# In[ ]:


plot_confusion_matrix(tclf,X_test_dt,y_test_dt,"Decision Tree Classification Using Accelerometer And Gyroscope Data",is_ml=True)


# #### We can see that the model accuracy on the test set is around 50% this is much better than our naive model.<br> This gives us an idea on the starting point from which we will try to improve!<br><br> Notice that the model is misclassifing activities like 'running' and 'rope_jumping' which have similar measurements,<br> and how it is generalizing the activity of 'lying' very well.

# ## Building our second model - LSTM
# ---
# #### since our data compromized on time-series we will use the notion of Deep Neural Network with LSTM embedding.<br> We will start with a simple LSTM model and based on the result will lean toward better models with a greater accuracy.

# ### Data preprocessing
# ----
# #### We will start by creating a dataset based on sliding window as shown in class.<br><br>The *timesteps* represant the samples we want to sample in a batch.<br> The *steps* represant the stride of the window.

# In[ ]:


from scipy import stats

def create_dataset(X, y, time_steps=1, step=1):
    Xs, ys = [], []
    for i in range(0, len(X) - time_steps, step):
        v = X.iloc[i:(i + time_steps)].values
        labels = y.iloc[i: i + time_steps]
        Xs.append(v)
        ys.append(stats.mode(labels)[0][0])
    return np.array(Xs), np.array(ys).reshape(-1, 1)


# In[ ]:


TIME_STEPS = 6
STEPS = 2

df_train_X = train_sc.drop('activityID', axis=1)
df_train_y = train_sc['activityID']

df_test_X = test_sc.drop('activityID', axis=1)
df_test_y = test_sc['activityID']

X_train_lstm, y_train_lstm = create_dataset(df_train_X,df_train_y,TIME_STEPS,STEPS)
X_test_lstm, y_test_lstm = create_dataset(df_test_X,df_test_y,TIME_STEPS,STEPS)


# In[ ]:


from sklearn.preprocessing import OneHotEncoder

# use of OneHotEncoder for the labels(classifaction task)
enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
enc.fit(y_train_lstm)

y_train_lstm = enc.transform(y_train_lstm)
y_test_lstm = enc.transform(y_test_lstm)


# ### Creating our model

# In[ ]:


lstm_model = Sequential()
lstm_model.add(LSTM(16,input_shape=(X_train_lstm.shape[1],X_train_lstm.shape[2])))
lstm_model.add(Dense(16,activation='relu'))
lstm_model.add(Dense(y_train_lstm.shape[1], activation='softmax'))

lstm_model.summary()
lstm_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


# ### Note that we will not suffule our data since each entry position in the dataset has a time meaning.<br>For example: entry 444 was measured 0.01[seconds] before entry 445.

# In[ ]:


np.random.seed(2018)
hist_lstm = lstm_model.fit(X_train_lstm,y_train_lstm,validation_split=0.2,epochs=6, verbose=1)


# In[ ]:


plot_accuracy_vs_loss(hist_lstm)


# In[ ]:


def map_one_hot_enc_positions_and_labels(model,X_test,y_test):
    preds = model.predict(X_test)
    preds = np.argmax(preds,axis=1)
    indicies = np.unique(preds).tolist()
    actual = np.unique(y_test).tolist()
    
    zipped = list(zip(actual, indicies))
    idx_dict = dict(zipped)
    y_test_e = []
    labelsDict = dict()
    for act in y_test.tolist():
        y_test_e.append(idx_dict[act[0]])
    
    for (act,idx) in zipped:
        labelsDict[idx] = activityIDdict[act]
    return np.array(y_test_e), labelsDict
    


# In[ ]:


y = enc.inverse_transform(y_test_lstm)
y,labels = map_one_hot_enc_positions_and_labels(lstm_model,X_test_lstm,y)
plot_confusion_matrix(lstm_model,X_test_lstm,y,"LSTM Model",labels=labels)


# ### As it can be seen very blantly our model is overfitting. <br> Also note that although his overfitting it is generalizing a bit better than our previous classic ML model, reaching about 54% accuracy on the test data.><br><br> It seems that this model is classifying 'walking', 'cycling' and 'lying' very well, but activities like 'running' is still hard for him to classify. 

# # Building or Third Model - Using Fine Tuning
# ---
# ### In this section we are going to try to use fine tuning in order to improve our model.<br> We will start by creating a regression model that predicts the heart rate of a subject based on the other measurements,<br> then use the weights of this model and apply it to our previous model.

# In[ ]:


# Note we are using the same TIME_STEPS and STEPS of the model we want to fine tune

TIME_STEPS = 6
STEPS = 2

# creating the relevant dataset

features_for_scale_hr = get_features_for_scale(list(train_data.columns),[])
train_sc_hr, test_sc_hr = scale_data(train_data,test_data,features_for_scale_hr)

df_train_X_hr = train_sc_hr.drop('heartrate', axis=1)
df_train_y_hr = train_sc_hr['heartrate']

df_test_X_hr = test_sc_hr.drop('heartrate', axis=1)
df_test_y_hr = test_sc_hr['heartrate']

X_train_hr, y_train_hr = create_dataset(df_train_X_hr,df_train_y_hr,TIME_STEPS,STEPS)
X_test_hr, y_test_hr = create_dataset(df_test_X_hr,df_test_y_hr,TIME_STEPS,STEPS)


# In[ ]:


# creating our regression model - predicting heartrate
from keras.layers import Dropout

hr_model = Sequential()
hr_model.add(LSTM(128,input_shape=(X_train_hr.shape[1],X_train_hr.shape[2])))
hr_model.add(Dropout(0.25))
hr_model.add(Dense(128,activation='relu'))
hr_model.add(Dense(y_train_hr.shape[1],activation='linear'))
hr_model.summary()
hr_model.compile(loss='mse', optimizer='adam', metrics=['mae'])


# In[ ]:


np.random.seed(2018)
hist_hr = hr_model.fit(X_train_hr,y_train_hr,validation_split=0.2,epochs=6, verbose=1)


# In[ ]:


ypred_hr = hr_model.predict(X_test_hr)


# In[ ]:


from sklearn.metrics import mean_absolute_error as mae
accuracy = mae(y_test_hr, ypred_hr)
print("heart rate regressor accuracy:" +str(accuracy))


# In[ ]:


# fine tuning our previous model
from keras.layers import Reshape

lstm_hr_model = Sequential()

for layer in hr_model.layers[:-1]:
    lstm_hr_model.add(layer)    

# Freeze the layers 
for layer in lstm_hr_model.layers:
    layer.trainable = False

lstm_hr_model.add(Reshape((128,1), input_shape=(128,)))
lstm_hr_model.add(LSTM(128))
lstm_hr_model.add(Dense(128,activation='relu'))
lstm_hr_model.add(Dense(y_train_lstm.shape[1], activation='softmax'))

lstm_hr_model.summary()
lstm_hr_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


# In[ ]:


np.random.seed(2018)
hist_lstm_hr = lstm_hr_model.fit(X_train_lstm,y_train_lstm,validation_split=0.2,epochs=3, verbose=1)


# In[ ]:


plot_accuracy_vs_loss(hist_lstm_hr)
y_hr = enc.inverse_transform(y_test_lstm)
y_hr,labels_hr = map_one_hot_enc_positions_and_labels(lstm_hr_model,X_test_lstm,y_hr)
plot_confusion_matrix(lstm_hr_model,X_test_lstm,y,"HR-LSTM Model",labels=labels_hr)


# ### We can that this model is worse than the previous model, also it takes a lot of time to run.<br>We will consider of dropping or tweeking it in order to reach a higher accuracy in short amount of time.

# # Building Our Fourth Model - LSTM
# ---
# ### Since our previous model did not yield the result we wanted we will try a new model.<br>In order to reach a greater accuracy we suggest the following:<br><br> 
# 1. #### Enlarge TIME_STEPS and STEPS - thus creating smaller dataset but large enough sequences of timestaps for the LSTM embedding
# 2. #### Making our network deeper and with more neurons, also dropping the fine-tuning from previous step
# 3. #### Trying to do another fine-tuning for the model
# 
# ### We will implement the first two suggestions in the following code

# In[ ]:


TIME_STEPS = 1000
STEPS = 200

df_train_X = train_sc.drop('activityID', axis=1)
df_train_y = train_sc['activityID']

df_test_X = test_sc.drop('activityID', axis=1)
df_test_y = test_sc['activityID']

X_train_lstm, y_train_lstm = create_dataset(df_train_X,df_train_y,TIME_STEPS,STEPS)
X_test_lstm, y_test_lstm = create_dataset(df_test_X,df_test_y,TIME_STEPS,STEPS)


# In[ ]:


from sklearn.preprocessing import OneHotEncoder

# use of OneHotEncoder for the labels(classifaction task)
enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
enc.fit(y_train_lstm)

y_train_lstm = enc.transform(y_train_lstm)
y_test_lstm = enc.transform(y_test_lstm)


# In[ ]:


from keras.layers import Dropout

lstm_model = Sequential()
lstm_model.add(LSTM(128,input_shape=(X_train_lstm.shape[1],X_train_lstm.shape[2])))
lstm_model.add(Dense(128,activation='relu'))
lstm_model.add(Dropout(0.25))
lstm_model.add(Dense(128,activation='relu'))
lstm_model.add(Dense(y_train_lstm.shape[1], activation='softmax'))

lstm_model.summary()
lstm_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


# In[ ]:


np.random.seed(2018)
hist_lstm = lstm_model.fit(X_train_lstm,y_train_lstm,validation_split=0.2,epochs=6, verbose=1)


# In[ ]:


plot_accuracy_vs_loss(hist_lstm)
y = enc.inverse_transform(y_test_lstm)
y,labels = map_one_hot_enc_positions_and_labels(lstm_model,X_test_lstm,y)
plot_confusion_matrix(lstm_model,X_test_lstm,y,"LSTM Model",labels=labels)


# #### looks like we managed to improve the model, reaching about 60% accuracy on the test set.<br> we can see that out validation accuracy reached 75% but our loss got higher.<br><br> We can see that our predictions are much more balanced and also that our number of samples is smaller,<br> this is due to the choice of TIMESTEPS and STEPS that define our sequences in our LSTM model.(we concluded that this parameter has the biggest impact on the results).<br><br> It seems that our model is classifing most of the activities in a much more even way, but still it's hard for it to classify 'running' and 'rope_jumping'.<br>
# ----
# ##### *Side note: after running the model several times we concluded that our accuracy on the test set is about 62%.*

# # Conclusions
# ---
# In our process of building an human activity classifier we used a methods and reached to the following conclusions:
# 
# 1.   We built a naive model that based on the standat deviation of the measurements given in the dataset.<br> This model didn't performed well but showed us a very basic benchmark for improving our results.
# 2.   We built a more complex model using decisions trees. This model yielded a better results and gave us an idea of a solid benchmark we want to improve.
# 3.   We built a basic LSTM model which splits the dataset into sequences and yielded a slightly better results than our classic ML model.
# 4.   We tried to improve our LSTM model by pretraining our model on a built heartrate regressor. This model didn't improve the results and took a long time to run.
# 5.   We dropped the heartrate regressor and splitted our data to larger sequences in order to take full advantage of our LSTM model. We also made our network even deeper. This notion improved our model, reaching about 66% accuracy(pervious model was about 50%).<br><br>
# ---
# #### The Following table summarizes our accuracy scores during the entire process:
# ![image.png](attachment:image.png)
# 
# 
# 
# 

# #### Final Note: few suggestions for next is maybe to decrease our STEPS so we can get more data, increase our training time and we can continue and try another pertraining concept,<br> such as predicting hand temperature or even improve our heart rate regressor
