#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, log_loss
import operator
import json
from IPython import display
import os
import warnings

np.random.seed(0)
warnings.filterwarnings("ignore")
THRESHOLD = 4


# Task: To predict whether the user likes the mobile phone or not. 
# Assumption: If the average rating of mobile >= threshold, then the user likes it, otherwise not.

# Missing values:
# 'Also Known As'(459),'Applications'(421),'Audio Features'(437),'Bezel-less display'(266),'Browser'(449),'Build Material'(338),'Co-Processor'(451),'Display Colour'(457),'Mobile High-Definition Link(MHL)'(472),'Music'(447) 'Email','Fingerprint Sensor Position'(174),'Games'(446),'HDMI'(454),'Heart Rate Monitor'(467),'IRIS Scanner'(467), 'Optical Image Stabilisation'(219),'Other Facilities'(444),'Phone Book'(444),'Physical Aperture'(87),'Quick Charging'(122),'Ring Tone'(444),'Ruggedness'(430),SAR Value(315),'SIM 3'(472),'SMS'(470)', 'Screen Protection'(229),'Screen to Body Ratio (claimed by the brand)'(428),'Sensor'(242),'Software Based Aperture'(473), 'Special Features'(459),'Standby time'(334),'Stylus'(473),'TalkTime'(259), 'USB Type-C'(374),'Video Player'(456), 'Video Recording Features'(458),'Waterproof'(398),'Wireless Charging','USB OTG Support'(159), 'Video ,'Recording'(113),'Java'(471),'Browser'(448)
# 
# Very low variance:
# 'Architecture'(most entries are 64-bit),'Audio Jack','GPS','Loudspeaker','Network','Network Support','Other Sensors'(28),'SIM Size', 'VoLTE'
# 
# Multivalued:
# 'Colours','Custom UI','Model'(1),'Other Sensors','Launch Date'
# 
# Not important:
# 'Bluetooth', 'Settings'(75),'Wi-Fi','Wi-Fi Features'
# 
# Doubtful:
# 'Aspect Ratio','Autofocus','Brand','Camera Features','Fingerprint Sensor'(very few entries are missing), 'Fingerprint Sensor Position', 'Graphics'(multivalued),'Image resolution'(multivalued),'SIM Size','Sim Slot(s)', 'User Available Storage', 'SIM 1', 'SIM 2','Shooting Modes', 'Touch Screen'(24), 'USB Connectivity'
# 
# To check:
# 'Display Type','Expandable Memory','FM Radio'
# 
# High Correlation with other features
# 'SIM Slot(s)' high correlation with SIM1 'Weight' has high high correlation with capacity , screen-to-body ratio 'Height' - screen size is also there
# 
# Given a mobile, we can't directly get these features
# 'Rating Count', 'Review Count'
# 
# Keeping:
# 'Capacity','Flash'(17),'Height'(22),'Internal Memory'(20, require cleaning),'Operating System'(25, require cleaning), 'Pixel Density'(1, clean it),'Processor'(22, clean it), 'RAM'(17, clean), 'Rating','Resolution'(cleaning), 'Screen Resolution','Screen Size', 'Thickness'(22), 'Type','User Replaceable','Weight'(cleaning),'Sim Size'(), 'Other Sensors'(28), 'Screen to Body Ratio (calculated)','Width',

# In[ ]:


# read data from file
train = pd.read_csv("../input/train.csv") 
test = pd.read_csv("../input/test.csv")

# check the number of features and data points in train
print("Number of data points in train: %d" % train.shape[0])
print("Number of features in train: %d" % train.shape[1])

# check the number of features and data points in test
print("Number of data points in test: %d" % test.shape[0])
print("Number of features in test: %d" % test.shape[1])


# In[ ]:


def data_clean(data):
    
    # Let's first remove all missing value features
    columns_to_remove = ['Also Known As','Applications','Audio Features','Bezel-less display'
                         'Browser','Build Material','Co-Processor','Browser'
                         'Display Colour','Mobile High-Definition Link(MHL)',
                         'Music', 'Email','Fingerprint Sensor Position',
                         'Games','HDMI','Heart Rate Monitor','IRIS Scanner', 
                         'Optical Image Stabilisation','Other Facilities',
                         'Phone Book','Physical Aperture','Quick Charging',
                         'Ring Tone','Ruggedness','SAR Value','SIM 3','SMS',
                         'Screen Protection','Screen to Body Ratio (claimed by the brand)',
                         'Sensor','Software Based Aperture', 'Special Features',
                         'Standby time','Stylus','TalkTime', 'USB Type-C',
                         'Video Player', 'Video Recording Features','Waterproof',
                         'Wireless Charging','USB OTG Support', 'Video Recording','Java']

    columns_to_retain = list(set(data.columns)-set(columns_to_remove))
    data = data[columns_to_retain]

    #Features having very low variance 
    columns_to_remove = ['Architecture','Audio Jack','GPS','Loudspeaker','Network','Network Support','VoLTE']
    columns_to_retain = list(set(data.columns)-set(columns_to_remove))
    data = data[columns_to_retain]

    # Multivalued:
    columns_to_remove = ['Architecture','Launch Date','Audio Jack','GPS','Loudspeaker','Network','Network Support','VoLTE', 'Custom UI']
    columns_to_retain = list(set(data.columns)-set(columns_to_remove))
    data = data[columns_to_retain]

    # Not much important
    columns_to_remove = ['Bluetooth', 'Settings','Wi-Fi','Wi-Fi Features']
    columns_to_retain = list(set(data.columns)-set(columns_to_remove))
    data = data[columns_to_retain]
    
    return data


# In[ ]:


train = data_clean(train)
test = data_clean(test)


# removing all those data points in which more than 15 features are missing

# In[ ]:


train = train[(train.isnull().sum(axis=1) <= 15)]
# You shouldn't remove data points from test set
#test = test[(test.isnull().sum(axis=1) <= 15)]


# In[ ]:


# check the number of features and data points in train
print("Number of data points in train: %d" % train.shape[0])
print("Number of features in train: %d" % train.shape[1])

# check the number of features and data points in test
print("Number of data points in test: %d" % test.shape[0])
print("Number of features in test: %d" % test.shape[1])


# In[ ]:


def for_integer(test):
    try:
        test = test.strip()
        return int(test.split(' ')[0])
    except IOError:
           pass
    except ValueError:
        pass
    except:
        pass

def for_string(test):
    try:
        test = test.strip()
        return (test.split(' ')[0])
    except IOError:
        pass
    except ValueError:
        pass
    except:
        pass

def for_float(test):
    try:
        test = test.strip()
        return float(test.split(' ')[0])
    except IOError:
        pass
    except ValueError:
        pass
    except:
        pass
def find_freq(test):
    try:
        test = test.strip()
        test = test.split(' ')
        if test[2][0] == '(':
            return float(test[2][1:])
        return float(test[2])
    except IOError:
        pass
    except ValueError:
        pass
    except:
        pass

    
def for_Internal_Memory(test):
    try:
        test = test.strip()
        test = test.split(' ')
        if test[1] == 'GB':
            return int(test[0])
        if test[1] == 'MB':
#             print("here")
            return (int(test[0]) * 0.001)
    except IOError:
           pass
    except ValueError:
        pass
    except:
        pass
    
def find_freq(test):
    try:
        test = test.strip()
        test = test.split(' ')
        if test[2][0] == '(':
            return float(test[2][1:])
        return float(test[2])
    except IOError:
        pass
    except ValueError:
        pass
    except:
        pass


# In[ ]:


def data_clean_2(x):
    data = x.copy()
    
    data['Capacity'] = data['Capacity'].apply(for_integer)

    data['Height'] = data['Height'].apply(for_float)
    data['Height'] = data['Height'].fillna(data['Height'].mean())

    data['Internal Memory'] = data['Internal Memory'].apply(for_Internal_Memory)

    data['Pixel Density'] = data['Pixel Density'].apply(for_integer)

    data['Internal Memory'] = data['Internal Memory'].fillna(data['Internal Memory'].median())
    data['Internal Memory'] = data['Internal Memory'].astype(int)

    data['RAM'] = data['RAM'].apply(for_integer)
    data['RAM'] = data['RAM'].fillna(data['RAM'].median())
    data['RAM'] = data['RAM'].astype(int)

    data['Resolution'] = data['Resolution'].apply(for_integer)
    data['Resolution'] = data['Resolution'].fillna(data['Resolution'].median())
    data['Resolution'] = data['Resolution'].astype(int)

    data['Screen Size'] = data['Screen Size'].apply(for_float)

    data['Thickness'] = data['Thickness'].apply(for_float)
    data['Thickness'] = data['Thickness'].fillna(data['Thickness'].mean())
    data['Thickness'] = data['Thickness'].round(2)

    data['Type'] = data['Type'].fillna('Li-Polymer')

    data['Screen to Body Ratio (calculated)'] = data['Screen to Body Ratio (calculated)'].apply(for_float)
    data['Screen to Body Ratio (calculated)'] = data['Screen to Body Ratio (calculated)'].fillna(data['Screen to Body Ratio (calculated)'].mean())
    data['Screen to Body Ratio (calculated)'] = data['Screen to Body Ratio (calculated)'].round(2)

    data['Width'] = data['Width'].apply(for_float)
    data['Width'] = data['Width'].fillna(data['Width'].mean())
    data['Width'] = data['Width'].round(2)

    data['Flash'][data['Flash'].isna() == True] = "Other"

    data['User Replaceable'][data['User Replaceable'].isna() == True] = "Other"

    data['Num_cores'] = data['Processor'].apply(for_string)
    data['Num_cores'][data['Num_cores'].isna() == True] = "Other"


    data['Processor_frequency'] = data['Processor'].apply(find_freq)
    #because there is one entry with 208MHz values, to convert it to GHz
    data['Processor_frequency'][data['Processor_frequency'] > 200] = 0.208
    data['Processor_frequency'] = data['Processor_frequency'].fillna(data['Processor_frequency'].mean())
    data['Processor_frequency'] = data['Processor_frequency'].round(2)

    data['Camera Features'][data['Camera Features'].isna() == True] = "Other"

    #simplifyig Operating System to os_name for simplicity
    data['os_name'] = data['Operating System'].apply(for_string)
    data['os_name'][data['os_name'].isna() == True] = "Other"

    data['Sim1'] = data['SIM 1'].apply(for_string)

    data['SIM Size'][data['SIM Size'].isna() == True] = "Other"

    data['Image Resolution'][data['Image Resolution'].isna() == True] = "Other"

    data['Fingerprint Sensor'][data['Fingerprint Sensor'].isna() == True] = "Other"

    data['Expandable Memory'][data['Expandable Memory'].isna() == True] = "No"

    data['Weight'] = data['Weight'].apply(for_integer)
    data['Weight'] = data['Weight'].fillna(data['Weight'].mean())
    data['Weight'] = data['Weight'].astype(int)

    data['SIM 2'] = data['SIM 2'].apply(for_string)
    data['SIM 2'][data['SIM 2'].isna() == True] = "Other"
    
    return data


# In[ ]:


train = data_clean_2(train)
test = data_clean_2(test)

# check the number of features and data points in train
print("Number of data points in train: %d" % train.shape[0])
print("Number of features in train: %d" % train.shape[1])

# check the number of features and data points in test
print("Number of data points in test: %d" % test.shape[0])
print("Number of features in test: %d" % test.shape[1])


# In[ ]:


def data_clean_3(x):
    
    data = x.copy()

    columns_to_remove = ['User Available Storage','SIM Size','Chipset','Processor','Autofocus','Aspect Ratio','Touch Screen',
                        'Bezel-less display','Operating System','SIM 1','USB Connectivity','Other Sensors','Graphics','FM Radio',
                        'NFC','Shooting Modes','Browser','Display Colour' ]

    columns_to_retain = list(set(data.columns)-set(columns_to_remove))
    data = data[columns_to_retain]


    columns_to_remove = [ 'Screen Resolution','User Replaceable','Camera Features',
                        'Thickness', 'Display Type']

    columns_to_retain = list(set(data.columns)-set(columns_to_remove))
    data = data[columns_to_retain]


    columns_to_remove = ['Fingerprint Sensor', 'Flash', 'Rating Count', 'Review Count','Image Resolution','Type','Expandable Memory',                        'Colours','Width','Model']
    columns_to_retain = list(set(data.columns)-set(columns_to_remove))
    data = data[columns_to_retain]

    return data


# In[ ]:


train = data_clean_3(train)
test = data_clean_3(test)

# check the number of features and data points in train
print("Number of data points in train: %d" % train.shape[0])
print("Number of features in train: %d" % train.shape[1])

# check the number of features and data points in test
print("Number of data points in test: %d" % test.shape[0])
print("Number of features in test: %d" % test.shape[1])


# In[ ]:


# one hot encoding

train_ids = train['PhoneId']
test_ids = test['PhoneId']

cols = list(test.columns)
cols.remove('PhoneId')
cols.insert(0, 'PhoneId')

combined = pd.concat([train.drop('Rating', axis=1)[cols], test[cols]])
print(combined.shape)
print(combined.columns)

combined = pd.get_dummies(combined)
print(combined.shape)
print(combined.columns)

train_new = combined[combined['PhoneId'].isin(train_ids)]
test_new = combined[combined['PhoneId'].isin(test_ids)]


# In[ ]:


train_new = train_new.merge(train[['PhoneId', 'Rating']], on='PhoneId')


# In[ ]:


# check the number of features and data points in train
print("Number of data points in train: %d" % train_new.shape[0])
print("Number of features in train: %d" % train_new.shape[1])

# check the number of features and data points in test
print("Number of data points in test: %d" % test_new.shape[0])
print("Number of features in test: %d" % test_new.shape[1])


# In[ ]:


train_new.head()


# In[ ]:


test_new.head()


# In[ ]:


print("train shape", train_new.shape)
print("test shape", test_new.shape)


# In[ ]:


train_new.columns


# In[ ]:


set(train_new)-set(test_new)


# In[ ]:


train_new['Class'] = train_new['Rating'].map(lambda x: 1 if x >= THRESHOLD else 0)


# In[ ]:


train_new.drop(['Rating', 'PhoneId'], axis=1, inplace=True)
train_new.Class.value_counts()


# In[ ]:


for col in ['RAM', 'Processor_frequency', 'Capacity', 'Resolution',
       'Height', 'Screen Size', 'Internal Memory', 'Weight', 'Pixel Density',
       'Screen to Body Ratio (calculated)']:
    train_new[col] = np.log1p(train_new[col])
    test_new[col] = np.log1p(test_new[col])


# In[ ]:


test_new.shape


# In[ ]:


train_new.shape


# In[ ]:


train_new.RAM.value_counts()


# In[ ]:


train_new.groupby('Class').mean()


# Identify variables

# In[ ]:


phoneid = test_new['PhoneId']
test_new.drop('PhoneId', axis=1, inplace=True)


# In[ ]:


# take only variables having significant difference in mean values (0.8)
# feature_0_1 will be assigned class 0 for values below 50% and class 1 above 50%
ds = train_new.groupby('Class').mean()
features_0_1 = []
for col in ds.columns:
    if ((ds[col][1]-ds[col][0])/ds[col][1]) > 0.81:
        features_0_1.append(col)
features_0_1


# In[ ]:


ds = train_new.groupby('Class').mean()
features_1_0 = []
for col in ds.columns:
    if ((ds[col][0]-ds[col][1])/ds[col][0]) > 0.81:
        features_1_0.append(col)
features_1_0


# In[ ]:


ds = train_new.groupby('Class').mean()


# In[ ]:


ds.columns


# Train Val Split

# In[ ]:


X_train, X_val, Y_train, Y_val = train_test_split(train_new.drop('Class', axis=1), train_new.Class, 
                                                  stratify=train_new.Class, test_size=0.25, 
                                                  random_state=1)
train_y = train_new.Class
train_new.drop('Class', axis=1, inplace=True)


# In[ ]:


print(Y_train.value_counts())
print(Y_val.value_counts())


# Normalisation

# In[ ]:


def scaled(df):
    scaler = MinMaxScaler()
#     for col in ['RAM', 'Resolution', 'Pixel Density', 'Capacity', 'Weight',
#            'Internal Memory', 'Processor_frequency', 'Screen Size', 'Height',
#            'Screen to Body Ratio (calculated)']:
    scaled_df = scaler.fit_transform(df)
    scaled_df = pd.DataFrame(scaled_df, columns=df.columns)
    return scaled_df


# Median

# In[ ]:


X_binarised_train = pd.DataFrame()
X_binarised_val = pd.DataFrame()
test_binarised = pd.DataFrame()
train_binarised = pd.DataFrame()
for col in features_0_1:
#     print(col)
    X_binarised_train[col] = pd.cut(X_train[col], bins=2, labels=[0, 1])
    X_binarised_val[col] = pd.cut(X_val[col], bins=2, labels=[0, 1])
    test_binarised[col] = pd.cut(test_new[col], bins=2, labels=[0, 1])
    train_binarised[col] = pd.cut(train_new[col], bins=2, labels=[0, 1])
for col in features_1_0:
    X_binarised_train[col] = pd.cut(X_train[col], bins=2, labels=[1, 0])
    X_binarised_val[col] = pd.cut(X_val[col], bins=2, labels=[1, 0])
    test_binarised[col] = pd.cut(test_new[col], bins=2, labels=[1, 0])
    train_binarised[col] = pd.cut(train_new[col], bins=2, labels=[1, 0])


# Mean

# In[ ]:


# X_binarised_train = pd.DataFrame()
# X_binarised_val = pd.DataFrame()
# test_binarised = pd.DataFrame()
# for col in features_0_1:
# #     print(col)
#     X_binarised_train[col] = X_train[col].map(lambda x: 1 if x > X_train[col].mean() else 0)
#     X_binarised_val[col] = X_val[col].map(lambda x: 1 if x > X_train[col].mean() else 0)
#     test_binarised[col] = test_new[col].map(lambda x: 1 if x > X_train[col].mean() else 0)
# for col in features_1_0:
#     X_binarised_train[col] = X_train[col].map(lambda x: 0 if x > X_train[col].mean() else 1)
#     X_binarised_val[col] = X_val[col].map(lambda x: 0 if x > X_train[col].mean() else 1)
#     test_binarised[col] = test_new[col].map(lambda x: 0 if x > X_train[col].mean() else 1)


# In[ ]:


X_train.describe()


# In[ ]:


X_binarised_val = X_binarised_val.values
X_binarised_train = X_binarised_train.values
test_binarised = test_binarised.values
train_binarised = train_binarised.values


# Modelling

# In[ ]:


Y_val.value_counts()


# In[ ]:


class MPNeuron:
  
  def __init__(self):
    self.b = None
    
  def model(self, x):
    pred = 1 if sum(x) >= self.b else 0 
    return(pred)
  
  def predict(self, X):
    Y = []
    for x in X:
      result = self.model(x)
      Y.append(result)
    return np.array(Y)
  
  def fit(self, X, Y):
    accuracy = {}
    
    for b in range(X.shape[1] + 1):
      self.b = b
      Y_pred = self.predict(X)
      accuracy[b] = accuracy_score(Y_pred, Y)
      print("b=",b,"accuracy=",accuracy[b])
      
    best_b = max(accuracy, key = accuracy.get)
    self.b = best_b
    
    print('Optimal value of b is', best_b)
    print('Highest accuracy is', accuracy[best_b])


# In[ ]:


mp_neuron = MPNeuron()
# mp_neuron.fit(X_binarised_train, Y_train)
mp_neuron.fit(train_binarised, train_y)


# In[ ]:


Y_val_pred = mp_neuron.predict(X_binarised_val)
accuracy_test = accuracy_score(Y_val_pred, Y_val)
print(accuracy_test)


# Preparing Submission File

# In[ ]:


test_pred = mp_neuron.predict(test_binarised)


# In[ ]:


type(test_pred)


# In[ ]:


submission = pd.DataFrame({'PhoneId':phoneid, 'Class':test_pred})
submission = submission[['PhoneId', 'Class']]
submission.Class.value_counts()


# In[ ]:


submission.to_csv("submission.csv", index=False)

