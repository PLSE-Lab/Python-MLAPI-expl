#!/usr/bin/env python
# coding: utf-8

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


# Task: To predict whether the user likes the mobile phone or not. <br>
# Assumption: If the average rating of mobile >= threshold, then the user likes it, otherwise not.

# <b>Missing values:</b><br>
# 'Also Known As'(459),'Applications'(421),'Audio Features'(437),'Bezel-less display'(266),'Browser'(449),'Build Material'(338),'Co-Processor'(451),'Display Colour'(457),'Mobile High-Definition Link(MHL)'(472),'Music'(447)
# 'Email','Fingerprint Sensor Position'(174),'Games'(446),'HDMI'(454),'Heart Rate Monitor'(467),'IRIS Scanner'(467),
# 'Optical Image Stabilisation'(219),'Other Facilities'(444),'Phone Book'(444),'Physical Aperture'(87),'Quick Charging'(122),'Ring Tone'(444),'Ruggedness'(430),SAR Value(315),'SIM 3'(472),'SMS'(470)', 'Screen Protection'(229),'Screen to Body Ratio (claimed by the brand)'(428),'Sensor'(242),'Software Based Aperture'(473),
# 'Special Features'(459),'Standby time'(334),'Stylus'(473),'TalkTime'(259), 'USB Type-C'(374),'Video Player'(456),
# 'Video Recording Features'(458),'Waterproof'(398),'Wireless Charging','USB OTG Support'(159), 'Video ,'Recording'(113),'Java'(471),'Browser'(448)
# 
# <b>Very low variance:</b><br>
# 'Architecture'(most entries are 64-bit),'Audio Jack','GPS','Loudspeaker','Network','Network Support','Other Sensors'(28),'SIM Size', 'VoLTE'
# 
# 
# <b>Multivalued:</b><br>
# 'Colours','Custom UI','Model'(1),'Other Sensors','Launch Date'
# 
# <b>Not important:</b><br>
# 'Bluetooth', 'Settings'(75),'Wi-Fi','Wi-Fi Features'
# 
# <b>Doubtful:</b><br>
# 'Aspect Ratio','Autofocus','Brand','Camera Features','Fingerprint Sensor'(very few entries are missing),
# 'Fingerprint Sensor Position', 'Graphics'(multivalued),'Image resolution'(multivalued),'SIM Size','Sim Slot(s)', 'User Available Storage', 'SIM 1', 'SIM 2','Shooting Modes', 'Touch Screen'(24), 'USB Connectivity'
#     
# <b>To check:</b><br>
# 'Display Type','Expandable Memory','FM Radio'
# 
# <b>High Correlation with other features</b><br>
# 'SIM Slot(s)' high correlation with SIM1
# 'Weight' has high high correlation with capacity , screen-to-body ratio
# 'Height' - screen size is also there
#     
# <b>Given a mobile, we can't directly get these features</b><br>
# 'Rating Count', 'Review Count'
# 
# <b>Keeping:</b><br>
# 'Capacity','Flash'(17),'Height'(22),'Internal Memory'(20, require cleaning),'Operating System'(25, require cleaning), 'Pixel Density'(1, clean it),'Processor'(22, clean it), 'RAM'(17, clean), 'Rating','Resolution'(cleaning), 'Screen Resolution','Screen Size', 'Thickness'(22), 'Type','User Replaceable','Weight'(cleaning),'Sim Size'(), 'Other Sensors'(28), 'Screen to Body Ratio (calculated)','Width',
# 

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


# # Removing features

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


# # Filling Missing values

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


# Not very important feature

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

#combined = pd.get_dummies(combined)
#print(combined.shape)
#print(combined.columns)

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


# ## Modelling Code

# In[ ]:


from random import randint
from sklearn.metrics import accuracy_score

class MPNeuron:
  
  def __init__(self):
    self.b = None
  
  def model(self,x):
    return(sum(x) >= self.b)
  
  def predict(self,X):
    Y = []
    for x in X:
      result = self.model(x)
      Y.append(result)
    return(np.array(Y))
  
  def fit(self, X, Y):
    accuracy = {}
    
    for b in range(X.shape[1] + 1):
      self.b = b
      Y_pred = self.predict(X)
      accuracy[b] = accuracy_score(Y_pred,Y)
    
    best_b = max(accuracy,key = accuracy.get)
    self.b = best_b
    
    print('Optimal value of b: ',best_b)
    print('Highest accuracy is: ',accuracy[best_b])
    
    return(accuracy[best_b])


# In[ ]:


train_new['Class'] = train_new['Rating'].map(lambda x: 0 if x < THRESHOLD else 1)
x = train_new.drop('Class',axis = 1)
x = x.drop('Rating',axis = 1)
y = train_new['Class']


# In[ ]:


x_train = x[:]
y_train = y[:]


# In[ ]:


#Split the data to test and train set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,stratify = y,
                                                random_state = 1)
print(x.shape,x_train.shape,x_test.shape)
print(y.shape,y_train.shape,y_test.shape)


# In[ ]:


#Copy train and test set
x_train_bin = x_train[:]
x_test_bin = x_test[:]
x_test_final_bin = test_new[:]


# In[ ]:


#Correcting name of brand
x_train_bin['Brand']  = x_train['Brand'].map(lambda x: 'Oppo' if x == 'OPPO' else x)
x_test_bin['Brand']  = x_test['Brand'].map(lambda x: 'Oppo' if x == 'OPPO' else x)
x_test_final_bin['Brand']  = test_new['Brand'].map(lambda x: 'Oppo' if x == 'OPPO' else x)


# In[ ]:


#Correcting RAM
x_train_bin['RAM']  = x_train['RAM'].map(lambda x: 0.5 if x == 512 else x)
x_test_bin['RAM']  = x_test['RAM'].map(lambda x: 0.5 if x == 512 else x)
x_test_final_bin['RAM']  = test_new['RAM'].map(lambda x: 0.5 if x == 512 else x)


# In[ ]:


#Removing PhoneId from data
x_train_bin = x_train_bin.drop('PhoneId',axis = 1)
x_test_bin = x_test_bin.drop('PhoneId',axis = 1)
x_test_final_bin = x_test_final_bin.drop('PhoneId',axis = 1)


# In[ ]:


#Defining auomated lambda function
def lambda_automated(lambda_list,col_name,test_train_object):
    test_train_object[col_name] = test_train_object[col_name].map(lambda x: 1 if x in lambda_list else 0)
    return(test_train_object)


# In[ ]:


#These are non-numeric objects. So we create a list of high accuracy values
major_phone = ['Apple','Comio','Google' ,'Huawei' ,'LeEco' ,'Meizu' ,'Motorola' ,'OnePlus' ,'Realme' ,'Ulefone' ,'Xiaomi Poco' ,'Oppo' ,'Vivo' ,'Xiaomi' ,'Honor' ,'Asus','Tecno' ,'Samsung',
              'Itel','Moto','Lava','Infinix','Nokia','Lenovo','Gionee']
major_os = ['iOS','Nokia','Other']
major_cores = ['312','Hexa','Other','Tru-Octa']
major_simslots = ['Dual SIM, GSM+GSM, Dual VoLTE','Single SIM, GSM']
sim1 = ['2G']
sim2 = ['2G']

lambda_cols = {'Brand':major_phone,'os_name':major_os,'Num_cores':major_cores,
               'SIM Slot(s)':major_simslots,'Sim1':sim1,'SIM 2':sim2}

for lambda_col in lambda_cols.keys():
    x_train_bin = lambda_automated(lambda_cols[lambda_col],lambda_col,x_train_bin)
    x_test_bin = lambda_automated(lambda_cols[lambda_col],lambda_col,x_test_bin)
    x_test_final_bin = lambda_automated(lambda_cols[lambda_col],lambda_col,x_test_final_bin)


# In[ ]:


#Defining auomated lambda function for numerics
def lambda_automated(val,col_name,test_train_object):
    test_train_object[col_name] = test_train_object[col_name].map(lambda x: 0 if x < val else 1)
    return(test_train_object)


# In[ ]:


remain_cols = {'Pixel Density': 240,'Resolution':16,'Height':155.6,'Weight': 78,'Capacity':3050,
               'Processor_frequency':2.2,'Screen Size':5.8,'RAM':6,'Internal Memory':4,
               'Screen to Body Ratio (calculated)':63.6}

for remain_col in remain_cols:
    x_train_bin = lambda_automated(remain_cols[remain_col],remain_col,x_train_bin)
    x_test_bin = lambda_automated(remain_cols[remain_col],remain_col,x_test_bin)
    x_test_final_bin = lambda_automated(remain_cols[remain_col],remain_col,x_test_final_bin)


# In[ ]:


max_list = ['Brand', 'Screen to Body Ratio (calculated)', 'os_name','Processor_frequency','Internal Memory']
mp_neuron = MPNeuron()
train_accuracy = mp_neuron.fit(np.array(x_train_bin[max_list]),y_train)
y_test_pred = mp_neuron.predict(np.array(x_test_bin[max_list]))
accuracy_test = accuracy_score(y_test_pred,y_test)
print(accuracy_test)


# In[ ]:


y_test_pred_final = mp_neuron.predict(np.array(x_test_final_bin[max_list]))


# In[ ]:


submission = pd.DataFrame({'PhoneId':test_new['PhoneId'], 'Class':y_test_pred_final.astype(int)})
submission = submission[['PhoneId', 'Class']]
submission.to_csv("Results.csv", index=False)

