#!/usr/bin/env python
# coding: utf-8

# <div style = "font-size: 24px">
#     <h1><center>
#         Titanic Disaster
#     </center></h1>
# </div>

# -----------------

# # 1. Importing the Libraries

# In[ ]:


from warnings import filterwarnings
filterwarnings('ignore')
import warnings
warnings.simplefilter(action = 'ignore', category = FutureWarning)


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from lightgbm import LGBMClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import GridSearchCV
from statistics import mean 


# -------------

# # 2. Load Titanic Disaster Data

# In[ ]:


titanic_train = pd.read_csv('../input/titanic/train.csv')
train         = titanic_train.copy()


# In[ ]:


titanic_test = pd.read_csv('../input/titanic/test.csv')
test         = titanic_test.copy()


# In[ ]:


dataset = pd.merge(train, test, how = 'outer')


# In[ ]:


dataset.head()


# In[ ]:


titanic_train.head()


# In[ ]:


titanic_test.head()


# -------------

# # 3. Information about the datas

# In[ ]:


dataset.info()


# In[ ]:


train.info()


# In[ ]:


test.info()


# In[ ]:


dataset.shape, train.shape, test.shape


# In[ ]:


dataset.isnull().sum()


# In[ ]:


train.isnull().sum()


# In[ ]:


test.isnull().sum()


# --------

# # 4. New columns : Surname and Title

# In[ ]:


dataset['Surname'] = 'Surname'


# In[ ]:


dataset['Title'] = 'Title'


# In[ ]:


dataset.head()


# In[ ]:


for counter in range(0, len(dataset)):
    full_name                   = dataset['Name'][counter].split(',')
    surname                     = full_name[0]
    name                        = full_name[1].split('.')
    title                       = name[0]
    dataset['Surname'][counter] = surname
    dataset['Title'][counter]   = title


# In[ ]:


dataset.head()


# In[ ]:


dataset.drop(columns = ['PassengerId', 'Survived', 'Name', 'Cabin'], inplace = True)


# In[ ]:


dataset.head()


# In[ ]:


train['Surname'] = 'Surname'


# In[ ]:


train['Title'] = 'Title'


# In[ ]:


train.head()


# In[ ]:


for counter in range(0, len(train)):
    full_name                 = train['Name'][counter].split(',')
    surname                   = full_name[0]
    name                      = full_name[1].split('.')
    title                     = name[0]
    train['Surname'][counter] = surname
    train['Title'][counter]   = title


# In[ ]:


train.drop(columns = ['PassengerId', 'Name', 'Cabin'], inplace = True)


# In[ ]:


train.head()


# In[ ]:


train.isnull().sum()


# In[ ]:


test['Surname'] = 'Surname'


# In[ ]:


test['Title'] = 'Title'


# In[ ]:


test.head()


# In[ ]:


for counter in range(0, len(test)):
    full_name                = test['Name'][counter].split(',')
    surname                  = full_name[0]
    name                     = full_name[1].split('.')
    title                    = name[0]
    test['Surname'][counter] = surname
    test['Title'][counter]   = title


# In[ ]:


test.drop(columns = ['Name', 'Cabin'], inplace = True)


# In[ ]:


test.head()


# In[ ]:


test.isnull().sum()


# --------

# # 5. Converting Distinct Values to Binary using Binary Encoder

# <div style = "font-size: 15px">
#     
# Thanks to the **binary encoder** function, too many distinct values in any column can be converted to binary format.<br>
# In this binary format, the stimulation and / or non-stimulation of any column means a **different situation**.<br><br>
# 
# For example; When there are 100 distinct values in a column, 100 columns are created with label encoder and one hot encoder.<br>
# But if binary encoder is used, 2 ^ 7 = 128 (the smallest binary number times greater than 100), only **7** columns can be used.
#     
# </div>    

# In[ ]:


dataset.isnull().sum()


# In[ ]:


dataset.Embarked.fillna(-1000, inplace = True)


# In[ ]:


dataset = dataset[(dataset.Embarked != -1000)]


# In[ ]:


dataset.isnull().sum()


# ## 1) Surname

# In[ ]:


dataset_surname_unique = list(dataset.Surname.unique())


# In[ ]:


dataset_surname_unique[:10]


# In[ ]:


len(dataset_surname_unique)


# In[ ]:


surname_unique_list = []

for i in range(len(dataset_surname_unique)):
    surname_unique_list.append(i)
    
dataset_surname_number_array = np.array(surname_unique_list)


# In[ ]:


dataset_surname_number_array[:10]


# In[ ]:


def binary_encoder(x_array):
    #---------------------------------------------------------------
    binary_list = []
    #---------------------------------------------------------------
    for i in range(0, len(x_array)):
        x_ = f"{x_array[i]:b}"
        str_list = []
        for j in x_:
            str_list.append(int(j))
        binary_list.append(str_list)
    #---------------------------------------------------------------
    lengths = [len(i) for i in binary_list]
    max_length = max(lengths)
    #---------------------------------------------------------------
    for i in range(0, len(binary_list)):
        if len(binary_list[i]) < max_length:
            x_ = binary_list[i]
            x_.reverse()
            for j in range(0, int(max_length - len(binary_list[i]))):
                x_.append(0)
            x_.reverse()
            binary_list[i] = x_
    #---------------------------------------------------------------
    binary_array = np.array(binary_list)      
    return binary_array


# In[ ]:


surname_binary = binary_encoder(dataset_surname_number_array)


# In[ ]:


surname_binary[:10]


# ## 2) Title

# In[ ]:


dataset_title_unique = list(dataset.Title.unique())


# In[ ]:


dataset_title_unique[:10]


# In[ ]:


len(dataset_title_unique)


# In[ ]:


title_unique_list = []

for i in range(len(dataset_title_unique)):
    title_unique_list.append(i)
    
dataset_title_array = np.array(title_unique_list)


# In[ ]:


dataset_title_array[:10]


# In[ ]:


title_binary = binary_encoder(dataset_title_array)


# In[ ]:


title_binary[:10]


# ## 3) Ticket

# In[ ]:


dataset_ticket_unique = list(dataset.Ticket.unique())


# In[ ]:


dataset_ticket_unique[:10]


# In[ ]:


len(dataset_ticket_unique)


# In[ ]:


ticket_unique_list = []

for i in range(len(dataset_ticket_unique)):
    ticket_unique_list.append(i)
    
dataset_ticket_array = np.array(ticket_unique_list)


# In[ ]:


dataset_ticket_array[:10]


# In[ ]:


ticket_binary = binary_encoder(dataset_ticket_array)


# In[ ]:


ticket_binary[:10]


# --------------

# # 6. Grouping Fare and Age columns for Train and Test set

# In[ ]:


max(dataset['Fare']), min(dataset['Fare'])


# In[ ]:


max(dataset['Age']), min(dataset['Age'])


# In[ ]:


train['Fare_Seg'] = 0
train['Age_Seg']  = 0


# In[ ]:


train.head()


# In[ ]:


test['Fare_Seg'] = 0
test['Age_Seg']  = 0


# In[ ]:


test.head()


# In[ ]:


train['Fare'].fillna(-1000, inplace = True)
train['Age'].fillna(-1000, inplace = True)


# In[ ]:


test['Fare'].fillna(-1000, inplace = True)
test['Age'].fillna(-1000, inplace = True)


# In[ ]:


for i in range(len(train)):
    if (train['Fare'][i] == -1000):
        train['Fare_Seg'][i] = 0
    if (train['Fare'][i] >= 0 and train['Fare'][i] < 10):
        train['Fare_Seg'][i] = 1
    if (train['Fare'][i] >= 10 and train['Fare'][i] < 30):
        train['Fare_Seg'][i] = 2
    if (train['Fare'][i] >= 30 and train['Fare'][i] < 70):
        train['Fare_Seg'][i] = 3
    if (train['Fare'][i] >= 70 and train['Fare'][i] < 150):
        train['Fare_Seg'][i] = 4
    if (train['Fare'][i] >= 150 and train['Fare'][i] < 250):
        train['Fare_Seg'][i] = 5
    if (train['Fare'][i] >= 250 and train['Fare'][i] < 375):
        train['Fare_Seg'][i] = 6
    if (train['Fare'][i] >= 375):
        train['Fare_Seg'][i] = 7


# In[ ]:


for i in range(len(test)):
    if (test['Fare'][i] == -1000):
        test['Fare_Seg'][i] = 0
    if (test['Fare'][i] >= 0 and test['Fare'][i] < 10):
        test['Fare_Seg'][i] = 1
    if (test['Fare'][i] >= 10 and test['Fare'][i] < 30):
        test['Fare_Seg'][i] = 2
    if (test['Fare'][i] >= 30 and test['Fare'][i] < 70):
        test['Fare_Seg'][i] = 3
    if (test['Fare'][i] >= 70 and test['Fare'][i] < 150):
        test['Fare_Seg'][i] = 4
    if (test['Fare'][i] >= 150 and test['Fare'][i] < 250):
        test['Fare_Seg'][i] = 5
    if (test['Fare'][i] >= 250 and test['Fare'][i] < 375):
        test['Fare_Seg'][i] = 6
    if (test['Fare'][i] >= 375):
        test['Fare_Seg'][i] = 7


# In[ ]:


for i in range(len(train)):
    if (train['Age'][i] == -1000):
        train['Age_Seg'][i] = 0
    if (train['Age'][i] >= 0 and train['Age'][i] < 6):
        train['Age_Seg'][i] = 1
    if (train['Age'][i] >= 6 and train['Age'][i] < 15):
        train['Age_Seg'][i] = 2
    if (train['Age'][i] >= 15 and train['Age'][i] < 21):
        train['Age_Seg'][i] = 3
    if (train['Age'][i] >= 21 and train['Age'][i] < 30):
        train['Age_Seg'][i] = 4
    if (train['Age'][i] >= 30 and train['Age'][i] < 45):
        train['Age_Seg'][i] = 5
    if (train['Age'][i] >= 45 and train['Age'][i] < 60):
        train['Age_Seg'][i] = 6
    if (train['Age'][i] >= 60):
        train['Age_Seg'][i] = 7


# In[ ]:


for i in range(len(test)):
    if (test['Age'][i] == -1000):
        test['Age_Seg'][i] = 0
    if (test['Age'][i] >= 0 and test['Age'][i] < 6):
        test['Age_Seg'][i] = 1
    if (test['Age'][i] >= 6 and test['Age'][i] < 15):
        test['Age_Seg'][i] = 2
    if (test['Age'][i] >= 15 and test['Age'][i] < 21):
        test['Age_Seg'][i] = 3
    if (test['Age'][i] >= 21 and test['Age'][i] < 30):
        test['Age_Seg'][i] = 4
    if (test['Age'][i] >= 30 and test['Age'][i] < 45):
        test['Age_Seg'][i] = 5
    if (test['Age'][i] >= 45 and test['Age'][i] < 60):
        test['Age_Seg'][i] = 6
    if (test['Age'][i] >= 60):
        test['Age_Seg'][i] = 7


# In[ ]:


train.drop(columns = ['Fare', 'Age'], inplace = True)


# In[ ]:


train.head()


# In[ ]:


test.drop(columns = ['Fare', 'Age'], inplace = True)


# In[ ]:


test.head()


# ------------

# # 7. Removing null values from train rows

# In[ ]:


train.isnull().sum()


# In[ ]:


train.Embarked.fillna(-1000, inplace = True)


# In[ ]:


train = train[(train.Embarked != -1000)]


# In[ ]:


train.isnull().sum()


# --------------

# # 8. Transferring binary values from dataset to train and test set

# ## 1a) Train['Surname']

# In[ ]:


len(dataset_surname_unique)


# In[ ]:


dataset_surname_unique[:10]


# In[ ]:


surname_binary[:10]


# In[ ]:


surnames_lists_train = []


# In[ ]:


for i in train['Surname']:
    if(i in dataset_surname_unique):
        index = dataset_surname_unique.index(i)
        surnames_lists_train.append(surname_binary[index])
        
surnames_arr_train = np.array(surnames_lists_train)


# In[ ]:


surnames_arr_train[:10]


# ## 1b) Test['Surname']

# In[ ]:


len(dataset_surname_unique)


# In[ ]:


dataset_surname_unique[:10]


# In[ ]:


surname_binary[:10]


# In[ ]:


surnames_lists_test = []


# In[ ]:


for i in test['Surname']:
    if(i in dataset_surname_unique):
        index = dataset_surname_unique.index(i)
        surnames_lists_test.append(surname_binary[index])
        
surnames_arr_test = np.array(surnames_lists_test)


# In[ ]:


surnames_arr_test[:10]


# ## 2a) Train['Title']

# In[ ]:


len(dataset_title_unique)


# In[ ]:


dataset_title_unique[:10]


# In[ ]:


title_binary[:10]


# In[ ]:


title_list_train = []


# In[ ]:


for i in train['Title']:
    if(i in dataset_title_unique):
        index = dataset_title_unique.index(i)
        title_list_train.append(title_binary[index])
        
title_arr_train = np.array(title_list_train)


# In[ ]:


title_arr_train[:10]


# ## 2b) Test['Title']

# In[ ]:


len(dataset_title_unique)


# In[ ]:


dataset_title_unique[:10]


# In[ ]:


title_binary[:10]


# In[ ]:


title_list_test = []


# In[ ]:


for i in test['Title']:
    if(i in dataset_title_unique):
        index = dataset_title_unique.index(i)
        title_list_test.append(title_binary[index])
        
title_arr_test = np.array(title_list_test)


# In[ ]:


title_arr_test[:10]


# ## 3a) Train['Ticket']

# In[ ]:


len(dataset_ticket_unique)


# In[ ]:


dataset_ticket_unique[:10]


# In[ ]:


ticket_binary[:10]


# In[ ]:


ticket_list_train = []


# In[ ]:


for i in train['Ticket']:
    if(i in dataset_ticket_unique):
        index = dataset_ticket_unique.index(i)
        ticket_list_train.append(ticket_binary[index])
        
ticket_arr_train = np.array(ticket_list_train)


# In[ ]:


ticket_arr_train[:10]


# ## 3b) Test['Ticket']

# In[ ]:


len(dataset_ticket_unique)


# In[ ]:


dataset_ticket_unique[:10]


# In[ ]:


ticket_binary[:10]


# In[ ]:


ticket_list_test = []


# In[ ]:


for i in test['Ticket']:
    if(i in dataset_ticket_unique):
        index = dataset_ticket_unique.index(i)
        ticket_list_test.append(ticket_binary[index])
        
ticket_arr_test = np.array(ticket_list_test)


# In[ ]:


ticket_arr_test[:10]


# -----------------

# # 9. Modelling

# In[ ]:


train.head()


# In[ ]:


train.drop(columns = ['Ticket', 'Surname', 'Title'], inplace = True)


# In[ ]:


train.head()


# In[ ]:


X = train.iloc[:, 1:].values


# In[ ]:


X[:10]


# In[ ]:


y = train.iloc[:, 0].values


# In[ ]:


y[:10]


# ## 1) Label Encoders

# In[ ]:


labelencoder_gender = LabelEncoder()
X[:, 1]             = labelencoder_gender.fit_transform(X[:, 1])


# In[ ]:


X[:10]


# In[ ]:


labelencoder_embarked = LabelEncoder()
X[:, 4]               = labelencoder_embarked.fit_transform(X[:, 4])


# In[ ]:


X[:10]


# ## 2) binary_encoder for label encoder

# In[ ]:


X_without_embarked = np.delete(X, 4, 1)


# In[ ]:


X_without_embarked[:10]


# In[ ]:


X_embarked_binary = binary_encoder(X[:, 4])


# In[ ]:


X_embarked_binary[:10]


# ## 3) Concatenation process

# In[ ]:


X = np.concatenate([X_without_embarked, X_embarked_binary], axis = 1)


# In[ ]:


X = np.concatenate([X, ticket_arr_train], axis = 1)


# In[ ]:


X = np.concatenate([X, surnames_arr_train], axis = 1)


# In[ ]:


X = np.concatenate([X, title_arr_train], axis = 1)


# In[ ]:


X[:5]


# ## 4) Standard Scaler

# In[ ]:


sc        = StandardScaler()
X_scaled  = sc.fit_transform(X)


# In[ ]:


X_scaled[:5]


# In[ ]:


X_scaled.shape


# ------

# ## 5) LGBM

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.2, random_state = 12)


# In[ ]:


lgbm_model = LGBMClassifier(n_estimators      = 10000,
                            learning_rate     = 0.01,
                            subsample         = 0.85,
                            max_depth         = 16,
                            min_child_samples = 32,
                            random_state      = 502,
                            reg_lambda        = 10,
                            reg_alpha         = 3,
                            num_leaves        = 250)

lgbm_model.fit(X_train, y_train)


# In[ ]:


y_pred_train = lgbm_model.predict(X_train)
accuracy_score(y_train, y_pred_train)


# In[ ]:


y_pred_test = lgbm_model.predict(X_test)
accuracy_score(y_test, y_pred_test)


# In[ ]:


splits = [10, 15, 20, 25]

cross_list = []

for i in range(4):
    kfold = KFold(n_splits = splits[i], random_state = 12)
    lgbm_cross_val = cross_val_score(lgbm_model, X_scaled, y, cv = kfold, scoring = 'accuracy')
    cross_list.append(lgbm_cross_val.mean())

cross_list


# In[ ]:


print(f'Accuracy : %{mean(cross_list) * 100}')


# ------------------

# # 10. Testing

# In[ ]:


test.head()


# In[ ]:


test.drop(columns = ['PassengerId', 'Ticket', 'Surname', 'Title'], inplace = True)


# In[ ]:


test.head()


# In[ ]:


test_arr = test.values


# In[ ]:


test_arr[:10]


# ## 1) Label Encoder

# In[ ]:


labelencoder_gender = LabelEncoder()
test_arr[:, 1]      = labelencoder_gender.fit_transform(test_arr[:, 1])


# In[ ]:


test_arr[:10]


# In[ ]:


labelencoder_embarked = LabelEncoder()
test_arr[:, 4]        = labelencoder_embarked.fit_transform(test_arr[:, 4])


# In[ ]:


test_arr[:10]


# ## 2) binary_encoder for label encoder

# In[ ]:


test_arr_without_embarked = np.delete(test_arr, 4, 1)


# In[ ]:


test_arr_without_embarked[:10]


# In[ ]:


test_arr_embarked_binary = binary_encoder(test_arr[:, 4])


# In[ ]:


test_arr_embarked_binary[:10]


# ## 3) Concatenation process

# In[ ]:


test_arr = np.concatenate([test_arr_without_embarked, test_arr_embarked_binary], axis = 1)


# In[ ]:


test_arr = np.concatenate([test_arr, ticket_arr_test], axis = 1)


# In[ ]:


test_arr = np.concatenate([test_arr, surnames_arr_test], axis = 1)


# In[ ]:


test_arr = np.concatenate([test_arr, title_arr_test], axis = 1)


# In[ ]:


test_arr[:5]


# ## 4) Standard Scaler

# In[ ]:


sc               = StandardScaler()
test_arr_scaled  = sc.fit_transform(test_arr)


# In[ ]:


test_arr_scaled[:5]


# In[ ]:


test_arr_scaled.shape


# ## 5) Predictions

# In[ ]:


y_pred = lgbm_model.predict(test_arr_scaled)


# In[ ]:


gender_sub = pd.read_csv('../input/titanic/gender_submission.csv')


# In[ ]:


gender_sub.head()


# In[ ]:


y_true = gender_sub.iloc[:, 1].values


# In[ ]:


accuracy_score(y_true, y_pred)


# -------------------

# # 11. My Submission

# In[ ]:


titanic_test.head()


# In[ ]:


submission = titanic_test.iloc[:, 0].values


# In[ ]:


submission[:10]


# In[ ]:


y_pred[:10]


# In[ ]:


submission.shape, y_pred.shape


# In[ ]:


submission = np.reshape(submission, (-1, 1))


# In[ ]:


submission[:10]


# In[ ]:


y_pred = np.reshape(y_pred, (-1, 1))


# In[ ]:


y_pred[:10]


# In[ ]:


submission = np.concatenate([submission, y_pred], axis = 1)


# In[ ]:


submission[:10]


# In[ ]:


sub_df = pd.DataFrame(data = submission, columns = ['PassengerId', 'Survived'])


# In[ ]:


sub_df.head(5)


# In[ ]:


sub_df.to_csv('My_Submission.csv', index = False)


# ---------------------

# In[ ]:




