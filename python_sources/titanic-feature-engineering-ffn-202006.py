#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
from numpy.random import seed

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.set_option('display.max_columns', 500)

from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn import linear_model

import tensorflow as tf #deep learning with keras
from tensorflow.keras import regularizers

from matplotlib import pyplot

import re


# In[ ]:


#ensure reproducibility
seed(1)
tf.random.set_seed(2)


# In[ ]:


#global variables

#data paths
train_path = "../input/titanic/train.csv"
test_path = "../input/titanic/test.csv"
output_prediction_path = "submission_titanic_v29_20200719.csv"

#preprocessing
list_col_remove = ["Ticket","Cabin"]
list_categ_col_to_encode = ["Cabin_letter", "Cabin_first_digit", "Embarked", "Title"]
list_ordinal_categ_col_to_encode = ['Fare_buckets', 'Age_buckets', 'Age_Class_buckets']
special_remove_later = ["Name"]
target_col = 'Survived'
components_PCA = 'full' #'full': no PCA applied, 'high': 90% of the total number of columns, 'medium': 75%, 'low': 50%, 'very_low': 33%
test_split_ratio = 0.25

#deep learning model
##initialization
number_nodes_initial = 4096 #4096
regul_L1 = 0.016 #0.02
regul_L2 = 0.014 #0.015
drop_out = 0.2
##training
epochs_max = 1000 #not much improvements past this point: 1000
epoch_max_full = 5 #after training on the train set, train on the validation set (to see all data)
patience_eval = int(round(epochs_max/1)) #change if we want to have an actual early stopping (e.g. 40)
batch_size = 15
checkpoint_filepath = '.mdl_wts.hdf5'
initial_learning_param = 0.0001


# In[ ]:


#create needed arrays for training and predicting
X_train, X_test, y_train, y_test, preprocessed_test_df =     create_train_val_predict_arrays(train_path,test_path, list_col_remove, list_categ_col_to_encode, list_ordinal_categ_col_to_encode,                                     special_remove_later, target_col, components_PCA, test_split_ratio)


# In[ ]:


#build and train model
model = model_initialization(number_nodes_initial, X_train, regul_L1, regul_L2, drop_out, initial_learning_param)
history, model = model_training(model, X_train, y_train, X_test, y_test, epochs_max, patience_eval, batch_size, checkpoint_filepath, initial_learning_param)


# In[ ]:


#Adjust model with full data (OPTIONAL)
# X_full = np.concatenate((X_train, X_test))
# y_full = np.concatenate((y_train, y_test))
# model.fit(X_full, y_full, epochs=epoch_max_full)


# In[ ]:


#evaluate model
train_acc, test_acc, confusion_matrix_results, classification_report_results = model_evaluation(model, X_train, y_train, X_test, y_test)
print("%s: %.2f%%" % (model.metrics_names[1], test_acc[1]*100))
print(confusion_matrix_results)
print(classification_report_results)

plot_history('accuracy')
# plot_history('loss')


# In[ ]:


#export model
export_pred(model, preprocessed_test_df, output_prediction_path)


# # FUNCTIONS
# ## Preprocessing
# ### 1. per feature

# In[ ]:


def add_features(df_prep, target_col, list_col_remove, n_ticket, tick_surv):
    '''
    combines all the functions below
    '''
    for function_adding_information in [cabin_information, name_information, family_information, fare_information]:
        df_prep = function_adding_information(df_prep)
        
    df_prep = ticket_information(df_prep, target_col, n_ticket, tick_surv)
    #df_prep = age_information(df_prep, list_col_remove, list_categ_col_to_encode)
 
    #remove non-numerical remaining columns
    df_prep = df_prep.drop(list_col_remove, axis=1)
    
    return df_prep

def cabin_information(df_prep):
    '''
    add first letter and first digit information wherever possible
    '''
    #feature to know whether the user has a cabin or not
    df_prep["Has_cabin"] = df_prep["Cabin"].isna().astype(int)
    
    #feature to know the length of the cabin string (some users have multiple cabins) -> useless
    #df_prep["Cabin_length"] = df_prep["Cabin"].str.len().fillna(0)
    
    #first letter of the cabin
    df_prep["Cabin_letter"] = df_prep["Cabin"].astype(str).str[0]

    #cabin digits (first digit may mean the floor on the boat)
    #Note: the number of digits may mean the passenger was far at the end -> NEXT STEP
    df_prep["Cabin_digits"] = df_prep["Cabin"].astype(str).str[1:]

    #get index of users who have a cabin digit after the first letter (most users who have a non-NaN value do)
    index_with_cabin_digits = df_prep.loc[df_prep['Cabin_digits'].str.isdigit(), :].index

    #initialize the column
    df_prep["Cabin_first_digit"] = "n"
    #fill in the column for the users with cabin digits
    df_prep.loc[index_with_cabin_digits, "Cabin_first_digit"] = df_prep.loc[index_with_cabin_digits,'Cabin_digits'].astype(str).str[0]
    df_prep.drop('Cabin_digits', axis=1, inplace=True)
    
    return df_prep

def name_information(df_prep):
    '''
    adds title information using regex from the name
    '''
    # Create a new feature Title, containing the titles of passenger names
    df_prep['Title'] = df_prep['Name'].apply(get_title)
    # Group all non-common titles into one single grouping "Rare"
    regex_mapping = {'Mlle': 'Miss', 'Major': 'Mr', 'Col': 'Rare', 'Sir': 'Rare', 'Rev': 'Rare',
               'Don': 'Mr', 'Mme': 'Mrs', 'Jonkheer': 'Rare', 'Lady': 'Mrs',
               'Capt': 'Rare', 'Countess': 'Rare', 'Ms': 'Miss', 'Dona': 'Rare'}
    df_prep.replace({'Title': regex_mapping}, inplace=True)
    # Mapping titles
    title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, "Master": 3, "Rare": 4, "None":5}
    df_prep['Title'] = df_prep['Title'].map(title_mapping)
    df_prep['Title'] = df_prep['Title'].fillna(5)

    return df_prep

def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
            return title_search.group(1)
    return "None"

def family_information(df_prep):
    '''
    adds 2 columns about how big the user's family is on board and whether he is alone or not
    '''
    # Create new feature FamilySize as a combination of SibSp and Parch
    df_prep['FamilySize'] = df_prep['SibSp'] + df_prep['Parch'] + 1

    # Create new feature IsAlone from FamilySize
    df_prep['isFamily'] = 1
    df_prep.loc[df_prep['FamilySize'] == 1, 'isFamily'] = 0
    
    return df_prep

def fare_information(df_prep):
    #special column for passengers who had a free tickets
    df_prep['Fare_free'] = 0
    index_free_fare = df_prep[df_prep['Fare'] == 0].index
    df_prep.loc[index_free_fare, 'Fare_free'] = 1
    
    #for these 'outliers', we change the value to NaN (handled just below)
    df_prep.loc[index_free_fare, 'Fare'] = np.nan
    
    #fill in missing values with median fare (not average as the ticket prices are not continuous)
    df_prep['Fare'] = df_prep['Fare'].fillna(df_prep['Fare'].median())
    
    #we transform the fare feature as it doesn't follow a normal distribution
    df_prep['Fare'] = np.log(np.log(df_prep['Fare']))
#     df_prep['Fare'] = np.log(df_prep['Fare'])
    
    #Create fare buckets:
    df_prep['Fare_buckets'] = 0
    df_prep.loc[ df_prep['Fare'] <= 7.91, 'Fare_buckets'] = 0
    df_prep.loc[(df_prep['Fare'] > 7.91) & (df_prep['Fare'] <= 14.454), 'Fare_buckets'] = 1
    df_prep.loc[(df_prep['Fare'] > 14.454) & (df_prep['Fare'] <= 31), 'Fare_buckets'] = 2
    df_prep.loc[ df_prep['Fare'] > 31, 'Fare_buckets'] = 3
    df_prep['Fare_buckets'] = df_prep['Fare_buckets'].astype(int)
    
    return df_prep

def get_train_ticket_information(train_df, target_col):
    '''
    make sure a ticket in test that we have data on train is better assigned
    We need to do this before the preprocessing happens as we will use these mappings in the preprocessing 
    '''
    
    boy = (train_df['Name'].str.contains('Master')) | ((train_df['Sex']==0) & (train_df['Age']<13))
    female = (train_df['Sex']==1)
    boy_or_female = boy | female   
    # no. females + boys on ticket
    n_ticket = train_df[boy_or_female].groupby('Ticket')[target_col].count()
    # survival rate amongst females + boys on ticket
    tick_surv = train_df[boy_or_female].groupby('Ticket')[target_col].mean()
    
    return n_ticket, tick_surv

def ticket_information(df_prep, target_col, n_ticket, tick_surv):
    '''
    types of ticket (i.e. location on the boat) and info about people on tickets (boy and female)
    Note: there is no empty ticket in either train or test set
    '''
    #get index of users who have an integer as ticket (some users have text included in their ticket ID)
    index_with_ticket_digits = df_prep.loc[df_prep['Ticket'].str.isdigit(), :].index
    #create a column equal to 1 (about 3/4 of all passengers) if integer and 0 if contains text as well
    df_prep["Ticket_integer"] = 0
    df_prep.loc[index_with_ticket_digits, "Ticket_integer"] = 1

    # if ticket exists in training data, fill NTicket with no. women+boys on that ticket in the training data
    df_prep = ticket_replace(df_prep, n_ticket, 'NTicket')
    # if ticket exists in training data, fill TicketSurv with women+boys survival rate in training data  
    df_prep = ticket_replace(df_prep, tick_surv, 'TicketSurv')
    
    return df_prep

def ticket_replace(df_prep, replace_param, new_col_name):
    '''
    if ticket exists in training data, fill the new column with women+boys count / survival rate (from training data)
    otherwise TicketSurv=0
    '''
    df_prep[new_col_name] = df_prep['Ticket'].replace(replace_param)
    df_prep.loc[~df_prep['Ticket'].isin(replace_param.index),new_col_name] = 0
    df_prep[new_col_name] = df_prep[new_col_name].fillna(0)
    
    return df_prep


def age_information(df_prep, special_remove_later, threshold):
    
    #fill in missing values
#     df_prep['Age'] = df_prep['Age'].fillna(df_prep['Age'].median())
    df_prep = age_fill_na(df_prep, special_remove_later)
    
    #add "boy" column (women and children usually would survive more)
    df_prep['Boy'] = (df_prep['Name'].str.contains('Master')) | ((df_prep['Sex']==0) & (df_prep['Age']<13))
    
    #Create Age buckets
    df_prep['Age_buckets'] = 0
    df_prep.loc[ df_prep['Age'] <= 16, 'Age_buckets'] = 5
    df_prep.loc[(df_prep['Age'] > 16) & (df_prep['Age'] <= 32), 'Age_buckets'] = 1
    df_prep.loc[(df_prep['Age'] > 32) & (df_prep['Age'] <= 48), 'Age_buckets'] = 2
    df_prep.loc[(df_prep['Age'] > 48) & (df_prep['Age'] <= 64), 'Age_buckets'] = 3
    df_prep.loc[ df_prep['Age'] > 64, 'Age_buckets'] = 4
    
    #add Age*Class buckets
    df_prep['Age_Class_buckets'] = df_prep['Age_buckets'] * df_prep['Pclass']
    
    #one-hot encoding on these -> NOT IN THIS VERSION FOR THE 'ORDINAL' CATEGORICAL COLUMNS
    df_prep = do_all_one_hot_encodings(df_prep,['Age_buckets','Age_Class_buckets'],threshold)
    
    return df_prep

def age_fill_na(df_prep, special_remove_later):
    '''
    use linear regression model to fill missing Age values
    '''
    list_col_usable = [col for col in df_prep.columns if col not in special_remove_later]
#     print(list_col_usable)

    index_nan = df_prep[df_prep['Age'].isna()].index
    index_no_nan = df_prep[~df_prep['Age'].isna()].index
    
    # Create linear regression object
    regr = linear_model.LinearRegression()
    # Train the model
    regr.fit(df_prep.loc[index_no_nan, list_col_usable].drop('Age', axis = 1), df_prep.loc[index_no_nan, 'Age'])
    # Make predictions
    df_prep.loc[index_nan, 'Age'] = regr.predict(df_prep.loc[index_nan, list_col_usable].drop('Age', axis = 1))

    return df_prep

def sex_renaming(df_prep):
    '''
    replace male, female respectively by 1 and 0 (there is no missing values in the "Sex" column)
    '''
#     print(df_prep["Sex"])
    df_prep["Sex"] = df_prep["Sex"].replace({"male": 1, "female": 0})
    
    return df_prep


# ### 2. for entire dataset

# In[ ]:


def handle_missing_values(df_prep):
    '''
    handle missing valuesfor the numerical columns (it is handled after the one hot encoding for the categorical columns)
    add a column wth the previously missing value data
    #impute with mean (as most of the remaining columns here do not have many missing values)
    '''
    list_col_with_missing_values = ['Has_cabin','Cabin_length','FamilySize','isFamily']
    list_actual_col_missing_values = [col for col in df_prep.columns if col in list_col_with_missing_values]
    for col in list_actual_col_missing_values:
        #fill in the missing values with the mean
        df_prep[col] = df_prep[col].fillna(df_prep[col].mean())
        
    #add a missing column before filling in the missing values
    list_add_na_col = ['Age']
    list_actual_add_na_col = [col for col in df_prep.columns if col in list_add_na_col]
    for col in list_actual_add_na_col:
        missing_col_name = col + "_na"
        df_prep[missing_col_name] = 0
        index_missing_values_for_col = df_prep[df_prep[col].isnull()].index
        df_prep.loc[index_missing_values_for_col, missing_col_name] = 1

    return df_prep

def do_all_one_hot_encodings(df_prep,list_categ_col_to_encode,threshold):
    '''
    one-hot encode the necessary columns (all the categorical columns with at least 3 significant values)
    '''
    for categ_col in list_categ_col_to_encode:
        df_prep = one_hot_significant(df_prep,categ_col,threshold)
    
    return df_prep

def one_hot_significant(df_prep,col_to_encode,threshold):
    '''
    we build a column for values with at least 'threshold' 10 instances
    '''
    #get the list of significant (enough) values to encode
    val_count = df_prep[col_to_encode].value_counts()
    list_one_hot = [val for val in val_count.index if val_count[val] >= threshold and val != "n"] #avoid NaN values

    #get users who have values within this list
    index_users_val_in_list = df_prep[df_prep[col_to_encode].isin(list_one_hot)].index

    #proceed to one-hot encode for these users
    dummy_temp = pd.get_dummies(df_prep.loc[index_users_val_in_list,col_to_encode], prefix=col_to_encode)

    #join with the original df and fill in the users not in this index with values of 0
    df_prep = pd.merge(df_prep, dummy_temp, left_index = True, right_index = True, how="left")
    df_prep.drop(col_to_encode, axis=1, inplace=True)
    list_one_hot_new_columns = [col for col in df_prep.columns if col_to_encode in col]
    df_prep[list_one_hot_new_columns] = df_prep[list_one_hot_new_columns].fillna(0)
    
    return df_prep

def apply_PCA(df_train, df_test, target_col, components_PCA='medium'):
    '''
    apply PCA on both training and test set to remove potential multi-colinearity
    '''
    
    #determines the number of components used based on the user desired reduction
    if components_PCA == 'full':
        return df_train, df_test
        
    elif components_PCA == 'high':
        no_component = int(round(len(df_test.columns) * 0.9))
    
    elif components_PCA == 'low':
        no_component = int(round(len(df_test.columns) * 0.5))
    
    elif components_PCA == 'very_low':
        no_component = int(round(len(df_test.columns) * 0.33))
        
    else: #when 'medium', i.e. default
        no_component = int(round(len(df_test.columns) * 0.75))
    
    # Separating out the features
    train = df_train.iloc[:,1:].values
    test = df_test.values

    pca = PCA(n_components = no_component)
    principalComponents_train = pca.fit_transform(train)
    
    principalComponents_test = pca.transform(test)
    
#     print("PCA variance explained with {} variables:".format(no_component), sum(pca.explained_variance_ratio_))
    
    principalDf_train = pd.DataFrame(data = principalComponents_train, index = df_train.index)
    finalDf_train = pd.concat([df_train[[target_col]], principalDf_train], axis = 1)
    
    finalDf_test = pd.DataFrame(data = principalComponents_test, index = df_test.index)
    
    return finalDf_train, finalDf_test


# ### 3. apply and get all required datasets

# In[ ]:


def create_train_val_predict_arrays(train_path, test_path, list_col_remove, list_categ_col_to_encode, list_ordinal_categ_col_to_encode,                                     special_remove_later, target_col, components_PCA, test_split_ratio):
    '''
    outputs the preprocessed training, validation and test sets based on original data
    '''
    #imports the original training and test sets
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    #preprocess training and test set
    preprocessed_train_df, preprocessed_test_df = preprocessing_train_test_df(train_df, test_df, list_col_remove, list_categ_col_to_encode, list_ordinal_categ_col_to_encode,                                                                               special_remove_later, target_col, components_PCA)

    #create 'train' and 'test' (i.e. validation) sets based on the original training data
    X = preprocessed_train_df.iloc[:,1:].values
    y = preprocessed_train_df.iloc[:,0].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split_ratio, random_state=42)

    return X_train, X_test, y_train, y_test, preprocessed_test_df


def preprocessing_train_test_df(train_df, test_df, list_col_remove, list_categ_col_to_encode, list_ordinal_categ_col_to_encode, special_remove_later, target_col, components_PCA):
    '''
    creates the preprocessed versions of the training and test sets
    '''
    #get n_ticket, tick_surv from the train data (to make sure a ticket in test that we have data on train is better assigned)
    n_ticket, tick_surv = get_train_ticket_information(train_df, target_col)
    
    #get the preprocessed df
    preprocessed_train_df = preprocessing_df(train_df, target_col, list_col_remove, list_categ_col_to_encode, list_ordinal_categ_col_to_encode, special_remove_later, n_ticket, tick_surv)
    preprocessed_test_df = preprocessing_df(test_df, target_col, list_col_remove, list_categ_col_to_encode, list_ordinal_categ_col_to_encode,special_remove_later, n_ticket, tick_surv,threshold=0)

    #remove the columns from preprocessed_test_df which are not in preprocessed_train_df
    list_col_test_and_train = [col for col in preprocessed_test_df.columns if col in preprocessed_train_df.columns]
    preprocessed_test_df = preprocessed_test_df[list_col_test_and_train]

    #handle the missing columns from preprocessed_test_df in preprocessed_train_df (add column and fill in with value of 0)
    list_col_train_not_test = [col for col in preprocessed_train_df.columns if col not in preprocessed_test_df.columns and col != target_col]
    for missing_col in list_col_train_not_test:
        preprocessed_test_df[missing_col] = 0
        
    #scale the data on train set and apply it on the test set. 
    #Note: RobustScaler should handle outliers better when scaling
#     scaler = MinMaxScaler()
    scaler = RobustScaler()
    preprocessed_train_df[preprocessed_test_df.columns] = scaler.fit_transform(preprocessed_train_df[preprocessed_test_df.columns])
    preprocessed_test_df[preprocessed_test_df.columns] = scaler.transform(preprocessed_test_df[preprocessed_test_df.columns])
    
    #PCA (to remove potential multi-colinearity)
    preprocessed_train_df, preprocessed_test_df = apply_PCA(preprocessed_train_df, preprocessed_test_df, target_col, components_PCA)
        
    return preprocessed_train_df, preprocessed_test_df


def preprocessing_df(df, target_col, list_col_remove, list_categ_col_to_encode, list_ordinal_categ_col_to_encode, special_remove_later, n_ticket, tick_surv, threshold=10):
    '''
    preprocesses a df
    '''
    #set index as the passenger ID
    df_prep = df.set_index("PassengerId")
    
    #enriches data with additional features
    df_prep = add_features(df_prep, target_col, list_col_remove, n_ticket, tick_surv)

    #rename necessary values to numerical
    df_prep = sex_renaming(df_prep)

    #categorical column handling with one-hot encoding
    list_actual_col_encode = [col for col in df_prep.columns if col in list_categ_col_to_encode]
    df_prep = do_all_one_hot_encodings(df_prep,list_actual_col_encode,threshold)

    #handle other missing values
    df_prep = handle_missing_values(df_prep)
    
    #special handling for age (including missing values, one hot encoding for Age bucket)
    df_prep = age_information(df_prep, special_remove_later, threshold)
    
    #ordinal column handling with one-hot encoding
    list_actual_ord_col_encode = [col for col in df_prep.columns if col in list_ordinal_categ_col_to_encode]
    df_prep = do_all_one_hot_encodings(df_prep,list_actual_ord_col_encode,threshold)
    
    #remove remaining useless columns
    df_prep.drop(special_remove_later, axis=1, inplace=True)


    return df_prep


# ## Modeling

# In[ ]:


def model_initialization(number_nodes_initial, X_train, regul_L1, regul_L2, drop_out, initial_learning_param): 
    '''
    define the model: deep learning classification model
    '''
    
    #define layers used in the DNN
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(number_nodes_initial,activation='relu', kernel_regularizer=regularizers.l1_l2(l1=regul_L1, l2=regul_L2), input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dropout(drop_out),
        tf.keras.layers.Dense(number_nodes_initial/2,activation='relu', kernel_regularizer=regularizers.l2(regul_L2)),
        tf.keras.layers.Dense(number_nodes_initial/2,activation='relu', kernel_regularizer=regularizers.l2(regul_L2)),
        tf.keras.layers.Dense(number_nodes_initial/2,activation='relu', kernel_regularizer=regularizers.l2(regul_L2)),
        tf.keras.layers.Dense(number_nodes_initial/2,activation='relu', kernel_regularizer=regularizers.l2(regul_L2)),
        tf.keras.layers.Dense(number_nodes_initial/2,activation='relu', kernel_regularizer=regularizers.l2(regul_L2)),
        tf.keras.layers.Dense(number_nodes_initial/2,activation='relu', kernel_regularizer=regularizers.l2(regul_L2)),
        tf.keras.layers.Dense(1, activation='sigmoid'),

    ])
    
    # optimizer_model = tf.keras.optimizers.RMSprop(initial_learning_param)
    optimizer_model = tf.keras.optimizers.Adam(initial_learning_param)

    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer=optimizer_model, metrics=['accuracy'])

    return model


def model_training(model, X_train, y_train, X_test, y_test, epochs_max, patience_eval, batch_size, checkpoint_filepath, initial_learning_param):
    '''
    train the model
    '''
    #early stopping
    es = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience = patience_eval, restore_best_weights=True)
    
    #save 
    mcp_save = tf.keras.callbacks.ModelCheckpoint(checkpoint_filepath, save_best_only=True, monitor='val_loss', mode='min')

    #training
    history = model.fit(X_train, y_train, epochs=epochs_max, batch_size=batch_size, validation_data=(X_test, y_test),callbacks=[es, mcp_save])
    
    # The model weights (that are considered the best) are loaded into the model.
    model.load_weights(checkpoint_filepath)
    
    return history, model


# ## Evaluating

# In[ ]:


def plot_history(parameter):
    '''
    plot training history
    '''
    pyplot.plot(history.history[parameter], label='train')
    pyplot.plot(history.history['val_' + parameter], label='test')
    pyplot.legend()
    pyplot.show()
    
    return None

def model_evaluation(model, X_train, y_train, X_test, y_test):
    '''
    evaluate model based on training and validation sets
    '''
    
    train_acc = model.evaluate(X_train, y_train, verbose=0)
    test_acc = model.evaluate(X_test, y_test, verbose=0)

    Y_pred = model.predict(X_test)
    y_pred = np.around(Y_pred,0).astype(int)
    
    #confusion matrix (rows: predicted, columns: actual)
    confusion_matrix_results = confusion_matrix(y_test, y_pred)
    classification_report_results = classification_report(y_test, y_pred)
    
    return train_acc, test_acc, confusion_matrix_results, classification_report_results


# ## Exporting

# In[ ]:


def export_pred(model, preprocessed_test_df, output_prediction_path):
    '''
    export actual submission
    '''
    
    #predict from the original test set (after it has been preprocessed the same way as the original training set)
    Y_pred_final = model.predict(preprocessed_test_df.values)
    y_pred_final = np.around(Y_pred_final,0).astype(int)
    
    #handle potential nan values / values with issues (there should not be any)
    y_pred_final[y_pred_final < 0] = 0 
    y_pred_final[y_pred_final > 1] = 1 
    
    #export actual submission
    final_output_df = pd.DataFrame(y_pred_final, index = preprocessed_test_df.index, columns=[target_col])
    final_output_df.to_csv(output_prediction_path)
    
    return None


# # Score & Ranking
# 
# - Score: 0.79665
# - Public Ranking: between top 8% and top 9%
