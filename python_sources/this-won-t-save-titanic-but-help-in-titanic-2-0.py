#!/usr/bin/env python
# coding: utf-8

# **My idea is to provide clean code for beginners to get started
# # 
# # I have submitted the first version to just get started.
# # I will keep on working on this (add more generic code) to get better results
# # **

# In[ ]:


def read_train_test_files(train_file,test_file):
    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)
    return df_train, df_test


# In[ ]:


def get_combined_data(df_train,df_test,label_name,label_data_type,
                      is_train_column):    
    # add new column is_train which will help in merge and split of 
    # Train and Test datasets
    df_train[is_train_column] = 1
    df_test[is_train_column] = 0
    df_test[label_name] = 0 if label_data_type=='int64' else ""
    df_combined=pd.concat([df_train, df_test],sort=True).reset_index(drop=True)
    return df_combined


# In[ ]:


def perform_data_preprocessing(df_combined, features, label,
                               is_train_column):
    #### TODO : Cleaning of data on df_combined ###########
    df_combined_dummies = pd.get_dummies(df_combined[features])
    df_combined_dummies[label] = df_combined[label].values
    df_combined_dummies[is_train_column]=df_combined[is_train_column].values
    df_train = df_combined_dummies[df_combined[is_train_column] == 1]
    df_test = df_combined_dummies[df_combined[is_train_column] == 0]
    df_train = df_train.drop([is_train_column], axis=1)
    df_test = df_test.drop([is_train_column], axis=1)    
    return df_train,df_test


# In[ ]:


def Create_confusion_matrix(model_name,y_test,test_predictions):
    # Compute and print the confusion matrix
    from sklearn.metrics import confusion_matrix
    print(f"confusion_matrix :")
    from sklearn.metrics import accuracy_score, f1_score 
    from sklearn.metrics import precision_score, recall_score
    print(f"------------ Test Metrics for {model_name}------------------------")
    print("Accuracy:  {:.3f}".format(accuracy_score(y_test,test_predictions)))
    print("Precision: {:.3f}".format(precision_score(y_test,test_predictions,average="macro")))
    print("Recall:    {:.3f}".format(recall_score(y_test,test_predictions,average="macro")))
    print("F1-Score:  {:.3f}".format(f1_score(y_test,test_predictions,average="macro")))
    print("--------------------------------------------------\n")


# In[ ]:


def get_GridSearchCV_best_estimator(classifier,params,best_classifiers,
                                    X_train, X_valid, y_train, y_valid):
    from sklearn.model_selection import GridSearchCV, cross_val_score
    model_name = classifier.__class__.__name__    
    grid_model = GridSearchCV(classifier, params, cv=5, refit=True, 
                              return_train_score=True)    
    best_model = grid_model.fit(X_train, y_train)
    best_estimator = grid_model.best_estimator_
    score=cross_val_score(best_estimator, X_train,y_train,cv=5,
                          scoring='recall')
    best_model_cv_score = round(score.mean() * 100, 3)
    new_row = {'Model_Name': model_name, 'Best_Model': best_estimator,
               'Best_Params': best_model.best_params_,
               'Best_training_score': best_model.best_score_,
               'Best_Model_CV_Score(%)': best_model_cv_score}
    best_classifiers = best_classifiers.append(new_row,ignore_index=True)
    best_classifiers = best_classifiers.sort_values(by='Best_Model_CV_Score(%)',ascending=False)
    print(
        f"ClassifierTrainer : {model_name}"
        f"\n\t Best Score: {best_model.best_score_}"
        f"\n\t Cross Validation Score: {best_model_cv_score}%"
        f"\n\t Best parameters : {best_model.best_params_}"
        f"\n\t Best Model : {best_estimator}")
    # Predict test set labels
    test_predictions = best_model.predict(X_valid)
    Create_confusion_matrix(model_name,y_valid,test_predictions)
    return best_classifiers, best_model


# In[ ]:


def get_best_model(X_train, X_valid, y_train, y_valid):
    best_model = None
    import pandas as pd
    columns=['Model_Name','Best_Model','Best_Params',
             'Best_training_score','Best_Model_CV_Score(%)']
    best_classifiers = pd.DataFrame(columns=columns)
    
    classifiers_list = []
    params_list = []
    
    
    from sklearn.linear_model import LogisticRegression       
    params = {"penalty": ['l1'],
              'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
              "solver": ['saga', 'liblinear']}
    classifiers_list.append(LogisticRegression())
    params_list.append(params)
    
    
    from xgboost import XGBClassifier 
    params = {'learning_rate': [0.001, 0.005, 0.01, 0.02, 0.05, 0.1],
              'n_estimators': [100, 500, 700, 1000, 1200, 2000],
              'max_depth': [5, 10, 15, 20, 50, 100, 500]}
    best_params = {'learning_rate': [0.005],
              'n_estimators': [1200],
              'max_depth': [10]}
    classifiers_list.append(XGBClassifier())
    params_list.append(best_params)
    
    index = -1
    for classifier in classifiers_list: 
        index += 1
        best_classifiers,best_model         =get_GridSearchCV_best_estimator(classifier,
                                         params_list[index],
                                         best_classifiers,
                                         X_train, 
                                         X_valid, 
                                         y_train, 
                                         y_valid)
        
    
    return best_model


# In[ ]:


def split_train_test(df_data, label_name):
    from sklearn.model_selection import train_test_split
    X = df_data.drop([label_name], axis=1).values
    y = df_data[label_name].values
    X_train, X_valid, y_train, y_valid=train_test_split(X,y,
                                                        test_size=0.3,
                                                        random_state=42,
                                                        stratify=y)
    return X_train, X_valid, y_train, y_valid


# In[ ]:


def submit_predictions(predictions, df_test_original,x_col_name,
                       y_col_name,filename,label_data_type):
    # creating submission file
    submission = pd.DataFrame({x_col_name: df_test_original[x_col_name],
                               y_col_name: predictions})
    submission[y_col_name] = submission[y_col_name].astype(label_data_type)
    print(submission.head())
    submission.to_csv(filename,index=False)
    print(f"Saved submission file: {filename}")


# In[ ]:


# Necessary imports
import numpy as np
import pandas as pd

# define constants
train_file = '/kaggle/input/titanic/train.csv'
test_file = '/kaggle/input/titanic/test.csv'
label_name = "Survived"
input_features = ['Pclass','Sex','SibSp','Parch']
is_train_column = 'is_train'
label_data_type = 'int64'

submission_filename = 'Titanic submission.csv'
submission_file_x_col_name="PassengerId"

# read train and test data files and merge the data 
df_train,df_test = read_train_test_files(train_file,test_file)
df_test_original = df_test.copy()

# so that we can perform cleaning on complete data
df_combined = get_combined_data(df_train,df_test,label_name,
                                label_data_type,is_train_column)

# Perform data cleaning on combined data and and split 
# back to train and test data
df_train,df_test = perform_data_preprocessing(df_combined,
                                              input_features,
                                              label_name,
                                              is_train_column)

# get X and y ready for model development
X_train, X_valid, y_train, y_valid = split_train_test(df_train,
                                                      label_name)

# train couple of algorithms on X and y and get the best model 
best_model = get_best_model(X_train, X_valid, y_train, y_valid)

# get predictions
X_test = df_test.drop([label_name], axis=1).values
predictions = best_model.predict(X_test)

# submit the predictions fro best model
submit_predictions(predictions,df_test_original, 
                   x_col_name=submission_file_x_col_name,
                   y_col_name=label_name,
                   filename = submission_filename,
                   label_data_type=label_data_type)

