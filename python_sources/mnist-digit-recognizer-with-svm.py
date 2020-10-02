#!/usr/bin/env python
# coding: utf-8

# # MNIST - Digit Recognizer with SVM

# In[ ]:


import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import seaborn as sns
sns.set(rc={'figure.figsize':(11, 4)})

#pandas option to display all columns
pd.set_option('max_columns',None)


# In[ ]:


#Read the data files
train_df=pd.read_csv('../input/train.csv')
train_df.head()


# In[ ]:


#Read the test dataset
test_df=pd.read_csv('../input/test.csv')
test_df.head()


# ## Basic Data Quality Check

# In[ ]:


#Check if there are any nulls
train_df.isnull().sum(axis=0) >0


# There doesn't appear to be any rows that have null values or any columns that have null values

# In[ ]:


train_df.describe()


# Since the pixel values can vary bewteen 0 and 255, there is not much we can gather from the above table. But we notice that the pixels number 0-11, 16-31, 52-57 etc are all 0's. This is because most of the digits are centerd and the spaces in the corners are blank. These pixels correspond to such spaces where there is nothing written.

# In[ ]:


#define a function to draw the image
#x-> dataframe with data
#y-> dataframe with labels
def draw_digit(x,y, label='Actual'):
    plt.figure(figsize=(20,5))
    #check the length of input dataset and divide by 8 to print 10 digits on each line
    nrows=(len(x)//10)+1
    #ncols=(len(x)%10)
    ncols=10
    print('Lenght of input: {}, nrows: {}, ncols: {}, label: {}'.format(len(x), nrows, ncols, y.shape))
    #print(x.shape)
    #iterate over all the digits passed in the array
   
    for idx,i in enumerate(x.index):
        #loop to iterate over blocks of 10 digits            
            plt.subplot(nrows,10,idx+1)  #subplots start with 1
            plt.subplots_adjust(top=1.3)
            plt.imshow(x.loc[i].values.reshape(28,28), cmap=plt.cm.gray, interpolation='nearest',clim=(0, 255))
            plt.title(label+' %i\n' % y.iloc[idx,:].values, fontsize = 11)
            #plt.title('Actual {}\n'.format(y.iloc[idx,:].values), fontsize = 11)
           
    plt.show()


# In[ ]:


#Display image of digits
draw_digit(train_df.iloc[:15,1:], train_df.iloc[:15,:1],'Actual')


# ### Check the distribution of labels

# In[ ]:


#check the distribution of dataset labels
sns.countplot(train_df['label'], palette='viridis')


# We see that there is a uniform distribution of different classes, and there is no class imbalance or skewness.

# ## Take a subset of data.
# __Since the data set is quite large, and tuning all the hyperparameters will be expensive, we will run the training on a subset of this data.__

# In[ ]:


import random
subset_percent=25


# In[ ]:


#np.floor(len(digi_df)*21/100), len(digi_df)*21/100
#list(digi_df.index)
int(len(train_df)*subset_percent/100)


# In[ ]:


print ('Sampling {}% of training, Labels and test data'.format(subset_percent))
train_idx=random.sample(list(train_df.index),int(len(train_df)*subset_percent/100))
test_idx=random.sample(list(test_df.index),int(len(test_df)*subset_percent/100))


# In[ ]:


print('Training Data size: {}, test data size: {}'.format(len(train_idx), len(test_idx)))


# In[ ]:


train_df.iloc[train_idx].head()


# In[ ]:


train_df.loc[train_idx,'label'].head()


# ## Split the training dataset into training and validation
# 

# In[ ]:


#Seprate the label from the training data 
train_label=train_df['label']
train_df.drop('label', axis=1, inplace=True)


# In[ ]:


print('Data: {}, Label: {}'.format(train_df.shape, train_label.shape))


# In[ ]:


from sklearn.model_selection import train_test_split
# from the original data, we will take a subset (25% calculated above) and split this subset into training and validation set

train_x, test_x, train_y, test_y = train_test_split(train_df.iloc[train_idx], train_label.iloc[train_idx],                                                     train_size=0.7, test_size=0.3, random_state=4)


# In[ ]:


print('Data - training Set: {}, test set: {}'.format(train_x.shape, test_x.shape))
print('Label - training Set: {}, test set: {}'.format(train_y.shape, test_y.shape))


# ## Use Cross Validation to train the model
# 
# __We already have a test set, but to tune the hyperparameters, we will use K-fold cross validaton on the training set__

# In[ ]:


from sklearn.model_selection import GridSearchCV


# ### Standardize the training and validation dataset

# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()


# In[ ]:


#Range of different hyperparameters
#kernel=['rbf','linear','poly']
#C=[0.001,0.01,0.1,1.0,10]
#gamma=[0.001,0.01,0.1,1.0]

kernel=['rbf']
C=[0.001, 0.01, 1, 10,30]
gamma=[0.001, .01, 1, 10]

params={'SVM__kernel':kernel, 'SVM__C':C, 'SVM__gamma':gamma}


# ### We will use a Pipeline with feature scaling and model training as it's steps

# In[ ]:


#Prepare a pipeline
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


# In[ ]:


#instantiate a class of SVM
svm = SVC(cache_size=7000, random_state=42)


# In[ ]:


#Prepare a pipeline
steps=[('Scaler',StandardScaler()), ('SVM',svm)]
pipeline=Pipeline(steps)


# In[ ]:


#use GridSearch for hyperparameter tuning
grid=GridSearchCV(n_jobs=-1, 
                  estimator=pipeline,
                  param_grid=params,
                  return_train_score=True,                   
                  cv=3, 
                  verbose=1)


# In[ ]:


grid.estimator.get_params


# In[ ]:


#
start=timer()
#Train on traning dataset
grid.fit(train_x, train_y)
end=timer()


# In[ ]:


print('Time taken to fit on train data (in min): {} '.format((end-start)/60))


# In[ ]:


grid.best_params_


# In[ ]:


grid.best_score_, grid.best_params_['SVM__C'], grid.best_params_['SVM__gamma']


# In[ ]:





# In[ ]:


cv_result= pd.DataFrame(grid.cv_results_)
cv_result.head()


# In[ ]:


cv_result.columns


# In[ ]:


cv_result[['rank_test_score','mean_test_score','mean_train_score','param_SVM__C','param_SVM__gamma']].sort_values(by='rank_test_score')


# In[ ]:





# In[ ]:


#Use the above values of C and gamma to train the model
svm_1 = SVC(C=grid.best_params_['SVM__C'], gamma=grid.best_params_['SVM__gamma'], kernel='rbf', random_state=21)


# In[ ]:


#Re-create the pipeline model, with the SVM class corresponding to one with best parameters
steps=[('Scaler',StandardScaler()), ('SVM',svm_1)]
pipeline=Pipeline(steps)


# In[ ]:


start=timer()
pipeline.fit(train_x, train_y)
end=timer()


# In[ ]:


print('Time taken to fit on train data using best hyperparameter (in min): {} '.format((end-start)/60))


# In[ ]:


test_pred=pipeline.predict(test_x)


# In[ ]:


from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report, accuracy_score


# In[ ]:


cmat=confusion_matrix(test_y, test_pred)


# In[ ]:


cmat


# In[ ]:


accuracy_score(test_y, test_pred)


# In[ ]:


pd.options.display.float_format= '${:,.4f}'.format


# In[ ]:


plt.figure(figsize=(15,8))
sns.heatmap(cmat, annot=True, fmt='g')


# In[ ]:


print(classification_report(test_y, test_pred))


# '**micro'**: Calculate metrics globally by counting the total true positives, false negatives and false positives.<br>
# **'macro':** Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.

# In[ ]:





# In[ ]:


test_result=pd.concat([test_y, pd.Series(test_pred, name='predicted', index=test_y.index)],axis=1)


# In[ ]:


test_result.head()


# In[ ]:


test_x.head()


# In[ ]:


test_result.iloc[:21,1:]


# In[ ]:


train_df.iloc[:15,:1].shape[1], test_result.iloc[:15,1:].shape[1]


# In[ ]:


#Let's visualize how the predictions come out to be
draw_digit(test_x.iloc[:21,:],test_result.iloc[:21,1:],'Predicted')
#draw_digit(train_df.iloc[:15,1:], train_df.iloc[:15,:1])


# In[ ]:


#digits that were incorrectly classified
incorrect_dig=test_result[test_result.label != test_result['predicted']].index
#draw_digit(test_x.iloc[:21,:],test_result.iloc[:21][['predicted']])


# In[ ]:


test_xypred=pd.concat([test_x, test_result], axis=1)


# In[ ]:


test_xypred.head()


# In[ ]:


test_xypred[test_xypred.label != test_xypred.predicted].head()


# **Images of digits that were incorrectly predicted by the algorithm**

# In[ ]:


draw_digit(test_xypred[test_xypred.label != test_xypred.predicted].iloc[:21,:784],           test_xypred[test_xypred.label != test_xypred.predicted].iloc[:21,785:],'Predicted')


# ~190 (out of 7000) digits were incorrectly predicted

# In[ ]:


#Save the model for further testing
from joblib import dump, load
dump(pipeline, 'mnist_svm_pipeline.joblib') 


# In[ ]:


test_df.head()


# In[ ]:


start=timer()
test_predict=test_pred=pipeline.predict(test_df)
end=timer()


# In[ ]:


print('Time take to predict: {}'.format((end-start)/60))


# In[ ]:


test_predict


# In[ ]:


test_df.index


# In[ ]:


submit_df =pd.concat([pd.Series(test_df.index, name='ImageId'), pd.Series(test_predict, name='Label')], axis=1)


# In[ ]:


submit_df.head()


# In[ ]:


submit_df.to_csv('../submission.csv')

