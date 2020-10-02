#!/usr/bin/env python
# coding: utf-8

# ## This Kernal is to show the initial data Analysis and build the baseline model and machine learning model using Titanic Data Set. 

# ### Understanding titanic dataset
# 
# |#|Feature        |Meaning          | 
# |:---:|:---:|:---:|
# |1|Passenger ID  | Unique passenger id |
# |2| Survived     | If Survived(1-yes, 0-no) |
# |3| Name         | Name of the passenger |
# |4| Sex          | Gender |
# |5| Age          | Age of passenger |
# |6| SibSp        | Number of siblings / spouses aboard |
# |7| Parch        | Number of parents / childer aobard |
# |8| Ticket       | Ticket Number |
# |9| Fare         | Passenger fare |
# |10| Cabin       | Cabin number |
# |11| Embarked    | Point of embarkment (C=Cherbourg; Q=Queenstown; S= Southampton) |

# # Data Orgnizantion (Data Preprocessing Script)

# In[ ]:


import numpy as np
import pandas as pd
import os


# In[ ]:


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
    
#read the data with all default parameters
train_df = pd.read_csv("/kaggle/input/titanic/train.csv", index_col = 'PassengerId')   
test_df = pd.read_csv("/kaggle/input/titanic/test.csv", index_col = 'PassengerId')
    
test_df['Survived'] = -888
df = pd.concat((train_df, test_df), axis=0, sort=True) 


# ## Missing Value Embarked Handeling. 

# In[ ]:


# extract rows with Embarked as Null
df[df.Embarked.isnull()]


# In[ ]:


# how may people embarked at differnt placess and find out the most common embarkment point
df.Embarked.value_counts()

# we can fill the missing values with the S as most people embarked from S.


# In[ ]:


# we can analyze further and see both of the passenger survived the disaster. So we should find out from which
#Embarked point most people survived that will be more logical. 
pd.crosstab(df[df.Survived!=-888].Survived, df[df.Survived!=-888].Embarked)


# ***We can see both the missing passenger Fare is 80 and both are 1st class passenger. We can explore that from which Embarked value people have paid near 80 fare in first class.***

# In[ ]:


#option 2 : explore the fare of each class of each embarkment point 
df.groupby(['Pclass', 'Embarked']).Fare.median()


# ***We can see here that class 1 passenger who embarked from the C has paid 76.7292 Fare which is very close to 80. There may be possibility that those two passenger who does not have embarked data have boarded from C.***

# ## Feature: Fare missing value issues

# In[ ]:


#see the row where fare is missing. 
df[df.Fare.isnull()]


# In[ ]:


df.groupby(['Embarked', 'Pclass']).Fare.median()


# In[ ]:


#calculate the median fare for Embarked in S and Pclass is 3.
median_fare = df.loc[(df.Embarked == 'S') & (df.Pclass == 3),'Fare'].median()


# In[ ]:


median_fare


# ### Fare will be updated with median fare. 

# ## Feature: Age Missing value
# Replace with median age of title
# Extract the title from the name column and create a separate column title. Then check median age of each title. Title of a name mostly gives us the age information. if its master then it is kid and its sir or dr means its old person

# In[ ]:


df.Name


# In[ ]:


# Function to extract the title from the name. There is pattern in the titel. Starts with last name, title. first name middle name. 
def getTitle(name):
    first_name_with_title = name.split(',')[1]
    title = first_name_with_title.split('.')[0]
    title = title.strip().lower()  # strip out all of the whitespaces and coverting the title to lower case
    return title


# ## Binning Fare_Bin

# In[ ]:


#Binning - qcut() performs quantile based binning. We are splitting the Fare in 4 bins here, Where each bins contains almost equal number of observations.
pd.qcut(df.Fare, 4)


# In[ ]:


#Specify the name of each bins
pd.qcut(df.Fare, 4, labels=['very_low', 'low', 'high', 'very_high'])  # discretization


# ***Clearly we have converted a numerical feature Fare to categorical feature, wher each bin is one category. Such techniques are called Discritization.***

# In[ ]:


# lets see the number of obervations in each bins
pd.qcut(df.Fare, 4, labels=['very_low', 'low', 'high', 'very_high']).value_counts().plot(kind='bar', color='c', rot=0);


# ## Read Data Function 
#   - This function reads the data from the file.
#   - create train_df and test_df with index column as PassengerID
#   - we also create a column Survived in the test_df as its not given in the original data so that we can add that to the   train_df to organize all the data at once. This way we do not have to perform same function twice on different data set. 
#   - merge both data and return the dataframe as df. 
#   

# In[ ]:


def read_data():

    for dirname, _, filenames in os.walk('/kaggle/input'):
        for filename in filenames:
            print(os.path.join(dirname, filename))
    
    #read the data with all default parameters
    train_df = pd.read_csv("/kaggle/input/titanic/train.csv", index_col = 'PassengerId')   
    test_df = pd.read_csv("/kaggle/input/titanic/test.csv", index_col = 'PassengerId')
    
    test_df['Survived'] = -888
    df = pd.concat((train_df, test_df), axis=0, sort=True) 
    return df


# In[ ]:


# use map function to apply the function on each Name value row i
df.Name.map(lambda x : getTitle(x))   #alternatively we can use : df.Name.map(getTitle)


# In[ ]:


# find out how many unique title we have
df.Name.map(lambda x : getTitle(x)).unique()


# In[ ]:


# We will modify our getTitle function here and we will introduce the dictionary to create custome tile.
# this is to club some of the title
def getTitle(name):
    title_group = {'mr' : 'Mr',
                'mrs' : 'Mrs',
                'miss' : 'Miss',
                'master' : 'Master',
                'don' : 'Sir',
                'rev' : 'Sir', 
                'dr' : 'Officer',
                'mme' : 'Mrs',
                'ms' : 'Mrs',
                'major' : 'Officer',
                'lady' : 'Lady',
                'sir' : 'Sir', 
                'mlle' : 'Miss',
                'col' : 'Officer',
                'capt' : 'Officer',
                'the countess' : 'Lady',
                'jonkheer' : 'Sir', 
                'dona' : 'Lady'
    }
    first_name_with_title = name.split(',')[1]
    title = first_name_with_title.split('.')[0]
    title = title.strip().lower()
    return title_group[title]


# In[ ]:


# Create title feature
df['Title'] = df.Name.map(lambda x : getTitle(x))


# In[ ]:


df.head()


# In[ ]:


#Box plot of Age with title
df[df.Age.notnull()].boxplot('Age', 'Title');


# ***We can see that each title has a different median value and they also have different Age range. It makes most sense to replace the Age median to the median of each title.***

# ## Get Title function. 
#   - Fetched the title from the name column and created a categorical feature column title. 
#   - This is required to fill the null age value. 
#   - Median age value for every title can be different. Like master title is for kids so the median age for kids can be smaller than the median age of old person. 
#   -  Also clubbed the title to make it compact. Mr, Mrs, Miss, Master, Sir, Officer, Lady titles used. 

# ### Missing value 
#  -  Function missing value to fill all missing values. 
#  -  Embarked missing value is filled with 'C'
#  -  Age missing value is filled with title midean. 

# In[ ]:


def fill_missing_values(df):
    # Embarked
    df.Embarked.fillna('C', inplace=True)
    # Fare
    median_fare = df.loc[(df.Embarked == 'S') & (df.Pclass == 3),'Fare'].median()
    df.Fare.fillna(median_fare, inplace=True)        
    # age
    title_age_median = df.groupby('Title').Age.transform('median')
    df.Age.fillna(title_age_median, inplace=True)   
    return df


# ## Get Deck function
#    - Create a new feature deck from cabin. fill all the missing value with value z. 

# In[ ]:


def get_deck(cabin):
    return np.where(pd.notnull(cabin), str(cabin)[0].upper(),'z')      


# ## Reorder function
#     - Put the column survived at the beginnig. 
#     - Its good practice to include the target column either at the begining or at the end. 

# In[ ]:


def reorder_columns(df):
    columns = [column for column in df.columns if column != 'Survived']
    columns = ['Survived'] + columns
    df = df[columns]
    return df


# ## Process Data
# This function process all the data. fill null values, create new features drop columns and on hot encoding for catgorical feature. all included in this function. 
#   - follwoing new features has been created. 
#   -  Fare_Bin =  Devided the fare into 4 categories 'very_low', 'low', 'high', 'very_high'. 
#   -  Age_State = 'Adult', 'Child' - As child has the more chances of survival than adults. They get priority in the life  boats.
#   -  FamilySize - Small family has better chance of survival than big families.
#   -  IsMother - Mother has better chances of survival as they get preference in boat. 
#   -  Deck column - Which place passenger is situated in the deck can also reveal how far they are from the lifeboat. 

# In[ ]:


def process_data(df):
    #using the method chaining concept
    return (df
         # create title attribute - then add this
         .assign(Title = lambda x: x.Name.map(get_title))
         # working missing values - start with this
         .pipe(fill_missing_values)   
         # Create Fare_Bin Feature
         .assign(Fare_Bin = lambda x: pd.qcut(x.Fare, 4, labels=['very_low', 'low', 'high', 'very_high']))
         # Create ageState
         .assign(AgeState = lambda x : np.where(x.Age >= 18, 'Adult', 'Child'))
         # Creat FamilySize
         .assign(FamilySize = lambda x : x.Parch + x.SibSp + 1)
         # Create IsMother   
         .assign(IsMother = lambda x : np.where(((x.Sex == 'female') & (x.Parch > 0) & (x.Age > 18) & (x.Title != 'Miss')), 1, 0))  
         # Create Deck feature
         .assign(Cabin = lambda x: np.where(x.Cabin == 'T', np.nan, x.Cabin))
         .assign(Deck = lambda x : x.Cabin.map(get_deck))   
         # Feature Encoding
         .assign(IsMale = lambda x : np.where(x.Sex == 'male', 1,0))
         .pipe(pd.get_dummies, columns=['Deck', 'Pclass', 'Title', 'Fare_Bin', 'Embarked', 'AgeState'])
         # Add code to drop unnecessarey columns
         .drop(['Cabin', 'Name', 'Ticket', 'Parch', 'SibSp', 'Sex'], axis = 1)   
         # Reorder columns
         .pipe(reorder_columns)
           )


# In[ ]:


df = read_data()


# In[ ]:


df = process_data(df)


# In[ ]:


#train data
train_df = df.loc[df.Survived != -888]


# In[ ]:


#test data
columns = [column for column in df.columns if column != 'Survived']
test_df = df.loc[df.Survived == -888, columns]


# # Preprocessed data
#   - Training data Set
#   - Test Data Set
#   
#   You can apply machine learning algorithm now. 

# In[ ]:


train_df.head()


# In[ ]:


test_df.head()


# # Machine Learning 
#   - Performance Matrices. 
#       - Performance Matrics, Which Classifier is better?
#           - Accuracy
#           - Precision
#           - Recall
# 
#    - Accuracy - Compare predicted output with actual output.
#         -  Accuracy = Correct / Total Count
#    - Precision & Recall:
#      
#     **Confution Metrix**
#  
# | | Predicted Negtive  |  Predicted Possitive |
# |:---:|:---:|:---:|
# |Actual Negative | True Negative (TN) | False Positive (FP)  |
# |Actual Possitve | False Negative (FN)| True Positive (TP)   |
#         
#    - Precision: What fraction of positive predictions are correct? TP / Total Positive Prediction = TP / TP + FP
#    - Recall : What fraction of positive cases you predicted correctly? TP / Total Positive Cases = TP / TP + FN

# ## Baseline Model
# 
# - First Step - Create baseline Model which does not use the machine leanring at all - It is a best practice to do that. It will help to compare our machine learning model. If Machine leanring model gives the better result then it makes sense to build machine learning model other wise it does not make sense at all. 
# 
# #### Baseline Model for classification 
# 
# - Always give the output as majority class
# 
# |Class | Count |
# |:---:|:---:|
# |1|60|
# |0|40|
# 
# If training data set has the majority class 60 percent as 1. Then our model will always return class 1 as a result.
# Baseline Model Accuracy = 60 / (60+40) = 0.6  = 60%
# 
# ***Predictive model should have better performance than baseline otherwise it does not make sense to build the machine learning model.***

# In[ ]:


import pandas as pd
import numpy as np
import os


# In[ ]:


X = train_df.loc[:,'Age':].as_matrix().astype('float')
y = train_df['Survived'].ravel()


# In[ ]:


print(X.shape, y.shape)


# In[ ]:


# train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# In[ ]:


# Average survival in train and test 
print(f"Mean Survival in train : {np.mean(y_train)}")
print(f"Mean Survival in test : {np.mean(y_test)}")  


# - First observation: Survival rate in both train and test data are almost same that means training and test data. We want positive cases to be distributed evenly in both trainig and test data.
# - Second obeservation: Only 39 % of data are positive cases and 61% are the negative classes. In this scenario 39 vs 61 is still good scenario however in some case like marketing - Attract, Engage and Convert ratio can be very small. 2-3% cutomer can be attracted. This highly imbalance problem has different approach to build the model.

# In[ ]:


from sklearn.dummy import DummyClassifier
# performance matrics
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score


# In[ ]:


# create Model
model_dummy = DummyClassifier(strategy='most_frequent', random_state=0)

# Most frequest will output the majority class. 


# In[ ]:


# train model
model_dummy.fit(X_train, y_train)


# In[ ]:


print(f"Score for baseline Model: {model_dummy.score(X_test, y_test)}")
print(f"Accurancy for baseline model: {accuracy_score(y_test, model_dummy.predict(X_test))}")
print(f"Confusion Matrix for baseline model: {confusion_matrix(y_test, model_dummy.predict(X_test))}")
print(f"Precision for baseline model: {precision_score(y_test, model_dummy.predict(X_test))}")
print(f"Recall for baseline model: {recall_score(y_test, model_dummy.predict(X_test))}")


# ## Logistic Regression 

# In[ ]:


#import function
from sklearn.linear_model import LogisticRegression


# In[ ]:


# Create model
model_lr_1 = LogisticRegression(random_state=0)
# train model
model_lr_1.fit(X_train, y_train)


# In[ ]:


print(f"Score for LogisticRegression Model: {model_lr_1.score(X_test, y_test)}")
print(f"Accurancy for LogisticRegression model: {accuracy_score(y_test, model_lr_1.predict(X_test))}")
print(f"Confusion Matrix for LogisticRegression model: {confusion_matrix(y_test, model_lr_1.predict(X_test))}")
print(f"Precision for LogisticRegression model: {precision_score(y_test, model_lr_1.predict(X_test))}")
print(f"Recall for LogisticRegression model: {recall_score(y_test, model_lr_1.predict(X_test))}")


# ## We can apply other machine algorithms as above. 
