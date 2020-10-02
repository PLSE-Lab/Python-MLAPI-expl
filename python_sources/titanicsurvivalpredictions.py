# NumPy
import numpy as np

# Dataframe operations
import pandas as pd

# Data visualization
import seaborn as sns
import matplotlib.pyplot as plt

# Scalers
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

# Models
from sklearn.linear_model import LogisticRegression #logistic regression
from sklearn.linear_model import Perceptron
from sklearn import svm #support vector Machine
from sklearn.ensemble import RandomForestClassifier #Random Forest
from sklearn.neighbors import KNeighborsClassifier #KNN
from sklearn.naive_bayes import GaussianNB #Naive bayes
from sklearn.tree import DecisionTreeClassifier #Decision Tree
from sklearn.model_selection import train_test_split #training and testing data split
from sklearn import metrics #accuracy measure
from sklearn.metrics import confusion_matrix #for confusion matrix
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier

# Cross-validation
from sklearn.model_selection import KFold #for K-fold cross validation
from sklearn.model_selection import cross_val_score #score evaluation
from sklearn.model_selection import cross_val_predict #prediction
from sklearn.model_selection import cross_validate

# GridSearchCV
from sklearn.model_selection import GridSearchCV

#Common Model Algorithms
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process

#Common Model Helpers
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics

#Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
from pandas.tools.plotting import scatter_matrix


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
#loading data files, both training and test data and then preproecess it
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
data_df = train_df.append(test_df)
data_df.columns
data_df['Title'] = data_df['Name']
#create a new feature called Title, it helps with the age imputation
#Extract the title from the name after a little clean up
for name_string in data_df['Name']:
    data_df['Title'] = data_df['Name'].str.extract('([A-Za-z]+)\.',expand=True) # the regular expression relies on the period (.) at the end of the title to extract the title
data_df['Title'].unique() #look at all the titles and see if any uncommon ones are there
#map the uncommon titles into standard ones to estimate the missing ages
#create a temporary mapping dictionary
mapping = {'Dr':'Mr','Rev':'Mr','Mlle':'Miss','Major':'Mr','Col':'Mr','Sir':'Mr','Don':'Mr','Mme':'Miss','Jonkheer':'Mr','Lady':'Mrs','Capt':'Mr','Countess':'Mrs','Ms':'Miss','Dona':'Mrs'}
data_df.replace({'Title':mapping},inplace=True)

titles = ['Master', 'Miss', 'Mr', 'Mrs'] #create an array of titles based on the unique values from the list
age_to_impute = data_df.groupby('Title')['Age'].median()
 
for title in titles:
    #age_to_impute = data_df.groupby('Title')['Age'].median() #calculates the mean age for each title
    print(titles.index(title),age_to_impute[titles.index(title)])
    data_df.loc[(data_df['Age'].isnull()) & (data_df['Title'] == title), 'Age'] = age_to_impute[titles.index(title)] #assigns the mean age to the missing one based on title
#update the trainign and test data sets with age caclulated above
train_df['Age'] = data_df['Age'][:891]
test_df['Age'] = data_df['Age'][891:]

#discard the title as we no longer need it
data_df.drop('Title',axis = 1, inplace = True)

#Family size is a combination of parents and siblings Parch + SibSp
data_df['Family_Size'] = data_df['Parch'] + data_df['SibSp']
#data_df.Family_Size[(data_df['Family_Size'] < 5)] = "Small"
#data_df.Family_Size[(data_df['Family_Size'] !="Small")] = "Large"
train_df['Family_Size'] = data_df['Family_Size'][:891]
test_df['Family_Size'] = data_df['Family_Size'][891:]
#test_df['Family_Size'].replace(['Small','Large'],[0,1],inplace=True)
#train_df['Family_Size'].replace(['Small','Large'],[0,1],inplace=True)
#Family survival as a unit
#group people byy their last name
data_df['Last_Name'] = data_df['Name'].apply(lambda x:str.split(x,",")[0]) #multiple Python features in play here. How cool is that!!
#data_df['Fare'].fillna(data_df['Fare'].mean(),inplace=True)

DEFAULT_SURVIVAL_VALUE = 0.5
data_df['Family_Survival'] = DEFAULT_SURVIVAL_VALUE
# Read notes from the other kernel about the logic below
for grp, grp_df in data_df[['Survived','Name', 'Last_Name', 'Fare', 'Ticket', 'PassengerId','Parch','SibSp', 'Age', 'Cabin']].groupby(['Last_Name','Fare']):
    if (len(grp_df) != 1): #more than one in a group
       for ind, row in grp_df.iterrows():
           smax = grp_df.drop(ind)['Survived'].max()
           smin = grp_df.drop(ind)['Survived'].min()
           passID = row['PassengerId']
           if (smax == 1.0):
               data_df.loc[data_df['PassengerId'] == passID,'Family_Survival'] = 1
           elif (smin == 0.0):
               data_df.loc[data_df['PassengerId']== passID, 'Family_Survival'] = 0
print("Number of passengers with family survival data:", data_df.loc[data_df['Family_Survival'] != 0.5].shape[0])
           
#In addition to travelling as a family, we can also check if someone is travelling as a group. Have similar survival possibilities
for _, grp_df in data_df.groupby('Ticket'):
    if (len(grp_df) != 1):
        for ind, row in grp_df.iterrows():
            if (row['Family_Survival'] == 0) | (row['Family_Survival'] == 0.5):
                smax = grp_df.drop(ind)['Survived'].max()
                smin = grp_df.drop(ind)['Survived'].min()
                passID = row['PassengerId']
                if (smax == 1.0):
                    data_df.loc[data_df['PassengerId'] == passID,'Family_Survival'] = 1
                elif(smin == 0.0):
                    data_df.loc[data_df['PassengerId'] == passID,'Family_Survival'] = 0
print("Number of passengers with family/group survival information: " + str(data_df[data_df['Family_Survival'] != 0.5].shape[0]))                               

#Add this family survival data to the train/test data sets
train_df['Family_Survival'] = data_df['Family_Survival'][:891]
test_df['Family_Survival'] = data_df['Family_Survival'][891:]

#Time for binning Age and Fares as continuous values are no good
#Let's tackle fare first
pclasses = [3,1,2]
fare_to_impute = data_df.groupby('Pclass')['Fare'].median()

for pcls in pclasses:
    #fare_to_impute = data_df.groupby('Pclass')['Fare'].median()[pclasses.index(pcls)]
    print(pclasses.index(pcls),fare_to_impute[pclasses.index(pcls)+1])
    data_df.loc[(data_df['Fare'].isnull()) & (data_df['Pclass'] == pcls), 'Fare'] = fare_to_impute[pclasses.index(pcls)+1]  #assigns the mean age to the missing one based on title

data_df['FareBin'] = pd.qcut(data_df['Fare'],5)

label = LabelEncoder()
data_df['FareBin_Code'] = label.fit_transform(data_df['FareBin'])
train_df['FareBin_Code'] = data_df['FareBin_Code'][:891]
test_df['FareBin_Code'] = data_df['FareBin_Code'][891:]

#Discard the raw fare data as it is no longer useful
train_df.drop(['Fare'], 1, inplace=True)
test_df.drop(['Fare'], 1, inplace=True)

#Binning Age
data_df['AgeBin'] = pd.qcut(data_df['Age'],4)

label = LabelEncoder()
data_df['AgeBin'].head()
data_df['AgeBin_Code'] = label.fit_transform(data_df['AgeBin'])
train_df['AgeBin_Code'] = data_df['AgeBin_Code'][:891]
test_df['AgeBin_Code'] = data_df['AgeBin_Code'][891:]


#mapping sex to code and then clean up any unnecessary stuff
train_df.Sex[(train_df['Age'] < 16) ] = 'female'
test_df.Sex[(test_df['Age'] < 16)] = 'female'
train_df.Sex.unique()
train_df['Sex'].replace(['male','female','Child'],[0,3,6],inplace=True)
test_df['Sex'].replace(['male','female','Child'],[0,3,6],inplace =True)

#Drop the original age feature from the dataset
train_df.drop(['Age'], 1, inplace=True)
test_df.drop(['Age'], 1, inplace=True)

#dropping a whole bunch of unused featureds
train_df.drop(['Name', 'PassengerId', 'SibSp', 'Parch', 'Ticket', 'Cabin',
               'Embarked'], axis = 1, inplace = True)
test_df.drop(['Name','PassengerId', 'SibSp', 'Parch', 'Ticket', 'Cabin',
              'Embarked'], axis = 1, inplace = True)
test_df.to_csv("Rajesh_train_data.csv")

#Training begins
#Separate labels in the datasets
X = train_df.drop('Survived',1)
y = train_df['Survived']

X_test = test_df.copy()
#Scaling features; very important steps as models require all of them in the same range
std_scaler = StandardScaler()
X = std_scaler.fit_transform(X) # How cool that one line can do all your scaling 
X_test = std_scaler.transform(X_test)
temp_df = pd.DataFrame(data=X_test)
temp_df.to_csv("raj_scaled_test.csv")


#Grid Search cross validations
#K neighnours network
n_neighbors = [6,7,8,9,10,11,12,14,16,18,20,22]
algorithm  = ['auto']
weights = ['uniform','distance']
leaf_size = list(range(1,50,5)) # 1-50 in the increments of 5
#Define hyperparamenters
hyperparams = {'algorithm': algorithm, 'weights': weights, 'leaf_size': leaf_size, 
               'n_neighbors': n_neighbors}
gd=GridSearchCV(estimator=KNeighborsClassifier(),param_grid=hyperparams,verbose=True,cv=10,scoring="roc_auc")
gd.fit(X,y)



#Grid Search cross validations
#K neighnours network
#n_neighbors = [6,7,8,9,10,11,12,14,16,18,20,22]
#algorithm  = ['auto']
#weights = ['uniform','distance']
#leaf_size = list(range(1,50,5)) # 1-50 in the increments of 5
#Define hyperparamenters
#hyperparams = {'algorithm': algorithm, 'weights': weights, 'leaf_size': leaf_size, 
#               'n_neighbors': n_neighbors}
#rfc = RandomForestClassifier()
#gd=GridSearchCV(estimator=RandomForestClassifier(),param_grid = {}, verbose=True,cv=10,scoring="roc_auc")
#gd.fit(X,y)

print(gd.best_score_)
print(gd.best_estimator_)
#Pick the bet estimator chosen by the grid search and use it to train our model
gd.best_estimator_.fit(X,y)
y_pred = gd.best_estimator_.predict(X_test)

print(gd.best_score_)
print(gd.best_estimator_)
#Pick the bet estimator chosen by the grid search and use it to train our model
gd.best_estimator_.fit(X,y)
y_pred = gd.best_estimator_.predict(X_test)

#The above line is commented out as there is no label for the test data
temp = pd.DataFrame(pd.read_csv("../input/test.csv")['PassengerId'])
temp['Survived'] = y_pred
temp.to_csv("../working/submission.csv", index = False)
# Any results you write to the current directory are saved as output.