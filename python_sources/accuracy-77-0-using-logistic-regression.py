# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

import matplotlib as mpl
import seaborn as sb
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import datawig as dw
from sklearn.metrics import f1_score, classification_report
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier

)


#loading the csv files as input for data. rather simple when you use Pandas.
#csv is a common datatype used in Machine Learning. Json and sql etc other complex
#ones wto get introduced later. need to know that language for it.
#Csv can be opened with Xcel and you can look at the database like that.

#NOTE FOR TEST FILE:- this is the file that has no data of survived or not
#This is the database that has to be filled up with predictions table and submitted
#into kaggle for your score. YOU SHALL NOT USE IT AS TRAIN TEST SPLITS TEST VARIABLE
#its for testing and submitting your model in kaggle. you cannot evaluate it
#basically its new unseen data. KAGGLE WILL EVALUATE YOUR MODEL BASED ON IT
ted= pd.read_csv('..input/test.csv')
trd= pd.read_csv('..input/train.csv')

#now once we load files, note that csv files can literally be made of
#anything. even our images we saved as csv. now we should ideally check
#what its datatype is.

#here what we have is a type called dataframe. Fancy words for a tabular
#column really. no big deal. its tabulated data.

print(type(trd))
print(type(ted))

#now we can obviously skip everything just make features out of every table
#and just train test split the train data but that makes for bad data
#analyisis and bad models. lets first open the data print it and see various
#columns, then compare and see what effect it has on survival and death.

#we do this by data visualization. Making different plots of different
#categories in our data. obviously different plots reveal different charecters.

print(trd) #to display the table and see various subtables

'''
exploratory data analysis. this portion is to be done by you and you must
vary different kind of plots, note anything interesting about it like
gaussian distribution etc (we have limited knowledge of this so far but its
stil something that will be important so try) and also which ones you see
major changes etc. first see each individual ones by comparing with
passenger id etc

now always remember why you are doing this. YOU NEED TO SEE A RELATION
BETWEEN SURVIVAL AND OTHER DATA COLUMNS. BECAUSE THATS WHAT YOU FINALLY
WANT TO KNOW. SURIVIVAL OR DEATH. THATS THE THING TO PREDICT HERE. ANALYZE
ACCORDINGLY.
'''
#sb.scatterplot(data=trd,x= 'Sex' ,y= 'Age',hue='Survived')
#hashing things because i dont want multiple plots at once

#sb.barplot(data=trd,x= 'PassengerId' ,y= 'Pclass',hue='Survived')
#sb.barplot(data=trd,x= 'PassengerId' ,y= 'Embarked',hue='Survived')
#sb.Barplot(data=trd,x= 'PassengerId' ,y= 'SibSp',hue='Survived')
#sb.scatterplot(data=trd,x= 'PassengerId' ,y= 'Fare',hue='Survived')
#sb.scatterplot(data=trd,x= 'Age' ,y= 'Parch',hue='SibSp')
#sb.scatterplot(data=trd,x= 'PassengerId' ,y= 'Fare',hue='Survived')
#plt.show()

#noticed too many missing terms in cabin. wondering how to fill it up
#checking how many missing terms in each table
print(trd.info())

'''
before going any further, note that i realised a personal LOL that happened.
while training i realized that i should have performed all the actions and
data manipulations i performed on Train.csv on Test.csv as well. Since i have
created new features, when your model is being fitted , it has those new features
which are required for prediciton. therefore it should be present in test as well
therefore all actions are repeated on Test.csv (ted) as well
'''
#age has a few missing values which need to be filled
'''
corr = trd.corr().abs().unstack().sort_values(kind="quicksort", ascending=False).reset_index()
corr.rename(columns={"level_0": "Feature 1", "level_1": "Feature 2", 0: 'Correlation Coefficient'}, inplace=True)
corr[corr['Feature 1'] == 'Age']
print[corr]

found this on a kaggle kernel. This guy used correlation to find features
which correlate to age the most thereby have a good predictable relation.
he took mean along Pclass which had best corrbut i found something more
interesting. So im testing that out, but im leaving Correlation here
so that you realise why i used these features. honestly i thought age
will depend on Parch and SibSp because family = age bla bla but i made a mistake
by forgetting they also take in consideration kids. still important and useful
'''       
df_train, df_test = dw.utils.random_split(trd)

#This method is called Imputation. its directly done from a library
#which is Datawigs. imputation is the process by which you fill in for
#missing information. here it uses deep learning to fill in missing info.

#Initialize a SimpleImputer model
imputer = dw.SimpleImputer(
    input_columns=['SibSp','Fare','Parch','Age','Pclass'], # column(s) containing information about the column we want to impute
    output_column='Age', # the column we'd like to impute values for
    output_path = 'imputer_model' # stores model data and metrics
    )

#Fit an imputer model on the train data
imputer.fit(train_df=trd, num_epochs=50)

#Impute missing values and return original dataframe with predictions
imputed = imputer.predict(trd)

#REPEATING THE SAME FOR TEST FILE
imputerTest = dw.SimpleImputer(
    input_columns=['SibSp','Fare','Parch','Age','Pclass'],
    output_column='Age', 
    output_path = 'imputer_model' 
    )

imputerTest.fit(train_df=ted, num_epochs=50)

imputedTest = imputerTest.predict(ted)

'''explaination of what happened above and below
What the above code did is it created a deeplearning model. dw.simple imputer did it
the parameters input_columns are the columns that have some information
corresponding to the table whose values you want to fill

since that correlation predicted SibSp, fare Parch and Pclass have decent Corr
i used them. I introduced Age so that the accuracy isnt horrifying. since
obviously age itself is a very strong data for you know, predicting age!

now imputer fit is same as any ML, DL , you fit data to the model
Here the train test split thing is just for checking accuracy.
when finally using it for real, you need to enter all data and
imputer.predict also for all data. because obviously YOU want to FILL
ALL the NA values.
This concludes for what happened above.

for what happens below this, i checked by printing imputed.
realised it prints whole dataframe (table with a fancy name)
So now i wanted to Fill in the missing values which i PREDICTED?
so to do that theres a function of dataframe ( here dataframe is trd)
(trd[age] because you are working on that column in trd)
the function is fillna. now i wanted to fill the NA values from my predictions.
prediction DataFrame is imputed and i want only Age_imputed column from it.

therefore the below code does exacly that.
'''

#EDIT FOR TRAIN

trd['Age'] = trd['Age'].fillna(imputed['Age_imputed'])

#EDIT FOR TEST

ted['Age'] = ted['Age'].fillna(imputed['Age_imputed'])

#seperating titles and surnames from peoples names

#EDIT FOR TRAIN
trd['Surname']=trd["Name"].str.split(',',expand=True)[0]
trd['Title']=trd['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]

trd['Title']=trd['Title'].replace(['Mlle','Lady','Mme','the Countess','Dona'],'FNobility')
trd['Title'] = trd['Title'].replace(['Dr', 'Col', 'Major', 'Jonkheer', 'Capt', 'Sir', 'Don', 'Rev'], 'RareM')

#EDIT FOR TEST
ted['Surname']=ted["Name"].str.split(',',expand=True)[0]
ted['Title']=ted['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]

ted['Title']=ted['Title'].replace(['Mlle','Lady','Mme','the Countess','Dona'],'FNobility')
ted['Title'] = ted['Title'].replace(['Dr', 'Col', 'Major', 'Jonkheer', 'Capt', 'Sir', 'Don', 'Rev'], 'RareM')

#Creating a new databasec called deck which only has cabin Alphabets. missing ones taken as M
trd['Deck'] = trd['Cabin'].apply(lambda s: s[0] if pd.notnull(s) else 'M')

#Doing Same For Train
ted['Deck'] = ted['Cabin'].apply(lambda s: s[0] if pd.notnull(s) else 'M')

#finally list of features selected
#PassengerId dropped because it has no information
#name dropped because all info from it has been used to create surname and title
#cabin dropped because used it to create Deck

features = ['Fare', 'Age', 'Sex', 'Surname', 'Title','Embarked','Pclass','Deck','Ticket','SibSp','Parch']
tot = pd.concat([trd[features], ted[features]])
print(ted.info())


#encoding labels for all features

for feature in features:
    le = preprocessing.LabelEncoder()
    le = le.fit(tot[feature])
    trd[feature] = le.transform(trd[feature])
    ted[feature] = le.transform(ted[feature])

trd.head()

#trd.drop means dropping those tables rest all become features
#had to do this for train test split. rest all are my features
#the ones dropped were the ones not on my features array
Xtrain = trd.drop(['Survived', 'PassengerId','Name','Cabin'], axis=1)
#testing feature is for obvious reasons survived because we are testing if dead or survived.

Ytrain = trd['Survived']

Size = 0.2
X_train, X_test,y_train, y_test = train_test_split(Xtrain,
                                                   Ytrain,
                                                   test_size=Size,
                                                   random_state=100
                                                   )
RandF= RandomForestClassifier(n_estimators=100,
                              criterion="entropy",
                              max_depth=7,
                              )
RandF.fit(X_train,y_train)
RandPred= RandF.predict(X_test)
RandScore=accuracy_score(y_test, RandPred)
print("Training accuracy of random forest",RandScore)

LogReg = LogisticRegression(random_state=200,
                            solver= 'liblinear',
                            multi_class='ovr',
                            penalty='l2',
                            C=1,
                            )
LogReg.fit(X_train,y_train)
LogRegPred= LogReg.predict(X_test)
LogRegScore=accuracy_score(y_test, LogRegPred)
print("Training accuracy of Logistic Regression",LogRegScore)
'''
now you have trained on the train dataset, tested on the train
dataset correct? tuned etc all done. time to predict using your
model for the unknown dataset.
'''
#Xtest here is the variable for Test.csv . since data from test.csv is in ted
#we are using ted. obviously we do have to drop the features we dropped (survived is excluded because test doesnt have it)in Train
#in test as well so its done. Now previously we had predicted for a test created
#from Train.csv itself by using test train split. Ie it was not data outside Train.csv
#now we are predicting for data we have no way to calculate accuracy for. accuracy only
#calculable by sending it to Kaggle.

Xtest = ted.drop(['PassengerId','Name','Cabin'], axis=1)


RealPrediction= LogReg.predict(Xtest)

#The line below is simply creating a dataframe (a fancy named table). theres a parameter called
#data, thats why the data is inside {brackets like this} survived has RealPrediction because
#we just used it to predict survival for Test.csv from the above line.

submission = pd.DataFrame({"PassengerId" : ted["PassengerId"],"Survived" : RealPrediction})
#.to_csv is a function of any dataframe. it basically stores the values. note adress of the file must be given
#index false because it adds an index otherwise. we already have passenger id for it.

submission.to_csv('submission.csv', index=False)




# Any results you write to the current directory are saved as output.