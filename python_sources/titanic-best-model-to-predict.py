#!/usr/bin/env python
# coding: utf-8

# ## Titanic : Machine Learning from Disaster

# The sinking of the RMS Titanic is one of the most infamous shipwrecks in history.  On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This sensational tragedy shocked the international community and led to better safety regulations for ships.
# 
# One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew. Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class.
# 
# In this challenge, we ask you to complete the analysis of what sorts of people were likely to survive. In particular, we ask you to apply the tools of machine learning to predict which passengers survived the tragedy.

# # Importing the libraries

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# ### Importing the Titanic dataset

# In[ ]:


df = pd.read_csv('../input/train.csv')


# ## Data understanding and Exploration

# ### Understanding the dataset by Descriptive statistic and Exploring the data through visualization 
# #### Describe >> Gives the overview of the distribution of the data by looking mean and std and how the quartiles are distributed
# #### Info >> Gives the information that is any Null values in the dataset and gives understanding of  datatypes and structure
# #### head >> Gives the top five observations of the dataset
# #### columns >>  contain columns in the dataset

# In[ ]:


df.head()


# In[ ]:


df.columns


# In[ ]:


df.describe()


# ### Graph Interpretation

# In[ ]:


f,ax = plt.subplots(1,2,figsize=(18,5))
#ax[0], ax[1]
## First Plot
df['Survived'].value_counts().plot.pie(explode=[0,0.1],
                                         autopct='%1.1f%%',
                                         shadow=True,
                                         colormap = 'Accent',
                                         ax = ax[0])
ax[0].set_title('Survived')

### second plot
sns.countplot('Survived',data=df,ax=ax[1])
ax[1].set_title('Survived')


# #### Here we observed that out of total passengers only 38.4% is survived and 61.6% of the people are not survived 

# In[ ]:


sns.countplot('Sex',data =df,hue = 'Survived',palette ='rocket_r')


# Here we can distinguish that above 400 of  male are dead and only 100 are able to survived but in case of female 
# most of female are able to survived and less no. of female are dead it is because female and child given the first preference to abroad from Titanic Ship as we can observed.

# In[ ]:


df.info()


# In this Titanic dataset we have 891 observations and 12 columns where age and Cabin  columns contain the null values and datatypes is of float,int and object.

# In[ ]:


sns.pairplot(df,diag_kind = 'kde')


# In[ ]:


sns.swarmplot(x='Embarked',y='Age',data =df,hue = 'Sex',palette = 'magma')


# In this above swarm plot show us that distribution of the age into the Embarked. 
# And most of the people of age between 19 to 50 is belong to Embarked S 

# In[ ]:


f,ax=plt.subplots(1,3,figsize=(18,8))
## first plot
df['Pclass'].value_counts().plot.bar(color=['r','c','g'],ax=ax[0])

ax[0].set_title('Number Of Passengers By Pclass')
ax[0].set_ylabel('Count')
# second plot
pd.crosstab(df.Pclass,df.Survived).plot.bar(ax=ax[1],stacked = True,colormap ='rainbow_r')
ax[1].set_title('Pclass:Survived vs Dead')
# third plot
sns.countplot('Survived',hue='Pclass',data=df,ax=ax[2],palette ='prism_r')
ax[2].set_title('Survived vs Pclass')


# Here from above graph we can observed from 1st graph  that the number of travelling passenger is belong to the Pclass 3 as the Fare is very low for that class.
# In second graph the most dead people is from Pclass 3 it is around 350
# And in the graph it is stated that the number of most dead people from Pclass 3 and number of most survived from Pclass 1 as that after female and child the perference is given to the Pclass wise to aboard from the ship.

# ### Reponse variable or dependent variable

# In[ ]:


y= df.loc[:,['Survived']]


# ### Features or independent variables

# In[ ]:



X=df.loc[:,['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']]


# In[ ]:


X.head()


# In[ ]:


df.corr()


# In[ ]:


sns.heatmap(df.corr(),cmap ='summer',annot = True,lw = 0.5,linecolor='black')


# Heatmap is use to find the multicolinearity withing the predictor as we can see there is no multicolinearity
# within the predictor and if the correlation between predictor and response is strongly correlated then 
# we can say it is significance variable with respective to response.

# In[ ]:


# Checking for null values in the dataset
X.isnull().sum()


# As we can see above Age,Cabin and Embarked  has null values

# In[ ]:


y.Survived.value_counts()


# ### Handling the null values i.e Age,Cabin and Embarked

# Handling the null values of Age columns 
# As I can do the fillna with mean but it will make no sense as the mean of the age column is 29 
# like I  am replacing it with 4 year old child which is meaningless so instead I can find out 
# the number of child,women,men in the dataset by extracting the Title from the name columns and will accordingly fill the values
# like if the Title in the name is master then we can fill it will 4 years old and similarily we do for

# ##### Here in the dataset the Name column has the title like Mr,Miss,Master,Mrs and so on.
# 
data.Name
master --> young boy
Mr ---> adult
miss --> young girls
mrs ---> married
ms
m
# #### Extracting the Title from the name column and finding the columns

# In[ ]:


X['Title_split'] = X.Name.str.split('[.,]').str.get(1)
X.Title_split = X.Title_split.str.strip()
X.Title_split.value_counts()


# In[ ]:





# In[ ]:


# Creating the new columns Title_reg for the title which is extracted from the name column
X['Title_reg']=''
for i in df:
    X['Title_reg']=X.Name.str.extract('([A-Za-z]+)\.')
    #lets extract the Salutationsd\.


# In[ ]:


X[['Title_reg','Name']].head()


# In[ ]:


X['Title'] = X.Title_split


# In[ ]:


X.columns


# In[ ]:


# Dropping the columns from the dataset Title_split and Title_reg
X.drop(['Title_split','Title_reg'], axis = 1, inplace = True)


# In[ ]:


## no. of unique 
print(X.Title.describe())
print(X.Title.nunique())


# In[ ]:


# Unique value count from the column name Title from dataset
X.Title.value_counts()


# In[ ]:


#Checking the Initials with the Sex
pd.crosstab(X.Title,X.Sex,margins=True).T.style.background_gradient(cmap='summer_r') 


# In this we can see the Title according to the gender and can see the distribution of the gender with the Title 
# So with this knowledge we can fill the null values accordingly to the Title like if some observation say with 
# the title name of Master then we can fill the null value with 5 years old and accordingly we do it for all

# In[ ]:


# Classify according to the Title
# Miss >> Mlle,Mme
# Mrs >> Ms,Lady,the countess
# Mr >> Dr,Major,Capt,Sir,Don,Jonkheer,Col,Rev

X['Title_1']=X['Title']
X['Title_1'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','the Countess','Jonkheer','Col','Rev','Capt','Sir','Don'],
                                       ['Miss','Miss','Mrs','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr'],
                      inplace=True)


# In[ ]:


# calculating the total counts of each unique Title
X['Title_1'].value_counts()


# In[ ]:


X['Title'] =X.Title_1


# In[ ]:


X.columns


# In[ ]:


X.drop(['Title_1'],axis =1,inplace = True)


# In[ ]:


X.columns


# In[ ]:


#lets check the average age by Initials
X.groupby('Title')['Age'].mean() 


# In[ ]:


X[X.Title=='Master'].Age.isnull().sum()


# In[ ]:


## Assigning the NaN Values with the Ceil values of the mean ages
X.loc[(X.Age.isnull())&(X.Title=='Mr'),'Age']=33
X.loc[(X.Age.isnull())&(X.Title=='Mrs'),'Age']=36
X.loc[(X.Age.isnull())&(X.Title=='Master'),'Age']=5
X.loc[(X.Age.isnull())&(X.Title=='Miss'),'Age']=22
X.loc[(X.Age.isnull())&(X.Title=='Other'),'Age']=46


# In[ ]:


X.Age.isnull().sum()


# In[ ]:


f,ax=plt.subplots(1,2,figsize=(18,5))
df[df['Survived']==0].Age.plot.hist(ax=ax[0],
                                        bins=20,
                                        edgecolor='black',
                                        color='red')
ax[0].set_title('Dead')
x1=list(range(0,85,10))
ax[0].set_xticks(x1)

df[df['Survived']==1].Age.plot.hist(ax=ax[1],
                                        color='green',
                                        bins=20,
                                        edgecolor='black')
ax[1].set_title('Survived')
x2=list(range(0,85,5))
ax[1].set_xticks(x2)
plt.show()


# In[ ]:


# Replacing the null values for Embarked
X['Embarked'].value_counts()


# As we see above the most occurance value in Column Embarked is S so we will replace Null values with S

# In[ ]:


X['Embarked'].fillna('S',inplace=True)


#  Checking for the null values as we see the columns Age and Embarked there is no null values
#  in case of Cabin columns 50% of the values are null so will drop the column and has not giving
#  any meaning to the Dataset
# 

# In[ ]:


X.isnull().sum()


# In[ ]:


X.head()


# Dropping the columns which are meaningless and of dtype as object as Machine Learning model only understand the 
# float,int,bool values

# In[ ]:


X.drop(['PassengerId','Name','Ticket','Cabin'], axis=1,inplace =True)


# In[ ]:


X.isnull().sum()


# In[ ]:


# When cleaning the test data 
# X.Fare.fillna(35.6271,inplace =True)


# In[ ]:


X.drop(['Title'],axis =1,inplace = True)


# In[ ]:


X.head()


# In[ ]:





# In[ ]:


sns.pairplot(X,diag_kind = 'kde')


# In[ ]:


X.SibSp.value_counts()


# In[ ]:


#pd.get_dummies(df_train['Sex'], drop_first=True)
df_train_dummied = pd.get_dummies(X, columns=["Sex"])


# In[ ]:


df_train_dummied.head()


# In[ ]:


X = pd.get_dummies(df_train_dummied, columns=["Embarked"])


# In[ ]:


X.head()


# In[ ]:





# In[ ]:


X.Sex_female.value_counts()


# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X = scaler.fit_transform(X)


# ## Independent variables

# In[ ]:


X


# ## Dependent variable

# In[ ]:


y = y['Survived'].values


# ### Splitting the dataset into training and testing with the ratio of 70:30 

# ### 70% of the data is goes for training 

# ##### X_train is the independent variables training dataset which is 70% of the total.
# ##### y_train is the dependent varaible dataset for training which is 70% of the total

# ### 30% of the data is reserved for testing

# #### X_test is the independent variables testing dataset which is 30% of the total
# #### y_test is the dependent variables testing dataset which is 30% of the total

# In[ ]:


get_ipython().run_line_magic('time', '')
# Splitting the dataset
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =0.3,random_state=1)


# # Logistic Regression Model

# In[ ]:


get_ipython().run_line_magic('time', '')
# Importing the LR model from scikit learn linear model
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()


# In[ ]:


# Fitting the LR model on training dataset
classifier.fit(X_train,y_train)


# In[ ]:


# Predicting the values on independent variables testing dataset
y_pred = classifier.predict(X_test)


# In[ ]:


# Confusion matrix for evaluation to get the accuracy of the model
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
cm


# In[ ]:


sns.heatmap(cm,annot = True,cmap ='plasma')


# In[ ]:


# Accuracy of the LR model is base on Actual values and predicting values by the model
from sklearn.metrics import accuracy_score
model_accuracy = accuracy_score(y_test,y_pred)
print('Accuracy of the model : ',model_accuracy)


# In[ ]:


# Cross validation score of 10 Kfolds
from sklearn.model_selection import cross_val_score
accuracy = cross_val_score(estimator = classifier,X=X_train,y=y_train,cv=10)
accuracy


# In[ ]:


print("The mean accuracy for  10 Kfolds :  ",accuracy.mean())


# In[ ]:


print("The Std deviation of the model :",accuracy.std())


# # Decision Tree Classifier

# In[ ]:


get_ipython().run_line_magic('time', '')
# Importing the model from scikit learn tree
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()


# In[ ]:


# fitting the DT model on training dataset
classifier.fit(X_train,y_train)


# In[ ]:


# Predicting the values on independent variables testing dataset
y_pred = classifier.predict(X_test)


# In[ ]:


# Confusion matrix for evaluation to get the accuracy of the model
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)


# In[ ]:


#Confusion Matrix
cm


# In[ ]:


sns.heatmap(cm,annot = True)


# In[ ]:


# Accuracy of the DT model is base on Actual values and predicting values by the model
from sklearn.metrics import accuracy_score
model_accuracy = accuracy_score(y_test,y_pred)
print("The accuracy of the DT model :",model_accuracy)


# In[ ]:


# Cross validation score of 10 Kfolds
from sklearn.model_selection import cross_val_score
accuracy = cross_val_score(estimator = classifier,X=X_train,y=y_train,cv=10)
accuracy


# In[ ]:


print('The mean accuracy for 10 Kfold :',accuracy.mean())


# In[ ]:


print("The standard deviation of the DT model :",accuracy.std())


# # Random Forest Classifier

# In[ ]:


get_ipython().run_line_magic('time', '')
# Importing the RF model from scikit learn ensemble 
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier()
model = classifier.fit(X_train,y_train)
model


# In[ ]:


# Predicting the values on independent variables testing dataset
y_pred = classifier.predict(X_test)


# In[ ]:


# Confusion matrix for evaluation to get the accuracy of the RF model
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
cm


# In[ ]:


sns.heatmap(cm,annot = True,cmap = 'viridis')


# In[ ]:


# Accuracy of the DT model is base on Actual values and predicting values by the model
from sklearn.metrics import accuracy_score
model_accuracy = accuracy_score(y_test,y_pred)
print('The accuracy of the RF model : ',model_accuracy)


# In[ ]:


# Cross validation score of 10 Kfolds
from sklearn.model_selection import cross_val_score
accuracy = cross_val_score(estimator = classifier,X=X_train,y=y_train,cv=10)
accuracy


# In[ ]:


print('The mean accuracy for 10 Kfold :',accuracy.mean())


# In[ ]:


print("The standard deviation of the DT model :",accuracy.std())


# # XGBoost Model

# In[ ]:


get_ipython().run_line_magic('time', '')
# Importing the XGBoost model from scikit learn ensemble 
from xgboost import XGBClassifier
classifier = XGBClassifier(n_estimator =1000)
classifier.fit(X_train,y_train)


# In[ ]:


# Predicting the values on independent variables testing dataset
y_pred = classifier.predict(X_test)


# In[ ]:


# Confusion matrix for evaluation to get the accuracy of the XGBoost model
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
cm


# In[ ]:


sns.heatmap(cm,annot = True,cmap="cividis")


# In[ ]:


# Accuracy of the XGBoost model is base on Actual values and predicting values by the model
from sklearn.metrics import accuracy_score
model_accuracy = accuracy_score(y_test,y_pred)
model_accuracy


# In[ ]:


# Cross validation score of 10 Kfolds
from sklearn.model_selection import cross_val_score
accuracy = cross_val_score(estimator = classifier,X=X_train,y=y_train,cv=10)
accuracy


# In[ ]:


print('The mean accuracy for 10 Kfold :',accuracy.mean())


# In[ ]:


print("The standard deviation of the DT model :",accuracy.std())


# # The Champion model out of all Models is XGBoost
