#!/usr/bin/env python
# coding: utf-8

# ## Import Moduels

# In[1]:


# data analysis 
import pandas as pd
import numpy as np
 
# visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(style="whitegrid")
# preprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
# algorithm of machine learning and evaluation the models
from sklearn.model_selection import train_test_split #to create validation data set
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score 


# ## Problem Definition
# - from (https://www.kaggle.com/c/titanic#description)

# - The sinking of the RMS Titanic is one of the most infamous shipwrecks in history.  On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This sensational tragedy shocked the international community and led to better safety regulations for ships. One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew. Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class.

# ### Refrence Resourses used by me to get idea , sequance and flow work:
# - https://www.kaggle.com/startupsci/titanic-data-science-solutions
# - https://www.kaggle.com/samsonqian/titanic-guide-with-sklearn-and-eda
# - https://www.kaggle.com/yassineghouzam/titanic-top-4-with-ensemble-modeling
# * and other documented inside the cells in this NoteBook.
# 

# ## Get training and testing data.

# In[4]:


# get training and testing data.
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')


# In[5]:


# save the ids for later to use in the test
Id_passenger=df_test['PassengerId'] 


# ## Prepare, cleanse the data.

# In[6]:


#check train df
df_train.head()


# In[8]:


#check trest df
df_test.head()


# In[9]:


#check train,test df keys
print(df_train.keys())
print(df_test.keys())


# In[10]:


#check null values
print('Train null value')
print(df_train.isnull().sum())
print('TEST null value')
print(df_test.isnull().sum())


# In[11]:


#check train,test df shapes
df_train.shape,df_test.shape


# In[12]:


#check train,test df dtypes
print(df_train.dtypes)
print(df_test.dtypes)


# In[13]:


# get Summarie and statistics
df_train.describe()


# - The Cabin column contains nan values > than the 1/2 of the rows no so it will delete it in bothe test and train df.

# In[14]:


df_train=df_train.drop(['Cabin','Ticket'],axis=1)
df_test=df_test.drop(['Cabin','Ticket'],axis=1)


# - The age column in both train and test df contains nan values and to fill this nan and after looking to the age distrbution below its more save to use median ro fill nan more than mean because the distrbution is right skewnes.

# In[15]:


sns.kdeplot(df_train['Age'], shade=True);


# In[16]:


df_train['Age']=df_train['Age'].fillna(df_train['Age'].median())
df_test['Age']=df_test['Age'].fillna(df_test['Age'].median())


# In[17]:


df_train['Age'].isnull().sum(),df_test['Age'].isnull().sum()


# - The Fare column has the same issue as age column regarding skewness in the test dfso, the mediian will be used here to fill the nan. 

# In[18]:


sns.kdeplot(df_test['Fare'], shade=True);


# In[19]:


df_test['Fare']=df_test['Fare'].fillna(df_test['Fare'].median())


# In[20]:


df_train['Fare'].isnull().sum(),df_test['Fare'].isnull().sum()


# In[21]:


print('Train null value')
print(df_train.isnull().sum())
print('TEST null value')
print(df_test.isnull().sum())


# - Last column contains nan value is Embarked in the train df and as the value of this column are categrical, it will be fielled with Q = Queenstown.

# In[22]:


df_train['Embarked']=df_train['Embarked'].fillna('Q')


# ## Analyze and explore the data

# ## Numerical Data

# In[23]:


# Heatmap to see correlation between numerical values (SibSp Parch Age and Fare values) and Survived 
fig, ax = plt.subplots(figsize=(10, 8))
corr = df_train[["Survived","SibSp","Parch","Age","Fare"]].corr()
sns.heatmap(corr, cmap=sns.diverging_palette(220, 10, as_cmap=True),square=True, ax=ax)


# Fare has correlation with survaived feature more than other numeric features.

# In[24]:


parchXsurvived = sns.barplot(x="Parch",y="Survived",data=df_train)


# - Minimum member of family are likely to survive more than single passengers and large family ,but this is not the case with 3 parents/children .

# In[25]:


sibSPXsurvived = sns.barplot(x="SibSp",y="Survived",data=df_train)


# - Passengers with less sibling and single ones are more likely to survive than the passengers with many number of siblings. 

# In[26]:


ageXsurvived = sns.kdeplot(df_train['Age'][(df_train["Survived"] == 1)])


ageXsurvived = sns.kdeplot(df_train['Age'][(df_train["Survived"] == 0)])

ageXsurvived=ageXsurvived.legend(['Survived',"Not Survived"])


# - The age distrbution regarding survive or not survive is not the same you , there is apeak in the survive curve shows that younger passenger are more likely to survive. Moreover,the range between 60 - 80 passengers have less chance to survive.It conclude that there is some catogrical ranges that are more likely to survive. 

# ## Categrical values

# In[27]:


sexXsurvived = sns.barplot(x="Sex",y="Survived",data=df_train)


# In[28]:


df_train[["Sex","Survived"]].groupby('Sex').mean()


# - From above female have a big chance to servive than male.

# In[29]:


sns.barplot(x="Pclass", y="Survived", data=df_train);


# In[30]:


sns.barplot(x="Pclass", y="Survived", hue="Sex", data=df_train);


# - First class passenger are more likely to survive more than the other two class's, and also the are is large different between female and male survivers in the same class too.

# In[31]:


sns.barplot(x="Embarked", y="Survived",data=df_train);


# - It seems like people from who come from Cherbourg are more likely to survive that the ,Queenstown and Southampton.But this is information need to be investigate more so what about seeing where are the  class's  that those people have?!

# In[32]:


sns.catplot("Pclass", col="Embarked",  data=df_train,kind="count");


# - The 3 class contains more passengers survive who come from Southampton and Queenstown , but Cherbourg than passengers are in class 1 where there survival rate is heigher.
# 
#  

# ## Feature Engineering
# ### Using One-Hot-Encoding for catogrical values (not binary dummies )(https://medium.com/@michaeldelsole/what-is-one-hot-encoding-and-how-to-do-it-f0ae272f1179).
# 
#   - Values will for Sex column :
#            - "male" = 0
#            - "female" = 1
#   - Values will for Embarked column:
#            - "S"= 0
#            - "C"= 1
#            -  "Q"= 2
#   

# In[33]:


#set up a labelencoder
label_train_sex= LabelEncoder()
# convert columns [Sex,Embarked] to numerical data
df_train['Sex']=label_train_sex.fit_transform(df_train['Sex'])
#set up a labelencoder
label_train_embarked= LabelEncoder()
df_train['Embarked']=label_train_embarked.fit_transform(df_train['Embarked'])


# In[34]:


#set up a labelencoder
label_test_sex= LabelEncoder()
# convert columns [Sex,Embarked] to numerical data
df_test['Sex']=label_test_sex.fit_transform(df_test['Sex'])
#set up a labelencoder
label_test_embarked= LabelEncoder()
df_test['Embarked']=label_test_embarked.fit_transform(df_test['Embarked'])


# ### Extracting new features from exesting variable.

# - In previouse sections its appear that if passenger have family either siblings or parents/children , so using the 2 feature SibSp and Parch will create new feature called family_member_no, to help in identifing the affect of no of family members on surviving.

# In[35]:


# start creating family_member_no" column in train and test df by adding 2 columns values SibSp+Parch + 1 (which is the current passenger)
df_train["family_member_no"] = df_train["SibSp"] + df_train["Parch"] + 1
df_test["family_member_no"] = df_test["SibSp"] + df_test["Parch"] + 1


# In[36]:


# if passenger is single traveller we might get more accurate info because its a general case so I will set single traveller as 1
df_train["family_member_no"] = df_train["SibSp"] + df_train["Parch"] + 1

df_test["family_member_no"] = df_test["SibSp"] + df_test["Parch"] + 1


# In[37]:


# I will get this info from family_member_no and i will use apply and lambda to do it
df_train["Single"] = df_train.family_member_no.apply(lambda a: 1 if a == 1 else 0)
df_test["Single"] = df_test.family_member_no.apply(lambda a: 1 if a == 1 else 0)


# - Age and Fare column have a large differense in terms of value distrbution so I will not delete ore replace these values with any thing I will just rescall them in both test and train.

# In[38]:


#Inshalize standerscaler 
ss= StandardScaler()
# change value as array and reshap them after that to pass them to the scaler fit_transform method
age_tr = np.array(df_train["Age"]).reshape(-1, 1)
fare_tr = np.array(df_train["Fare"]).reshape(-1, 1)
age_ts = np.array(df_test["Age"]).reshape(-1, 1)
fare_ts = np.array(df_test["Fare"]).reshape(-1, 1)
# fit_and transform column value and reduce their magnitude using ss
df_train["Age"] = ss.fit_transform(age_tr)
df_train["Fare"]= ss.fit_transform(fare_tr )
df_test["Age"]["Age"] = ss.fit_transform(age_ts)
df_test["Fare"] = ss.fit_transform(fare_ts)


# - Also Name feature are not numeric and has useless info unless we take the classification word Mr and Mess so I will choose to drop it for now this is my approach .

# In[39]:


# drop name feature from the 2 df
df_train = df_train.drop('Name', axis=1) 
df_test = df_test.drop('Name', axis=1)


# In[40]:


df_train.head()


# In[41]:


df_test.head()


# ## Model, predict and solve the problem.

# In[42]:


#define features for  training and test
X_train = df_train.drop(["PassengerId", "Survived"], axis=1) 
y_train = df_train["Survived"]  
X_test = df_test.drop("PassengerId", axis=1) 


# In[43]:


X_train.head()


# In[44]:


# to ensure that model doesn't overfit with the data  train_test_split well be used. 
X_for_train, X_for_test, y_for_train, y_for_test= train_test_split(X_train, y_train, test_size=0.3, random_state=42) 


# In[45]:


# RandomForestClassifier
rd_f = RandomForestClassifier(n_estimators=20, criterion='entropy',random_state=42)
rd_f.fit(X_for_train, y_for_train)
predictions = rd_f.predict(X_for_test)

rd_f_model=accuracy_score(y_for_test,predictions)
print("RandomForestClassifier accurecy score: " ,rd_f_model)


# In[46]:


# LogisticRegression
log_model= LogisticRegression()
log_model.fit(X_for_train, y_for_train)
predictions = log_model.predict(X_for_test)
acc_log_model=accuracy_score(y_for_test,predictions)
print("Logistic Regression score: " ,acc_log_model)


# In[47]:


#KNeighborsClassifier
clf=KNeighborsClassifier(p=2, n_neighbors=10)
clf.fit(X_for_train,y_for_train)
predictions=clf.predict(X_for_test)
print('KNeighborsClassifier score: ',accuracy_score(y_for_test,predictions))


# ## Submit the results.
# - the final solution used RandomForestClassifier over LogisticRegression who gives more accurate score because RandomForestClassifier gives result 0,1 ,and It does not suffer from the overfitting problem because it takes the average of all the predictions, which cancels out the biases.(https://www.datacamp.com/community/tutorials/random-forests-classifier-python). 
# 
# 

# In[48]:


#Supply or submit the results.
final_prediction= pd.Series(rd_f.predict(X_test), name="Survived")

results = pd.concat([Id_passenger,final_prediction],axis=1)

results.to_csv("results.csv",index=False)


# ### Future work :
# - may use gridsearch and other tuning methods to get Hyper-parameters for the estimator classes where it will be optimized in this manner and get more better results.
