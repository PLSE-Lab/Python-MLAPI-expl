#!/usr/bin/env python
# coding: utf-8

# # EDA & ML Modelling Report on TITANIC Dataset
# * Analysis By: NEELESH DUGAR 
# * Email: dugar.nilesh23@gmail.com 
# * Mob: +91-7838823636

# # I. Exploratory Descriptive Analysis (EDA)

# In[ ]:


# We are importing WARNINGS class to suppress warnings for now
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


# We are importing all necessary classes which we will use in thsi report.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ### 1. Load the dataset

# In[ ]:


titanic = pd.read_csv('../input/titaniceda/titanic_dataset.csv')


# ##### (a) To see how many rows and columns are there in Titanic Dataset

# In[ ]:


titanic.shape


# ##### (b) To see the first 2 rows of the dataset

# In[ ]:


titanic.head(2)


# # ---INFERENCES---
# 1. There are 1309 rows and 14 columns.
# 2. Out of 14 columns, some are not of much importance, like, name, ticket, cabin, boat, body & home.dest. 
# > >    So I am removing them from our dataframe.

# ### 2. Dropping unwanted columns (This step is part of Data Cleansing)

# ##### (a) To drop the unwanted columns

# In[ ]:


titanic.drop(["name","ticket","cabin","boat","body","home.dest"],axis=1,inplace=True)


# ##### (b) To see the first 2 rows of the dataset, after dropping the columns

# In[ ]:


titanic.head(2)


# ##### (c) .describe() is used to get 5 Number Summary of the data (column-wise)

# In[ ]:


titanic.describe()


# ##### (d) if we use include="all" as argument with .describe() method, it gives additional info about the data

# In[ ]:


titanic.describe(include="all")


# ### 3. Filling empty rows/columns with appropriate data
# > (This step is part of Data Pre-processing)

# In[ ]:


titanic.isnull().any()
titanic.age.fillna(titanic.age.mean(), inplace=True)
titanic.fare.fillna(titanic.fare.mean(), inplace=True)
titanic.embarked.fillna(titanic.embarked.mode()[0], inplace=True)


# ### 4. Now Analysis Begins....

# ##### (a) To see how many males and females were there on the ship

# In[ ]:


print("Male vs Females count on ship:\n")
print(titanic.Gender.value_counts())

pd1 = pd.DataFrame(list(titanic.Gender.value_counts()), index=["Male", "Female"])
pd1.plot(kind="bar", width=0.5, title="Male - Female Distribution", legend=False)
plt.show()


# ##### (b) To see how many survived

# In[ ]:


print("Not Survived(0) vs Survived(1):\n")
print(titanic.survived.value_counts())

pd1 = pd.DataFrame(list(titanic.survived.value_counts()), index=["Not Survived", "Survived"])
pd1.plot.bar(width=0.5, alpha=0.5, title="How many survived?", color="green")
plt.show()

print("% of People who survived is ",(round((500/(500+809))*100,2)), "%")


# ##### (c) To see how many survived (Gender-wise)

# In[ ]:


male = titanic[titanic.Gender=="male"].Gender
female = titanic[titanic.Gender=="female"].Gender
male_s = titanic[titanic.survived==1].Gender[titanic.Gender=="male"]
female_s = titanic[titanic.survived==1].Gender[titanic.Gender=="female"]

plt.figure(figsize=(16,10))

plt.subplot(2,2,1)
plt.title("Total Males and Females")
plt.ylim(0,900)
plt.bar("female",female.size, color="pink", width=0.4)
plt.text("female",female.size+7,female.size)
plt.bar("male",male.size, width=0.4)
plt.text("male",male.size+7,male.size)

plt.subplot(2,2,2)
plt.title("Total Males and Females who Survived")
plt.ylim(0,900)
plt.bar("female_s",female_s.size, color="pink", width=0.4)
plt.text("female_s",female_s.size+3,female_s.size)
plt.bar("male_s",male_s.size, width=0.4)
plt.text("male_s",male_s.size+3,male_s.size)

plt.subplot(2,2,3)
sns.swarmplot(x="Gender", y="age", data=titanic)

plt.subplot(2,2,4)
sns.swarmplot(x="Gender", y="age", data=titanic, hue="survived", dodge=True)

plt.show()

print("Males who survived = ",round((161/843)*100,2), "%")
print("Females who survived = ",round((339/466)*100,2), "%")


# ##### (d) To see Age distribution using Dist plot

# In[ ]:


# dataset = sns.load_dataset(r"C:\Users\User\Documents\LearnBay Material\Datasets\titanic_dataset.csv")
plt.figure(figsize=(16,4))
sns.distplot(titanic.age)
# ax = sns.catplot(x="Gender",y="age", data="titanic_dataset")
plt.show()


# ##### (e) To see how many survived (Age-wise)

# In[ ]:


child = titanic.age[titanic.age<13]
print("Child: age<13")
teen = titanic.age[(titanic.age>12) & (titanic.age<20)]
print("Teen: age>12 and age<20")
adult = titanic.age[(titanic.age>19) & (titanic.age<60)]
print("Adult: age>19 and age<60")
old= titanic.age[titanic.age>59]
print("Old: age>59")

child_s = titanic[titanic.survived==1].age[titanic.age<13]
teen_s = titanic[titanic.survived==1].age[(titanic.age>12) & (titanic.age<20)]
adult_s = titanic[titanic.survived==1].age[(titanic.age>19) & (titanic.age<60)]
old_s= titanic[titanic.survived==1].age[titanic.age>59]

plt.figure(figsize=(16,4))

plt.subplot(1,2,1)
plt.title("Total passengers_age wise")
plt.ylim(0,1200)
plt.bar("child",child.size)
plt.text("child",child.size+5,child.size)
plt.bar("teen",teen.size)
plt.text("teen",teen.size+5,teen.size)
plt.bar("adult",adult.size)
plt.text("adult",adult.size+5,adult.size)
plt.bar("old",old.size)
plt.text("old",old.size+5,old.size)

plt.subplot(1,2,2)
plt.title("Total passengers who survived_age wise")
plt.ylim(0,1200)
plt.bar("child_s",child_s.size)
plt.text("child_s",child_s.size+5,child_s.size)
plt.bar("teen_s",teen_s.size)
plt.text("teen_s",teen_s.size+5,teen_s.size)
plt.bar("adult_s",adult_s.size)
plt.text("adult_s",adult_s.size+5,adult_s.size)
plt.bar("old_s",old_s.size)
plt.text("old_s",old_s.size+10,old.size)

plt.show()

print("Children who survived = ",round((54/94)*100,2), "%")
print("Teenagers who survived = ",round((52/131)*100,2), "%")
print("Adults who survived = ",round((382/1044)*100,2), "%")
print("Old people who survived = ",round((40/40)*100,2), "%")


# ##### (f) To see the distribution of age and passenger class

# In[ ]:


plt.figure(figsize=(16,4))
titanic.age[titanic.pclass==1].plot(kind="kde")
titanic.age[titanic.pclass==2].plot(kind="kde")
titanic.age[titanic.pclass==3].plot(kind="kde")
plt.legend(('1st Class','2nd Class','3rd Class'),loc = 'best')
plt.xlabel("Age Distribution")
plt.ylabel("No. of people")

plt.show()


# ##### (g) To see Age distribution with Scatter plot

# In[ ]:


plt.figure(figsize=(16,4))
for i in range(titanic.age.size):
    if(titanic.age[i]>0 and titanic.age[i]<20):
        plt.scatter(i, titanic.age[i], alpha=0.5, color="cyan")
    elif(titanic.age[i]>19 and titanic.age[i]<40):
        plt.scatter(i, titanic.age[i], alpha=0.5, color="orange")
    elif(titanic.age[i]>39 and titanic.age[i]<60):
        plt.scatter(i, titanic.age[i], alpha=0.5, color="green")
    else:
        plt.scatter(i, titanic.age[i], alpha=0.5, color="red")
plt.xlabel("No of people")
plt.ylabel("Age Distribution")
plt.show()


# ##### (h) To see how many passengers survived (Passenger Class-wise)

# In[ ]:


pc1 = titanic[titanic.pclass==1].pclass
pc2 = titanic[titanic.pclass==2].pclass
pc3 = titanic[titanic.pclass==3].pclass
pc1_s = titanic[titanic.survived==1].pclass[titanic.pclass==1]
pc2_s = titanic[titanic.survived==1].pclass[titanic.pclass==2]
pc3_s = titanic[titanic.survived==1].pclass[titanic.pclass==3]

plt.figure(figsize=(16,4))

plt.subplot(1,2,1)
plt.title("Total passengers (pclass-wise)")
plt.bar("1",pc1.size, color="gold")
plt.text("1",pc1.size+7,pc1.size)
plt.bar("2",pc2.size)
plt.text("2",pc2.size+5,pc2.size)
plt.bar("3",pc3.size)
plt.text("3",pc3.size+5,pc3.size)

plt.subplot(1,2,2)
plt.title("Total passengers who survived (pclass-wise)")
plt.bar("1",pc1_s.size)
plt.text("1",pc1_s.size+2,pc1_s.size)
plt.bar("2",pc2_s.size)
plt.text("2",pc2_s.size+2,pc2_s.size)
plt.bar("3",pc3_s.size)
plt.text("3",pc3_s.size+2,pc3_s.size)

plt.show()

print("P_Class=1 who survived = ",round((200/323)*100,2), "%")
print("P_Class=2 who survived = ",round((119/277)*100,2), "%")
print("P_Class=3 who survived = ",round((181/709)*100,2), "%")


# **INSIGHTS:-**
#     
#     1. % of people who survived is 38.19%.
#     2. 72.75% of all females survived, whereas, only 19.10% of all males survived.
#     3. All the old age people (whose age>59) survived, and more than 50% of kids (whose age<13) also survived.
#     4. Survival Rate (Passenger-Class wise) = PC1 > PC2 > PC3. This means who paid more were the ones who survived more.

# # II. Prediction using various ML Models

# ### 1. Load the Test & Training Data

# In[ ]:


train_df = pd.read_csv('/kaggle/input/train-testing-dataset/train.csv')
test_df = pd.read_csv('/kaggle/input/train-testing-dataset/test.csv')


# ### 2. Data Cleaning of "Training" Dataset 

# In[ ]:


#we will drop "Cabin" column as it is of no use in ML modelling
train_df = train_df.drop('Cabin', axis=1)

#we will now create a function to fill missing values in Age bt taking average age of each Pclass
def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 28
        else:
            return 24
    else:
        return Age
    
train_df['Age'] = train_df[['Age', 'Pclass']].apply(impute_age, axis=1)

#we will drop all columns with null attributes
train_df.dropna(inplace=True)

#we will remove all the non numerical columns and then use pd.get_dummies to get the dummy values of the Pclass & Sex column
sex = pd.get_dummies(train_df['Sex'], drop_first=True)
embarked = pd.get_dummies(train_df['Embarked'], drop_first=True)

#as we have the dummy values of Pclass & Sex, so we will drop these (Name, Sex, Ticket, Embarked) columns 
train_df.drop(train_df[['Name', 'Sex', 'Ticket', 'Embarked']], axis=1, inplace=True)
train_df = pd.concat([train_df, sex, embarked], axis=1)


# In[ ]:


print("TRAINING Dataset:-")
train_df.head()


# ### 3. Data Cleaning of "Test" Dataset

# ##### (We will do the same cleaning as we did in Training dataset)

# In[ ]:


#we will now create a function to fill missing values in Age bt taking average age of each Pclass
test_df['Age'] = test_df[['Age', 'Pclass']].apply(impute_age, axis=1)

#we will fill NULL values in "Fare" column by taking Mean of Fare column
test_df['Fare'].fillna(test_df['Fare'].mean(),inplace=True)

#we will remove all the non numerical columns and then use pd.get_dummies to get the dummy values of the Pclass & Sex column
sex_test = pd.get_dummies(test_df['Sex'], drop_first=True)
embarked_test = pd.get_dummies(test_df['Embarked'], drop_first=True)

#as we have the dummy values of Pclass & Sex, so we will drop these (Name, Sex, Ticket, Embarked) columns
test_df.drop(['Name', 'Sex', 'Embarked', 'Ticket', 'Cabin'], axis=1, inplace=True)
test_df = pd.concat([test_df, sex_test, embarked_test],  axis=1)


# In[ ]:


print("TESTING Dataset:-")
test_df.head()


# ### 4. PREDICTIONS (Finding the best ML model)

# In[ ]:


# We will create a Train-Test split of the Train_df dataset in order to find the best Classification Model.
X_train = train_df.drop(['Survived', 'PassengerId'], axis=1)
y_train = train_df['Survived']
X_test = test_df.drop('PassengerId', axis=1)


# ##### ***1. Logistic Regression***

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
log_pred = log_reg.predict(X_test)
log_score = round(log_reg.score(X_train, y_train) *100, 2)


# #### *2. K-Nearest Neighbor*

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)
knn_score = round(knn.score(X_train, y_train) * 100, 2)


# #### *3. Decision Tree*

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
d_tree = DecisionTreeClassifier()
d_tree.fit(X_train, y_train)
dt_pred = d_tree.predict(X_test)
dt_score = round(d_tree.score(X_train, y_train) * 100, 2)


# #### *4. Random Forest*

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_score = round(rf.score(X_train, y_train) * 100,2)


# #### *5. Naive Bayes*

# In[ ]:


from sklearn.naive_bayes import GaussianNB
guassian = GaussianNB()
guassian.fit(X_train, y_train)
gua_pred = guassian.predict(X_test)
gua_score = round(guassian.score(X_train, y_train)*100, 2)


# #### *6. Support Vector Machine*

# In[ ]:


from sklearn.svm import SVC
svm = SVC()
svm.fit(X_train, y_train)
svm_predict = svm.predict(X_test)
svm_score = round(svm.score(X_train, y_train)*100, 2)


# In[ ]:


# We will now check scores of all the models by comparing against each other.


# In[ ]:


models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 
              'Decision Tree'],
    'Score': [svm_score, knn_score, log_score, 
              rf_score, gua_score, dt_score]})
models.sort_values(by='Score', ascending=False)


# In[ ]:


#We will now save a copy of submission file which will have just PassengerId and Survived as columns and exactly 418 rows exluding the header.


# In[ ]:


pid = test_df['PassengerId'].tolist()
pid
dict = {'PassengerId': pid, 'Survived': dt_pred}  
     
submission = pd.DataFrame(dict) 
submission
submission.to_csv('submission.csv') 


# ## We can see that the DECISION TREE Classifier gives the "Best Accuracy" with a score of 98.09.

# #### I hope this report is easy to understand for the beginners in this line. I will start posting more Notebooks in public on Kaggle to help those in need. And you can contact me for any queries/collaboration/discussion. My contact details are available at the Top. 
# 
# Welcome to Data Science & Machine Learning Club! All the best for future endeavours! :)
