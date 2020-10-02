#!/usr/bin/env python
# coding: utf-8

# # The goals of this notebook :

# * **[Part 1: Exploratory Data Analysis](#Part-1:-Exploratory-Data-Analysis)**  
# * **[Part 2: Machine Learning](#Part-2:-Machine-Learning)**
# * **[Part 3: Model Selection And Boosting](#Part-3:-Model-Selection-And-Boosting)**

# Start by importing the necerrary libraries

# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os

DATA_DIR='../input'
print(os.listdir(DATA_DIR))


# In[5]:


# Dummy variables to hold dataset file names on my local machine
TRAIN_CSV_FILE = "../input/train.csv"
TEST_CSV_FILE = "../input/test.csv"
df_train = pd.read_csv(TRAIN_CSV_FILE)
df_test = pd.read_csv(TEST_CSV_FILE)


# ## Part 1: Exploratory Data Analysis

# In[6]:


df_train.head()


# In[7]:


df_train.info()


# In[8]:


df_test.head()


# In[9]:


df_test.info()


# Both datasets have missing values for Age and Cabin

# In[10]:


df_train.isna().sum()


# In[11]:


df_test.isna().sum()


# Do some basic statistics

# In[12]:


df_train.describe()


# In[13]:


df_test.describe()


# How many survived?

# In[14]:


sns.countplot(data=df_train, x='Survived')


# **Who has more survivors? Males or females?**

# In[15]:


df_train.groupby(['Survived', 'Sex'])['Survived'].count()


# In[16]:


sns.catplot(x='Sex', col='Survived', kind='count', data=df_train)


# Show Gender survival percentages

# In[17]:


women_survived = df_train[df_train.Sex == 'female'].Survived.sum()
men_survived = df_train[df_train.Sex == 'male'].Survived.sum()
total_female_survived = df_train[df_train.Sex == 'female'].Survived.count()
total_male_survived = df_train[df_train.Sex == 'male'].Survived.count()
print(women_survived,men_survived,total_female_survived, total_male_survived, sep=' ')
print('Women Survived --> {:<7.3f}%'.format(women_survived/total_female_survived * 100))
print('Men Survived --> {:<7.3f}%'.format(men_survived/total_male_survived * 100))


# Do some pie charts for gender survival

# In[18]:


f,ax=plt.subplots(1,2,figsize=(16,7))
df_train['Survived'][df_train['Sex']=='male'].value_counts().plot.pie(explode=[0,0.2],autopct='%1.1f%%',ax=ax[0],shadow=True)
df_train['Survived'][df_train['Sex']=='female'].value_counts().plot.pie(explode=[0,0.2],autopct='%1.1f%%',ax=ax[1],shadow=True)
ax[0].set_title('Survived (male)')
ax[1].set_title('Survived (female)')

plt.show()


# **Let's see if survivors depends of passenger class**

# In[19]:


df_train.groupby(['Survived', 'Pclass'])['Survived'].count()


# In[20]:


sns.catplot(x='Pclass', col='Survived', kind='count', data=df_train)


# The majority of those who did not survive were from the 3rd class, while the survivors are almost the same for all classes.

# In[21]:


pd.crosstab(df_train.Pclass, df_train.Survived, margins=True).style.background_gradient(cmap='autumn_r')


# **Display some percentages again**

# In[22]:


pd_class_p = pd.crosstab(df_train.Pclass, df_train.Survived, margins=True,normalize='index')
pd_class_p


# In[23]:


pd_class_p[1][3]


# In[24]:


print('Survivals per class percentages :')
for i in range(3):
    print('Class {} --> {:<7.3f}%'.format(i+1,pd_class_p[1][i+1]*100))


# In[25]:


sns.catplot('Pclass','Survived', kind='point', data=df_train)


# **Passenger Class and Sex :**

# In[26]:


pd.crosstab([df_train.Sex, df_train.Survived], df_train.Pclass, margins=True).style.background_gradient(cmap='autumn_r')


# In[27]:


cl_sex_sur_per = pd.crosstab([df_train.Sex, df_train.Survived], df_train.Pclass, margins=True)
cl_sex_sur_per


# In[28]:


sns.catplot('Pclass','Survived',hue='Sex', kind='point', data=df_train);


# It seems like almost women in Pclass 1 and Pclass 2 survived 
# and almost men in Pclass 2 and Pclass 3 not survived.

# **Lets examine the relationship between Survived and Embarked**

# In[29]:


sns.catplot(x='Survived', col='Embarked', kind='count', data=df_train);


# In[30]:


sns.catplot('Embarked','Survived', kind='point', data=df_train);


# **Embarked and Sex**

# In[31]:


sns.catplot(x='Sex',y='Survived', col='Embarked', kind='bar', data=df_train)


# In[32]:


sns.catplot('Embarked','Survived', hue= 'Sex', kind='point', data=df_train);


# **Embarked, Pclass and Sex :**

# In[33]:


sns.catplot('Embarked','Survived', col='Pclass', hue= 'Sex', kind='point', data=df_train)


# * All women of Pclass 2 that embarked in C and Q survived, also nearly all women of Pclass 1 survived.
# 
# * All men of Pclass 1 and 2 embarked in Q have not survived, survival rate for men in Pclass 2 and 3 is always below 0.2
# * For the remaining men in Pclass 1 that embarked in S and Q, survival rate is approx. 0.4

# In[34]:


pd.crosstab([df_train.Survived], [df_train.Sex, df_train.Pclass, df_train.Embarked], margins=True)


# Let's look at the survivors based on their age and we will slowly 
# add the other factors we have studied so far.
# 
# First we 'll create 8 bins with age.

# In[35]:


for df in [df_train, df_test]:
    df['Age_bin']=np.nan
    for i in range(8,0,-1):
        df.loc[ df['Age'] <= i*10, 'Age_bin'] = i


# In[36]:


df_train[['Age', 'Age_bin']].head(20)


# In[37]:


sns.catplot(x='Age_bin',y='Survived',  kind='bar', data=df_train)


# In[38]:


sns.catplot(x='Age_bin',y='Survived',col='Sex',  kind='bar', data=df_train)


# In[39]:


sns.catplot('Age_bin','Survived',hue='Sex',kind='point',data=df_train)


# In[40]:


sns.catplot('Age_bin','Survived', col='Pclass', row = 'Sex', kind='point', data=df_train);


# In[41]:


pd.crosstab([df_train.Sex, df_train.Survived], [df_train.Age_bin, df_train.Pclass], margins=True).style.background_gradient(cmap='autumn_r')


# **Conclusions**
# 
# * All males in Age_bin 1 (age <= 10 ) in Pclass 1 and Pclass 2 survived.
# * All females in Pclass 3 with 50 <= age < 60 died.

# **SibSp and Parch**

# In[42]:


sns.catplot('SibSp','Survived', col='Pclass' , row = 'Sex', kind='point', data=df_train)


# In[43]:


pd.crosstab([df_train.Sex, df_train.Survived], [df_train.SibSp, df_train.Pclass], margins=True).style.background_gradient(cmap='autumn_r')


# For males, no survival, rate above 0.5 for any values of SibSp. For females, passengers with SibSp = 3 and Pclass = 3 died, also all females with SibSp > 4 died. For females with SibSp = 1 and Pclass = 3 survival rate is below 0.5

# In[44]:


sns.catplot('Parch','Survived', col='Pclass' , row = 'Sex', kind='point', data=df_train)


# In[45]:


pd.crosstab([df_train.Sex, df_train.Survived], [df_train.Parch, df_train.Pclass], margins=True).style.background_gradient(cmap='autumn_r')


# Very similar to SibSp, but different values.
# * For females with Parch = 2 and Pclass = 3 survival rate is below 0.5  
# * All females with Parch = 4 and Pclass = 3 died.
# * All females with Parch > 4 died.
# * For females with Parch = 1 and Pclass = 3 survival rate is below 0.5
# * For males,all survival rates below 0.5 for any values of Parch, except for Parch = 2 and Pclass = 1.

# **Continue with Fare**
# 
# Let see it's distribution

# In[46]:


sns.distplot(df_train['Fare'])


# Create 12 bin of Fares

# In[47]:


for df in [df_train, df_test]:
    df['Fare_bin']=np.nan
    for i in range(12,0,-1):
        df.loc[ df['Fare'] <= i*50, 'Fare_bin'] = i


# In[48]:


df_train['Fare_bin'].head(10)


# In[49]:


sns.catplot(x='Fare_bin',y='Survived',col='Sex',  kind='bar', data=df_train)


# In[50]:


sns.catplot('Fare_bin','Survived', col='Pclass' , row = 'Sex', kind='point', data=df_train)


# * All males in Pclass 1 and Fare_bin = 11 survived.
# * For males in Pclass 2 survival rates < 20%
# * For males in Pclass 3 survival rates < 50%
# * Females in Pclass 1 and Fare_bin = 2.0, 3.0, 5.0, 6.0, 11.0 survived
# * Females in Pclass 2 and Fare_bin = 2.0 survived.
# * Females in Pclass 2 and 3 regardless of fares survival rates <= 0.5

# In[51]:


pd.crosstab([df_train.Sex, df_train.Survived], [df_train.Fare_bin, df_train.Pclass], margins=True).style.background_gradient(cmap='autumn_r')


# ## Part 2: Machine Learning

# **Data Preparation**
# 
# 1. Load the original data from csv
# 2. Encode categorical data
# 3. Drop columns that we don't need
# 4. Get the independent and dependent variable
# 5. Take care of missing data
# 6. Split the dataset into the trainning and test set

# In[52]:


df_train_ml = pd.read_csv(TRAIN_CSV_FILE)
df_test_ml = pd.read_csv(TEST_CSV_FILE)


# In[53]:


df_train_ml.head()


# In[54]:


df_test_ml.head()


# In[55]:


# Encoding categorical data
df_train_ml = pd.get_dummies(data=df_train_ml, columns=['Sex', 'Embarked'], drop_first=True)
df_train_ml.drop(['Name','Ticket', 'Cabin'],axis=1, inplace=True) 

passenger_id = df_test_ml['PassengerId']
df_test_ml = pd.get_dummies(data=df_test_ml, columns=['Sex', 'Embarked'], drop_first=True)
df_test_ml.drop(['Name','Ticket', 'Cabin'],axis=1, inplace=True) 


# In[56]:


df_train_ml.head()


# In[57]:


df_test_ml.head()


# In[58]:


X = df_train_ml.iloc[:, 2:].values
y = df_train_ml.iloc[:, 1].values


# In[59]:


# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:2])
X[:, 1:2] = imputer.transform(X[:, 1:2])


# In[60]:


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 101)


# In[61]:


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# **All data for transmision**

# In[62]:


X_train_all = X
y_train_all = y
X_test_all = df_test_ml.iloc[:,1:].values


# In[63]:


# Take care of NaNs in all data
imputer = imputer.fit(X_test_all[:, [1,4]])
X_test_all[:, [1,4]] = imputer.transform(X_test_all[:, [1,4]])


# **Feature scaling for all data**

# In[64]:


sc_all = StandardScaler()
X_train_all = sc_all.fit_transform(X_train_all)
X_test_all = sc_all.transform(X_test_all)


# Use this utility function to show metrics

# In[65]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, accuracy_score

def show_metrics(y_test, y_pred,msg='Summary'):
    cm = confusion_matrix(y_test,y_pred)
    cm = sns.heatmap(cm, annot=True, fmt='d')
    print(msg)
    print(classification_report(y_test, y_pred))
    print('Accuracy : {}'.format(accuracy_score(y_test, y_pred)))


# **Logistic Regression**

# In[66]:


from sklearn.linear_model import LogisticRegression
lg_classifier = LogisticRegression(random_state = 101)
lg_classifier.fit(X_train, y_train)


# In[67]:


# Predicting the Test set results
lg_y_pred = lg_classifier.predict(X_test)


# In[68]:


# Making the Confusion Matrix
cm = confusion_matrix(y_test, lg_y_pred)
sns.heatmap(cm, annot=True, fmt='d')


# In[69]:


# Print some metrics
print(classification_report(y_test, lg_y_pred))
print(accuracy_score(y_test, lg_y_pred))


# **Train again for all data and submit**

# In[70]:


lg_classifier.fit(X_train_all, y_train_all)
lg_y_pred_all = lg_classifier.predict(X_test_all)


# In[71]:


sub_logreg = pd.DataFrame()
sub_logreg['PassengerId'] = df_test['PassengerId']
sub_logreg['Survived'] = lg_y_pred_all
#sub_logmodel.to_csv('logmodel.csv',index=False)


# **K-Nearest Neighbors (K-NN)**

# In[72]:


# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
knn_classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
knn_classifier.fit(X_train, y_train)


# In[73]:


# Predicting the Test set results
knn_y_pred = knn_classifier.predict(X_test)


# In[74]:


#confusion matrix
knn_cm = confusion_matrix(y_test, knn_y_pred)
sns.heatmap(knn_cm, annot=True, fmt='d')


# In[75]:


print('K-NN Summary')
print(classification_report(y_test, knn_y_pred))
print(accuracy_score(y_test, knn_y_pred))


# Fit again for all data

# In[76]:


knn_classifier.fit(X_train_all, y_train_all)
knn_y_pred_all= knn_classifier.predict(X_test_all)


# In[77]:


sub_knn = pd.DataFrame()
sub_knn['PassengerId'] = df_test['PassengerId']
sub_knn['Survived'] = knn_y_pred_all
#sub_knn.to_csv('knn.csv',index=False)


# **Support Vector Machine (SVM)**

# In[78]:


# Fitting SVM to the Training set
from sklearn.svm import SVC
svm_classifier = SVC(kernel = 'linear', random_state = 101)
svm_classifier.fit(X_train, y_train)


# In[79]:


# Predicting the Test set results
svm_y_pred = svm_classifier.predict(X_test)


# In[80]:


#confusion matrix
svm_cm = confusion_matrix(y_test, svm_y_pred)
sns.heatmap(svm_cm, annot=True, fmt='d')
print('SVM Summary')
print(classification_report(y_test, svm_y_pred))
print(accuracy_score(y_test, svm_y_pred))


# Fit again for all data

# In[81]:


svm_classifier.fit(X_train_all, y_train_all)
svm_y_pred_all= svm_classifier.predict(X_test_all)


# In[82]:


sub_svm = pd.DataFrame()
sub_svm['PassengerId'] = df_test['PassengerId']
sub_svm['Survived'] = svm_y_pred_all
#sub_svm.to_csv('svm.csv',index=False)


# **Kernel SVM**

# In[83]:


# Fitting Kernel SVM to the Training set
ksvm_classifier = SVC(kernel = 'rbf', random_state = 101)
ksvm_classifier.fit(X_train, y_train)


# In[84]:


# Predicting the Test set results
ksvm_y_pred = ksvm_classifier.predict(X_test)


# In[85]:


#confusion matrix and metrics for kernel SVM
ksvm_cm = confusion_matrix(y_test, ksvm_y_pred)
sns.heatmap(ksvm_cm, annot=True, fmt='d')
print('Kernel SVM Summary')
print(classification_report(y_test, ksvm_y_pred))
print(accuracy_score(y_test, ksvm_y_pred))


# Fit again for all data

# In[86]:


ksvm_classifier.fit(X_train_all, y_train_all)
ksvm_y_pred_all= ksvm_classifier.predict(X_test_all)


# In[87]:


sub_ksvm = pd.DataFrame()
sub_ksvm['PassengerId'] = df_test['PassengerId']
sub_ksvm['Survived'] = ksvm_y_pred_all
#sub_svm.to_csv('svm.csv',index=False)


# **Naive Bayes**

# In[88]:


# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)


# In[89]:


# Predicting the Test set results
nb_y_pred = nb_classifier.predict(X_test)


# Confusion matrix and metrics

# In[90]:


show_metrics(y_test, nb_y_pred, msg='Naives Bayes Summary')


# **Random Forest Classification**

# In[91]:


# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
rf_classifier.fit(X_train, y_train)


# In[92]:


# Predicting the Test set results
rf_y_pred = rf_classifier.predict(X_test)


# Confusion matrix and metrics

# In[93]:


show_metrics(y_test, rf_y_pred, msg='Random Forest Summary')


# ## Part 3: Model Selection And Boosting

# In[94]:


# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = ksvm_classifier, X = X_train, y = y_train, cv = 10)
print(accuracies.mean())
print(accuracies.std())


# In[95]:


# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
              {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]
grid_search = GridSearchCV(estimator = ksvm_classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)


# In[96]:


grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
print('Best Accuracy : {}\n'.format(best_accuracy))
print('Best Parameters : {}\n'.format(best_parameters))


# 1. <p>Lets fit it again using the best parameters</p>

# In[97]:


# Fitting Kernel SVM to the Training set
ksvm_classifier = SVC(kernel = 'rbf', C=1, gamma=0.6,random_state = 101)
ksvm_classifier.fit(X_train, y_train)


# In[98]:


# Predicting the Test set results
ksvm_y_pred = ksvm_classifier.predict(X_test)


# Fitting again to all data and submit

# In[99]:


ksvm_classifier.fit(X_train_all, y_train_all)
ksvm_y_pred_all= ksvm_classifier.predict(X_test_all)


# In[100]:


sub_ksvm = pd.DataFrame()
sub_ksvm['PassengerId'] = df_test['PassengerId']
sub_ksvm['Survived'] = ksvm_y_pred_all
sub_svm.to_csv('svm.csv',index=False)


# In[ ]:




