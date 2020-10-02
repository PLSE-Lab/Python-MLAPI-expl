#!/usr/bin/env python
# coding: utf-8

# ### Problem Statement:
# Titanic is a widely known disaster. However, it can be useful to learn from disasters. In this project, we acquired the dataset of the passengers of Titanic. It contains 10 features about approximately 1300 passengers; their name, ticket class, sex, age, ticket number, number of children and parents, number of siblings, ticket fare, cabin number, port of embarkation, and whether or not they survived.<br>
# We wanted to predict whether or not a specific passenger has survived the tragedy. Therefore, We trained some classification models to predict that.

# In[ ]:


import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

sns.set(font_scale=1.5)
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Data Cleaning and EDA

# In[ ]:


train=pd.read_csv('/kaggle/input/titanic/train.csv')


# In[ ]:


test=pd.read_csv('/kaggle/input/titanic/test.csv')


# #### print the head of train and test data

# In[ ]:


train.head()


# In[ ]:


test.head()


# #### print the shape of train and test data

# In[ ]:


train.shape


# In[ ]:


test.shape


# #### Display the columns with null values and number of nulls

# In[ ]:


train.isnull().sum()


# In[ ]:


test.isnull().sum()


# #### print summary statistics

# In[ ]:


train.describe()


# In[ ]:


test.describe()


# #### Correlation heatmap to see how features are correlated with SalePrice

# In[ ]:


corr=train.corr()


# In[ ]:


sns.heatmap(corr, annot=True)


# ### Filling null values

# #### For port of embarkation: Replacing missing values with S because it is the most repetitve value

# In[ ]:


train['Embarked'].value_counts()


# In[ ]:


train['Embarked']=train['Embarked'].fillna('S')


# #### For cabin: Grouping cabins by Pclass, then selecting one cabin for each class randomly to fill the missing values 

# In[ ]:


train['Cabin']= train['Cabin'].fillna('Unknown')


# In[ ]:


train['Cabin']= train['Cabin'].apply(lambda x: x[0])


# In[ ]:


train['Cabin'].head()


# In[ ]:


train.groupby('Pclass')['Cabin'].value_counts()


# In[ ]:


train['Cabin'] = np.where((train.Pclass==1) & (train.Cabin=='U'),'T',
                                            np.where((train.Pclass==2) & (train.Cabin=='U'),'D',
                                                                        np.where((train.Pclass==3) & (train.Cabin=='U'),'E',train.Cabin
                                                                                                    )))


# #### For sex: Replacing male and female with 0 and 1

# In[ ]:


train['Sex']=train['Sex'].apply(lambda x: 1 if x=='female' else 0 )


# #### For age, we created an array of 6 values of age. They are the median of age for each gender of each Pclass

# In[ ]:


guess_ages = np.zeros((2,3))
guess_ages


# In[ ]:



for i in range(0, 2):
    for j in range(0, 3):
        guess_df = train[(train['Sex'] == i) & (train['Pclass'] == j+1)]['Age'].dropna()

            # age_mean = guess_df.mean()
            # age_std = guess_df.std()
            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

        age_guess = guess_df.median()

            # Convert random age float to nearest .5 age
        guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
            
for i in range(0, 2):
    for j in range(0, 3):
        train.loc[ (train.Age.isnull()) & (train.Sex == i) & (train.Pclass == j+1),'Age'] = guess_ages[i,j]

train['Age'] = train['Age'].astype(int)


# #### Rechecking remaining missing values if any

# In[ ]:


train.isnull().sum()


# #### Filling the test data in the same way that we did with the train data 

# In[ ]:


test['Cabin']= test['Cabin'].fillna('Unknown')
test['Cabin']= test['Cabin'].apply(lambda x: x[0])    


# In[ ]:


test.groupby('Pclass')['Cabin'].value_counts()


# In[ ]:


test['Cabin'] = np.where((test.Pclass==1) & (test.Cabin=='U'),'T',
                                            np.where((test.Pclass==2) & (test.Cabin=='U'),'D',
                                                                        np.where((test.Pclass==3) & (test.Cabin=='U'),'E',test.Cabin
                                                                                                    )))


# In[ ]:


test.groupby('Pclass')['Cabin'].value_counts()


# In[ ]:


guess_ages = np.zeros((2,3))
guess_ages


# In[ ]:


test['Sex']=test['Sex'].apply(lambda x: 1 if x=='female' else 0 )


# In[ ]:


for i in range(0, 2):
    for j in range(0, 3):
        guess_df = test[(test['Sex'] == i) & (test['Pclass'] == j+1)]['Age'].dropna()

            # age_mean = guess_df.mean()
            # age_std = guess_df.std()
            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

        age_guess = guess_df.median()

            # Convert random age float to nearest .5 age
        guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
            
for i in range(0, 2):
    for j in range(0, 3):
        test.loc[ (test.Age.isnull()) & (test.Sex == i) & (test.Pclass == j+1),'Age'] = guess_ages[i,j]

test['Age'] = test['Age'].astype(int)


# In[ ]:


test['Fare']=test['Fare'].fillna('unknown')


# In[ ]:


test.loc[test['Fare'] == 'unknown']


# In[ ]:


train.groupby('Pclass')['Fare'].mean()


# In[ ]:


test['Fare'].replace('unknown', 14, inplace=True)


# In[ ]:


test.isnull().sum()


# ### Data Visualization

# #### Distributions of the features in the dataset

# In[ ]:


def subplot_histograms(dataframe, list_of_columns, list_of_titles, list_of_xlabels):
    nrows = int(np.ceil(len(list_of_columns)/2)) # Makes sure you have enough rows
    fig, ax = plt.subplots(nrows=nrows, ncols=2, figsize=(14,24)) # You'll want to specify your figsize
    ax = ax.ravel() # Ravel turns a matrix into a vector, which is easier to iterate
    for i, column in enumerate(list_of_columns): # Gives us an index value to get into all our lists
        ax[i].hist(dataframe[column], color='skyblue') # feel free to add more settings
        ax[i].set_title(list_of_titles[i])
        ax[i].set_xlabel(list_of_xlabels[i])
       


# In[ ]:


cols=['Pclass','Sex','Age','SibSp','Parch','Fare','Survived']
tit=['Pclass','Sex','Age','Siblings','Parents and children','Fare','Survived']
xs=['Pclass','Sex','Age','SibSp','Parch','Fare','Survived']
subplot_histograms(train,cols,tit,xs)


# #### Distributions of the features in the dataset for the survivors only

# In[ ]:


surv=pd.DataFrame(train.loc[train['Survived'] == 1])


# In[ ]:




cols=['Pclass','Sex','Age','SibSp','Parch','Fare']
tit=['Pclass','Sex','Age','Siblings','Parents and children','Fare']
xs=['Pclass','Sex','Age','SibSp','Parch','Fare']
subplot_histograms(surv,cols,tit,xs)


# In[ ]:


sns.heatmap(train.corr(),annot=True)


# In[ ]:


fig, ax = plt.subplots()
ax.scatter(x = train['Survived'], y = train['Age'])
plt.ylabel('Age', fontsize=13)
plt.xlabel('Survived', fontsize=13)
plt.title('')
plt.show()


# In[ ]:


#Deleting outliers
train = train.drop(train[(train['Survived']== 1) & (train['Age']>79)].index)

#Check the graphic again
fig, ax = plt.subplots()
ax.scatter(x = train['Survived'], y = train['Age'])
plt.ylabel('Age', fontsize=13)
plt.xlabel('Survived', fontsize=13)
plt.title('')
plt.show()


# #### Display pie plot for percentage of Male Vs. Female passengers

# In[ ]:


def percentage(part):
    whole= train['Sex'].value_counts().sum()
    percentage= (part/whole)
    return percentage

percentage= train['Sex'].value_counts().apply(lambda x : percentage(x))
print (percentage)



labels = 'Male','Female' 
sizes = [65, 35]
c=['#b39ab0','#e6e6fa']
fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%',colors=c,
        shadow=True, startangle=90)
ax1.axis('equal')  
plt.title('Percentage of Male Vs. Female passengers')
plt.show()


# #### Display pie plot for percentage of Male Vs. Female  survivors

# In[ ]:


def percentage(part):
    whole= surv['Sex'].value_counts().sum()
    percentage= (part/whole)
    return percentage

percentage= surv['Sex'].value_counts().apply(lambda x : percentage(x))
print (percentage)



labels = 'Female', 'Male'
sizes = [68, 32]
c=['#b39ab0','#e6e6fa']
fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%',colors=c,
        shadow=True, startangle=90)
ax1.axis('equal')  
plt.title('Percentage of Male Vs. Female survivors')
plt.show()


# #### Display count plot for range of ages in each Pclass

# In[ ]:


cat=train['Age']
cat=cat.apply(lambda x:'<10' if x<11 else '11-18' if x<18  else '19-35' if x<36  else '36-60' if x<61  else '61-80')

c= {'male': '#8c9fff' , 'female': '#ffb68c'}
sns.countplot(x='Pclass', data = train , hue=cat, palette= 'Set2' )
plt.title('Range of ages in each Pclass')
plt.legend(bbox_to_anchor=(1,1), loc=2)


# #### Display count plot for range of ages of survivors in each Pclass

# In[ ]:


cat=surv['Age']
cat=cat.apply(lambda x:'<10' if x<11 else '11-18' if x<18  else '19-35' if x<36  else '36-60' if x<61  else '61-80')

c= {'male': '#8c9fff' , 'female': '#ffb68c'}
sns.countplot(x='Pclass', data = surv , hue=cat, palette= 'Set2' )
plt.title('Range of ages of survivors in each Pclass')
plt.legend(bbox_to_anchor=(1,1), loc=2)


# #### Display count plot for range of ages of ages for survivors

# In[ ]:


c= {'male': '#8c9fff' , 'female': '#ffb68c'}
sns.countplot(x='Survived', data = train , hue=cat, palette= 'Set2' )
plt.title('Range of ages for survivors')
plt.legend(bbox_to_anchor=(1,1), loc=2)


# #### Display a plot for range of survivors for each Pclass

# In[ ]:


df_plot=train.groupby(['Pclass', 'Survived']).size().reset_index().pivot(columns='Pclass', index='Survived', values=0)
df_plot.plot(kind='bar', stacked=True,colormap='Set2')
plt.title('Number of survivors for each Pclass')
plt.legend(bbox_to_anchor=(1,1), loc=2)


# In[ ]:



sns.pairplot(train)


# ## Preprocessing and Modeling

# #### Split data to train and test

# In[ ]:


features_drop = ['PassengerId','Name', 'Ticket', 'Survived','Embarked','Cabin']
selected_features=[c for c in train if c not in features_drop]
selected_features


# In[ ]:


X_train = train[selected_features]
y_train = train['Survived']
X_test= test[selected_features]


# #### We imported StandardScaler and applied it to both X_train and X_test.

# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(scaler.fit_transform(X_test), columns=X_test.columns)


# In[ ]:


# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler(feature_range=(0, 1))

# x_train_scaled = scaler.fit_transform(X_train)
# X_train = pd.DataFrame(x_train_scaled)

# x_test_scaled = scaler.fit_transform(X_test)
# X_test = pd.DataFrame(x_test_scaled)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score


# #### Calculating the baseline

# In[ ]:


baseline=y_train.value_counts(normalize=True)
baseline[0]


# #### In this model, we created a K neighbors classifier and applied the model on the entire features in X_train.

# In[ ]:


knn = KNeighborsClassifier(n_neighbors=5) 
knn.fit(X_train, y_train)
m_score= knn.score(X_train, y_train)
print('Model score: ', m_score)


# In[ ]:


predictions = pd.DataFrame(knn.predict(X_test))
predictions['PassengerId']=[i for i in range(892, 1310)]
predictions.rename(columns={0:'Survived'},inplace=True)
predictions.set_index('PassengerId',inplace=True)


# In[ ]:


predictions.to_csv('predictions_knn.csv')


# ![model1.png](attachment:model1.png)

# #### In this model, we created a K neighbors classifier after performing a grid search. We applied the model on the entire features in X_train.

# In[ ]:


from sklearn.model_selection import GridSearchCV
knn_params = {
    'n_neighbors': range(1,100),
    'weights':['uniform','distance'],
    'metric':['euclidean','manhattan']}
print('Initialized parameters for Grid Search')
print(knn_params)


# In[ ]:


knn_gridsearch = GridSearchCV(KNeighborsClassifier(), 
                              knn_params, 
                              n_jobs=1, cv=5) # try verbose!


knn_gridsearch.fit(X_train, y_train)


# In[ ]:


best_knn = knn_gridsearch.best_estimator_
best_knn.score(X_train, y_train)
predictions = pd.DataFrame(best_knn.predict(X_test))


# In[ ]:


predictions['PassengerId']=[i for i in range(892, 1310)]
predictions.rename(columns={0:'Survived'},inplace=True)
predictions.set_index('PassengerId',inplace=True)
predictions.to_csv('predictions_gs_knn_afteroutliers.csv')


# ![model2.png](attachment:model2.png)

# #### In this model, we created a decision tree after performing a grid search. We applied the model on the entire features in X_train.

# In[ ]:


from sklearn.tree import DecisionTreeClassifier


# In[ ]:


dtc_params = {
    'max_depth': range(1,20),
    'max_features': [None, 'log2', 'sqrt'],
    'min_samples_split': range(5,30),
    'max_leaf_nodes': [None],
    'min_samples_leaf': range(1,10)
}

from sklearn.model_selection import GridSearchCV
# set the gridsearch
dtc_gs = GridSearchCV(DecisionTreeClassifier(), dtc_params,  n_jobs=-1, cv=5)


# In[ ]:


dtc_gs.fit(X_train, y_train)


# In[ ]:


predictions = dtc_gs.best_estimator_.predict(X_test)
# predictions.to_csv('predictions_dt_gs.csv')


# In[ ]:


predictions = pd.DataFrame(predictions)
predictions['PassengerId']=[i for i in range(892, 1310)]
predictions.rename(columns={0:'Survived'},inplace=True)
predictions.set_index('PassengerId',inplace=True)
predictions.to_csv('predictions_dt_gs.csv')


# ![model3.png](attachment:model3.png)

# #### In this model, we created a Random Forest after performing a grid search. We applied the model on the entire features in X_train.

# In[ ]:


from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

rf_params = {
      'n_estimators': range(1,100),
#     'max_features':[2, 3, 5, 7, 8],
      'max_depth': range(1,20),
     'criterion':['gini', 'entropy'],
}


# In[ ]:


rf_g = RandomForestClassifier() 


# In[ ]:


gs = GridSearchCV(rf_g, param_grid=rf_params, cv=5, verbose = 1)#, refit=False) 


# In[ ]:


gs=gs.fit(X_train, y_train)


# In[ ]:


predictions = gs.best_estimator_.predict(X_test)


# In[ ]:


predictions = pd.DataFrame(predictions)
predictions['PassengerId']=[i for i in range(892, 1310)]
predictions.rename(columns={0:'Survived'},inplace=True)
predictions.set_index('PassengerId',inplace=True)
predictions.to_csv('predictions_RF_gs.csv')


# ![model4.png](attachment:model4.png)

# #### In this model, we created a Extra trees classifier after performing a grid search. We applied the model on the entire features in X_train.

# In[ ]:


rf_params = {
      'n_estimators': range(1,100),
#     'max_features':[2, 3, 5, 7, 8],
      'max_depth': range(1,20),
     'criterion':['gini', 'entropy'],
}


# In[ ]:


et_g = ExtraTreesClassifier()


# In[ ]:


gs_et = GridSearchCV(rf_g, param_grid=rf_params, cv=5, verbose = 1)#, refit=False) 


# In[ ]:


gs_et =gs_et.fit(X_train, y_train)


# In[ ]:


predictions = gs_et.best_estimator_.predict(X_test)


# In[ ]:


predictions = pd.DataFrame(predictions)
predictions['PassengerId']=[i for i in range(892, 1310)]
predictions.rename(columns={0:'Survived'},inplace=True)
predictions.set_index('PassengerId',inplace=True)
predictions.to_csv('predictions_ET_gs.csv')


# ![model5.png](attachment:model5.png)

# #### In this model, we created a Logistic regression with lasso after performing a grid search. We applied the model on the entire features in X_train.

# In[ ]:


from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
model = LogisticRegression()
params = {'C':np.logspace(-5,5,15),
          'penalty':['l1'],
          'fit_intercept':[True,False]}
gs = GridSearchCV(estimator=model,
                  param_grid=params,
                  cv=5,
                  scoring='accuracy',
                  return_train_score=True)
gs.fit(X_train,y_train)
print(gs.best_score_)
print(gs.score(X_train,y_train))


# In[ ]:


predictions= gs.predict(X_test)
predictions = pd.DataFrame(predictions)
predictions['PassengerId']=[i for i in range(892, 1310)]
predictions.rename(columns={0:'Survived'},inplace=True)
predictions.set_index('PassengerId',inplace=True)
predictions.to_csv('predictions_log_l1.csv')


# ![Screen%20Shot%202019-11-05%20at%2011.18.35%20AM.png](attachment:Screen%20Shot%202019-11-05%20at%2011.18.35%20AM.png)

# #### In this model, we created a Logistic regression with ridge after performing a grid search. We applied the model on the entire features in X_train.

# In[ ]:


model = LogisticRegression()
params = {'C':np.logspace(-5,5,15),
          'penalty':['l2'], #Ridge
          'fit_intercept':[True,False]}
gs = GridSearchCV(estimator=model,
                  param_grid=params,
                  cv=5,
                  scoring='accuracy',
                  return_train_score=True)
gs.fit(X_train,y_train)
print(gs.best_score_)
print(gs.score(X_train,y_train))


# In[ ]:


predictions= gs.predict(X_test)
predictions = pd.DataFrame(predictions)
predictions['PassengerId']=[i for i in range(892, 1310)]
predictions.rename(columns={0:'Survived'},inplace=True)
predictions.set_index('PassengerId',inplace=True)
predictions.to_csv('predictions_log_l2.csv')


# ![Screen%20Shot%202019-11-05%20at%2011.18.35%20AM.png](attachment:Screen%20Shot%202019-11-05%20at%2011.18.35%20AM.png)

# In[ ]:


X_train_s = train[['Sex','Pclass']]
y_train_s = train['Survived']
X_test_s= test[['Sex','Pclass']]


# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_s = pd.DataFrame(scaler.fit_transform(X_train_s), columns=X_train_s.columns)
X_test_s = pd.DataFrame(scaler.fit_transform(X_test_s), columns=X_test_s.columns)


# In[ ]:


rf_params = {
      'n_estimators': range(1,100),
#     'max_features':[2, 3, 5, 7, 8],
      'max_depth': range(1,20),
     'criterion':['gini', 'entropy'],
}
rf_g = RandomForestClassifier()
gs = GridSearchCV(rf_g, param_grid=rf_params, cv=5, verbose = 1)#, refit=False) 
gs=gs.fit(X_train_s, y_train_s)
predictions = gs.best_estimator_.predict(X_test_s)


# In[ ]:


predictions = pd.DataFrame(predictions)
predictions['PassengerId']=[i for i in range(892, 1310)]
predictions.rename(columns={0:'Survived'},inplace=True)
predictions.set_index('PassengerId',inplace=True)
predictions.to_csv('predictions_RF_gs_s.csv')


# ![Screen%20Shot%202019-11-05%20at%2011.19.40%20AM.png](attachment:Screen%20Shot%202019-11-05%20at%2011.19.40%20AM.png)

# In[ ]:


from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

X_train_s = train[['Sex','Pclass','Age']]
y_train_s = train['Survived']
X_test_s= test[['Sex','Pclass','Age']]

scaler = StandardScaler()
X_train_s = pd.DataFrame(scaler.fit_transform(X_train_s), columns=X_train_s.columns)
X_test_s = pd.DataFrame(scaler.fit_transform(X_test_s), columns=X_test_s.columns)

rf_params = {
      'n_estimators': range(1,50),
#     'max_features':[2, 3, 5, 7, 8],
      'max_depth': range(1,20),
     'criterion':['gini', 'entropy'],
}
rf_g = RandomForestClassifier()
gs = GridSearchCV(rf_g, param_grid=rf_params, cv=5, verbose = 1)#, refit=False) 
gs=gs.fit(X_train_s, y_train_s)
predictions = gs.best_estimator_.predict(X_test_s)

predictions = pd.DataFrame(predictions)
predictions['PassengerId']=[i for i in range(892, 1310)]
predictions.rename(columns={0:'Survived'},inplace=True)
predictions.set_index('PassengerId',inplace=True)
predictions.to_csv('predictions_RF_gs_s2.csv')


# ![model9.png](attachment:model9.png)

# ### Conclusion
# In this project, we created classification models; LogisticRegression, ExtraTreesClassifier, KNeighborsClassifier, RandomForestClassifier and DecisionTreeClassifier to predict whether or not a Titanic passenger has survived. We tested all models by submitting them individually in Kaggle. The best score was for RandomForestClassifier and DecisionTreeClassifier, in which we got a score of 0.76555.

# In[ ]:




