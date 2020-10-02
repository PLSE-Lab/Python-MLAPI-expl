#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn as sk
import matplotlib as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# <font size="5">**Let's firstly get the data**</font>

# In[ ]:


columns = ['Id', 'age', 'workclass', 'final_weight', 'education', 'education_num', 'marital_status',
            'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week',
            'native_country', 'income']
#with this we put headers in the data, instaead of it's column 0 being the header

adult = pd.read_csv('/kaggle/input/adult-pmr3508/train_data.csv', names = columns, sep=',', engine='python',
                     na_values="?").drop(0, axis = 0).reset_index(drop = True)

adult.head()


# <font size="5">**Now we'll be analying the data**</font>

# In[ ]:


adult.describe(include = 'all')


# In[ ]:


num_columns = ['age', 'final_weight', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
cat_columns = ['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country', 'income']

for col in num_columns:
    adult[col] = pd.to_numeric(adult[col])
    
sns.set()
sns.pairplot(adult, vars = num_columns, hue = 'income', palette = 'husl', dropna = True)


# <font size="4">We see that the most import ones are age, final_weight, education_num, capital_gain and hours_per_week </font>

# <font size="4">Now we'll be analying the categorical data</font>

# In[ ]:


for col in cat_columns:
    adult[col] = pd.Categorical(adult[col])
    
num_columns = ['age', 'final_weight', 'education_num', 'hours_per_week', 'capital_gain'] #updating the list because we'll use these atributes

for numeric in num_columns:
    for col in cat_columns:
        sns.set()
        sns.catplot(x = col, y = numeric, height = 6, aspect = 1.75, data=adult);


# <font size="4">We can see that some variables have less interference in the object of study of this evaluation, so this means that those with high variance won't be usefull when traing the classifier, especially beacuse it isn't a huge amount of data, therefore it would interfere negatively in the traning process.</font>

# In[ ]:


cat_columns = ['marital_status', 'race', 'sex', 'native_country', 'income'] #updating the categorical atributes used
#workclass doesn't seen to change much the results in the income, as well as ocupation and relationship. 
#The education column isn;t useful because there's the education num column being used already


# <font size="5">**Now we'll be analying dealing with missing values and preparating the data**</font>

# <font size="4">Firstly the missing data </font>

# In[ ]:


#the numeric values will be dealt by analysing the frequency, standard deviation and speculating some inferences
for col in num_columns:
    print(col + ":")
    print(adult[col].describe())
    print()
#the categorical values will be dealt by inference and commom sense


# <font size="4">OK, we will put the mean for the atributes age and hours per week, since it's standard deviation isn't high.
# Now, for the final weight we will assume that people won't tell the census about it because either they are up the mean, and this depends biologically if it's a male or female, because of society imposed standards, so we'll check if it is a woman or a man and treat with this information.
# For the education num we can assume it's beacause they've quitted school early,so we'll assume the first quart as it's number.
# For the capital gain we will assume 40000.
# </font>

# <font size="4">For the categorical data: 
#     The marital status will be filled with the most frequent one (Married-civ-spouse), as seen in the .describe() method above;
#     The race will be filled with Black because this could be a source of racism;
#     The sex will be filled with man, since it's the most frequent and the ones that didn't answer this question wouldn't identify with eitheer;
#     The native country will be replaced with a country other than USA, because this could be a away of avoiding xenophobia.
# </font>

# In[ ]:


#just getting the data of women and men separately
men, women = [], []

for index, row in adult.iterrows():
    if row['sex'] == "Male":
        men.append(row)
    else:
        women.append(row)
women = pd.DataFrame(women, columns = columns)
men = pd.DataFrame(men, columns = columns) 

print(women.describe())
print(men.describe())


# In[ ]:


values = {'age': 39, 'education.num': 9, 'hours.per.week': 40, 'sex': 'Male', 'race': 'Black', 'marital_status': 'Married-civ-spouse',
          'native_country': 'Hungary', 'capital_gain': 40000} 
adult.fillna(values)
adult['final_weight'] = adult.apply(lambda row: ('2.411530e+05' if row['sex']== 'Male' else '2.283315e+05') if np.isnan(row['final_weight']) else row['final_weight'],
    axis=1)


# <font size="5">Now, we will create auxiliar columns to use, based on inference and aspects analysed earlier</font>

# In[ ]:


#the columns we will create are: 'graduated', 'maried' and 'american', using the columns 'education_num', 'marital_status' and 'native_country'
graduated = []
for i in range(len(adult)): 
    if adult['education_num'][i] <= 8:
        graduated.append(0)
    elif adult['education_num'][i] <= 12:
        graduated.append(1)
    else:
        graduated.append(2)
adult['graduated'] = graduated

maried = []
for i in range(len(adult)): 
    if adult['marital_status'][i] == 'Married-civ-spouse' or adult['marital_status'][i] == 'Maried-AF-spouse':
        maried.append(1)
    else:
        maried.append(0)
adult['maried'] = maried

american = []
for i in range(len(adult)): 
    if adult['native_country'][i] == 'United-States':
        american.append(1)
    else:
        american.append(0)
adult['american'] = american

adult.head()


# In[ ]:


#In order to use Male or Female (Sex) data, we will create a new column that is a numeric view of this categorical data
Male_or_Female = []
for i in range(len(adult)): 
    if adult["sex"][i] == "Male":
        Male_or_Female.append(1)
    else:
        Male_or_Female.append(0)
    
adult["Male_or_Female"] = Male_or_Female


# In[ ]:


#before moving on to the classifiers it is important to use the same methods on the testing data

adultTest = pd.read_csv('/kaggle/input/adult-pmr3508/test_data.csv', names = columns, sep=',', engine='python',
                     na_values="?").drop(0, axis = 0).reset_index(drop = True)

values = {'age': 39, 'education.num': 9, 'hours.per.week': 40, 'sex': 'Male', 'race': 'Black', 'marital_status': 'Married-civ-spouse',
          'native_country': 'Hungary'}
adultTest.fillna(values)
adultTest['final_weight'] = adultTest.apply(lambda row: ('2.411530e+05' if row['sex']== 'Male' else '2.283315e+05') if pd.isnull(row['final_weight']) else row['final_weight'],
    axis=1)

graduated = []
for i in range(len(adultTest)): 
    num = int(adultTest['education_num'][i])
    if num <= 8:
        graduated.append(0)
    elif num <= 12:
        graduated.append(1)
    else:
        graduated.append(2)
adultTest['graduated'] = graduated

maried = []
for i in range(len(adultTest)): 
    if adultTest['marital_status'][i] == 'Married-civ-spouse' or adultTest['marital_status'][i] == 'Maried-AF-spouse':
        maried.append(1)
    else:
        maried.append(0)
adultTest['maried'] = maried

american = []
for i in range(len(adultTest)): 
    if adultTest['native_country'][i] == 'United-States':
        american.append(1)
    else:
        american.append(0)
adultTest['american'] = american

Male_or_Female = []
for i in range(len(adultTest)): 
    if adultTest["sex"][i] == "Male":
        Male_or_Female.append(1)
    else:
        Male_or_Female.append(0)
    
adultTest["Male_or_Female"] = Male_or_Female


# In[ ]:


#now, just separating the X and Y
adult = adult[columns[:-1] + ['graduated', 'maried', 'american', 'Male_or_Female', 'income']] #reordering
dataset = adult.to_numpy()
adultX = dataset[:,:-1]
adultY = dataset[:,len(dataset[0])-1]
adultX = pd.DataFrame(adultX)
adultY = pd.DataFrame(adultY)

#and now for the testing data
adultTest = adultTest[columns[:-1] + ['graduated', 'maried', 'american', 'Male_or_Female']] #reordering
datasetTest = adultTest.to_numpy()
adultTestX = datasetTest[:,:]
adultTestX = pd.DataFrame(adultTestX)


# In[ ]:


#Now,transforming categorical data into numeric data
#Just remembering we're suign the columns: ['age', 'final_weight', 'education_num', 'hours_per_week', 'maried', 
#'race', 'Male_or_Female', 'American', 'income']
        
for index,row in adultX.iterrows():
    if row[9] == 'White':
        row[9] = 0
    elif row[9] == 'Asian-Pac-Islander':
        row[9] = 1
    elif row[9] == 'Amer-Idian-Eskimo':
        row[9] = 2
    elif row[9] == 'Black':
        row[9] = 3
    else:
        row[9] = 4
        
for index, row in adultTestX.iterrows():
    if row[9] == 'White':
        row[9] = 0
    elif row[9] == 'Asian-Pac-Islander':
        row[9] = 1
    elif row[9] == 'Amer-Idian-Eskimo':
        row[9] = 2
    elif row[9] == 'Black':
        row[9] = 3
    else:
        row[9] = 4
        
for index, row in adultY.iterrows():
    if row[0] == '>50K':
        row[0] = 1
    else:
        row[0] = 0
        
adultY = adultY.astype('int')


# In[ ]:


print(adultX.head())
adultX = adultX[[1, 3, 9, 11, 13, 15, 16, 17, 18]]
adultTestX = adultTestX[[1, 3, 9, 11, 13, 15, 16, 17, 18]]
#separating only the attributes that were treated

#the testing dataframe had and error that some numbers were considered strings
for index,row in adultTestX.iterrows():
    for i in [1,3,9,11, 13,15,16,17,18]:
        row[i] = int(row[i])


# <font size="5">Classifiers</font>

# <font size="4">SVM</font>

# In[ ]:


#for svm me will standardize the data
normalized_X = (adultX - adultX.mean())/adultX.std()
normalized_TestX = (adultTestX - adultTestX.mean())/adultTestX.std()

from sklearn.svm import SVC
SVM = SVC(C=1.0, kernel='rbf', gamma='scale', coef0=0.0, shrinking=True, probability=False,
                 tol=0.001, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', random_state=None)


# In[ ]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(SVM, normalized_X, adultY.to_numpy().ravel(), cv=10)
print("scores = {}". format(scores))
print("accuracy = {}".format(scores.mean()))


# In[ ]:


SVM.fit(normalized_X, adultY.to_numpy().ravel())


# In[ ]:


results_SVM = SVM.predict(normalized_TestX)


# In[ ]:


ResultsSVM = []
for i in range(len(results_SVM)):
    if results_SVM[i] == 1:
        ResultsSVM.append('>50K')
    else:
        ResultsSVM.append('<=50K')

id_index = pd.DataFrame({'Id' : list(range(len(ResultsSVM)))})
income = pd.DataFrame({'income' : ResultsSVM})
resultSVM = income
resultSVM


# <font size="4">Random Forest</font>

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

RandomForest = RandomForestClassifier(n_estimators=750, criterion='gini', max_depth=11, min_samples_split=2,
                                      min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None,
                                      min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None, 
                                      random_state=None, verbose=0, warm_start=False, class_weight=None)


# In[ ]:


scores = cross_val_score(RandomForest, normalized_X, adultY.to_numpy().ravel(), cv=10)
print("scores = {}". format(scores))
print("accuracy = {}".format(scores.mean()))


# In[ ]:


RandomForest.fit(normalized_X, adultY.to_numpy().ravel())


# In[ ]:


results_RandomForest = RandomForest.predict(normalized_TestX)
ResultsRandomForest = []
for i in range(len(results_RandomForest)):
    if results_RandomForest[i] == 1:
        ResultsRandomForest.append('>50K')
    else:
        ResultsRandomForest.append('<=50K')

id_index = pd.DataFrame({'Id' : list(range(len(ResultsRandomForest)))})
income = pd.DataFrame({'income' : ResultsRandomForest})
resultRandomForest = income
resultRandomForest


# <font size="4">Logistic Regression</font>

# In[ ]:


LogisticRegression = sk.linear_model.LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, 
                                   class_weight=None, random_state=None, solver='lbfgs', max_iter=100, multi_class='auto', 
                                   verbose=0, warm_start=False, n_jobs=None, l1_ratio=None)


# In[ ]:


scores = cross_val_score(LogisticRegression, normalized_X, adultY.to_numpy().ravel(), cv=10)
print("scores = {}". format(scores))
print("accuracy = {}".format(scores.mean()))


# In[ ]:


LogisticRegression.fit(normalized_X, adultY.to_numpy().ravel())


# In[ ]:


results_LogisticRegression = LogisticRegression.predict(normalized_TestX)
ResultsLogisticRegression = []
for i in range(len(results_LogisticRegression)):
    if results_LogisticRegression[i] == 1:
        ResultsLogisticRegression.append('>50K')
    else:
        ResultsLogisticRegression.append('<=50K')

id_index = pd.DataFrame({'Id' : list(range(len(ResultsLogisticRegression)))})
income = pd.DataFrame({'income' : ResultsLogisticRegression})
resultLogisticRegression = income
resultLogisticRegression


# <font size="5">Conclusion</font>

# <font size="3">So, the three classifiers chosen have been SVM, Random Forest and Logistic Regression because all of them are relatively easy to implement, and are also very commonly used by professionals. They're avalible in the sklearning packet and have lots of online tutorials that can help users to get started.</font>

# The SVM wasn't fast to run and it used a lot of memory, but it was easy to implement once all the data was normalized and treated. The accuracy was high but not too much. Overall this classifier is used when more complex system is needed, a lot of data is avaliable and requires no further explanation.

# The Random Forest got the highest accuracy but also took a lot of time to run, it wasn't so easy to implement but it wasn't rocket science. It has the best overall results because of it's high accuracy. It also used a lot of memory. The interpretability isn't its best part, because when a lot of tress are used together it makes things harder to understand.
# 

# The logistic regression is (almost) a must try, because it is the simplest and has huge accuracy when considering its use of memory and time to run. It didn't had a high accuracy, considering the others, but it had by far the lowest run time and the loest memory use. It's also very simple to interpret and be understood.
# 

# In[ ]:


resultSVM.to_csv("submissionSVM.csv", index = True, index_label = 'Id')
resultRandomForest.to_csv("submissionRandomForest.csv", index = True, index_label = 'Id')
resultLogisticRegression.to_csv("submissionLogisticRegression.csv", index = True, index_label = 'Id')


# <font size="5">Extra Activity: CaliforniaHousing</font>

# In[ ]:


train = pd.read_csv('../input/atividade-3/train.csv', na_values='?').reset_index(drop = True)
test = pd.read_csv('../input/atividade-3/test.csv', na_values='?').reset_index(drop = True)
train.head()


# <font size="4">Analysing the data</font>

# In[ ]:


train.describe()


# since it has few attributes we will use all of them to train our classifier. Also, it will be a 'simple' classifier for time purposes and for the lack of a big amount of data

# In[ ]:


columns = ['longitude', 'latitude', 'median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income']
Ycolumn = 'median_house_value'


# In[ ]:


trainX = train[columns]
trainY = train[[Ycolumn]]


# In[ ]:


#we will use a KNN and a Random Forest, testing the classifiers
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

RandomForestHousehold = RandomForestClassifier(n_estimators=400, criterion='gini', max_depth=9, min_samples_split=2,
                                      min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None,
                                      min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None, 
                                      random_state=None, verbose=0, warm_start=False, class_weight=None)

from sklearn.model_selection import cross_val_score
scores = cross_val_score(RandomForestHousehold, trainX , trainY.to_numpy().ravel() , cv=5)
print("scores = {}". format(scores))
print("accuracy = {}".format(scores.mean()))


# In[ ]:


KNN = sk.neighbors.KNeighborsClassifier(n_neighbors=15, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski',
                                       metric_params=None, n_jobs=None)

scores = cross_val_score(KNN, trainX , trainY.to_numpy().ravel(), cv=5)
print("scores = {}". format(scores))
print("accuracy = {}".format(scores.mean()))


# <font size="4">Conclusion</font>

# The tests run in the KNN and in the RandomForest show that the Random Forest got a higher accuracy (as in the adult database) but used more memory and took longer to run than the KNN. The KNN got good results despite its simplicity
