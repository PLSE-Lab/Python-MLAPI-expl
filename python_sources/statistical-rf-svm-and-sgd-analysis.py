# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import linear_model
import warnings

warnings.simplefilter("ignore")

pd.set_option('display.max_columns', 10)

#Importing data
data = pd.read_csv('../input/StudentsPerformance.csv')

#Updating columns names
data.rename(columns={'race/ethnicity':'groups', 'parental level of education':'parentEd', 'test preparation course':'prep', 'math score':'mathScore','reading score':'readingScore','writing score':'writingScore'}, inplace=True)
print(data.head(5))

#Displaying shape and description of data
print("Data Shape:", data.shape)
print("Data Description:\n", data.describe())
print("\n")

#Showing grouped scores w.r.t Parents Education, Race/Enthnicity, and Gender
print("Mean scores w.r.t parents qualification:")
dataPED = data.groupby(['parentEd']).mean().sort_values(by=['mathScore', 'readingScore', 'writingScore'],ascending=False)
print(dataPED)
print("\n")

print("Mean scores w.r.t race/enthnicity groups:")
dataGrps = data.groupby(['groups']).mean().sort_values(by=['mathScore', 'readingScore', 'writingScore'],ascending=False)
print(dataGrps)
print("\n")

print("Mean scores w.r.t gender:")
dataGend = data.groupby(['gender']).mean().sort_values(by=['mathScore', 'readingScore', 'writingScore'],ascending=False)
print(dataGend)
print("\n")

'''Boxplots for scores w.r.t groups'''
sns.set(context='notebook', style='whitegrid')
#Plot of mathscores w.r.t groups
sns.boxplot(x="mathScore", y="groups", data=data, whis="range", palette="vlag")
#Add in points to show each observation
sns.swarmplot(x="mathScore", y="groups", data=data, linewidth=0)
plt.title('Math Score (Max, Min, and Spread) w.r.t Ethnic groups')
plt.show()

#Plot of readingscores w.r.t groups
sns.boxplot(x="readingScore", y="groups", data=data, whis="range", palette="vlag")
#Add in points to show each observation
sns.swarmplot(x="readingScore", y="groups", data=data, linewidth=0)
plt.title('Reading Score (Max, Min, and Spread) w.r.t Ethnic groups')
plt.show()

#Plot of writingscores w.r.t groups
sns.boxplot(x="writingScore", y="groups", data=data, whis="range", palette="vlag")
#Add in points to show each observation
sns.swarmplot(x="writingScore", y="groups", data=data, linewidth=0)
plt.title('Writing Score (Max, Min, and Spread) w.r.t Ethnic groups')
plt.show()

#Plot of mathScore w.r.t parents qualification
sns.boxplot(x="mathScore", y="parentEd", data=data, whis="range", palette="vlag")
#Add in points to show each observation
sns.swarmplot(x="mathScore", y="parentEd", data=data, linewidth=0)
plt.title('Math Score (Max, Min, and Spread) w.r.t Parents Qualification')
plt.show()

#Plot of readingScore w.r.t parents qualification
sns.boxplot(x="readingScore", y="parentEd", data=data, whis="range", palette="vlag")
#Add in points to show each observation
sns.swarmplot(x="readingScore", y="parentEd", data=data, linewidth=0)
plt.title('Reading Score (Max, Min, and Spread) w.r.t Parents Qualification')
plt.show()

#Plot of writingscores w.r.t parents qualification
sns.boxplot(x="writingScore", y="parentEd", data=data, whis="range", palette="vlag")
#Add in points to show each observation
sns.swarmplot(x="writingScore", y="parentEd", data=data, linewidth=0)
plt.title('Writing Score (Max, Min, and Spread) w.r.t Parents Qualification')
plt.show()

#Comparing count of males/females above means scores
meanMS = data['mathScore'].mean()
print("Math Mean Score:", meanMS)
meanRS = data['readingScore'].mean()
print("Reading Mean Score:", meanMS)
meanWS = data['writingScore'].mean()
print("Writing Mean Score:", meanMS)
mIndx = list(np.where(data['gender']=='male'))
fIndx = list(np.where(data['gender']=='female'))
print("\n")

# pie charts
labels = ['Males Above Mean Score', 'Females Above Mean Score']
sizes = [sum([s for s in mIndx[0] if data['mathScore'][s]>meanMS]), sum([s for s in fIndx[0] if data['mathScore'][s]>meanMS])]
fig1, ax1 = plt.subplots()
ax1.pie(sizes, autopct='%1.1f%%')
ax1.axis('equal')
plt.title('Male/Female students Math Scores above mean')
plt.legend(labels)
plt.show()

labels = ['Males Above Mean Score', 'Females Above Mean Score']
sizes = [sum([s for s in mIndx[0] if data['readingScore'][s]>meanRS]), sum([s for s in fIndx[0] if data['readingScore'][s]>meanRS])]
fig1, ax1 = plt.subplots()
ax1.pie(sizes, autopct='%1.1f%%')
ax1.axis('equal')
plt.title('Male/Female students Reading Scores above mean')
plt.legend(labels)
plt.show()

labels = ['Males Above Mean Score', 'Females Above Mean Score']
sizes = [sum([s for s in mIndx[0] if data['writingScore'][s]>meanWS]), sum([s for s in fIndx[0] if data['writingScore'][s]>meanWS])]
fig1, ax1 = plt.subplots()
ax1.pie(sizes, autopct='%1.1f%%')
ax1.axis('equal')
plt.title('Male/Female students Writing Scores above mean')
plt.legend(labels)
plt.show()

'''Data transformations'''
dataC = data.copy()
dataC = dataC.drop(['gender'], axis=1)

dataC['lunch']=np.where(dataC['lunch']=='standard', 1, 0)
dataC['parentEd']=np.where(np.logical_or(dataC['parentEd']=='high school', dataC['parentEd']=='some high school'), 0, 1)
dataC['prep']=np.where(dataC['prep']=='none', 0, 1)
for i in range(0, len(dataC['groups'])):
    if dataC['groups'][i]=='group A':
        dataC['groups'][i] = 0
    elif dataC['groups'][i]=='group B':
        dataC['groups'][i] = 1
    elif dataC['groups'][i]=='group C':
        dataC['groups'][i] = 2
    elif dataC['groups'][i]=='group D':
        dataC['groups'][i] = 3
    elif dataC['groups'][i]=='group E':
        dataC['groups'][i] = 4
print("Transformed Data:")
print(dataC.head(10))

#Labels and featureSet columns
columns = dataC.columns.tolist()
columns = [c for c in columns if c not in ['prep']]
target = 'prep'

X = dataC[columns]
y = dataC[target]

#Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

print("\n")
print("Training FeatureSet:", X_train.shape)
print("Training Labels:", y_train.shape)
print("Testing FeatureSet:", X_test.shape)
print("Testing Labels:", y_test.shape)
print("\n")

'''Using random forrest Model'''
#Initializing the model with some parameters.
model = RandomForestClassifier(n_estimators=100, min_samples_leaf=10, random_state=1)
#Fitting the model to the data.
model.fit(X_train, y_train)
#Generating predictions for the test set.
predictions = model.predict(X_test)
#Computing the Model Accuracy
print("Random Forrest Accuracy:",metrics.accuracy_score(y_test, predictions))
#Computing the error.
print("Mean Absoulte Error:", mean_absolute_error(predictions, y_test))
#Computing classification Report
print("Classification Report:\n", classification_report(y_test, predictions))
#Plotting confusion matrix
print("Confusion Matrix:")
df = pd.DataFrame(
    confusion_matrix(y_test, predictions),
    index = [['actual', 'actual'], ['0','1']],
    columns = [['predicted', 'predicted'], ['0', '1']])
print(df)
print("\n")

'''Using SVM'''
#Initializing the model with some parameters.
model = SVC(gamma='auto')
#Fitting the model to the data.
model.fit(X_train, y_train)
#Generating predictions for the test set.
predictions = model.predict(X_test)
#Computing the Model Accuracy
print("SVM Accuracy:",metrics.accuracy_score(y_test, predictions))
#Computing the error.
print("Mean Absoulte Error:", mean_absolute_error(predictions, y_test))
#Computing classification Report
print("Classification Report:\n", classification_report(y_test, predictions))
#Plotting confusion matrix
print("Confusion Matrix:")
df = pd.DataFrame(
    confusion_matrix(y_test, predictions),
    index = [['actual', 'actual'], ['0','1']],
    columns = [['predicted', 'predicted'], ['0', '1']])
print(df)
print("\n")

'''Using linear classifier model(stochastic gradient descent (SGD))'''
#Initializing the model with some parameters.
model = linear_model.SGDClassifier(max_iter=1000, tol=1e-3)
#Fitting the model to the data.
model.fit(X_train, y_train)
#Generating predictions for the test set.
predictions = model.predict(X_test)
#Computing the Model Accuracy
print("SGD Accuracy:",metrics.accuracy_score(y_test, predictions))
#Computing the error.
print("Mean Absoulte Error:", mean_absolute_error(predictions, y_test))
#Computing classification Report
print("Classification Report:\n", classification_report(y_test, predictions))
#Plotting confusion matrix
print("Confusion Matrix:")
df = pd.DataFrame(
    confusion_matrix(y_test, predictions),
    index = [['actual', 'actual'], ['0','1']],
    columns = [['predicted', 'predicted'], ['0', '1']])
print(df)