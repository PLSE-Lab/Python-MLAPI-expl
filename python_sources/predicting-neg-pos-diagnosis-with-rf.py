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
import warnings

warnings.simplefilter("ignore")

#Setting Max Columns to 20
pd.set_option('display.max_columns', 20)

#importing data
data = pd.read_csv('../input/heart.csv')

#Displaying columns
print("Data Columns:",data.columns)

#Displaying shape and first 10 records
print("Shape:",data.shape)
print("First 10 Records:\n", data.head(10))

'''BarPlot Representation of some parameters relationships'''
#Displaying Top 30 individuals(same age & gender) with specific chest pain
data.groupby(['sex', 'age', 'cp']).cp.count().nlargest(30).plot(kind='barh')
plt.xlabel('Total No. of Individuals(same age & gender) with specific Chest Pain Type')
plt.ylabel('Sex:[0:F;1:M], Age & CP:[0:typical angina;1:atypical angina;2:non-anginal pain;3:asymptomatic]')
plt.title('Top 30 Individuals with Specific Chest Pain')
plt.show()

#Displaying Top 30 individuals(same age & gender) with fasting blood sugar/Non
data.groupby(['sex', 'age', 'fbs']).fbs.count().nlargest(30).plot(kind='barh')
plt.xlabel('Total No. of Individuals(same age & gender) with fbs/non')
plt.ylabel('Sex:[0:F;1:M], Age & fbs(fbs > 120 mg/dl):[1:true;0:false]')
plt.title('Top 30 Individuals with fasting blood sugar')
plt.show()

#Displaying Top 30 individuals(same age & gender) with specific resting electrocardiographic results
data.groupby(['sex', 'age', 'restecg']).restecg.count().nlargest(30).plot(kind='barh')
plt.xlabel('Total No. of Individuals(same age & gender) with specific resting electrocardiographic results')
plt.ylabel('Sex:[0:F;1:M], Age & restecg:[0:normal;1:ST-T wave abnormality;2:Probable/Definite left ventricular hypertrophy]')
plt.title('Top 30 Individuals specific resting electrocardiographic results')
plt.show()

#Displaying Top 30 individuals(same age & gender) with exercise induced angina/non
data.groupby(['sex', 'age', 'exang']).exang.count().nlargest(30).plot(kind='barh')
plt.xlabel('Total No. of Individuals(same age & gender) with exercise induced angina/non')
plt.ylabel('Sex:[0:F;1:M], Age & exang(exercise induced angina)[1:yes;0:no]')
plt.title('Top 30 Individuals with exercise induced angina/non')
plt.show()

#Displaying Top 30 individuals(same age & gender) with specific slope of the peak exercise ST segment
data.groupby(['sex', 'age', 'slope']).slope.count().nlargest(30).plot(kind='barh')
plt.xlabel('Total No. of Individuals(same age & gender) with specific slope of the peak exercise ST segment')
plt.ylabel('Sex:[0:F;1:M], Age & slope(specific slope of the peak exercise ST segment)[0:upsloping;1:flat;2:downsloping]')
plt.title('Top 30 Individuals with specific slope of the peak exercise ST segment')
plt.show()

#Displaying Top 30 individuals(same age & gender) with specific thal type
data.groupby(['sex', 'age', 'thal']).thal.count().nlargest(30).plot(kind='barh')
plt.xlabel('Total No. of Individuals(same age & gender) with specific thal type')
plt.ylabel('Sex:[0:F;1:M], Age & thal[0:normal;1:fixed defect;2:reversable defect]')
plt.title('Top 30 Individuals with specific thal type')
plt.show()

#Displaying Top 30 individuals(same age & gender) with specific angiographic disease status
data.groupby(['sex', 'age', 'target']).target.count().nlargest(30).plot(kind='barh')
plt.xlabel('Total No. of Individuals(same age & gender) with specific angiographic disease status')
plt.ylabel('Sex:[0:F;1:M], Age & Status[0:<50% diameter narrowing;1:>50% diameter narrowing]')
plt.title('Top 30 Individuals with specific angiographic disease status')
plt.show()

print("\nMales of 52 years age are prone to getting diagnosed with heart disease")
print("Females of 54 years age are prone to getting diagnosed with heart disease")

datac=data
datac=datac.sort_values(by=['sex'], ascending=True)
print("\nFirst 10 sorted records gender wise:\n", datac.head(10))

'''Boxplots'''
#Boxplots of females w.r.t major symptoms of getting diagnosed with heart disease
sns.boxplot(x="cp", y=datac['age'][:96], data=datac)
sns.swarmplot(x="cp", y=datac['age'][:96], data=datac, color=".25")
plt.xlabel('CP:[0:typical angina; 1:atypical angina; 2:non-anginal pain; 3:asymptomatic]')
plt.ylabel('Age(Female)')
plt.title('Types of chest pain in females')
plt.show()

sns.boxplot(x="fbs", y=datac['age'][:96], data=datac)
sns.swarmplot(x="fbs", y=datac['age'][:96], data=datac, color=".25")
plt.xlabel('Fbs(fbs > 120 mg/dl):[1:true;0:false]')
plt.ylabel('Age(Female)')
plt.title('Fasting blood sugar in females')
plt.show()

sns.boxplot(x="restecg", y=datac['age'][:96], data=datac)
sns.swarmplot(x="restecg", y=datac['age'][:96], data=datac, color=".25")
plt.xlabel('restecg:[0:normal; 1:ST-T wave abnormality; 2:Probable/Definite left ventricular hypertrophy]')
plt.ylabel('Age(Female)')
plt.title('Resting Electrocardiographic Results of females')
plt.show()

sns.boxplot(x="exang", y=datac['age'][:96], data=datac)
sns.swarmplot(x="exang", y=datac['age'][:96], data=datac, color=".25")
plt.xlabel('exang(exercise induced angina)[1:yes; 0:no]')
plt.ylabel('Age(Female)')
plt.title('Exercise induced angina in females')
plt.show()

sns.boxplot(x="slope", y=datac['age'][:96], data=datac)
sns.swarmplot(x="slope", y=datac['age'][:96], data=datac, color=".25")
plt.xlabel('slope(specific slope of the peak exercise ST segment)[0:upsloping; 1:flat; 2:downsloping]')
plt.ylabel('Age(Female)')
plt.title('Slope of the peak exercise ST segment in females')
plt.show()

sns.boxplot(x="thal", y=datac['age'][:96], data=datac)
sns.swarmplot(x="thal", y=datac['age'][:96], data=datac, color=".25")
plt.xlabel('thal[0:normal; 1:fixed defect; 2:reversable defect]')
plt.ylabel('Age(Female)')
plt.title('Thal types in females')
plt.show()

sns.boxplot(x="target", y=datac['age'][:96], data=datac)
sns.swarmplot(x="target", y=datac['age'][:96], data=datac, color=".25")
plt.xlabel('Status[0:<50% diameter narrowing; 1:>50% diameter narrowing(in any major vessel: attributes 59 through 68 are vessels)]')
plt.ylabel('Age(Female)')
plt.title('Heart disease diagnosis in females')
plt.show()

print("\nMean age of female for getting diagnosed with heart disease:",54)

#Boxplots of males w.r.t major symptoms of getting diagnosed with heart disease
sns.boxplot(x="cp", y=datac['age'][96:303], data=datac)
sns.swarmplot(x="cp", y=datac['age'][96:303], data=datac, color=".25")
plt.xlabel('CP:[0:typical angina; 1:atypical angina; 2:non-anginal pain; 3:asymptomatic]')
plt.ylabel('Age(Male)')
plt.title('Types of chest pain in males')
plt.show()

sns.boxplot(x="fbs", y=datac['age'][96:303], data=datac)
sns.swarmplot(x="fbs", y=datac['age'][96:303], data=datac, color=".25")
plt.xlabel('Fbs(fbs > 120 mg/dl):[1:true;0:false]')
plt.ylabel('Age(Male)')
plt.title('Fasting blood sugar in males')
plt.show()

sns.boxplot(x="restecg", y=datac['age'][96:303], data=datac)
sns.swarmplot(x="restecg", y=datac['age'][96:303], data=datac, color=".25")
plt.xlabel('restecg:[0:normal; 1:ST-T wave abnormality; 2:Probable/Definite left ventricular hypertrophy]')
plt.ylabel('Age(Male)')
plt.title('Resting Electrocardiographic Results of males')
plt.show()

sns.boxplot(x="exang", y=datac['age'][96:303], data=datac)
sns.swarmplot(x="exang", y=datac['age'][96:303], data=datac, color=".25")
plt.xlabel('exang(exercise induced angina)[1:yes; 0:no]')
plt.ylabel('Age(Male)')
plt.title('Exercise induced angina in males')
plt.show()

sns.boxplot(x="slope", y=datac['age'][96:303], data=datac)
sns.swarmplot(x="slope", y=datac['age'][96:303], data=datac, color=".25")
plt.xlabel('slope(specific slope of the peak exercise ST segment)[0:upsloping; 1:flat; 2:downsloping]')
plt.ylabel('Age(Male)')
plt.title('Slope of the peak exercise ST segment in males')
plt.show()

sns.boxplot(x="thal", y=datac['age'][96:303], data=datac)
sns.swarmplot(x="thal", y=datac['age'][96:303], data=datac, color=".25")
plt.xlabel('thal[0:normal; 1:fixed defect; 2:reversable defect]')
plt.ylabel('Age(Male)')
plt.title('Thal types in males')
plt.show()

sns.boxplot(x="target", y=datac['age'][96:303], data=datac)
sns.swarmplot(x="target", y=datac['age'][96:303], data=datac, color=".25")
plt.xlabel('Status[0:<50% diameter narrowing; 1:>50% diameter narrowing(in any major vessel: attributes 59 through 68 are vessels)]')
plt.ylabel('Age(Male)')
plt.title('Heart disease diagnosis in males')
plt.show()

print("Mean age of male for getting diagnosed with heart disease:",52,"\n")

#Correlation matrix & Heatmap - Finding correlation
corrmat = data.corr()
sns.heatmap(corrmat, annot=True, fmt='.1f', vmin=0, vmax=1, square=True);
plt.show()

#Pairplot of parameters
sns.pairplot(datac, kind="reg")
plt.show()
print("No need for dropping any of the parameters")

#Labels and featureSet columns
columns = data.columns.tolist()
columns = [c for c in columns if c not in ['target']]
target = 'target'

X = data[columns]
y = data[target]

#Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

print("\n")
print("Training FeatureSet:", X_train.shape)
print("Training Labels:", y_train.shape)
print("Testing FeatureSet:", X_test.shape)
print("Testing Labels:", y_test.shape)
print("\n")

#Using random forrest Model
#Initializing the model with some parameters.
model = RandomForestClassifier(n_estimators=100, min_samples_leaf=10, random_state=1)
#Fitting the model to the data.
model.fit(X_train, y_train)
#Generating predictions for the test set.
predictions = model.predict(X_test)
#Computing the Model Accuracy
print("Random Forrest Accuracy:",round((metrics.accuracy_score(y_test, predictions))*100,2))
#Computing the error.
print("Mean Absoulte Error:",round((mean_absolute_error(predictions, y_test))*100,2))
#Computing classification Report
print("Classification Report:\n", classification_report(y_test, predictions))
#Plotting confusion matrix
print("Confusion Matrix:")
df = pd.DataFrame(
    confusion_matrix(y_test, predictions),
    index = [['actual', 'actual'], ['0','1']],
    columns = [['predicted', 'predicted'], ['0', '1']])
print(df)

#Generating predictions for the whole dataset.
print("\nExecuting trained Random Forrest Model for the whole dataset...")
predictions = model.predict(X)
#Computing the Model Accuracy
print("Random Forrest Accuracy:",round((metrics.accuracy_score(y, predictions))*100,2))
#Computing the error.
print("Mean Absoulte Error:",round((mean_absolute_error(predictions, y))*100,2))
#Computing classification Report
print("Classification Report:\n", classification_report(y, predictions))
#Plotting confusion matrix
print("Confusion Matrix:")
df = pd.DataFrame(
    confusion_matrix(y, predictions),
    index = [['actual', 'actual'], ['0','1']],
    columns = [['predicted', 'predicted'], ['0', '1']])
print(df)

data['prediction']=predictions
data['prediction']=np.where(data['prediction']==0, "Diagnosis-Negative(0)", "Diagnosis-Positive(1)")

print("\nFirst 10 records after appending predictions:\n",data.head(10))