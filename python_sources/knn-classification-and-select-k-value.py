import pandas as pd
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

print(os.listdir("../input"))
# read csv data file
df=pd.read_csv('../input/pulsar_stars.csv')
df.head(10)
df.shape
df.columns

# Exploratory Data Analysis
# create a histogram for every feature
meantarget_class = df.groupby('target_class').mean().transpose()
print(meantarget_class)
df.groupby('target_class').groups
df['target_class'].value_counts()
df.hist()
plt.savefig('histogram for pulsar stars features.png')


#calculate correlation coefficient between each other and make a half heatmap plot
corr = df.corr()
corr1=corr[abs(corr)>0.5]
plt.figure(figsize=(20,70))

# Default correlation heatmap plot
plt.clf()
plt.subplot(2, 1, 1)
mask = np.zeros_like(corr)
mask[np.triu_indices_from(corr)] = True
sns.heatmap(corr,annot=True,mask=mask,linewidths=0.5, linecolor='black')

# Customized 2nd plot that abs value of correlation>0.5
plt.subplot(2, 1, 2)
sns.heatmap(corr1,annot=True, mask=mask,linewidths=0.5, linecolor='black')
plt.show()
plt.savefig('Correlation heatmap.png')

#convert target_class data type to boolean type
df['target_class'] = df['target_class'].astype('bool')
df.info()

# select first 8 columns as x and 9th column as y
x = df.iloc[:,0:8]
y= df.target_class

#SPLITTING THE DATASET INTO THE TRAINING SET AND TEST SET
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25,random_state = 0, stratify=df.target_class)

#FEATURE SCALING
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train_z = sc.fit_transform(x_train)
x_test_z = sc.transform(x_test)
x_train_z.shape

#FITTING CLASSIFIER TO THETRAINING SET
from sklearn.neighbors import KNeighborsClassifier
knn =KNeighborsClassifier(n_neighbors=round(13423**(0.5)))
knn.fit(x_train_z,y_train)
y_predKNN = knn.predict(x_test)
knn.score(x_test_z, y_test)  # 0.9758659217877095

# make confusion matrix
from sklearn.metrics import confusion_matrix,classification_report
cmKNN = confusion_matrix(y_test, y_predKNN)
crKNN=classification_report(y_test, y_predKNN)
print(cmKNN)
print(crKNN) # accuracy=0.91

# make confusion matrix plot
plt.clf()  #clear current figure
plt.imshow(cmKNN, interpolation='nearest', cmap=plt.cm.Wistia) # display an image based on cmdt1, insert value nearest place, and wistia color
classNames = ['Positive','Negative']
plt.title('BernoulliNB Confusion Matrix - Test Data')  # add title name
plt.ylabel('True label')  # add y label name
plt.xlabel('Predicted label')  # add x label name
tick_marks = np.arange(len(classNames))
plt.xticks(tick_marks, classNames)  #set the current tick locations and labels of the x-axis
plt.yticks(tick_marks, classNames)  #set the current tick locations and labels of the y-axis
s = [['TP', 'FN'], ['FP', 'TN']]

for i in range(2):
    for j in range(2):
        plt.text(j, i, str(s[i][j]) + " = " + str(cmKNN[i][j]))   # Add text to the axes.
plt.show()

errorKNN=[]
# Calculating error for K values between 1 and 40
for i in range(1, round(17898**(0.5))):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train, y_train)
    pred_i = knn.predict(x_test)
    errorKNN.append(np.mean(pred_i != y_test))

plt.figure(figsize=(12, 6))
plt.plot(range(1, round(17898**(0.5))), errorKNN, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate of K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')
plt.show()
plt.savefig('Error Rate of K Value.png')  # when k = 60 is the best


# make roc curve
import scikitplot as skplt
import matplotlib.pyplot as plt
predicted_probas = knn.predict_proba(x_test_z)
skplt.metrics.plot_roc_curve(y_test, predicted_probas)
plt.show()

