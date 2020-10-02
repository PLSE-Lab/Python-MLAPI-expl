# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from sklearn.metrics import accuracy_score,confusion_matrix, f1_score
import pandas as pd
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import matplotlib.pyplot as plt
import seaborn as sns


neg = 0
pos = 1
dataset_train = pd.read_csv('../input/aps_failure_training_set_processed_8bit.csv')
dataset_test = pd.read_csv('../input/aps_failure_test_set_processed_8bit.csv')
dataset_train.replace([-0.9921875,0.9921875],[neg,pos],inplace = True)
dataset_test.replace([-0.9921875,0.9921875],[neg,pos],inplace = True)

dataset_train=dataset_train.rename(columns={"class": "label"})
dataset_test=dataset_test.rename(columns={"class": "label"})

#split dataset for baseline 
y_train_bsln = dataset_train.label
x_train_bsln = dataset_train.drop(labels='label', axis=1)
y_test_bsln = dataset_test.label
x_test_bsln = dataset_test.drop(labels='label', axis=1)

#function for print confusion matrix
def plot_CM(cm, title): 
    plt.figure(figsize=(5,5))
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, fmt='d', ax=ax,cmap="YlOrRd"); #annot=True to annotate cells
    # labels, title and ticks
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title(title)
    ax.xaxis.set_ticklabels(['neg', 'pos'])
    ax.yaxis.set_ticklabels(['pos', 'neg']);
    plt.show()
    cost = cm[0,1]*10+cm[1,0]*500
    print("Cost:",cost)
    print('-------------------------------//------------------------------')
    
print('-------------------------QDA baseline----------------------------')
QDA = QuadraticDiscriminantAnalysis()
QDA.fit(x_train_bsln,y_train_bsln)
y_pred = QDA.predict(x_test_bsln)
f1_val = f1_score(y_test_bsln, y_pred);
accuracy = accuracy_score(y_test_bsln, y_pred);
print("F1 QDA:",f1_val, "Accuracy QDA:",accuracy)
cm = confusion_matrix(y_test_bsln, y_pred)
plot_CM(cm,'CM QDA baseline')

#some functions for find and sum the values on the histograms for every samples
def find_ist_col(data):
    istograms_columns=list([])
    columns = data.columns
    for i in range(0,columns.size):
        first_2_char = columns[i][0:2]
        if(i<(len(columns)-1)):
            if first_2_char==columns[i+1][0:2]:
                istograms_columns.append(columns[i])
            elif first_2_char==columns[i-1][0:2]:
                istograms_columns.append(columns[i])
    return istograms_columns

def sum_isto_values(data,plot): 
    istograms_columns = find_ist_col(data)            
    divided_istogram = list([]) 
    for i in range(0,len(istograms_columns),10):
        single_ist = list([])
        for j in range(0,10):
            l=i+j
            single_ist.append(istograms_columns[l])
        divided_istogram.append(single_ist)
    sum_of_istograms = pd.DataFrame()
    for element in divided_istogram:
        sum_of_single_isto = data[element].sum(axis=1)
        sum_of_istograms = pd.concat([sum_of_istograms,sum_of_single_isto],axis=1)
    sum_of_istograms.columns=['1','2','3','4','5','6','7']
    sum_of_istograms=pd.concat([sum_of_istograms,data['label']],axis=1)
    if plot:
        correlation = sum_of_istograms.corr()
        print("sum_of_istograms.describe()",sum_of_istograms.describe())
        print("sum_of_istograms.head()",sum_of_istograms.head())           
        print("correlation",correlation)   
        plt.figure(figsize=(20,5))
        for i in range (1,8):
            plt.subplot(1,8,i)
            ax = sns.stripplot(y=str(i),x='label',data=sum_of_istograms, jitter=True)
            ax.set_title(i)
        plt.show()
    #add column sum to the two dataset
    mean_of_sum = sum_of_istograms.mean(axis=1)
    col = list(data.columns)
    data=data[col]
    col.append('sum')
    data=pd.concat([data,mean_of_sum], axis=1)
    data.columns=col
    return data

#add the new feature sum_of_hist to the train and test dataset
dataset_train=sum_isto_values(dataset_train,False)
dataset_test=sum_isto_values(dataset_test,False)

y_train = (dataset_train.label)
x_train = pd.DataFrame(dataset_train.drop(labels='label', axis=1))
y_test = (dataset_test.label)
x_test = pd.DataFrame(dataset_test.drop(labels='label', axis=1))

print('------------------Quadratic Discriminant Analysis--------------------')
QDA = QuadraticDiscriminantAnalysis()
QDA.fit(x_train,y_train)
y_pred = QDA.predict(x_test)
f1_val = f1_score(y_test, y_pred);
accuracy = accuracy_score(y_test, y_pred);
print("F1 QDA:",f1_val, "Accuracy QDA:",accuracy)
cm = confusion_matrix(y_test, y_pred)
plot_CM(cm,'CM QDA baseline')

# Any results you write to the current directory are saved as output.