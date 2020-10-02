# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

 # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


import matplotlib.pyplot as plt
from sklearn import tree #for decision tree classifier
from sklearn import datasets #for data
from sklearn import metrics #for confusion matric 
from sklearn import cross_validation #for test and train data split
from sklearn.naive_bayes import GaussianNB
import csv
import numpy 


content=[]
file="../input/voice.csv"
with open(file, encoding="utf8") as csvfile:
    read=csv.reader(csvfile)
    for row in read:
        content.append(row)
sig=[]
def analyse(j): 
    men=[]
    women=[]
    for i in content[1:]:
        if i[20]=="male":
            men.append(float(i[j]))
        else:
            women.append(float(i[j]))
    m=numpy.average(men)
    w= numpy.average(women)
    print("men" , content[0][j], m)
    print("women", content[0][j],w)
    print(m-w)
    if(m-w>0.05):
        sig.append(j)
    if (m-w <-0.05):
        sig.append(j)
        
for i in range(20): 
    analyse(i)


#Using only attributes for which mean diff is either >0.05 or <0.05
import math
data=[]
digit=0
meanM=[]
meanW=[]
for i in content:
    if i[20]=='male':
        digit=0
        meanM.append(float(i[0])*100)
    elif i[20]=='female':
        digit=1
        meanW.append(float(i[0])*100)
    data.append([  i[5], i[17], i[7],i[18],i[12], i[6],i[15], i[9], digit ])



#even without meanfreq, accuracy is 96% 

#for visualization of mean frequency of male (meanM) and female (meanW)
#there is no clear cut line differentiating male and female voice
#0 for male mean frequency distribution
#1 for female mean frequency distribution  
#def visualize(i):
#    if i=1: 
#        plt.hist(meanW)
#    elif i=0:
#        plt.hist(meanM)
#    plt.title("Gaussian Histogram")
#    plt.xlabel("Value")
#    plt.ylabel("Frequency")
#    plt.show()





X=[]
y=[]
for i in data[1:]:
    X.append(i[:8])
    y.append(i[8])
    
X_train, X_test, y_train, y_test = cross_validation.train_test_split(
    X, y, test_size=0.33, random_state=0)
classifier = tree.DecisionTreeClassifier()
classifier.fit(X_train,y_train)
expected=y_test
predict=classifier.predict(X_test)
print(metrics.confusion_matrix(expected, predict))
score = metrics.accuracy_score(y_test, classifier.predict(X_test))
print(score)



