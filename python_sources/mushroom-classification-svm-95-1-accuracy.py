# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.manifold import Isomap
from sklearn.model_selection import train_test_split,cross_val_score,cross_val_predict

def coding(col,codedict):
    colcoded = pd.Series(col,copy=True)
    for key,value in codedict.items():
        colcoded.replace(key,value,inplace=True)
    return colcoded

data = pd.read_csv('../input/mushrooms.csv')
data_y = data['class']
data_X = data.drop(['class','veil-type','stalk-root'],axis=1)
'''z = data['cap-shape'].value_counts()
z.plot(kind='bar')
plt.show()'''
#plotting bar graphs for frequency of first 20 columns
fig = plt.figure(figsize=(10,9))
fig.subplots_adjust(left=0,right=1,bottom=0,top=1,hspace=0.05,wspace=0.05)
for i in range(20):
    ax = fig.add_subplot(5,4,i+1,xticks=[],yticks=[])
    z = data_X.iloc[:,i].value_counts()
    z.plot(kind='bar')
plt.show()

# converting data to numeric for further exploration
data_X['cap-shape'] = coding(data_X['cap-shape'],{'x':1,'f':2,'k':3,'b':4,'s':5,'c':6})
data_X['cap-surface'] = coding(data_X['cap-surface'],{'y':1,'s':2,'f':3,'g':4})
data_X['cap-color'] = coding(data_X['cap-color'],{'n':1,'g':2,'e':3,'y':4,'w':5,'b':6,'p':7,'c':8,'u':9,'r':10})
data_X['odor'] = coding(data_X['odor'],{'n':1,'f':2,'s':3,'y':4,'a':5,'l':6,'p':7,'c':8,'m':9})
data_X['gill-attachment'] = coding(data_X['gill-attachment'],{'f':1,'a':2})
data_X['gill-spacing'] = coding(data_X['gill-spacing'],{'c':1,'w':2})
data_X['gill-size'] = coding(data_X['gill-size'],{'b':1,'n':2})
data_X['gill-color'] = coding(data_X['gill-color'],{'b':1,'p':2,'w':3,'n':4,'g':5,'h':6,'u':7,'k':8,'e':9,'y':10,'o':11,'r':12})
data_X['stalk-shape'] = coding(data_X['gill-size'],{'t':1,'e':2})
data_X['stalk-surface-above-ring'] = coding(data_X['stalk-surface-above-ring'],{'s':1,'k':2,'f':3,'y':4})
data_X['stalk-surface-below-ring'] = coding(data_X['stalk-surface-below-ring'],{'s':1,'k':2,'f':3,'y':4})
data_X['stalk-color-above-ring'] = coding(data_X['stalk-color-above-ring'],{'w':1,'p':2,'g':3,'n':4,'b':5,'o':6,'e':7,'c':8,'y':9})
data_X['stalk-color-below-ring'] = coding(data_X['stalk-color-below-ring'],{'w':1,'p':2,'g':3,'n':4,'b':5,'o':6,'e':7,'c':8,'y':9})
data_X['veil-color'] = coding(data_X['veil-color'],{'w':1,'o':2,'n':3,'y':4})
data_X['ring-number'] = coding(data_X['ring-number'],{'o':1,'t':2,'n':3})
data_X['ring-type'] = coding(data_X['ring-type'],{'p':1,'e':2,'l':3,'f':4,'n':5})
data_X['spore-print-color'] = coding(data_X['spore-print-color'],{'w':1,'n':2,'k':3,'h':4,'r':5,'u':6,'o':7,'y':8,'b':9})
data_X['population'] = coding(data_X['population'],{'v':1,'y':2,'s':3,'n':4,'a':5,'c':6})
data_X['habitat'] = coding(data_X['habitat'],{'d':1,'g':2,'p':3,'l':4,'u':5,'m':6,'w':7})
data_X['bruises'] = coding(data_X['bruises'],{'f':1,'t':2})
data_y = coding(data_y,{'p':0,'e':1})

#dimension reduction
pca = PCA(n_components = 2)
reduced_data = pca.fit_transform(data_X)

colors=['blue','red']
#plotting initial data
'''for i in range(2):
    t = np.where(data_y == i)
    X = reduced_data[t,0]
    Y = reduced_data[t,1]
    plt.scatter(X,Y,c=colors[i],s=3)
plt.title('initial distribution after applying pca')
plt.show()'''

#applying svm (95.1% accuracy)
X_train,X_test,y_train,y_test = train_test_split(data_X,data_y,test_size=0.3,random_state=25)
svc_model = svm.SVC(gamma=0.001,C=100,kernel='linear')
svc_model.fit(X_train,y_train)
y_pred = svc_model.predict(X_test)
score = svc_model.score(X_test,y_test)
#scores = cross_val_score(svc_model,X_train,y_train,cv=5)
#predicted = cross_val_predict(svc_model,X_train,y_train,cv=5)

y_test = y_test.as_matrix()
X_test_iso = Isomap(n_neighbors=10).fit_transform(X_test)
fig,ax = plt.subplots(1,2,figsize=(8,4))
fig.subplots_adjust(top=0.85)
for i in range(2):
    t1 = np.where(y_test==i)
    t2 = np.where(y_pred==i)
    X1 = X_test_iso[t1,0]
    Y1 = X_test_iso[t1,1]
    X2 = X_test_iso[t2,0]
    Y2 = X_test_iso[t2,1]
    ax[0].scatter(X1,Y1,c=colors[i],s=3)
    ax[1].scatter(X2,Y2,c=colors[i],s=3)
ax[0].set_title('actual_labels')
ax[1].set_title('predicted_labels')
plt.show()
    