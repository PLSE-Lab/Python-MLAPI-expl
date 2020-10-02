from __future__ import division,print_function
import pandas as pd
import numpy as np
import random


def norm(data, avg_age=40):
    data['Gender']= data['Sex'].map({'female':0,'male':1})
    ports={'C':0,'Q':1,'S':2}
   
    for i in range(3):
        mean_fare=data.loc[data.Pclass==i+1, 'Fare'].mean()
        data.loc[(data.Pclass==i+1) & (data.Fare.isnull()), 'Fare']=mean_fare
        
    #data['Boarding'] = data['Embarked'].map(lambda x: ports.get(x, random.choice(ports.values()))).astype(int)
    
    # this was not working well
    fare_cat=[10,20,30,40,50,100,200,500]   
    def farebin(x):
        for i in range(len(fare_cat)):
            if x<fare_cat[i]:
                return i
        return i+1
    #data['AgeCat'] = data['Age'].map(lambda x: 1 if x <15 else 2 if x > 65 else 0)
    ##########
    data.Age.fillna(avg_age, inplace=True)
    data.SibSp.fillna(0, inplace=True)
    data.Parch.fillna(0, inplace=True)
    #data['Family']=(data.Parch+data.SibSp).map(lambda x: 1 if x>0 else 0).astype(int)
    return data.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'SibSp', 'Parch'], axis=1)


res_f1=[]
res_accu=[]
NTRIES=100

for i in range(NTRIES):
    print('Test %d' % (i+1))
    data=pd.read_csv('../input/train.csv')
    avg_age=data.Age.median()
    data=norm(data, avg_age)
    split=np.random.uniform(size=len(data))<0.25
    tdf=data[split]
    t=data[split].values
    l=data[(~ split)].values
    assert len(t)+len(l)==len(data)
    print('Splited to', len(t), len(l))

    from sklearn.ensemble import RandomForestClassifier
    forest=RandomForestClassifier(n_estimators=200, criterion='entropy', class_weight='auto' )
    forest.fit(l[::, 2:], l[::, 1])


    to=forest.predict(t[::, 2:]).astype(int)
    from collections import defaultdict
    count=defaultdict(lambda: 0)
    for i in range(len(tdf)):
        count[(tdf['Survived'].iloc[i], to[i])]+=1
        
    print('Correct',count[(0,0)]+count[(1,1)], 'False Possitives', count[(0,1)], 'False Negatives', count[(1,0)])
    accu=(count[(0,0)]+count[(1,1)])/len(t)
    res_accu.append(accu)
    print('Accuracy', accu)
    prec= count[(1,1)]/ (count[(1,1)] + count[(0,1)])
    print('Precission', prec)
    recall= count[(1,1)]/ (count[(1,1)] + count[(1,0)])
    print('Recall',  recall)
    f1=2* prec*recall / (prec + recall)
    res_f1.append(f1)
    print('F1 Score',  f1)


r=pd.DataFrame({'Accuracy':res_accu, 'F1': res_f1})
with open('res.txt','w') as f:
    f.write(str(r.describe()))
print(r.describe())


x=np.arange(1,NTRIES+1)
import matplotlib.pyplot as plt
plt.plot(x,res_accu, 'r-',label="Accuracy")
plt.plot(x,res_f1, 'g-', label='F1 Rate')
plt.title('Stability of random forrest classifier')
plt.legend()
plt.savefig('pic.png', format='png')
plt.show()








    



    
