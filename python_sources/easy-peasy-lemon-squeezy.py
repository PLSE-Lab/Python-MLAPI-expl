import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
    

# The competition datafiles are in the directory ../input
# Read competition data files:
train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")
alg_type='LR'
# Write to the log:
print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))
print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))
# Any files you write to the current directory get shown as outputs

nTrees,nSplits,nLeaves=10,4,2
par_str=str(nTrees)+'_'+str(nSplits)+'_'+str(nLeaves)

target= ['label']
predictors=[x for x in train.columns if x not in target]
trainX=pd.DataFrame(train[predictors])
trainY=pd.DataFrame(train['label'])
train=''

if alg_type=='RF':
    print('Starting RF training')
    RFR=RandomForestRegressor(n_estimators=nTrees, min_samples_split=nSplits, min_samples_leaf=nLeaves)
    RFR.fit(trainX[predictors],trainY[target])
    print('Done Training RF')

    print('Testing RF')
    y_pred=list(RFR.predict(test[predictors]))
    print('Testing done.')
    test=''
if alg_type=='NB':
    par_str=alg_type
    print('Starting NB training')
    #NB=GaussianNB() #Does not work as well (50 % Accuracy)
    NB=MultinomialNB() #>80% accuracy
    NB.fit(trainX[predictors],trainY[target])
    print('Done Training NB')

    print('Testing NB')
    y_pred=list(NB.predict(test[predictors]))
    print('Testing done.')
    test=''  
if alg_type=='K_n':
    par_str=alg_type
    print('Starting Nearest Neighbors training')
    print('Failed with 5 neighbors, as it took too long.')
    NN=KNeighborsClassifier(n_neighbors=3)
    NN.fit(trainX[predictors],trainY[target])
    print('Done Training the Neighbors')

    print('Testing Nearest Neighbors')
    y_pred=list(NN.predict(test[predictors]))
    print('Testing done.')
    test=''  
if alg_type=='SVC':
    par_str=alg_type
    print('Starting SVCs training')
    supVec=SVC()
    supVec.fit(trainX[predictors],trainY[target])
    print('Done Training the SVC')

    print('Testing SVC')
    y_pred=list(supVec.predict(test[predictors]))
    print('Testing done.')
    test=''      
if alg_type=='LR':
    par_str=alg_type
    print('Starting LRs training')
    LR=LogisticRegression()
    LR.fit(trainX[predictors],trainY[target])
    print('Done Training the LR')

    print('Testing LR')
    y_pred=list(LR.predict(test[predictors]))
    print('Testing done.')
    test=''          
    
    
print('Prepare for Submission')
y_pred=[int(round(x)) for x in y_pred]
for i in range(len(y_pred)):
    if y_pred[i] >9:
        y_pred[i]=9
    if y_pred[i] < 0:
        y_pred[i]=0

image_id_n=np.arange(1,len(y_pred)+1,1)
print(len(image_id_n)/27997.,len(y_pred)/27997.)
subm=pd.DataFrame({'ImageId': image_id_n,'Label': y_pred})
print( subm.head(3))
subm.to_csv(par_str+'_RF_out.csv',index=False)