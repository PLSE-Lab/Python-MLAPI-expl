##########################################################
################## DS Project ############################
###  FEATURE DELECTION USING LINEAR CLASSIFIER WEIGHTS ########
##########################################################3

from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

names = ['pregnent', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'clas']
data = pd.read_csv('../input/breast-cancer.csv',names=names)
f1 = open("Naive_Bayers.txt","w+")
f2 = open("Perceptron.txt","w+")
f3 = open("Linear_SVM.txt","w+")

datavalues = data.values
tsize=0.33
##########################################################
################## Naive Bayers Implementation ############################


def naivebayers(xtrain , xtest , ytrain , ytest,fname):
    f=fname
    gmodel = GaussianNB()
    yprediction = gmodel.fit(xtrain, ytrain).predict(xtest)
    print('\n\nNAIVE BAYERS ALGORITHM')
    print("Number of mislabeled points out of a total %d points : %d" % (len(ytest) , (ytest != yprediction).sum()))
    print('Accuracy of Naive Bayers: %.2f' %accuracy_score(ytest, yprediction))
    print('\n')
    f.write('\n\nNAIVE BAYERS ALGORITHM')
    f.write("\nNumber of mislabeled points out of a total %d points : %d" % (len(ytest) , (ytest != yprediction).sum()))
    f.write('\nAccuracy of Naive Bayers: %.2f' %accuracy_score(ytest, yprediction))
    f.write("\n====================================================================================\n")
    
    return accuracy_score(ytest,yprediction)

#def perceptron():


##########################################################
################## Linear SVM Implementation ############################


def linearsvm(xtrain , xtest , ytrain , ytest,fname):
    f=fname
    print('\nLINEAR SVM ALGORITHM')
    f.write('\nLINEAR SVM ALGORITHM')
    
    ######################################################################3
    ##############33 Training for linear classifier weight ################
    #######################################################################
    linsvmodel = svm.LinearSVC()
    linsvmodel.fit(xtrain, ytrain) 
    
    svm.LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
    intercept_scaling=1, loss='squared_hinge', max_iter=1000,
    multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,verbose=0)
    
    lsvmpredict = linsvmodel.predict(xtest)
    print("Number of mislabeled points out of a total %d points : %d" % (len(ytest) , (ytest != lsvmpredict).sum()))
    print('Accuracy of Linear SVM: %.2f' %accuracy_score(ytest, lsvmpredict))
    print('\n')
    f.write("\nNumber of mislabeled points out of a total %d points : %d" % (len(ytest) , (ytest != lsvmpredict).sum()))
    f.write('\nAccuracy of Linear SVM: %.2f' %accuracy_score(ytest, lsvmpredict))
    f.write('\n')
    f.write("====================================================================================\n")
    return accuracy_score(ytest,lsvmpredict)

##########################################################
################## PERCEPTRON Implementation ############################


def perceptron(xtrain , xtest , ytrain , ytest,fnae):
    f=fnae
    print('\n')
    print('PERCEPTRON ALGORITHM')
    f.write('\n')
    f.write('\nPERCEPTRON ALGORITHM')
    
    xtrain = xtrain.astype(float)
    xtest = xtest.astype(float) 
    ytrain = ytrain.astype(float) 
    ytest = ytest.astype(float)
    
    ######################################################################3
    ##############33 Training for linear classifier weight ################
    #######################################################################
    
    scalermodel = StandardScaler() 
    scalermodel.fit(xtrain)
    xtrainscaler = scalermodel.transform(xtrain)
    xtestscaler = scalermodel.transform(xtest)

    perceptronmodel = Perceptron(eta0=0.1, max_iter=40, random_state=0)    
    perceptronmodel.fit(xtrainscaler, ytrain)    
    perceptpred = perceptronmodel.predict(xtestscaler)
    
    print("Number of mislabeled points out of a total %d points : %d" %(len(ytest), (ytest != perceptpred).sum()))
    print('accuracy: %.2f' %accuracy_score(ytest, perceptpred))
    print('\n')
    f.write("\nNumber of mislabeled points out of a total %d points : %d" %(len(ytest), (ytest != perceptpred).sum()))
    f.write('\naccuracy: %.2f' %accuracy_score(ytest, perceptpred))
    f.write('\n')
    f.write("====================================================================================\n")
    return accuracy_score(ytest,perceptpred)

##########################################################
################## UNIVARIATE SELECTION Implementation ############################


def fsunvariateselection(X , Y,fname):
    f=fname
    print("Using Univariate Seletion for feature selection...")
    print("Choosing 4 attributes out of 7\n")
    f.write("\nUsing Univariate Seletion for feature selection...")
    f.write("\nChoosing 4 attributes out of 7\n")
    
    vsmodel = SelectKBest(score_func=chi2, k=4)  #choosing 4 attributes
    vsmodel = vsmodel.fit(X, Y)
    
    np.set_printoptions(precision=3)
    print('Calculated scores for given features')
    f.write('\nCalculated scores for given features')
    count = 0
    for i in vsmodel.scores_:
        print("Column Name: ", names[count] ," Weight: ", i)
        f.write("\nColumn Name: %s Weight: %f " % (names[count] ,i))
        count+=1
    
    arr = vsmodel.scores_
    arr1 = names

  ###############################################################################
  ##### Obtaining those colmun through sorting which yields highest weight######
  ###############################################################################    

    for i in range(0,len(arr)-1):
        for j in range(i,len(arr)):
            if arr[i] < arr[j]:
                temp1=arr1[j]
                temp= arr[j]
                arr[j]= arr[i]
                arr1[j]= arr1[i]
                arr[i]= temp
                arr1[i]= temp1
        
    print("\nSelected Features are: ")
    f.write("\nSelected Features are: ")
    
    for i in range(0,4):
        print(arr1[i])
        f.write(arr1[i])
        f.write('\n')   
## returning the selected features identified by the scheme   #####
    
    features = vsmodel.transform(X)
    return features   


##########################################################
################## RECURSIVE ELIMINATION Implementation ############################
   
   
def recursiveelimination(X,Y,fname):
    f=fname
    print("Using Recursive Elimination for feature selection...")
    print("Choosing 4 attributes out of 7\n")
    f.write("\nUsing Recursive Elimination for feature selection...")
    f.write("\nChoosing 4 attributes out of 7\n")
    
    
    remodel = LogisticRegression()
    rfemodel = RFE(remodel, 4)  #selecting top 4 features
    fitted = rfemodel.fit(X, Y)
    
    count = 0
    print('Calculated scores for given features')
    f.write('\nCalculated scores for given features')
    
    for i in fitted.ranking_:
        print("Column Name: ", names[count] ," Weight: ", i)
        f.write("\nColumn Name: %s Weight: %f " % (names[count] ,i))
        count+=1
    print("Selected Features: %s" % fitted.support_)
    f.write("\nSelected Features: %s" % fitted.support_)
    
    arr= fitted.ranking_
    arr1 = names
    arr2 = [0,0,0,0]
    
    
    ###############################################################################
    ##### Obtaining those colmun through sorting which yields highest weight######
    ###############################################################################
    
    count = 0
    for i in range(0,len(arr)):
        if arr[i] == 1:
            print(i)
            arr2[count]= i
            temp = arr1[i]
            arr1[i] = arr1[count]
            arr1[count] = temp
            count += 1
    
    print("\nSelected Features are: ")
    f.write("\nSelected Features are: ")
    
    for i in range(0,4):
        print(arr1[i])
        f.write(arr1[i])
        f.write('\n')
    
    S = X[:, [arr2[0],arr2[1],arr2[2],arr2[3]]] ##selecting the 4 highest weight columns 
    
    return S   ## returning the selected features identified by the scheme   #####

def resetvalues(var):
    X = datavalues[:,0:8]
    Y = datavalues[:,8]
    tsize=0.44
    return (X,Y,tsize)

##########################################################
################## ODDS RATIO Implementation ############################


def oddsratio(datasetcopy,fname):    
    f=fname
    print("Using Odds Ratio for feature selection...")
    print("Choosing 4 attributes out of 7\n")
    f.write("\nUsing Odds Ratio for feature selection...")
    f.write("\nChoosing 4 attributes out of 7\n")


    x1 = datasetcopy.pregnent.values.reshape(699,1)
    x2 = datasetcopy.plas.values.reshape(699,1)
    x3 = datasetcopy.pres.values.reshape(699,1)
    x4 = datasetcopy.skin.values.reshape(699,1)
    x5 = datasetcopy.test.values.reshape(699,1)
    x6 = datasetcopy.mass.values.reshape(699,1)
    x7 = datasetcopy.pedi.values.reshape(699,1)
    x8 = datasetcopy.age.values.reshape(699,1)
    y1 = datasetcopy.clas.values
    
################################################################################################
###### calculating the odds ratio for each individual column ###################################    
################################################################################################


###########################################################################3
####### By taking the exponent of the coeffecients we can get the odds ratio  #################3
############################################################################    
    
    
    ormodel = LogisticRegression(C=1e5)
    ormodel.fit(x1,y1)
    
    ormodel1 = LogisticRegression(C=1e5)
    ormodel1.fit(x2,y1)
    
    ormodel2 = LogisticRegression(C=1e5)
    ormodel2.fit(x3,y1)
    
    ormodel3 = LogisticRegression(C=1e5)
    ormodel3.fit(x4,y1)
    np.exp(ormodel3.coef_)
    
    ormodel4 = LogisticRegression(C=1e5)
    ormodel4.fit(x5,y1)
    np.exp(ormodel4.coef_)
    
    ormodel5 = LogisticRegression(C=1e5)
    ormodel5.fit(x6,y1)
    np.exp(ormodel5.coef_)
    
    ormodel6 = LogisticRegression(C=1e5)
    ormodel6.fit(x7,y1)
    np.exp(ormodel6.coef_)
    
    ormodel7 = LogisticRegression(C=1e5)
    ormodel7.fit(x8,y1)
    np.exp(ormodel7.coef_)
    
    
    #d = np.concatenate((np.exp(ormodel.coef_) , np.exp(ormodel1.coef_)),axis=0)
    #d = np.concatenate((d , np.exp(ormodel2.coef_)),axis=0)
    #d = np.concatenate((d , np.exp(ormodel3.coef_)),axis=0)
    #d = np.concatenate((d , np.exp(ormodel4.coef_)),axis=0)
    #d = np.concatenate((d , np.exp(ormodel5.coef_)),axis=0)
    #d = np.concatenate((d , np.exp(ormodel6.coef_)),axis=0)
    #d = np.concatenate((d , np.exp(ormodel7.coef_)),axis=0)
    
    #print(d)
    
    
    multimap = []
    multimap.append( np.exp(ormodel.coef_))
    multimap.append( np.exp(ormodel1.coef_))
    multimap.append( np.exp(ormodel2.coef_))
    multimap.append( np.exp(ormodel3.coef_))
    multimap.append( np.exp(ormodel4.coef_))
    multimap.append( np.exp(ormodel5.coef_))
    multimap.append( np.exp(ormodel6.coef_))
    multimap.append( np.exp(ormodel7.coef_))
    
    
    print('Calculated scores for given features')
    f.write('\nCalculated scores for given features')
    count = 0
    for i in multimap:
        print("Column Name: ", names[count] ," Weight: ", i)
        f.write("\nColumn Name: %s Weight: %f " % (names[count] ,i))
        count+=1
    arr = multimap
    arr1 = names
    arr2 = [0,1,2,3,4,5,6,7]
  
    ###############################################################################
    ##### Obtaining those colmun through sorting which yields highest weight######
    ###############################################################################
    
    for i in range(0,len(arr)-1):
        for j in range(i,len(arr)):
            if arr[i] < arr[j]:
                temp2=arr2[j]
                temp1=arr1[j]
                temp= arr[j]
                arr[j]= arr[i]
                arr1[j]= arr1[i]
                arr2[j]= arr2[i]
                arr[i]= temp
                arr1[i]= temp1
                arr2[i]= temp2
     
       
    print("\nSelected Features are: ")
    f.write("\nSelected Features are: ")
    
    for i in range(0,4):
        print(arr1[i])
        f.write(arr1[i])
        f.write('\n')
        
    
    S = X[:, [arr2[0],arr2[1],arr2[2],arr2[3]]] ##selecting the 4 highest weight columns 
    
    return S   ## returning the selected features identified by the scheme   #####
    

##########################################################
################## INFORMATION GAIN Implementation ############################

def informationgain(X , Y,fname):
    f=fname
    print("Using Information Gain for feature selection...")
    print("Choosing 4 attributes out of 7\n")
    f.write("\nUsing Information Gain for feature selection...")
    f.write("\nChoosing 4 attributes out of 7\n")
    
    
    etcmodel = ExtraTreesClassifier()
    etcmodel.fit(X, Y)
    print('Calculated scores for given features')
    
    f.write('\nCalculated scores for given features')
    count = 0
    for i in etcmodel.feature_importances_:
        print("Column Name: ", names[count] ," Weight: ", i)
        f.write("\nColumn Name: %s Weight: %f " % (names[count] ,i))
        count+=1
    
    arr = etcmodel.feature_importances_
    arr1 = names
    arr2 = [0,1,2,3,4,5,6,7]
  
    ###############################################################################
    ##### Obtaining those colmun through sorting which yields highest weight######
    ###############################################################################
    
    for i in range(0,len(arr)-1):
        for j in range(i,len(arr)):
            if arr[i] < arr[j]:
                temp2=arr2[j]
                temp1=arr1[j]
                temp= arr[j]
                arr[j]= arr[i]
                arr1[j]= arr1[i]
                arr2[j]= arr2[i]
                arr[i]= temp
                arr1[i]= temp1
                arr2[i]= temp2
        
    print("\nSelected Features are: ")
    f.write("\nSelected Features are: ")
    
    for i in range(0,4):
        print(arr1[i])
        f.write(arr1[i])
        f.write('\n')
        
    
    S = X[:, [arr2[0],arr2[1],arr2[2],arr2[3]]] ##selecting the 4 highest weight columns 
    
    return S   ## returning the selected features identified by the scheme   #####


    

##########################################################
################## Reading Data ############################

X = datavalues[:,0:8]
Y = datavalues[:,8]

##############  Odds Ratio    ################################################3
##############################################################################
X = oddsratio(data,f2)
X_train , X_test , Y_train , Y_test = train_test_split(X, Y, test_size=tsize, random_state=42, stratify=Y)
p1 = perceptron(X_train , X_test , Y_train , Y_test,f2)

X , Y ,tsize= resetvalues(1)
X = oddsratio(data,f1)
X_train , X_test , Y_train , Y_test = train_test_split(X, Y, test_size=tsize, random_state=42, stratify=Y)
nb1 = naivebayers(X_train , X_test , Y_train , Y_test,f1)

X , Y ,tsize= resetvalues(1)
X = oddsratio(data,f3)
X_train , X_test , Y_train , Y_test = train_test_split(X, Y, test_size=tsize, random_state=42, stratify=Y)
ls1 = linearsvm(X_train , X_test , Y_train , Y_test,f3)


##############  Recursive Elimination    ################################################3
##############################################################################

X , Y ,tsize= resetvalues(2)
X=recursiveelimination(X,Y,f2)
X_train , X_test , Y_train , Y_test = train_test_split(X, Y, test_size=tsize, random_state=42, stratify=Y)
p2 = perceptron(X_train , X_test , Y_train , Y_test,f2)

X , Y ,tsize= resetvalues(2)
X=recursiveelimination(X,Y,f1)
X_train , X_test , Y_train , Y_test = train_test_split(X, Y, test_size=tsize, random_state=42, stratify=Y)
nb2 = naivebayers(X_train , X_test , Y_train , Y_test,f1)

X , Y ,tsize= resetvalues(2)
X=recursiveelimination(X,Y,f3)
X_train , X_test , Y_train , Y_test = train_test_split(X, Y, test_size=tsize, random_state=42, stratify=Y)
ls2 = linearsvm(X_train , X_test , Y_train , Y_test,f3)

##############  Information Gain    ################################################3
##############################################################################

X , Y ,tsize= resetvalues(3)
X = informationgain(X,Y,f2)
X_train , X_test , Y_train , Y_test = train_test_split(X, Y, test_size=tsize, random_state=42, stratify=Y)
p3 = perceptron(X_train , X_test , Y_train , Y_test,f2)

X , Y ,tsize= resetvalues(3)
X = informationgain(X,Y,f1)
X_train , X_test , Y_train , Y_test = train_test_split(X, Y, test_size=tsize, random_state=42, stratify=Y)
nb3 = naivebayers(X_train , X_test , Y_train , Y_test,f1)

X , Y ,tsize= resetvalues(3)
X = informationgain(X,Y,f3)
X_train , X_test , Y_train , Y_test = train_test_split(X, Y, test_size=tsize, random_state=42, stratify=Y)
ls3 = linearsvm(X_train , X_test , Y_train , Y_test,f3)


#########   LINEAR CLASSIFIER WEIGHTS #########################################
##############   i.e training perceptron and SVM moidel   #####################3
##############################################################################

print('Using Linear Classifier Weights: ')
f2.write('\nUsing Linear Classifier Weights: ')

X , Y ,tsize= resetvalues(4)
X_train , X_test , Y_train , Y_test = train_test_split(X, Y, test_size=tsize, random_state=42, stratify=Y)
p4 = perceptron(X_train , X_test , Y_train , Y_test,f2)


X , Y ,tsize= resetvalues(4)
X_train , X_test , Y_train , Y_test = train_test_split(X, Y, test_size=tsize, random_state=42, stratify=Y)
print('Using Linear Classifier Weights: ')
f1.write('\nUsing Linear Classifier Weights: ')
nb4 = naivebayers(X_train , X_test , Y_train , Y_test,f1)

X , Y ,tsize= resetvalues(4)
X_train , X_test , Y_train , Y_test = train_test_split(X, Y, test_size=tsize, random_state=42, stratify=Y)

print('Using Linear Classifier Weights: ')
f3.write('\nUsing Linear Classifier Weights: ')
ls4 = linearsvm(X_train , X_test , Y_train , Y_test,f3)

##############  Univariate Selection    ################################################3
##############################################################################

X , Y ,tsize= resetvalues(5)
X = fsunvariateselection(X,Y,f2)
X_train , X_test , Y_train , Y_test = train_test_split(X, Y, test_size=tsize, random_state=42, stratify=Y)
p5 = perceptron(X_train , X_test , Y_train , Y_test,f2)

X , Y ,tsize= resetvalues(5)
X = fsunvariateselection(X,Y,f1)
X_train , X_test , Y_train , Y_test = train_test_split(X, Y, test_size=tsize, random_state=42, stratify=Y)
nb5 = naivebayers(X_train , X_test , Y_train , Y_test,f1)

X , Y ,tsize= resetvalues(5)
X = fsunvariateselection(X,Y,f3)
X_train , X_test , Y_train , Y_test = train_test_split(X, Y, test_size=tsize, random_state=42, stratify=Y)
ls5 = linearsvm(X_train , X_test , Y_train , Y_test,f3)


############   Naive bayers Graph  ##########################333

obj=("OR","RE","IG","LCF","UCF")
ypos=np.arange(len(obj))
s=pd.Series(
    [nb1,nb2,nb3,nb4,nb5],
    index=["OR","RE","IG","LCF","UCF"]
)

plt.ylabel('Accuracy Percentage')
plt.title('Naive Bayers')
plt.xticks(ypos,obj)

ypos=np.arange(len(obj))
perf = [nb1,nb2,nb3,nb4,nb5]

plt.bar(ypos, perf, align='center', alpha=1,color='rgby',)
plt.show()


############    Linear SVM Graph  ##########################333

obj=("OR","RE","IG","LCF","US")
ypos=np.arange(len(obj))
s=pd.Series(
    [ls1,ls2,ls3,ls4,ls5],
    index=["OR","RE","IG","LCF","US"]
)

plt.ylabel('Accuracy Percentage')
plt.title('Lnear SVM')
plt.xticks(ypos,obj)

ypos=np.arange(len(obj))
perf = [ls1,ls2,ls3,ls4,ls5]

plt.bar(ypos, perf, align='center', alpha=1,color='rgby',)
plt.show()

############    Perceptron Graph  ##########################333

obj=("OR","RE","IG","LCF","US")
ypos=np.arange(len(obj))
s=pd.Series(
    [p1,p2,p3,p4,p5],
    index=["OR","RE","IG","LCF","US"]
)

plt.ylabel('Accuracy Percentage')
plt.title('Peerceptron')
plt.xticks(ypos,obj)

ypos=np.arange(len(obj))
perf = [p1,p2,p3,p4,p5]

plt.bar(ypos, perf, align='center', alpha=1,color='rgby',)
plt.show()

f1.close
f2.close
f3.close