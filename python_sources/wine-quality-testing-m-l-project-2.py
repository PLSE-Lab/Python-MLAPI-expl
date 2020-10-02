# %% [code]
# White Wine Quality Machine Learning Project 2
import pandas as pd
pd.set_option("display.max_columns",200)
pd.set_option('display.max_colwidth', None)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split

wine = pd.read_csv('S:/SURAJ_STUFF/Spyder/wine_dataset.csv')
wine.head()
white_wine = wine[wine['style']=='white']

white_wine.head()
#%%

#------# Exploratory Data Analysis (EDA)----
white_wine.info()
white_wine = white_wine.iloc[:,0:-1]
white_wine.isnull().sum()
white_wine.describe() #gives mean,max,avg,quintiles
# Obs 1 = Outliers in "residual sugar", "free sulphur dioxide", "total sulphur dioxide"

#--seeing dependable variable
white_wine['quality'].unique() # there is no 1,2,10
white_wine['quality'].value_counts()
# Obs 2 = imbalanced Data set

#  Detecting features which are strongly co-related or not co-related, this can be found by heatmap and .corr() function
plt.rcParams['figure.figsize'] = (50.0, 20.0)
sns.heatmap(white_wine.corr(method ='pearson'), cmap='coolwarm',annot=True)

# Obs 2 = "citric acid" and "free sulphur dioxide" are not corelated to quality
# Obs 3 = "density" and "residual sugar" are highly correlated

# Detecting whether data is skewed or not, this can be done by histograms for individual variable using sub plots
fig, ((ax1, ax2, ax3, ax4),(ax5, ax6, ax7, ax8),(ax9, ax10, ax11, ax12)) = plt.subplots(3, 4,figsize=(12,12))
axs = [ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10,ax11,ax12]
cols = list(white_wine.columns)
cols
for n in range(0,len(axs)):
    axs[n].hist(white_wine.iloc[:,n],bins=50)
    axs[n].set_title('name={}'.format(cols[n])) 
# Obs 4 = "volatile acidity" and "residual sugar" are right skewed

###-----Since this is imbalanced class dataset we will try to achieve higher recall and use roc_auc as evaluation metric----###
    
#%%
    
#-- removing features which are highly correlated or zero correlated:  ('citric acid','free sulfur dioxide')
col = ['fixed_acidity','volatile_acidity','residual_sugar','chlorides','total_sulfur_dioxide','density','sulphates', 'alcohol']    
white_wine.head()
# Working with right skewed data--> cube root 
white_wine_mod = pd.DataFrame()
for n in col:
    white_wine_mod[col] = white_wine[col].apply(lambda x: np.cbrt(x))
white_wine_mod['pH']=white_wine['pH']
white_wine_mod['quality']=white_wine['quality']

# Working with normal data
white_wine_nor = white_wine[['fixed_acidity','volatile_acidity','residual_sugar','chlorides','total_sulfur_dioxide','density','sulphates', 'alcohol','pH','quality']]  

#%%
#-- detecting outliers, transforming data can also help in handling outliers.
z = np.abs(stats.zscore(white_wine_mod)) # Calculating Z-score
threshold = 3
np.where(z > 3)

white_wine_mod_o = white_wine_mod[(z < 3).all(axis=1)]
white_wine_mod_o.shape # (4616, 10) # without outliers
white_wine_mod.shape #(4898, 10)  # with outliers

z1 = np.abs(stats.zscore(white_wine_nor))
white_wine_nor_o = white_wine_nor[(z1 < 3).all(axis=1)] 
white_wine_nor_o.shape #(4573,10)

#%%
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from statistics import mean
# Defining evaluate model

def evaluate_model(X, y, model):
	# define evaluation procedure
	cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=2, random_state=1)
	# evaluate model
	scores = cross_val_score(model, X, y, scoring='roc_auc_ovr', cv=cv, n_jobs=-1)
	return scores

X = white_wine_nor_o.iloc[:,0:9] #using normal data set without outliers
y = white_wine_nor_o.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


# Defining Dummy Base Line result
model = DummyClassifier(strategy='most_frequent')   
dummy_train_scores = mean(evaluate_model(X, y, model)) #0.5


#%%
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
# Defining function to evaluate different models to get roc_auc 
def get_models():
    models, names = list(), list()
    #SVM
    models.append(SVC(kernel='rbf',gamma='auto',probability=True))
    names.append('SVM')
    #KNN
    models.append(KNeighborsClassifier())
    names.append('KNN')
    #RandomForrest
    models.append(RandomForestClassifier(n_estimators=200))
    names.append('RF')
    return models, names

models, names = get_models()
results = pd.DataFrame(columns=['Model_name','ROC_AUC'])
dn=[]

# evaluate each model
for i in range(len(models)):
	# evaluate the model and store results
    print(models[i])
    scores = mean(evaluate_model(X, y, models[i]))
    df1 = pd.DataFrame(data = [names[i],scores]) 
    dn.append(df1)                              
results = pd.concat(dn,axis=1)

results
#%%
# Since Random Forrest has highest roc_auc, we will evaluate and optimize RF 

# Random Forrest Model-1

#Confusion Matrix
from sklearn.metrics import confusion_matrix
model_RF = RandomForestClassifier(n_estimators=200)
# evaluate the model
scores = mean(evaluate_model(X, y, model_RF))

model_RF.fit(X_train,y_train)
RF_predicted = model_RF.predict(X_test)
confusion_mc = confusion_matrix(y_test, RF_predicted)
df_cm = pd.DataFrame(confusion_mc, 
                     index = [ 4, 5, 6, 7, 8], columns = [ 4, 5, 6, 7, 8])

plt.figure(figsize=(8,8))
sns.heatmap(df_cm, annot=True, fmt='g')
plt.title('RFLinear Kernel Accuracy:{0:.3f} \n Roc_Auc:{0:.3f}'.format(accuracy_score(y_test,RF_predicted)))
plt.ylabel('True label')
plt.xlabel('Predicted label')

#%%
from sklearn import metrics

# Clssification report

print(metrics.classification_report(y_test, RF_predicted.round(), digits=3))

# Calculate cv score with 'roc_auc_ovr' scoring and 5 folds

accuracy = cross_val_score(model_RF, X, y,scoring = 'roc_auc_ovr',cv=5)
print('cross validation score with roc_auc_ovr scoring',accuracy.mean())

#%%
# Since this is imbalanced dataset we will add samples which has minority in the dataset
# Random Forrest Model-2 with SMOTE 

from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state = 2) 
X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel())

model_RF2 = RandomForestClassifier(n_estimators=200,random_state = 0)
model_RF2.fit(X_train_res, y_train_res)
RF2_predicted = model_RF2.predict(X_test)

# Classification report
print(metrics.classification_report(y_test, RF2_predicted.round(), digits=3))
# Calculate cv score with 'roc_auc_ovr' scoring and 10 folds
accuracy = cross_val_score(model_RF2, X, y,scoring = 'roc_auc_ovr',cv=10)
print('cross validation score with roc_auc_ovr scoring',accuracy.mean())

