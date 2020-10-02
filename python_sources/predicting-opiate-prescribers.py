
# coding: utf-8

# # Detecting Frequent Opiate Prescription
#This classification model based upon to check whether the doctor Opiate prescriber or not
# # Importing all the packages

# In[1]:

import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier,GradientBoostingClassifier
from sklearn.metrics import roc_curve, auc, accuracy_score, log_loss, roc_auc_score,confusion_matrix
from sklearn.linear_model import LogisticRegression


# # Data Preprocessing 

# Upload the csv files and remove NaN values

# In[2]:

pd.set_option('display.max_columns',None,'display.max_rows',None)


# In[3]:

prescriber = pd.read_csv('prescriber-info.csv')


# In[4]:

prescriber.columns


# In[5]:

prescriber.isnull().sum()


# In[6]:

prescriber.dropna(inplace=True)


# In[7]:

len(prescriber['State'].unique())


# In[8]:

len(prescriber['Credentials'].unique())


# In[9]:

len(prescriber['Specialty'].unique())


# In[10]:

prescriber.dtypes


# In[11]:

opioid = pd.read_csv('opioids.csv')


# In[12]:

opioid


# In[13]:

opioid[opioid['Generic Name'].str.contains('NOPHEN',case=False)]


# In[14]:

prescriber['Opioid.Prescriber'].value_counts()


# In[15]:

prescriber.columns


# In[16]:

prescriber.dtypes


# # Feature Selection 

# Feature Selection Using Light GBM

# In[17]:

dfcopy=prescriber.copy()


# In[18]:

dfcopy.dtypes


# In[19]:

dfcopy.shape


# In[20]:

dfcopy.drop(['Opioid.Prescriber'],axis=1,inplace=True)


# In[21]:

catList= ['Gender', 'State', 'Credentials', 'Specialty']
for i in catList:
    dfcopy[i]=dfcopy[i].astype('category')


# In[22]:


d_train = lgb.Dataset(dfcopy, label = prescriber['Opioid.Prescriber'])
params = {"max_depth":5, "learning_rate":0.1,"num_leaves":900,"n_estimators":100}
model2 = lgb.train(params = params,train_set = d_train,categorical_feature = catList)
ax = lgb.plot_importance(model2,max_num_features=256,figsize=(100,60))
plt.show()


# In[23]:

prescriber.isnull().sum()


# In[24]:

model2.feature_importance()


# The features ranked above 28 are considered for the model

# In[25]:

feature = list(model2.feature_importance())
effe_cols = []
for i in feature:
    if i >= 28:
        effe_cols.append(model2.feature_name()[feature.index(i)])
        feature[feature.index(i)]=0
prescrib = prescriber[effe_cols]
prescrib.head()


# In[26]:

prescrib['State'].unique()


# In[27]:

len(prescrib['Credentials'].unique())


# In[28]:

len(prescrib['Specialty'].unique())


# In[29]:

del prescrib['NPI']


# In[30]:

prescrib.columns


# In[31]:

categorical=pd.read_csv('categorical.csv')


# In[32]:

categorical.head()


# # Label Encoding

# In[33]:

lb = LabelEncoder()


# In[34]:

pres_copy = prescrib.copy()


# In[35]:

pres_copy.head()


# In[36]:

pres_copy.iloc[:,0]=lb.fit_transform(pres_copy.iloc[:,1])


# In[37]:

pres_copy.iloc[:,1]=lb.fit_transform(pres_copy.iloc[:,2].astype(str))


# In[38]:

pres_copy.iloc[:,2]=lb.fit_transform(pres_copy.iloc[:,3])


# In[39]:

pres_copy.head(10)


# In[40]:

pres_copy['Opioid.Prescriber']=prescriber['Opioid.Prescriber']


# In[41]:

pres_copy.isnull().sum()


# In[42]:

del pres_copy['Opioid.Prescriber']


# In[43]:

pres_copy.dtypes


# In[44]:

pres_copy.head()


# In[45]:

x = pres_copy.values
x.shape


# In[46]:

y = prescriber.iloc[:,-1].values
y.shape


# In[47]:

x_train,x_test,y_train,y_test = model_selection.train_test_split(x,y,test_size=0.2,random_state=42)


# In[48]:

x_train


# In[49]:

len(prescrib.columns)


# In[50]:

pres_copy.columns


# # Model Evaluation

# In[51]:

models = []
name = []
results = []
models.append(('Decision Tree',DecisionTreeClassifier()))
models.append(('Random Forest',RandomForestClassifier()))
models.append(('AdaBoost', AdaBoostClassifier()))
models.append(('Gradient Boosting',GradientBoostingClassifier()))
for name, model in models:
    kfold = model_selection.KFold(n_splits = 10, random_state = 42)
    cv_results = model_selection.cross_val_score(model,x_train,y_train,cv = kfold, scoring = 'accuracy')
    results.append(cv_results)
    msg = "%s: %f (std: %f)" % (name,cv_results.mean(),cv_results.std())
    print(msg)


# Graph showing comparison among different algorithms

# In[52]:

sns.boxplot( x = ['Decsion\nTree','Random\nForest','AdaBoost','Gradient\nBoosting'] , y= results, palette="rainbow")


# Comparison of different algorithms for roc_auc scores

# In[53]:

models = []
name = []
models.append(('Decision Tree',DecisionTreeClassifier()))
models.append(('Random Forest',RandomForestClassifier()))
models.append(('AdaBoost', AdaBoostClassifier()))
models.append(('Gradient Boosting',GradientBoostingClassifier()))
for name, model in models:
    kfold = model_selection.KFold(n_splits = 10, random_state = 42)
    cv_results = model_selection.cross_val_score(model,x_train,y_train,cv = kfold, scoring = 'roc_auc')
    msg = "%s: %f (std: %f)" % (name,cv_results.mean(),cv_results.std())
    print(msg)


# As ROC score for Gradient Boosting classifier looks better, this classifier is chosen for model building.
# Hyper Parameter tuning could be done to improve accuracy further.

# # Model Building

# In[54]:

gb = GradientBoostingClassifier()
gb.fit(x_train, y_train)
predictions = gb.predict(x_test)
print('Accuracy: ',accuracy_score(y_test, predictions))
print('Log Loss: ',log_loss(y_test, gb.predict_proba(x_test)[:,1]))
print('ROC-AUC: ',roc_auc_score(y_test, gb.predict_proba(x_test)[:,1]))
print("Confusion Matrix:\n",confusion_matrix(y_test, predictions))
pd.crosstab(y_test,predictions,rownames=["True"],colnames=["False"],margins= True)


# In[55]:

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, gb.predict_proba(x_test)[:,1])
roc_auc = auc(false_positive_rate, true_positive_rate)


# In[56]:

plt.plot(false_positive_rate,true_positive_rate, color='fuchsia', label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([-0.01, 1.0])
plt.ylim([0.0, 1.03])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic applied for Gradient Boosting\n')
plt.legend(loc="lower right")
plt.show()


# In[57]:

import matplotlib.pyplot as plt
prescrib.plot(kind='density', subplots=True, layout=(6,6), sharex=False,figsize=(30,10))
plt.show()


# In[58]:

corr = pres_copy.corr()
f, ax = plt.subplots(figsize=(11, 9))
plot = sns.heatmap(corr, cmap='Paired', vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
ax.set_title("Correlation among features")


# In[59]:

import pickle
import os
# create the file paths
model_file_path = os.path.join(os.path.pardir,'final_data','Opiate','gb.pkl')
# open the files to write 
model_file_pickle = open(model_file_path, 'wb')
np.isnan(x_test)
pickle.dump(gb, model_file_pickle)
model_file_pickle.close()


# In[60]:

categorical.to_csv('categorical_columns.csv',index=False)


# In[61]:

prescrib.info()


# In[62]:

model = LogisticRegression()
ac =model.fit(x_train, y_train)
predictions = ac.predict(x_test)
print('Accuracy: ',accuracy_score(y_test, predictions))
print('Log Loss: ',log_loss(y_test, model.predict_proba(x_test)[:,1]))
print('ROC-AUC: ',roc_auc_score(y_test, model.predict_proba(x_test)[:,1]))
print("Confusion Matrix:\n",confusion_matrix(y_test, predictions))
pd.crosstab(y_test,predictions,rownames=["True"],colnames=["False"],margins= True)
model.coef_


# In[63]:

import pickle
import os
# create the file paths
model_file_path = os.path.join(os.path.pardir,'final_data','Opiate','Logistic.pkl')
# open the files to write 
model_file_pickle = open(model_file_path, 'wb')
np.isnan(x_test)
pickle.dump(model, model_file_pickle)
model_file_pickle.close()
model.coef_


# In[64]:

load_model_logreg = pickle.load(open('Logistic.pkl', 'rb'))


# In[65]:

load_model_logreg.coef_


# In[66]:

load_model_logreg.coef_.tolist()


# In[67]:

prescrib.columns


# # User Inputs

# In[68]:

def load_model():
    """This function load the pickled model as API for flask
         :returns: loaded_model
         :rtype: float64
         method's invoked:  None"""
    loaded_model = pickle.load(open('Gradient.pkl', 'rb'))
    return loaded_model
def load_model_lr():
    """This function load the pickled model as API for flask
         :returns: loaded_model
         :rtype: float64
         method's invoked:  None"""
    load_model_logreg = pickle.load(open('Logistic.pkl', 'rb'))
    return load_model_logreg


def total_weight_find(col_values, col_intercept):
    total_weight = 0
    for i in range(len(col_values)):
        total_weight = total_weight + (col_values[i] * col_intercept[0][i])
    return total_weight


# In[69]:

Opiate=pd.DataFrame()


# In[70]:

Opiate


# In[73]:

print("Enter state")
Opiate.loc[0,'state'] = (input())


# In[72]:

print("Enter Credentials")
Opiate.loc[0,'Credentials'] = (input())


# In[74]:

print("Enter Specialty")
Opiate.loc[0,'Specialty'] = (input())


# In[75]:

print("Enter ACETAMINOPHEN.CODEINE")
Opiate.loc[0,'ACETAMINOPHEN.CODEINE'] = (input())


# In[76]:

print("Enter AMOXICILLIN")
Opiate.loc[0,'AMOXICILLIN'] = (input())


# In[77]:

print("Enter FUROSEMIDE")
Opiate.loc[0,'FUROSEMIDE'] = (input())


# In[78]:

print("Enter GABAPENTIN")
Opiate.loc[0,'GABAPENTIN'] = (input())


# In[79]:

print("Enter HYDROCODONE.ACETAMINOPHEN")
Opiate.loc[0,'HYDROCODONE.ACETAMINOPHEN'] = (input())


# In[80]:

print("Enter LEVOTHYROXINE.SODIUM")
Opiate.loc[0,'LEVOTHYROXINE.SODIUM'] = (input())


# In[81]:

print("Enter OMEPRAZOLE")
Opiate.loc[0,'OMEPRAZOLE'] = (input())


# In[82]:

print("Enter OXYCODONE.ACETAMINOPHEN")
Opiate.loc[0,'OXYCODONE.ACETAMINOPHEN'] = (input())


# In[83]:

print("Enter PREDNISONE")
Opiate.loc[0,'PREDNISONE'] = (input())


# In[84]:

print("Enter TRAMADOL.HCL")
Opiate.loc[0,'TRAMADOL.HCL'] = (input())


# In[85]:

state = Opiate.iat[0,0]
Credentials = Opiate.iat[0,1]
Specialty = Opiate.iat[0,2]
Ac = int(Opiate.iat[0,3])
AMOXICILLIN = int(Opiate.iat[0,4])
FUROSEMIDE = int(Opiate.iat[0,5])
GABAPENTIN = int(Opiate.iat[0,6])
Ha = int(Opiate.iat[0,7])
Ls = int(Opiate.iat[0,8])
OMEPRAZOLE = int(Opiate.iat[0,9])
Oa = int(Opiate.iat[0,10])
PREDNISONE = int(Opiate.iat[0,11])
Th = int(Opiate.iat[0,12])


# In[88]:

categorical = pd.read_csv('categorical.csv')
state1 = int(categorical[categorical['State'] == state].iloc[0][3])
Credentials1 = int(categorical[categorical['Credentials'] == Credentials].iloc[1][4])
Specialty1 = int(categorical[categorical['Specialty'] == Specialty].iloc[2][5])
dataframe = ([[state1, Credentials1, Ac, AMOXICILLIN, FUROSEMIDE, GABAPENTIN,
          Ha, Ls, OMEPRAZOLE, Oa, PREDNISONE, Th, Specialty1]])
pdf = pd.DataFrame(dataframe)


# In[89]:

pdf = pdf.rename(columns={0:'State', 1:'Credentials', 2:'Specialty', 3:'ACETAMINOPHEN.CODEINE',
       4:'AMOXICILLIN', 5:'FUROSEMIDE', 6:'GABAPENTIN', 7:'HYDROCODONE.ACETAMINOPHEN',
       8:'LEVOTHYROXINE.SODIUM', 9:'OMEPRAZOLE', 10:'OXYCODONE.ACETAMINOPHEN',
       11:'PREDNISONE', 12:'TRAMADOL.HCL'})


# In[90]:

data = np.array(dataframe)
data


# In[91]:

model = load_model()
data = model.predict(data)


# In[92]:

model_lr = load_model_lr()
col_list = [state1, Credentials1,Specialty1,Ac, AMOXICILLIN, FUROSEMIDE, GABAPENTIN,
            Ha, Ls, OMEPRAZOLE, Oa, PREDNISONE, Th]
intercept_values = model_lr.coef_.tolist()
total_weigh_data = total_weight_find(col_list, intercept_values)


# In[93]:

w_state = ((state1 * intercept_values[0][0]) / total_weigh_data)
w_cred = ((Credentials1 * intercept_values[0][1]) / total_weigh_data)
w_spe = ((Specialty1 * intercept_values[0][2]) / total_weigh_data)
w_Ac = ((Ac * intercept_values[0][3]) / total_weigh_data)
w_am = ((AMOXICILLIN * intercept_values[0][4]) / total_weigh_data)
w_fu = ((FUROSEMIDE * intercept_values[0][5]) / total_weigh_data)
w_ga = ((GABAPENTIN * intercept_values[0][6]) / total_weigh_data)
w_ha = ((Ha * intercept_values[0][7]) / total_weigh_data)
w_ls = ((Ls * intercept_values[0][8]) / total_weigh_data)
w_om = ((OMEPRAZOLE * intercept_values[0][9]) / total_weigh_data)
w_oa = ((Oa * intercept_values[0][10]) / total_weigh_data)
w_pr = ((PREDNISONE * intercept_values[0][11]) / total_weigh_data)
w_th = ((Th * intercept_values[0][12]) / total_weigh_data)


# In[94]:

total_pv = abs(w_state) + abs(w_cred) + abs(w_Ac) + abs(w_spe) + abs(w_am) + abs(w_fu) + abs(w_ga) + abs(
    w_ha) + abs(w_ls) + abs(w_om) + abs(w_oa) + abs(w_pr) + abs(w_th)
p_state = (w_state / total_pv) * 100
p_cred = (w_cred / total_pv) * 100
p_spe = (w_spe / total_pv) * 100
p_Ac = (w_Ac / total_pv) * 100
p_am = (w_am / total_pv) * 100
p_fu = (w_fu / total_pv) * 100
p_ga = (w_ga / total_pv) * 100
p_ha = (w_ha / total_pv) * 100
p_ls = (w_ls / total_pv) * 100
p_om = (w_om / total_pv) * 100
p_oa = (w_oa / total_pv) * 100
p_pr = (w_pr / total_pv) * 100
p_th = (w_th / total_pv) * 100


# In[95]:

vb = [abs(round(p_state, 2)), abs(round(p_cred, 2)), abs(round(p_spe, 2)), abs(round(p_Ac, 2)),
      abs(round(p_am, 2)), abs(round(p_fu, 2)),abs(round(p_ga, 2)), abs(round(p_ha, 2)),
      abs(round(p_ls, 2)), abs(round(p_om, 2)), abs(round(p_oa, 2)),abs(round(p_pr, 2)), abs(round(p_th, 2))]
pro_col = [{"State": abs(round(p_state, 2)), "Credentials": abs(round(p_cred, 2)),
             "Specialty": abs(round(p_spe, 2)), "ACETAMINOPHEN.CODEINE": abs(round(p_Ac, 2)),
             "AMOXICILLIN": abs(round(p_am, 2)), "FUROSEMIDE": abs(round(p_fu, 2)),
             "GABAPENTIN": abs(round(p_ga, 2)), "HYDROCODONE.ACETAMINOPHEN": abs(round(p_ha, 2)),
              "LEVOTHYROXINE.SODIUM": abs(round(p_ls, 2)), "OMEPRAZOLE": abs(round(p_om, 2)),
              "OXYCODONE.ACETAMINOPHEN": abs(round(p_oa, 2)),
               "PREDNISONE": abs(round(p_pr, 2)), "TRAMADOL.HCL": abs(round(p_th, 2))}]


# In[99]:

pro_col


# In[96]:

predicted_data = model.predict(pdf.values.astype('float'))
if data == 1:
    result = "Yes"
else:
    result = "No"


# Based Upon the user inputs the graph will be generated

# In[102]:

se=sns.barplot(y=vb,x=pdf.columns)
sns.set(rc={'figure.figsize':(40,8.27)})
k = 0
for p in se.patches:
    se.text(p.get_x()+p.get_width()/2.,
    p.get_height()+0.02,
    '{:1.1f}%'.format(vb[k]),
    ha="center",color='black',fontsize=30)
    k = k+1



# In[100]:




# In[ ]:



