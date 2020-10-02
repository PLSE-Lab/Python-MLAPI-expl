#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')
Data = data


# In[ ]:


data.head()
test_data.head()


# In[ ]:


data.info()


# In[ ]:


#finding categorical and continuous features
feature_values = {}
data = data.replace('?',np.nan)
test_data = test_data.replace('?', np.nan)
for i in data.columns:
    print(i,data[i].unique())
    feature_values[i] = len(data[i].unique())
    


# In[ ]:


print(feature_values)
print(len(feature_values))


# In[ ]:


data.isnull().sum()
null_columns = ['Worker Class', 'Enrolled', 'MIC', 'MOC', 'Hispanic', 'MLU', 'Reason', 'Area', 'State', 'MSA', 'REG', 'MOVE', 'Live', 'PREV', 'Teen', 'COB MOTHER', 'COB FATHER', 'COB SELF', 'Fill']
len(null_columns)


# In[ ]:


#plotting the correlation matrix to visualise correlation between the class and features
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
#corelation heat map
df = data
corr = df.corr(method="kendall")

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(25, 25))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0.5,
            square=True, linewidths=.5, cbar_kws={"shrink": .5},annot=True)

plt.show()


# In[ ]:


#count number of 0's and 1's
data.groupby('Class').size()


# In[ ]:


#feature selection
data = data.drop(['ID', 'Timely Income', 'Weight', 'WorkingPeriod', 'Detailed'], axis = 1)
data = data.drop(columns = null_columns)

test_data1 = test_data
test_data1 = test_data1.drop(['ID', 'Timely Income', 'Weight', 'WorkingPeriod', 'Detailed'], axis = 1)
test_data1 = test_data1.drop(columns = null_columns)
test_data1.head()


# In[ ]:


data.head()


# In[ ]:


#using get dummies to encode data
categorical = ['Schooling','Married_Life','Cast', 'Sex', 'Full/Part', 'Tax Status', 'Summary', 'Citizen']
data_one_hot_encoded = pd.get_dummies(data, columns = categorical)
#data_one_hot_encoded.info()


# In[ ]:


#label encoding

from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode

    def fit(self,X,y=None):
        return self # not relevant here

    def transform(self,X):
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)


# In[ ]:


data_label_encoded = MultiColumnLabelEncoder(columns = categorical).fit_transform(data)
test_data_label_encoded = MultiColumnLabelEncoder(columns = categorical).fit_transform(test_data1)

test_data_label_encoded.info()


# In[ ]:


y = data['Class']
X = data.drop(['Class'], axis=1)
X.head()

y_label_encoded = data_label_encoded['Class']
X_label_encoded = data_label_encoded.drop(['Class'], axis = 1)

#t_X_label_encoded = test_data_label_encoded.drop(['Class'], axis = 1)

y_one_hot = data_one_hot_encoded['Class']
X_one_hot = data_one_hot_encoded.drop(['Class'], axis = 1)
X_one_hot.shape


# In[ ]:


#Using undersampling
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(random_state=42)
X_res, y_res = rus.fit_resample(X_label_encoded, y_label_encoded)
X_res.shape


# In[ ]:


'''
from imblearn.under_sampling import TomekLinks
tl = TomekLinks(ratio='majority')
X_tl, y_tl = tl.fit_sample(X_label_encoded, y_label_encoded)


# In[ ]:


from imblearn.over_sampling import SMOTE

smote = SMOTE(ratio='minority')
X_sm, y_sm = smote.fit_sample(X_label_encoded, y_label_encoded)


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.20, random_state=42)


# In[ ]:


#Performing Min_Max Normalization
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()

np_scaled = min_max_scaler.fit_transform(X_train)
X_train = pd.DataFrame(np_scaled)

np_scaled_val = min_max_scaler.transform(X_test)
X_test = pd.DataFrame(np_scaled_val)
#X_train.head()

np_scaled_test = min_max_scaler.fit_transform(test_data_label_encoded)
x_final_test = pd.DataFrame(np_scaled_test)


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

#LogisticRegression
lg = LogisticRegression(solver = 'liblinear', C = 1, multi_class = 'ovr', random_state = 42)
lg.fit(X_train,y_train)
lg.score(X_test,y_test)
y_pred = lg.predict(X_test)
roc_auc_score(y_pred, y_test)


# In[ ]:


#Naive Bayes
from sklearn.naive_bayes import GaussianNB as NB
nb = NB()
nb.fit(X_train,y_train)
y_pred = nb.predict(X_test)
roc_auc_score(y_pred, y_test)


# In[ ]:


#Decision Tree Classifier 
from sklearn import tree
dtc = tree.DecisionTreeClassifier(random_state=1)
dtc = dtc.fit(X_train, y_train)
y_pred = dtc.predict(X_test)
roc_auc_score(y_pred, y_test)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
#RandomForestClassifier
rf = RandomForestClassifier(n_estimators=13, random_state = 42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
roc_auc_score(y_pred, y_test)


# In[ ]:


#Adaboost
from sklearn.ensemble import AdaBoostClassifier

ada = AdaBoostClassifier(rf,n_estimators=10)
ada.fit(X_train, y_train)
ada.score(X_test,y_test)
y_pred = ada.predict(X_test)
roc_auc_score(y_pred, y_test)


# In[ ]:


from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score

scorer = make_scorer(roc_auc_score)
cv_results = cross_validate(rf, X_train, y_train, cv=10, scoring=(scorer), return_train_score=True)
print cv_results.keys()
print"Train Accuracy for 3 folds= ",np.mean(cv_results['train_score'])
print"Validation Accuracy for 3 folds = ",np.mean(cv_results['test_score'])


# In[ ]:


from sklearn.model_selection import GridSearchCV

rf_temp = DecisionTreeClassifier(random_state=1)        #Initialize the classifier object

parameters = {'max_depth':[3,4, 5, 8, 10, 11, 12],'min_samples_split':[2, 3, 4, 5, 6, 7, 8, 9]}    #Dictionary of parameters

scorer = make_scorer(roc_auc_score)         #Initialize the scorer using make_scorer

grid_obj = GridSearchCV(rf_temp, parameters, scoring=scorer)         #Initialize a GridSearchCV object with above parameters,scorer and classifier

grid_fit = grid_obj.fit(X_train, y_train)        #Fit the gridsearch object with X_train,y_train

best_rf = grid_fit.best_estimator_         #Get the best estimator. For this, check documentation of GridSearchCV object

print(grid_fit.best_params_)


# In[ ]:


from sklearn.model_selection import GridSearchCV

rf_temp = RandomForestClassifier(n_estimators = 13)        #Initialize the classifier object

parameters = {'max_depth':[3, 5, 8, 10, 9, 11, 12],'min_samples_split':[2, 3, 4, 5, 6, 7, 8, 9]}    #Dictionary of parameters

scorer = make_scorer(roc_auc_score)         #Initialize the scorer using make_scorer

grid_obj = GridSearchCV(rf_temp, parameters, scoring=scorer)         #Initialize a GridSearchCV object with above parameters,scorer and classifier

grid_fit = grid_obj.fit(X_train, y_train)        #Fit the gridsearch object with X_train,y_train

best_rf = grid_fit.best_estimator_         #Get the best estimator. For this, check documentation of GridSearchCV object

print(grid_fit.best_params_)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
#RandomForestClassifier Final(?)
rf = RandomForestClassifier(n_estimators=13, random_state = 42, min_samples_split=7, max_depth=12)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
roc_auc_score(y_pred, y_test)
y_pred = rf.predict(x_final_test)
np.count_nonzero(y_pred)


# In[ ]:


#Decision Tree Classifier (Final?)
from sklearn import tree
dtc = tree.DecisionTreeClassifier(random_state=1, min_samples_split=2, max_depth=5)
dtc = dtc.fit(X_train, y_train)
y_pred = dtc.predict(X_test)
roc_auc_score(y_pred, y_test)
np.count_nonzero(y_pred)


# In[ ]:


#Adaboost
from sklearn.ensemble import AdaBoostClassifier

ada = AdaBoostClassifier(rf,n_estimators=10)
ada.fit(X_train, y_train)
ada.score(X_test,y_test)
y_pred = ada.predict(X_test)
roc_auc_score(y_pred, y_test)


# In[ ]:


y_pred_final = ada.predict(x_final_test)
y_pred_final.shape


# In[ ]:


test_data['Class'] = y_pred_final
test_data.groupby('Class').size()


# In[ ]:


df_submit = test_data[['ID', 'Class']]
df_submit.head()


# In[ ]:


df_submit.to_csv('df_submit4.csv', index = False)


# In[ ]:


from IPython.display import HTML
import pandas as pd
import numpy as np
import base64
def create_download_link(df, title = "Download CSV file", filename = "data.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html ='<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)
create_download_link(df_submit4)


# In[ ]:




