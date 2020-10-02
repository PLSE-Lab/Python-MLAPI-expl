#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


get_ipython().system('pip install plotly')


# In[ ]:


get_ipython().system('pip install cufflinks')


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.offline import iplot
import plotly as py
import plotly.tools as tls
import cufflinks as cf


# In[ ]:


train = pd.read_csv('../input/siim-isic-melanoma-classification/train.csv',na_values=['unknown'])
submission = pd.read_csv('../input/siim-isic-melanoma-classification/sample_submission.csv')
test = pd.read_csv('../input/siim-isic-melanoma-classification/test.csv')
train.head()


# In[ ]:


train.isnull().sum()


# In[ ]:


train.info()


# In[ ]:


py.offline.init_notebook_mode(connected = True)
cf.go_offline()


# ## Pandas-Bokeh is use full and quite handy. I previously tried to use Bokeh but because of its complexcity i gave up and started learning plotly.In this notebook i used both bokeh and plotly for comparison. Except that i have implimented stacking on the tabular data which is giving score of 0.685.

# In[ ]:


type(train.columns)


# In[ ]:


lists = ['anatom_site_general_challenge']


# In[ ]:


# cf.set_config_file(theme = 'solar')
from bokeh.models.widgets import DataTable, TableColumn
from bokeh.models import ColumnDataSource

data_table = DataTable(
    columns=[TableColumn(field=Ci, title=Ci) for Ci in lists],
    source=ColumnDataSource(train),
    height=300,
)


count_anatom = train['anatom_site_general_challenge'].value_counts().plot_bokeh(kind = 'barh',color='green',title="count of location of imaged site", 
    alpha=0.6,show_figure=False)

pandas_bokeh.plot_grid([[data_table, count_anatom]], plot_width=400, plot_height=350)


# In[ ]:


train['benign_malignant'].value_counts().plot_bokeh(kind='bar',alpha=0.6,color= 'red',title="count of benign and malignant",ylabel='count',xlabel='benign_malignant')


# In[ ]:


train['diagnosis'].value_counts().plot_bokeh(kind='bar',color ='magenta',alpha=0.6,vertical_xlabel=True,title="count of diagnosis",ylabel='count',xlabel='diagnosis')


# In[ ]:


train['age_approx'].value_counts().plot_bokeh(kind='bar',color = 'blue',title="count of age_approx",vertical_xlabel=True,ylabel='count',xlabel='age_approx',alpha=0.6)


# In[ ]:


train['sex'].value_counts().plot_bokeh(kind='bar',alpha=0.6,colormap=["#009933"],title="count of sex",ylabel='count',xlabel='sex')


# In[ ]:


p_vs_img = train.groupby('patient_id').image_name.count().to_frame().reset_index()


# In[ ]:


p_vs_img_plot = p_vs_img.sort_values(by=['image_name'],ascending=False).iloc[0:50]
p_vs_img_plot.plot_bokeh(kind='bar',alpha=0.6,color="brown",title="count of top 50 patient_id",ylabel='count',xlabel='patient_id',vertical_xlabel=True)


# In[ ]:



# patient_id
train['patient_id'].value_counts().plot_bokeh(kind='bar',alpha=0.6,color="blue",title="count of full patient_id",ylabel='count',xlabel='patient_id',vertical_xlabel=True,figsize=(1000, 600))


# In[ ]:


(train.groupby('patient_id').image_name.count()).max()


# In[ ]:


train.head()


# In[ ]:


train.groupby(['target','sex']).count()


# In[ ]:


train.groupby(['target','anatom_site_general_challenge'])['benign_malignant'].count().iplot(kind = 'bar')


# In[ ]:


train.groupby(['target','sex'])['benign_malignant'].count().iplot(kind = 'bar',color='red')


# In[ ]:


train.groupby(['target','age_approx'])['benign_malignant'].count().iplot(kind = 'bar',color='green')


# In[ ]:


train.groupby(['sex','anatom_site_general_challenge'])['benign_malignant'].count().iplot(kind = 'bar',color = 'blue')


# In[ ]:


train.groupby(['target','diagnosis'])['benign_malignant'].count().iplot(kind = 'bar')


# In[ ]:


train.groupby(['sex','diagnosis'])['benign_malignant'].count().iplot(kind = 'bar',color ='magenta')


# In[ ]:





# In[ ]:


amount = train.groupby('anatom_site_general_challenge')['anatom_site_general_challenge'].transform('count')


# In[ ]:


import plotly.graph_objs as go
labels = set(train['anatom_site_general_challenge'])
labels_list = list(labels)
trace = go.Pie(values=amount,labels = labels_list,hole=0.3,pull=[0, 0, 0.2, 0])
iplot([trace])


# In[ ]:


train['diagnosis'].value_counts()


# In[ ]:


train['diagnosis'].value_counts().iplot(kind='bar',color='yellow')


# # Feature Engineering:

# In[ ]:


train.head()


# ### Feature Engineering:

# In[ ]:


numerical_features = [i for i in train.columns if train[i].dtypes != 'O']
numerical_features


# In[ ]:


categorical_features = [i for i in train.columns if train[i].dtypes == 'O']
categorical_features


# In[ ]:


discreat_features = [i for i in numerical_features if len(train[i].unique())<25 ]
discreat_features


# So, we got some features which are neumerical and categorical. And from above cell it is clear that the neumerical features are discontinious.so there is no continious features.

# In[ ]:


import gc
gc.collect()


# In[ ]:


train.head()


# In[ ]:


train.isnull().sum()


# In[ ]:


nan_features = [i for i in train.columns if train[i].isnull().sum()>=1]
nan_features


# In[ ]:


# pd.pandas.set_option('display.max_columns',None)
# pd.pandas.set_option('display.max_rows',None)

train['age_approx'] = train['age_approx'].fillna(train['age_approx'].median())
train['age_approx'].isnull().sum()


# In[ ]:


train['anatom_site_general_challenge'].mode()


# In[ ]:


train['anatom_site_general_challenge']=train['anatom_site_general_challenge'].fillna('torso') 
train['anatom_site_general_challenge'].isnull().sum()


# In[ ]:


train['sex'] = train['sex'].fillna(str(train['sex'].mode()))
train['diagnosis'] = train['diagnosis'].fillna(str(train['diagnosis'].mode()))
train['sex'].isnull().sum()
train['diagnosis'].isnull().sum()


# In[ ]:


train.isnull().sum()


# ### Label Encoding

# In[ ]:


from sklearn.preprocessing import LabelEncoder
label_encod = LabelEncoder()

for i in categorical_features:
    train[i]=label_encod.fit_transform(train[i])


# In[ ]:


train.head()


# ### Droping some columns:

# In[ ]:


data = train.copy()
data = data.drop(['image_name','patient_id','diagnosis','benign_malignant'],axis = 1)
data.head()


# ### Train test split:

# In[ ]:


Y = data['target']
X = data.drop(['target'],axis = 1)


# In[ ]:


SEED = 42
from sklearn.model_selection import train_test_split

x_train,x_val,y_train,y_val = train_test_split(X,Y,test_size = 0.2,random_state = SEED)


# ### Model Building:(RF)

# In[ ]:


random_grid = {'bootstrap': [True, False],
               'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, None],
               'max_features': ['auto', 'sqrt'],
               'min_samples_leaf': [1, 2, 4],
               'min_samples_split': [2, 5, 10],
               'n_estimators': [130, 180, 230]}


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV

classifier_rf = RandomForestClassifier(random_state=SEED)
rf_random = RandomizedSearchCV(estimator = classifier_rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
rf_random.fit(x_train,y_train)


# In[ ]:


rf_random.best_estimator_


# In[ ]:


classifier_rf1 = RandomForestClassifier(max_depth=70, min_samples_leaf=4, min_samples_split=5,
                       n_estimators=180, random_state=42)
classifier_rf1.fit(x_train,y_train)


# In[ ]:


y_pred = classifier_rf1.predict_proba(x_val)
type(y_pred)


# In[ ]:





# ### Test data:

# In[ ]:


test.isnull().sum()


# In[ ]:


test['anatom_site_general_challenge'] = test['anatom_site_general_challenge'].fillna(str(test['anatom_site_general_challenge'].mode())) 


# In[ ]:


test.isnull().sum()


# In[ ]:





# In[ ]:


categorical_fea_test = [i for i in test.columns if test[i].dtypes == 'O']
for i in categorical_fea_test:
    test[i] = label_encod.fit_transform(test[i])
    
test.head()


# In[ ]:


test = test.drop(['image_name','patient_id'],axis = 1)
test.head()


# ### Evaluation:

# In[ ]:


from sklearn.metrics import roc_curve, roc_auc_score,auc
print(roc_auc_score(y_val, y_pred[:,1]))

fpr, tpr, _ = roc_curve(y_val, y_pred[:,1])

plt.clf()
plt.plot(fpr, tpr)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC curve')
plt.show()


# In[ ]:


y_pred


# In[ ]:


auc_rf = auc(fpr,tpr)
print(auc_rf)


# In[ ]:


y_pred_test_rf = classifier_rf1.predict_proba(test)
y_pred_test_rf[:,1].


# In[ ]:


submission_main = pd.read_csv('../input/siim-isic-melanoma-classification/sample_submission.csv')


# ### Image prediction file: 

# In[ ]:


img_sub = pd.read_csv('../input/1st-featuredcom-submission-baseline-keras-vgg16/submission.csv')
img_sub.head()


# ### Model Building:(XGB)

# In[ ]:


import xgboost as xgb
from scipy import stats
from scipy.stats import randint

xgb_clf = xgb.XGBClassifier()


param_dist = {'n_estimators': stats.randint(150, 1000),
              'learning_rate': stats.uniform(0.01, 0.6),
              'subsample': stats.uniform(0.3, 0.9),
              'max_depth': [3, 4, 5, 6, 7, 8, 9],
              'colsample_bytree': stats.uniform(0.5, 0.9),
              'min_child_weight': [1, 2, 3, 4]
             }
clf_xgb = RandomizedSearchCV(xgb_clf, param_distributions = param_dist, n_iter = 25, scoring = 'roc_auc', error_score = 0, verbose = 3, n_jobs = -1)
clf_xgb.fit(x_train,y_train)


# In[ ]:


clf_xgb.best_estimator_


# In[ ]:


xgbo = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.5791848525626986, gamma=0,
              gpu_id=-1, importance_type='gain', interaction_constraints='',
              learning_rate=0.47840118037023044, max_delta_step=0, max_depth=4,
              min_child_weight=4, monotone_constraints='()',
              n_estimators=585, n_jobs=0, num_parallel_tree=1, random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
              subsample=0.792826572247452, tree_method='exact',
              validate_parameters=1, verbosity=None)


# In[ ]:


xgbo.fit(x_train,y_train)
y_pred_xgb = xgbo.predict_proba(x_val)


# In[ ]:


print(roc_auc_score(y_val, y_pred_xgb[:,1]))


# #### So random Forest is giving me more better results than Xgboost, though the difference is slightly large.

# ### Stacking:

# In[ ]:


from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import (AdaBoostClassifier,GradientBoostingClassifier,ExtraTreesClassifier)
from sklearn.tree import DecisionTreeClassifier

base_learners = [
                 ('rf_1', RandomForestClassifier(n_estimators=100, random_state=SEED)),
                 ('adb',AdaBoostClassifier(n_estimators=100, random_state=SEED)),
                 ('ext',ExtraTreesClassifier(n_estimators=100, random_state=SEED)),
                 ('gbc',GradientBoostingClassifier(n_estimators=100,random_state=SEED)),
                 ('svc', SVC())
    
    
                ]

# Initialize Stacking Classifier with the Meta Learner
stk_clf = StackingClassifier(estimators=base_learners, final_estimator=LogisticRegression())

stk_clf.fit(x_train, y_train)


# In[ ]:


y_pred_stk = stk_clf.predict_proba(x_val)


# In[ ]:


print(roc_auc_score(y_val, y_pred_stk[:,1]))


# #### So stacking is producing far more better result than, normal xgboost or random forest. And it is found that we are getting same score with and without SVC.

# In[ ]:


get_ipython().system('pip install -U pandas_bokeh')


# In[ ]:


import pandas_bokeh
pd.set_option('plotting.backend', 'pandas_bokeh')
pandas_bokeh.output_notebook()


# In[ ]:


train['anatom_site_general_challenge'].value_counts().plot_bokeh(kind='barh')


# In[ ]:




