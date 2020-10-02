#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[1]:


import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import average_precision_score
from xgboost.sklearn import XGBClassifier
from xgboost import plot_importance, to_graphviz
from tqdm import tqdm


# In[2]:


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.savefig('confusion.png')
    plt.show()


# In[3]:


df = pd.read_csv('../input/PS_20174392719_1491204439457_log.csv')
df = df.rename(columns={'oldbalanceOrg':'oldBalanceOrig', 'newbalanceOrig':'newBalanceOrig',                         'oldbalanceDest':'oldBalanceDest', 'newbalanceDest':'newBalanceDest'})


# In[4]:


df.head()


# In[5]:


df.loc[df.isFraud == 1].type.drop_duplicates().values


# In[6]:


df = df[(df.type == 'TRANSFER') | (df.type == 'CASH_OUT')]


# In[7]:


len(df)


# In[8]:


df.loc[(df.isFraud == 1) & (df.type == 'TRANSFER')].amount.median()


# In[9]:


df.loc[(df.isFraud == 0) & (df.type == 'TRANSFER')].amount.median()


# In[10]:


df['Fraud_Heuristic'] = np.where(((df['type'] == 'TRANSFER') & 
                                  (df['amount'] > 200000)),1,0)


# In[11]:


df['Fraud_Heuristic'].sum()


# In[12]:


from sklearn.metrics import f1_score


# In[13]:


f1_score(y_pred=df['Fraud_Heuristic'],y_true=df['isFraud'])


# In[14]:


from sklearn.metrics import confusion_matrix


# In[15]:


cm = confusion_matrix(y_pred=df['Fraud_Heuristic'],y_true=df['isFraud'])


# In[16]:


plot_confusion_matrix(cm,['Genuine','Fraud'], normalize=False)


# In[17]:


df.shape


# In[18]:


df['hour'] = df['step'] % 24


# In[19]:


frauds = []
genuine = []
for i in range(24):
    f = len(df[(df['hour'] == i) & (df['isFraud'] == 1)])
    g = len(df[(df['hour'] == i) & (df['isFraud'] == 0)])
    frauds.append(f)
    genuine.append(g)


# In[ ]:





# In[20]:


sns.set_style("white")

fig, ax = plt.subplots(figsize=(10,6))
gen = ax.plot(genuine/np.sum(genuine), label='Genuine')
fr = ax.plot(frauds/np.sum(frauds),dashes=[5, 2], label='Fraud')
#frgen = ax.plot(np.devide(frauds,genuine),dashes=[1, 1], label='Fraud vs Genuine')
plt.xticks(np.arange(24))
legend = ax.legend(loc='upper center', shadow=True)
fig.savefig('time.png')


# In[21]:


sns.set_style("white")

fig, ax = plt.subplots(figsize=(10,6))
#gen = ax.plot(genuine/np.sum(genuine), label='Genuine')
#fr = ax.plot(frauds/np.sum(frauds),dashes=[5, 2], label='Fraud')
frgen = ax.plot(np.divide(frauds,np.add(genuine,frauds)), label='Share of fraud')
plt.xticks(np.arange(24))
legend = ax.legend(loc='upper center', shadow=True)
fig.savefig('time_comp.png')


# In[22]:


dfFraudTransfer = df[(df.isFraud == 1) & (df.type == 'TRANSFER')]


# In[23]:


dfFraudCashOut = df[(df.isFraud == 1) & (df.type == 'CASH_OUT')]


# In[24]:


dfFraudTransfer.nameDest.isin(dfFraudCashOut.nameOrig).any()


# In[25]:


dfNotFraud = df[(df.isFraud == 0)]


# In[26]:


dfFraud = df[(df.isFraud == 1)]


# In[27]:


dfFraudTransfer.loc[dfFraudTransfer.nameDest.isin(
    dfNotFraud.loc[dfNotFraud.type == 'CASH_OUT'].nameOrig.drop_duplicates())]


# In[28]:


len(dfFraud[(dfFraud.oldBalanceDest == 0) & (dfFraud.newBalanceDest == 0) & (dfFraud.amount)]) / (1.0 * len(dfFraud))


# In[29]:


len(dfNotFraud[(dfNotFraud.oldBalanceDest == 0) & (dfNotFraud.newBalanceDest == 0) & (dfNotFraud.amount)]) / (1.0 * len(dfNotFraud))


# In[30]:


dfOdd = df[(df.oldBalanceDest == 0) & 
           (df.newBalanceDest == 0) & 
           (df.amount)]


# In[31]:


len(dfOdd[(dfOdd.isFraud == 1)]) / len(dfOdd)


# In[32]:


len(dfOdd[(dfOdd.oldBalanceOrig <= dfOdd.amount)]) / len(dfOdd)


# In[33]:


len(dfOdd[(dfOdd.oldBalanceOrig <= dfOdd.amount) & (dfOdd.isFraud == 1)]) / len(dfOdd[(dfOdd.isFraud == 1)])


# In[34]:


dfOdd.columns


# In[35]:


dfOdd.head(20)


# In[36]:


df.head()


# In[37]:


df['type'] = 'type_' + df['type'].astype(str)


# In[38]:


# Get dummies
dummies = pd.get_dummies(df['type'])

# Add dummies to df
df = pd.concat([df,dummies],axis=1)

#remove original column
del df['type']


# Predictive modeling with Keras

# In[41]:


df = df.drop(['nameOrig','nameDest','Fraud_Heuristic'], axis= 1)


# In[49]:


df['isNight'] = np.where((2 <= df['hour']) & (df['hour'] <= 6), 1,0)


# In[50]:


df[df['isNight'] == 1].isFraud.mean()


# In[51]:


df.head()


# In[53]:


df = df.drop(['step','hour'],axis=1)


# In[54]:


df.head()


# In[56]:


df.columns.values


# In[58]:


y_df = df['isFraud']
x_df = df.drop('isFraud',axis=1)


# In[59]:


y = y_df.values
X = x_df.values


# In[60]:


y.shape


# In[61]:


X.shape


# In[62]:


from sklearn.model_selection import train_test_split


# In[84]:


X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.33, 
                                                    random_state=42)


# In[85]:


X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, 
                                                    test_size=0.1, 
                                                    random_state=42)


# In[86]:


from imblearn.over_sampling import SMOTE, RandomOverSampler


# In[96]:


sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_sample(X_train, y_train)


# In[194]:


from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD


# In[195]:


# Log reg
model = Sequential()
model.add(Dense(1, input_dim=9))
model.add(Activation('sigmoid'))


# In[196]:


model.summary()


# In[197]:


model.compile(loss='binary_crossentropy',
              optimizer=SGD(lr=1e-5), 
              metrics=['acc'])


# In[199]:


model.fit(X_train_res,y_train_res,
          epochs=5, 
          batch_size=256, 
          validation_data=(X_val,y_val))


# In[144]:


y_pred = model.predict(X_test)


# In[145]:


y_pred[y_pred > 0.5] = 1
y_pred[y_pred < 0.5] = 0


# In[146]:


f1_score(y_pred=y_pred,y_true=y_test)


# In[147]:


cm = confusion_matrix(y_pred=y_pred,y_true=y_test)


# In[149]:


plot_confusion_matrix(cm,['Genuine','Fraud'], normalize=False)


# In[200]:


model = Sequential()
model.add(Dense(16,input_dim=9))
model.add(Activation('tanh'))
model.add(Dense(1))
model.add(Activation('sigmoid'))


# In[203]:


model.compile(loss='binary_crossentropy',optimizer=SGD(lr=1e-4), metrics=['acc'])


# In[205]:


model.fit(X_train_res,y_train_res,
          epochs=5, batch_size=256, 
          validation_data=(X_val,y_val))


# In[206]:


y_pred = model.predict(X_test)


# In[207]:


y_pred[y_pred > 0.5] = 1
y_pred[y_pred < 0.5] = 0


# In[208]:


f1_score(y_pred=y_pred,y_true=y_test)


# In[138]:


cm = confusion_matrix(y_pred=y_pred,y_true=y_test)


# In[139]:


plot_confusion_matrix(cm,['Genuine','Fraud'], normalize=False)


# # Tree based methods

# In[150]:


from sklearn.tree import export_graphviz


# In[151]:


from sklearn.tree import DecisionTreeClassifier
dtree=DecisionTreeClassifier()
dtree.fit(X_train,y_train)


# In[162]:


from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
from subprocess import check_call
from PIL import Image, ImageDraw, ImageFont
from IPython.display import Image as PImage

#import pydotplus
dot_data = StringIO()
'''export_graphviz(dtree, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True)'''
with open("tree1.dot", 'w') as f:
     f = export_graphviz(dtree,
                              out_file=f,
                              max_depth = 3,
                              impurity = True,
                              feature_names = list(df.drop(['isFraud'], axis=1)),
                              class_names = ['Genuine', 'Fraud'],
                              rounded = True,
                              filled= True )
        
#Convert .dot to .png to allow display in web notebook
check_call(['dot','-Tpng','tree1.dot','-o','tree1.png'])

# Annotating chart with PIL
img = Image.open("tree1.png")
draw = ImageDraw.Draw(img)
font = ImageFont.truetype('/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf', 26)
img.save('sample-out.png')
PImage("sample-out.png")


# In[163]:


from sklearn.ensemble import  RandomForestClassifier


# In[176]:


rf = RandomForestClassifier(n_estimators=10,n_jobs=-1)
rf.fit(X_train,y_train)


# In[179]:


y_pred = rf.predict(X_test)


# In[180]:


f1_score(y_pred=y_pred,y_true=y_test)


# In[181]:


cm = confusion_matrix(y_pred=y_pred,y_true=y_test)
plot_confusion_matrix(cm,['Genuine','Fraud'], normalize=False)


# In[182]:


import xgboost as xgb


# In[186]:


booster = xgb.XGBClassifier(n_jobs=-1)
booster = booster.fit(X_train,y_train)


# In[188]:


y_pred = booster.predict(X_test)


# In[189]:


f1_score(y_pred=y_pred,y_true=y_test)


# In[190]:


cm = confusion_matrix(y_pred=y_pred,y_true=y_test)
plot_confusion_matrix(cm,['Genuine','Fraud'], normalize=False)


# # Entity embeddings

# In[ ]:


# Reload data
df = pd.read_csv('../input/PS_20174392719_1491204439457_log.csv')
df = df.rename(columns={'oldbalanceOrg':'oldBalanceOrig', 'newbalanceOrig':'newBalanceOrig',                         'oldbalanceDest':'oldBalanceDest', 'newbalanceDest':'newBalanceDest'})


# In[ ]:


df.head()


# In[ ]:


df = df.drop(['nameDest','nameOrig','step'],axis=1)


# In[ ]:


df['type'].unique()


# In[ ]:


map_dict = {}
for token, value in enumerate(df['type'].unique()):
    map_dict[value] = token   


# In[ ]:


map_dict


# In[ ]:


df["type"].replace(map_dict, inplace=True)


# In[ ]:


df.head()


# In[ ]:


other_cols = [c for c in df.columns if ((c != 'type') and (c != 'isFraud'))]


# In[ ]:


other_cols


# In[ ]:


from keras.models import Model
from keras.layers import Embedding, Merge, Dense, Activation, Reshape, Input, Concatenate


# In[ ]:


num_types = len(df['type'].unique())
type_embedding_dim = 3


# In[ ]:


inputs = []
outputs = []


# In[ ]:


type_in = Input(shape=(1,))
type_embedding = Embedding(num_types,type_embedding_dim,input_length=1)(type_in)
type_out = Reshape(target_shape=(type_embedding_dim,))(type_embedding)

type_model = Model(type_in,type_out)

inputs.append(type_in)
outputs.append(type_out)


# In[ ]:


num_rest = len(other_cols)


# In[ ]:


rest_in = Input(shape = (num_rest,))
rest_out = Dense(16)(rest_in)

rest_model = Model(rest_in,rest_out)

inputs.append(rest_in)
outputs.append(rest_out)


# In[ ]:


concatenated = Concatenate()(outputs)


# In[ ]:


x = Dense(16)(concatenated)
x = Activation('sigmoid')(x)
x = Dense(1)(concatenated)
model_out = Activation('sigmoid')(x)


# In[ ]:


merged_model = Model(inputs, model_out)
merged_model.compile(loss='binary_crossentropy', 
                     optimizer='adam', 
                     metrics=['accuracy'])


# In[ ]:


types = df['type']


# In[ ]:


rest = df[other_cols]


# In[ ]:


target = df['isFraud']


# In[ ]:


history = merged_model.fit([types.values,rest.values],target.values, 
                           epochs = 1, batch_size = 128)


# In[ ]:


merged_model.summary()


# In[ ]:




