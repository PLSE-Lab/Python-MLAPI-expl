#!/usr/bin/env python
# coding: utf-8

# # This is a simplistic attempt to get basic baseline from different models
# ## Models used include
# * XGBoost
# * Keras
# * RandomForest
# * DecisionTrees
# * AdaBoost
# * VotingClassifier (Hard + Soft)
# * Others tried but discarded include MLP, Gaussian, NB 
# 
# ## New Feature were created as well
# * Current code only keeps the features which provide a spearman correlation of +/-1 sigma to churn. Everything else is dropped. You can experiment with all features or change the sigma threshold in the code (I did not get any major difference in results)

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
from inspect import signature
from sklearn.metrics import *
from sklearn.utils.multiclass import unique_labels
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras import optimizers
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA


# In[ ]:


file1 = "/kaggle/input/telecom-customer/Telecom_customer churn.csv"


# In[ ]:


df = pd.read_csv(file1)
df_o = df.copy() #I just made a copy for easier experimentation during code writing.


# # Perform any data exploration below.

# In[ ]:


## all experiments for exploration are discarded as mainly it was a combination of excel and tableau before I decided to use all features.


# In[ ]:


##


# # We will now reset the df and start the feature processing. (Exploration work will be unsaved)

# In[ ]:


df = df_o.copy()


# In[ ]:


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=True,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

def plot_roc(y_true, y_pred, title):
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
    #print(fpr, tpr, prec, rec)
    plt.plot(fpr, tpr)
    plt.plot(fpr,tpr,linestyle = "dotted",
             color = "royalblue",linewidth = 2,
             label = "AUC = " + str(np.around(roc_auc_score(y_true,y_pred),3)))
    plt.legend(loc='best')
    plt.plot([0,1], [0,1])
    plt.xticks(np.arange(0,1.1,0.1))
    plt.yticks(np.arange(0,1.1,0.1))
    plt.grid(b=True, which='both')
    plt.title(title)
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.show()


# In[ ]:


# first of all, lets try to see the non-numeric fields.
# list them down and explore 1 by 1 please
df.columns[df.dtypes.values == "object"]


# In[ ]:


categ_nominal = ['new_cell', 'asl_flag', 'prizm_social_one', 'area', 'dualband', 'refurb_new', 'hnd_webcap',
                 'ownrent', 'dwlltype', 'marital', 'infobase', 'HHstatin', 'dwllsize', 'ethnic',
                 'kid0_2', 'kid3_5', 'kid6_10', 'kid11_15', 'kid16_17', 'creditcd']

categ_ordinal_ordered = ['crclscod']  #order this as alphabet sort and then assign numbers.


# In[ ]:


#pd.get_dummies(df['new_cell'], prefix="new_cell", dummy_na=True)

#df_p = df.copy()

for i in categ_nominal:
    df = pd.concat([df, pd.get_dummies(df[i], prefix=i, dummy_na=True)], sort=False, axis=1)


# In[ ]:


df = df.drop(categ_nominal, axis=1)


# In[ ]:


for i in categ_ordinal_ordered:
    s_a = sorted(df[i].unique())
    s_a_dict = {i:x for x,i in enumerate(s_a)}
    df[i] = df[i].map(s_a_dict)


# In[ ]:


df.columns[df.dtypes.values == "object"] #recheck if there are any more object types remaining in the dataframe


# # Below are the new features we are building

# In[ ]:


df['vce_blk_rate'] = 0
df.loc[ df['plcd_vce_Mean'] > 0, 'vce_blk_rate'] = df['blck_vce_Mean'] / df['plcd_vce_Mean']

df['vce_drp_rate'] = 0
df.loc[ df['plcd_vce_Mean'] > 0, 'vce_drp_rate'] = df['drop_vce_Mean'] / df['plcd_vce_Mean']

df['dat_blk_rate'] = 0
df.loc[ df['plcd_dat_Mean'] > 0, 'dat_blk_rate'] = df['blck_dat_Mean'] / df['plcd_dat_Mean']

df['dat_drp_rate'] = 0
df.loc[ df['plcd_dat_Mean'] > 0, 'dat_drp_rate'] = df['drop_dat_Mean'] / df['plcd_dat_Mean']

df['vce_cmpt_rate'] = 0
df.loc[ df['plcd_vce_Mean'] > 0, 'vce_cmpt_rate'] = df['comp_vce_Mean'] / df['plcd_vce_Mean']

df['dat_cmpt_rate'] = 0
df.loc[ df['plcd_dat_Mean'] > 0, 'dat_cmpt_rate'] = df['comp_dat_Mean'] / df['plcd_dat_Mean']

df['tot_cmpt_rate'] = 0
df.loc[ df['attempt_Mean'] > 0, 'tot_cmpt_rate'] = df['complete_Mean'] / df['attempt_Mean']

df['tot_drp_blk_rate'] = 0
df.loc[ df['attempt_Mean'] > 0, 'tot_drp_blk_rate'] = df['drop_blk_Mean'] / df['attempt_Mean']

df['vce_dat_ratio'] = 0
df.loc[ (df['plcd_vce_Mean'] + df['plcd_dat_Mean']) > 0, 'tot_drp_blk_rate'] = df['plcd_vce_Mean'] /  (df['plcd_vce_Mean'] + df['plcd_dat_Mean'])

df['diff_3mon_overall_mou'] = 0
df.loc[ (df['avgmou'] == df['avgmou']) & (df['avg3mou'] == df['avg3mou']), 'diff_3mon_overall_mou'] = (df['avg3mou'] - df['avgmou']) / df['avgmou']

df['diff_3mon_overall_qty'] = 0
df.loc[ (df['avgqty'] == df['avgqty']) & (df['avg3qty'] == df['avg3qty']), 'diff_3mon_overall_qty'] = (df['avg3qty'] - df['avgqty']) / df['avgqty']

df['diff_3mon_overall_rev'] = 0
df.loc[ (df['avgrev'] == df['avgrev']) & (df['avg3rev'] == df['avg3rev']), 'diff_3mon_overall_rev'] = (df['avg3rev'] - df['avgrev']) / df['avgrev']

df['diff_6mon_overall_mou'] = 0
df.loc[ (df['avgmou'] == df['avgmou']) & (df['avg6mou'] == df['avg6mou']), 'diff_6mon_overall_mou'] = (df['avg6mou'] - df['avgmou']) / df['avgmou']

df['diff_6mon_overall_qty'] = 0
df.loc[ (df['avgqty'] == df['avgqty']) & (df['avg6qty'] == df['avg6qty']), 'diff_6mon_overall_qty'] = (df['avg6qty'] - df['avgqty']) / df['avgqty']

df['diff_6mon_overall_rev'] = 0
df.loc[ (df['avgrev'] == df['avgrev']) & (df['avg6rev'] == df['avg6rev']), 'diff_6mon_overall_rev'] = (df['avg6rev'] - df['avgrev']) / df['avgrev']

df['total_nulls'] = 0
df.loc[:, 'total_nulls'] = np.sum(pd.isnull(df), axis=1)

df['eqpdays_digitized'] = np.digitize(df['eqpdays'], bins=[-10, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360, 390, 5000])


# In[ ]:


corr_mat = df.corr(method='spearman')
# nothing
#sorted(corr_mat['churn'], reverse=True)

# nothing


# In[ ]:


corr_mat.loc[ corr_mat['churn'] == 1, 'churn'] = np.nan
dev = 1
s_6 = corr_mat['churn'].mean() + corr_mat['churn'].std() * dev
s__6 = corr_mat['churn'].mean() - corr_mat['churn'].std() * dev
print(s_6, s__6)


# In[ ]:


reduced_features = corr_mat[(corr_mat['churn'] >= s_6) | (corr_mat['churn'] <= s__6)].index.values.tolist()
reduced_features.extend(['churn'])


# In[ ]:


df = df[reduced_features].copy() #reducing the features based on standard deviation on the spearman correlation


# In[ ]:


df_o = df.copy()


# # **ML Starting below**

# In[ ]:


RANDOM_STATE = 91
TEST_SIZE = 0.3
models = dict() #trained models will be kept in this dict as "ModelName": Model

classifiers = {
    "Decision trees" : DecisionTreeClassifier(max_depth=5),
    "Random Forest" : RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    "Adaboost" : AdaBoostClassifier(n_estimators=200, learning_rate=0.01),
}


# In[ ]:


df = df_o.copy() #restore the worked on dataset for multiple running of only below cells


# In[ ]:


from sklearn.impute import SimpleImputer
#Check different NA handling options.

OPTION = 4
TECH = 'median' #technique to apply for OPTION 2 setting

if OPTION == 1:
    #Option 1, drop na
    df = df.dropna(axis=1)
elif OPTION == 2:
    #Take a fill approach
    #NA handling using different techniques just to see the impact.
    if TECH == 'median':
        df = df.fillna(value=df.median())
        print("Applying median to all nan values.")
    elif TECH == 'mean':
        df = df.fillna(value=df.mean())
        print("Applying mean to all nan values.")
    elif TECH == 'mode':
        df = df.fillna(value=df.mode())
        print("Applying mode to all nan values.")
    else:
        df = df.fillna(value=0)
        print("Applying 0 to all nan values.")
elif OPTION == 3: #impute
    my_imputer = SimpleImputer()
    df_i = my_imputer.fit_transform(df)
    df = pd.DataFrame(df_i, columns=df.columns.values)
elif OPTION == 4:
    # make copy to avoid changing original data (when Imputing)
    new_data = df.copy()

    # make new columns indicating what will be imputed
    cols_with_missing = (col for col in new_data.columns if new_data[col].isnull().any())
    for col in cols_with_missing:
        new_data[col + '_was_missing'] = new_data[col].isnull()
    # Imputation
    my_imputer = SimpleImputer(strategy=TECH)
    #print("new_data now ", new_data.shape, " columns ", new_data.columns.values)
    new_data = pd.DataFrame(my_imputer.fit_transform(new_data), columns=new_data.columns)
    #print("new_data now ", new_data.shape, " columns ", new_data.columns.values)
    #new_data.columns = df.columns
    df = new_data.copy()
    #print("DF shape now ", df.shape, " columns ", df.columns.values)


# # Feature scaling. Simple approach to MinMaxScaler choosen here as the distribution of all the columns is almost similar with limited outliers.

# In[ ]:


from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()

df_X = sc.fit_transform(df)
df = pd.DataFrame(df_X, columns=df.columns.values)


# ### Separate the target label from the dataset.

# In[ ]:


y = df['churn']
# make sure that Customer_ID is not present in the data set as ideally this should not be a feature to predict churn if data was well randomized
# so we will drop the churn column and the Customer_ID (if exists) from the dataframe for training purposes.
df = df.drop([x for x in ['churn', 'Customer_ID'] if x in df.columns], axis=1)


# # Optional PCA plot below (2-d). tSNE is too time consuming.

# In[ ]:


pca = PCA(n_components=2, svd_solver='full')
df_pca = pca.fit_transform(df)



df_o["pca1"] = df_pca[:, 0]
df_o["pca2"] = df_pca[:, 1]

plt.subplots(figsize=(10,8))
sns.scatterplot(data=df_o, x="pca1", y="pca2", hue="churn")
plt.show()


# # Train test split. (For now no stratified splitting is happening)

# In[ ]:


X_train, X_test, y_train, y_test =         train_test_split(df, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)


# # XGBoost Below

# In[ ]:


import xgboost as xgb

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

param = {'max_depth': 9, 'objective':'binary:hinge', 'grow_policy' : 'lossguide', 'predictor' : 'gpu_predictor',
        'booster' : 'dart', 'rate_drop' : 0.02, 'tree_method' : 'gpu_hist'}

param_2 = {'booster': 'dart',
         'max_depth': 8, 'learning_rate': 0.1,
         'objective': 'binary:logistic',
         'sample_type': 'weighted',#select dropped trees based on weight
         'normalize_type': 'tree',
         'rate_drop': 0.1,
         'skip_drop': 0.5}

watchlist = [(dtest, 'eval'), (dtrain, 'train')]
num_round = 200


bst = xgb.train(param_2, dtrain, num_round, watchlist)
models['XGB'] = bst
plot_roc(y_test, bst.predict(dtest), "XGB")


# # Keras Model Below

# #### Warning, this is a pretty big model for a simple task. But, no matter the configurations tried, I was unable to make it perform better than the XGB. Please if someone can make a better DNN for this classification, kindly let me know.

# In[ ]:


## Warning, this is a pretty big model for a simple task. But, no matter the configurations tried, I was unable to make it perform better than the XGB.
model = Sequential()
model.add(Dense(512, input_shape=(X_train.shape[1],), activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=0.0001), metrics=['accuracy'], )

model.fit(X_train, y_train, epochs=20, batch_size=20, validation_data=(X_test, y_test))
models['Keras'] = model

plot_roc(y_test, model.predict(X_test), title="Keras ")


# # RandomForest, DecisionTrees, Adaboost below

# In[ ]:


incorrect_indexes = []


for name in classifiers.keys():
    print("Starting with ", name)
    clf = classifiers[name]
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    y_pred = clf.predict(X_test)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    CM = confusion_matrix(y_test, y_pred)
    print(X_test[y_pred != y_test].index.values)
    print("Name {}, Score {}, Precision {}, Recall {}".format(name, score, prec, rec))
    print(CM)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plot_confusion_matrix(y_test, y_pred, classes=["loyal", "churn"],
                          title='Confusion matrix, with normalization', normalize=True)
    plt.show()
    
    models[name] = clf
    
    plot_roc(y_test, y_pred, title="Clf:" + name)
    


# # Let us try a Voting Classifier as well

# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier

clflr = LogisticRegression(solver='lbfgs', multi_class='multinomial',
                          random_state=1)

clfrf = RandomForestClassifier(n_estimators=300, random_state=1)

clfgb = GradientBoostingClassifier(learning_rate=0.05)

clfad = AdaBoostClassifier(n_estimators=300)


# In[ ]:


eclf1 = VotingClassifier(estimators=[
        ('lr', clflr), ('rf', clfrf), ('gnb', clfgb), ('adab', clfad)], voting='soft')

eclf1 = eclf1.fit(X_train, y_train)

y_pred = eclf1.predict(X_test)

models['VotingSoft'] = eclf1
plot_roc(y_test, y_pred, "Voting - Soft")


# In[ ]:


eclf2 = VotingClassifier(estimators=[
        ('lr', clflr), ('rf', clfrf), ('gnb', clfgb), ('adab', clfad)], voting='hard')

eclf2 = eclf1.fit(X_train, y_train)

y_pred = eclf1.predict(X_test)

models['VotingHard'] = eclf1
plot_roc(y_test, y_pred, "Voting - Hard")


# # Finally the end. We will now try to plot the ROC curves for all the classifiers in one place

# In[ ]:


for mname in models.keys():
    print(mname)
    if mname.lower().find("xgb") > -1:
        plot_roc(y_test, models[mname].predict(dtest), title=mname) #xgb has a different dtest
    else:
        plot_roc(y_test, models[mname].predict(X_test), title=mname)
    


# #### I believe that seeing the PCA plots, the data correlation for churn and non-churn is too high for models to pick up better patterns. Another possibility is that I am missing some clear opportunities to make the model better. (feature engineering maybe??)

# In[ ]:




