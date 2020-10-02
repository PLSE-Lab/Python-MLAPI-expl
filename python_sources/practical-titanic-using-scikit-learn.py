#!/usr/bin/env python
# coding: utf-8

# <h1>Practical Titanic Using Scikit-Learn</h1>
# <h3> First principles of data science approach to the Hello World of ML <br> Kliment Minchev, Jan 2020</h3>

# <h4>TL;DR</h4>An overview of my personal approach to developing Data Science insights. Below is a top quartile, easily portable solution to the <a href="https://www.kaggle.com/c/titanic/overview">Kaggle Titanic problem.</a> I used a probabilistic algorithm (Gaussian Process) to achieve 87% training accuracy and 78.95% Kaggle testing accuracy.

# The <a href="https://www.kaggle.com/c/titanic/overview">task involves a training set with known survival outcomes and a test set with unknown survival outcomes </a>.

# Import the two best know data pre-processing libraries (pandas and numpy) and load the training data.

# In[ ]:


import pandas as pd
import numpy as np

df = pd.read_csv('train.csv')
df.head(6)


# <h2>Feature Selection </h2>
# Right off the bat, I discarded PassengerId, Ticket, and Cabin since they carry random (indiscernible) information. <br> Let's see survivability stats by class ticket.

# In[ ]:


def class_survival(Pclass):
    class_surv = 0
    total = 0
    classes = df[['Survived','Pclass']]
    for i, row in classes.iterrows():
        if classes['Pclass'][i] == Pclass:
            class_surv += classes['Survived'][i]
            total = total + 1
    surv_rate = class_surv/total
    print(f'Class {Pclass} survival rate: {surv_rate}')
    return surv_rate
class1_surival = class_survival(1)
class2_surival = class_survival(2)
class3_surival = class_survival(3)


# I replaced the NaN values in Age with the average age per class. The rationale is that Pclass is easiest to cross reference age with. Higher class tickets have a higher average age (as well as a higher chance of survival).

# In[ ]:


def agefill(Pclass):
    agefill = df[['Age', 'Pclass']].copy()
    for i, row in agefill['Pclass'].iteritems():
        if agefill['Pclass'][i] != Pclass:
            agefill = agefill.drop(i)
    Pclass_mean = np.nanmean(agefill.iloc[:,0].copy())
    return Pclass_mean
ones_mean = agefill(1)
twos_mean = agefill(2)
threes_mean = agefill(3)
print(f'Class 1 avg age: {ones_mean}\nClass 2 avg age: {twos_mean}\nClass 3 avg age: {threes_mean}')


# <h2>Feature Construction </h2>
# Let's do some <b>feature engineering</b> to give our algorithm more clues and improve accuracy. <br> At a glance, it seems that <u>females had a higher chance of survival than males</u>. So, we will <i>add a gender_column as a feature</i>.

# In[ ]:


malesurv, males, females, femalesurv = [0 for _ in range(4)]

gend = df[['Survived','Sex']]
for i1, row1 in gend.iterrows():
    if gend['Sex'][i1] == str('male'):
        malesurv += gend['Survived'][i1]
        males = males + 1
    else:
        femalesurv += gend['Survived'][i1]
        females = females + 1
print(f'Female survival rate: {femalesurv/females}\nMale survival rate: {malesurv/males}')


# A little biased, but I noticed everyone with a Southeastern European name (**I'm Bulgarian**) did not survive.<br> Specifically, 27/27 with a name ending in '-ff', 19/19 with a name ending in '-ic', as well as 5/5 edning in '-ulos'. <br>Granted most had 3rd class tickets, but I decided to include the ethnic names as a feature to test it out.

# In[ ]:


last_names = df['Name'].str.split(",", expand=True)[0]
is_balkan = list()
for name in range(len(last_names)):
    if last_names[name][-2:] != 'ic' and last_names[name][-2:] != 'ff' and last_names[name][-2:] != 'ulos':
        is_balkan.append(0) 
    else:
       is_balkan.append(1)
print(f'There were (at least) {np.sum(is_balkan)} Southeastern Europeans in the training set.')


# Many of the names had a title associated, i.e. Mister, Master, Miss, Mrs, etc. Let's see survivability stats by title:

# In[ ]:


titles = df['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
titles_investigate = pd.DataFrame(data=(df['Survived'],titles)).transpose()
types_titles = pd.DataFrame(data=set( val for dic in titles for val in titles.values)) #17 types of titles

def title_survive(name_title):
    name_title_survive = 0
    name_title_total = 0
    survival_rate = 0
    for title, rowz in titles_investigate.iterrows():
        if titles_investigate.iloc[title,1] == name_title:
            name_title_total += 1
            name_title_survive += titles_investigate.iloc[title,0]
    survival_rate = name_title_survive / name_title_total
    return survival_rate

titles2 = []
for type_title in range(len(types_titles)):
    titles2.append(title_survive(types_titles[0][type_title]))
    print(f'{types_titles[0][type_title]} had survivability chance: {titles2[type_title]}')


# Which port the passengers boarded the Titanic also seems to have an effect. Let's see survivability stats by port:

# In[ ]:


embark = df[['Survived','Embarked']]
def embarkchance(port):
    port_embark = 0
    port_survive = 0
    for ie, rowe in embark.iterrows():
        if embark.iloc[ie,1] == port:
            port_embark += 1
            if embark.iloc[ie,0] == 1:
                port_survive += 1
    survival_rate = port_survive / port_embark
    return survival_rate
    
S_embark = embarkchance('S')
C_embark = embarkchance('C')
Q_embark = embarkchance('Q')
print(f'Embarking at Port S had survivability chance: {S_embark}\nEmbarking at Port C had survivability chance: {C_embark}\nEmbarking at Port Q had survivability chance: {Q_embark}')


# Finally, let's give variable names to the 5 newly created features for survivability: Age, Gender, Is_Balkan, Title, Port 

# In[ ]:


# Age feature
agefill = df[['Age', 'Pclass']].copy()
for i, row in agefill.iterrows():
    if np.isnan(agefill['Age'][i]):
        if agefill['Pclass'].iloc[i] == 3:
            agefill['Age'].iloc[i] = threes_mean
        elif agefill['Pclass'].iloc[i] == 2:
            agefill['Age'].iloc[i] = twos_mean
        else:
            agefill['Age'].iloc[i] = ones_mean
    
agecol = agefill.iloc[:,0].copy()

# Gender feature
gendercol = df['Sex'].copy()
for i2, row2 in gendercol.iteritems():
    if gendercol[i2] == str('male'):
        gendercol[i2] = 1
    else:
        gendercol[i2] = 2

# Is_balkan feature 
# is_balkan

# Type of title associated
types_titles['Survival_Chance'] = titles2

titles3 = []
for title, row4 in titles.iteritems():
    for title_type, row5 in types_titles.iterrows():
        if titles[title] == types_titles.iloc[title_type,0]:
            titles3.append(types_titles.iloc[title_type,1])
            
# Port where passenger embarked
ports = []
for embark_port, rowport in embark.iterrows():
        if embark.iloc[embark_port,1] == 'S':
            ports.append(S_embark)
        elif embark.iloc[embark_port,1] == 'C':
            ports.append(C_embark)
        else:
            ports.append(Q_embark)


# <h2>Model Fitting</h2>

# Below are: our independent variable matrix X (all the features) and our dependent variable vector Y (our label, survival)

# In[ ]:


X = np.array(np.column_stack((df[['Pclass', 'Fare', 'Parch']],agecol, gendercol, is_balkan, titles3, ports))).copy()
y = np.array(df[['Survived']].copy()).ravel()


# Because all our features have different ranges of data, e.g. binary (gender), linear (age) etc, we will employ a <a href="https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html">scaler to have a unit range for our data</a>.

# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)


# To ensure we are leveraging the maximum of our training set, we will employ <a href="https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html">K-Fold Cross-Validation to shuffle our training and testing splits</a>. 

# In[ ]:


from sklearn.model_selection import KFold
kf = KFold(n_splits = 5)
for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]


# **At this stage, we are ready to fit any model of our choice and evaluate it using common metrics.** <br> <h4> As an example, I demonstrate SVM with accuracy_score and a confusion_matrix (to calculate Precision and Recall). <br>Any scikit-learn model of your choice can be fitted.</h4> <h5> My Kaggle submission was done using a Gaussian Process Classifier, as it yielded the highest test accuracy.

# In[ ]:


from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

model = SVC(gamma=1, C=2)
model.fit(X_train, y_train)
y_predict = model.predict(X_test)

cm = confusion_matrix(y_test, y_predict)
acc = accuracy_score(y_predict, y_test)
print('CM:\n {}'.format( cm))
print('Accuracy: {}'.format(acc))


# <h3><i>What the data says:</i></h3><br> Using SVM, we were able to predict 81.46% of the survival outcomes in our <i>Training set</i>.<br><br> The <a href="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html">Confusion Matrix</a> yields another two valuable metrics: <br>Precision (fraction of survivors from the pool we believed would survive) <br> Recall (fraction of the ones we believed would survive from the total pool of survivors). <br> Precision: 101/(101+14) = 0.8783 <br> Recall: 101/(101+19) = 0.8417

# <h4>Hyperparameter Tuning </h4>
# Tuning the hyperparameters of the model would help us achieve a hire training accuracy (though the model may start to overfit, i.e. be too closely fitted to the training data and perform poorly on test data). We can employ a <a href="https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html"> GridSearch technique </a> to test combinations of a set of hyperparameters of interest. Then, we will replace our model = SVC(gamma=a, C=b) with the desired combination of a and b.

# In[ ]:


# For SVM, an example optimization (for demonstration purposes) would be

from sklearn.model_selection import GridSearchCV
parameters = {
    'gamma': [1, 2],
    'C': [1, 2],
    }
gs = GridSearchCV(model, parameters, cv=3)
gs.fit(X, y)
print(gs.best_params_)


# <h2>Model Testing and Kaggle Test Accuracy</h2>

# The final step is to use the approach we developed above for the test set (separate file, named test.csv without a known Survival label). <br><br>Same exact functions developed above are used for constructing the features for our new Test set: Gender, Southeast European name, Title, and Port the passenger embarked the Titanic. <br>Finally, we generate an output csv file of our predictions in the desired Kaggle competition format shown below:

# In[ ]:


sub_format = np.column_stack((df_real['PassengerId'].astype(int),y_predict_test.astype(int)))
df_submit = pd.DataFrame(data=sub_format)
df_submit.columns = ['PassengerId','Survived']
df_submit.to_csv(r'~\titanic\submit.csv', index=False)


# I approached the problem with 3 separate families of classification algorithms :<br><a href="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoCV.html">LassoCV</a> (a cross-validated Lasso regression, where >=0.5 == 1 and <0.5 == 0 for survival).<br> <a href="https://xgboost.readthedocs.io/en/latest/python/python_api.html">XGBoost</a> classifier (an extreme gradient boosting ensemble method).<br><a href="https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessClassifier.html">Gaussian Process Classifier</a> (a probabilistic approach). 

# Of course, I ran several other algorithms which belong to these 3 families of classification algorithms (e.g. SVM, AdaBoost, Random Forest) to validate results.

# <h4>Performance Summary</h4>
# <table align="left" width="50%">
#   <tr>
#     <th>Model</th>
#     <th>Train Accuracy [%]</th>
#     <th>Test Accuracy [%]<br>(Kaggle score)</th>
#   </tr>
#     <tr>
#     <td>Gaussian Process Classifier</td>
#     <td>87.07</td>
#     <td>78.95</td>
#   </tr>
#   <tr>
#     <td>XGBoost</td>
#     <td>80.00</td>
#     <td>76.55</td>
#   </tr>
#     <tr>
#     <td>LassoCV</td>
#     <td>74.23</td>
#     <td>67.94</td>
#   </tr>
#     <tr>
#     <td>SVM</td>
#     <td>81.46</td>
#     <td></td>
#   </tr>
#     <tr>
#     <td>Random Forest</td>
#     <td>83.11</td>
#     <td></td>
#   </tr>
#     <tr>
#     <td>AdaBoost</td>
#     <td>86.53</td>
#     <td></td>
#   </tr>
# </table>
