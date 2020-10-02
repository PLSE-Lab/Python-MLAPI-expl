# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Handle table-like data and matrices
import numpy as np
import pandas as pd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

# machine learning
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import make_scorer, accuracy_score, precision_score, f1_score, recall_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler

# Acquire data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

#Shapes
print('Shapes:\n','train: ',train.shape,'\n test: ',test.shape)
ntrain = train.shape[0]
ntest = test.shape[0]

#Functions
def plot_confusion_matrix(y_test, y_pred):
    print('Confusion matrix :')
    cmap = sns.diverging_palette(220 , 10 , as_cmap = True)
    plt.figure(figsize = (8, 8))
    sns.heatmap(confusion_matrix(y_test, y_pred), cmap=cmap, square=True, annot=True, vmin=0, fmt="d")
    plt.xlabel('Predicted value')
    plt.ylabel('Real value')
    plt.show()

def print_scores(y_test, y_pred):
    #Distribution between actual and basic predicted:
    print('y actual : \n' + str(y_test.value_counts()))
    print('y predicted : \n' + str(y_pred.value_counts()))
    print()
    print('Accuracy score : ' + str(accuracy_score(y_test, y_pred)))
    print('Precision score : ' + str(precision_score(y_test, y_pred)))
    print('Recall score : ' + str(recall_score(y_test, y_pred)))
    print('f1 score : ' + str(f1_score(y_test, y_pred)))
    print()
    plot_confusion_matrix(y_test, y_pred)
    
#Preparing
data = pd.concat([train,test],ignore_index=True, sort=False)
data = data.drop(['Survived','PassengerId'], axis=1)

mean_fare_pclass = data[['Pclass', 'Fare']].groupby('Pclass').mean().reset_index()
pclass_1_mean_price = float(mean_fare_pclass[mean_fare_pclass['Pclass'] == 1].Fare)
pclass_2_mean_price = float(mean_fare_pclass[mean_fare_pclass['Pclass'] == 2].Fare)
pclass_3_mean_price = float(mean_fare_pclass[mean_fare_pclass['Pclass'] == 3].Fare)
data = data.replace({'Pclass': {1: pclass_1_mean_price,
                                2: pclass_2_mean_price,
                                3: pclass_3_mean_price}})
scaler = MinMaxScaler()
scaler.fit(data[['Pclass']])
data[['Pclass']] = scaler.transform(data[['Pclass']])

age_categories = ["0-8", "8-16", "16-45", "45-100"]
data['AgeCategory'] = pd.cut(data['Age'],
                             (0, 8, 16, 45, 100),
                             labels=age_categories)

fare_categories = ["0", "0-8", "8-14", "14-21", "21-60", "60-84", "84+"]
data['FareCategory'] = pd.cut(data['Fare'],
                             (-1, 0, 8, 14, 21, 60, 84, 1000),
                             labels=fare_categories)

sibsp_categories = ["0-1", "2-3", "3+"]
data['SibSPCategory'] = pd.cut(data['SibSp'],
                             (-1, 1, 3, 10),
                             labels=sibsp_categories)

parch_categories = ["0", "1-3", "4+"]
data['ParchCategory'] = pd.cut(data['Parch'],
                             (-1, 0, 3, 10),
                             labels=parch_categories)

data = data.drop(['Age', 'Fare', 'Embarked', 'Cabin', 'Name', 'Parch', 'SibSp', 'Ticket'],
                 axis=1)

prep_data = pd.get_dummies(data)
df_train_x = prep_data.iloc[:ntrain] 
df_test_x = prep_data.iloc[ntrain:]
df_train_y = train['Survived']

#First Model
X_train, X_test, y_train, y_test = train_test_split(df_train_x,
                                                    df_train_y,
                                                    random_state=7)

dtc = DecisionTreeClassifier(random_state=7)
dtc.fit(X_train, y_train)
y_pred_dtc = pd.Series(dtc.predict(X_train))

print_scores(y_train, y_pred_dtc)

results = dtc.predict(df_test_x)

ids = test.PassengerId
surv_df = pd.DataFrame({ 'PassengerId' : ids, 'Survived': results})
surv_df.to_csv('submission_titanic.csv', index=False)