import matplotlib.pyplot as plt
import pandas as pd
from pandas.tools.plotting import parallel_coordinates
import numpy as np
import re
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB

def visualize_data(X, Y, name):
    m_lbl_survived = Y['Survived'].apply(lambda e: 'Survived' if e == 1 else 'Dead')
    plt.figure()
    parallel_coordinates(X.assign(Survived=m_lbl_survived), 'Survived', color=["red","green"])
    plt.suptitle(name)
    plt.show()
    plt.savefig(name.join(".png"))
  
#==================================
print ("Reading training data ...")
#==================================
train_set = pd.read_csv ('../input/train.csv', sep=',', header=0, 
			skip_blank_lines=True, quotechar='"')

# Predict factors:
#   - Sex (1 - Male)
#   - Age
#   - SibSp
#   - Parch
#   - Pclass
#   - Floor

#==================================
print ("Extracting and numerizing features ...")
#==================================
m_train_set = pd.DataFrame(columns=[
			'Sex','Age', 'SibSp','Parch', 'Pclass',
			'Floor'])
m_survived = pd.DataFrame(columns=['Survived'])

for idx, row in train_set[['Survived','Sex','Age','SibSp',
			'Parch','Pclass','Cabin']].iterrows():
	cabins   = row['Cabin'].split(' ') if pd.notnull(row['Cabin']) else ['Z']
	sex      = 1 if row['Sex'] == 'female' else 0
	age      = row['Age'] if pd.notnull(row['Age']) else 0
	sibsp    = row['SibSp']
	parch    = row['Parch']
	pclass   = row['Pclass']
	survived = row['Survived']

	if len(cabins) > 1:
		for i in cabins:
			floor = ord(re.findall('[A-Z]',i)[0]) - 65
			new_row = {'Pclass':pclass, 'Floor':floor, 'Sex':sex, 'Age':age, 
								'SibSp':sibsp, 'Parch':parch}
			m_train_set.loc[len(m_train_set)] = new_row
			m_survived.loc[len(m_survived)] = {'Survived':survived}
	else:
		floor = ord(re.findall('[A-Z]',cabins[0])[0]) - 65
		new_row = {'Pclass':pclass, 'Floor':floor, 'Sex':sex, 
							'Age':age, 'SibSp':sibsp, 'Parch':parch}
		m_train_set.loc[len(m_train_set)] = new_row
		m_survived.loc[len(m_survived)] = {'Survived':survived}

# Normalizing and scaling data
s_train_set = (m_train_set - m_train_set.mean()) / (m_train_set.max() -
		m_train_set.min())

#==================================
print ("Visulizing training data")
#==================================
visualize_data (s_train_set, m_survived, "TrainingData")

#==================================
print ("Split training set to training and test sets")
#==================================
Y = m_survived['Survived'].values
X = s_train_set[list(s_train_set.columns)].values

#==================================
print ("Trying multiple classifiers:")
#==================================
scores = [0] * 7
names = ["KNN", "Linear SVM", "RBF SVM", "Decision Tree","Random Forest",
		"Neural Network", "AdaBoost", "Naive Bayes"]
classifiers = pd.DataFrame([[1,2,3,4,5,6,7,8],
              names,
               [KNeighborsClassifier(11),
                 SVC(kernel="linear", C=0.025),
							 SVC(gamma=2, C=1),
							 DecisionTreeClassifier(max_depth=6),
							 RandomForestClassifier(max_depth=6, n_estimators=10,
								 max_features=None),
							 MLPClassifier(alpha=1),
							 AdaBoostClassifier(),
							 GaussianNB()],
							 scores]).T.rename(columns={0:'Idx', 1:'Name', 2:'Clf', 3:'Score'})

X_train, X_test, Y_train, Y_test = train_test_split (X, Y)

for idx, clf in classifiers.iterrows():
	print (clf['Name'])
	clf['Clf'].fit(X_train, Y_train)
	clf['Score'] = clf['Clf'].score(X_test, Y_test)

plt.figure()
plt.plot (classifiers['Idx'], classifiers['Score'], 'ro')
plt.xticks(classifiers['Idx'], classifiers['Name'], rotation =
		'vertical')
plt.margins(0.2)
plt.subplots_adjust(bottom=0.3)
plt.suptitle("Classifiers scores")
plt.show()
plt.savefig("scores.png")

#==================================
print ("Pick the most accuracy model ...")
#==================================
selected_clf      = classifiers.loc[classifiers['Score'].idxmax()]
print (selected_clf['Name'])

#==================================
print ("Reading test data ...")
#==================================
test_set = pd.read_csv ('../input/test.csv', sep=',', header=0, 
			skip_blank_lines=True, quotechar='"')

#==================================
print ("Extracting and numerizing features and then classifying test data ...")
#==================================
m_test_set = pd.DataFrame(columns=[
			'Sex','Age', 'SibSp','Parch', 'Pclass',
			'Floor'])
m_pred_survived = pd.DataFrame(columns=['PassengerId', 'Survived'])

for idx, row in test_set[['PassengerId','Sex','Age','SibSp',
			'Parch','Pclass','Cabin']].iterrows():
    cabins   = row['Cabin'].split(' ') if pd.notnull(row['Cabin']) else ['Z']
    sex      = 1 if row['Sex'] == 'female' else 0
    age      = row['Age'] if pd.notnull(row['Age']) else 0
    sibsp    = row['SibSp']
    parch    = row['Parch']
    pclass   = row['Pclass']
    pid      = row['PassengerId']
    
    if len(cabins) > 1:
        tmp_predict = 0
        location = len(m_pred_survived)
        for i in cabins:
            floor = ord(re.findall('[A-Z]',i)[0]) - 65
            sample = pd.Series({'Pclass':pclass, 'Floor':floor, 'Sex':sex, 'Age':age, 
								'SibSp':sibsp, 'Parch':parch})
            sample = (sample - m_train_set.mean()) / (m_train_set.max() - m_train_set.min())
            if (tmp_predict == 0):
                m_test_set.loc[location] = sample
                tmp_predict = selected_clf['Clf'].predict(sample[['Sex','Age','SibSp','Parch','Pclass','Floor']].values.reshape(1,-1))[0]
                m_pred_survived.loc[location]={'PassengerId':pid,'Survived':tmp_predict}
    else:
        floor = ord(re.findall('[A-Z]',cabins[0])[0]) - 65
        sample = pd.Series({'Pclass':pclass, 'Floor':floor, 'Sex':sex, 'Age':age, 'SibSp':sibsp, 'Parch':parch})
        sample = (sample - m_train_set.mean()) / (m_train_set.max() - m_train_set.min())
        m_test_set.loc[len(m_test_set)] = sample
        m_pred_survived.loc[len(m_pred_survived)]={'PassengerId':pid,'Survived':selected_clf['Clf'].predict(sample[['Sex','Age','SibSp','Parch','Pclass','Floor']].values.reshape(1,-1))[0]}

#==================================
print ("Visulizing classified test data")
#==================================
visualize_data (m_test_set, m_pred_survived, "ClassifiedTest")
m_pred_survived.to_csv("clasified.csv")