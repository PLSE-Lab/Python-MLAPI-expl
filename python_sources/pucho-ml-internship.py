"""
Machine Learning Model to classify password based on their strength

Dataset used: Password Strength Classifier by Bhavik bansal
(https://www.kaggle.com/bhavikbb/password-strength-classifier-dataset)

Dataset structure: (670000 rows, 2 columns)
Col1: Password
Col2: Strength containing values 0/1/2, with 0 as weak password and 2 as strong password

Plots in Comments Section
"""
#importing modules
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


nRowsRead = 10000 # specify 'None' if want to read whole file
# data.csv has 669879 rows in reality, but we are only loading/previewing the first 10000 rows
df1 = pd.read_csv('../input/data.csv', error_bad_lines = False, nrows = nRowsRead)

#Visualising the count of labels
import seaborn as sns
sns.countplot(df1['strength'])
#From the countplot, we can see that class '1' is over 70% of the whole dataset
#Hence, by definition, we have an imbalanced dataset

"""
The data we have is a list of passwords, which pandas identify has 'object type'.
But, to work with a ML model, we have to convert the text into numerical features.
Here I generated a series of numerical features inspired from the common text mining techniques.
"""
df1['char_count'] = df1['password'].str.len() #Number of characters in password
df1['numerics'] = df1['password'].apply(lambda x: len([str(x) for x in list(x) if str(x).isdigit()])) #No. of numerals in password
df1['alpha'] = df1['password'].apply(lambda x: len([x for x in list(x) if x.isalpha()])) #No. of alphabets in password

vowels = ['a', 'e', 'i', 'o', 'u']
#To check if the password is an actual english word or name, I extracted the number of vowels and consonants in the text
#The basic idea is: The more the consonants, less meaning the password makes
df1['vowels'] = df1['password'].apply(lambda x: len([x for x in list(x) if x in vowels]))
df1['consonants'] = df1['password'].apply(lambda x: len([x for x in list(x) if x not in vowels and x.isalpha()]))

df1.head()

"""
Preparing for building a SVM Classifier
"""
#Separating the class label from feature matrix
y = df1['strength'] 
X = df1.drop('strength', axis = 1)

#Splitting into training data and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=np.random)

#Scaling the input data and Training the SVM
svm = SVC(kernel="rbf", C=1.0, degree=3, coef0=1.0, random_state=np.random.RandomState()) #Gaussian Kernel SVM
X_train = X_train.drop('password', axis = 1) 
scaler = StandardScaler().fit(X_train)
X_train_tr = scaler.transform(X_train)

svm.fit(X_train_tr, y_train)
X_test = X_test.drop('password', axis = 1)
X_test_tr = scaler.transform(X_test)

y_pred = svm.predict(X_test_tr)

#Accuracy scores
print("Accuracy of SVM Classifier: {0:.2f}".format(balanced_accuracy_score(y_test, y_pred)))
scores = cross_val_score(svm, X_test_tr, y_test, cv = 10)
print("Mean accuracy after 10-fold cross validation: {0:.2f}".format(scores.mean()))

"""
As one can see from the scores, the model has near-perfect accuracy. Models like this should always be taken with a pinch
of salt. There is no obvious data leakage in the process. In order to ensure the model is actually efficient, I plotted
the features and differentiated them based on the class label.
"""
g = sns.pairplot(data = df1, hue = 'strength', vars = df1.columns.difference(['password', 'strength']), 
                 diag_kind = {'kde'}, palette = 'husl') #Plotting only the engineered features
                 
#It is evident from the pairplot that the features are spatially well-distinguished.
#Hence, one can say that the model actually performs well and the near-perfect score is not an anomaly.




