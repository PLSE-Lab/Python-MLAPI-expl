#newbie
#For SMS spam detection i am using navie bais algorithm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer  #could have used tf-idf feature extractor
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import seaborn as sns

df = pd.read_csv("../input/spam.csv", encoding = "latin1")
#drop the unwanted rows and colmuns
df = df.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
df = df.rename(columns={"v1":"label", "v2":"text"})
# convert label to a numerical variable
df['label_num'] = df.label.map({'ham':0, 'spam':1})

X_train, X_test, y_train, y_test = train_test_split(df["text"],df["label"], test_size = 0.2, random_state = 10)


#tranfrom text into frequency
cv = CountVectorizer()
X_train_df = cv.fit_transform(X_train)
X_test_df = cv.transform(X_test)

#machine algo
prediction = dict()
clf = MultinomialNB()
clf.fit(X_train_df,y_train)

prediction["MultinomialNB"] = clf.predict(X_test_df)

a = accuracy_score(y_test,prediction["MultinomialNB"])
print('The accuracy using MutlinomailNB is:',format(a*100))


#Model tuning and cross-valdation
k_range = np.arange(1,30)
param_grid = dict(n_neighbors=k_range)
model = KNeighborsClassifier()

grid = GridSearchCV(model,param_grid)
grid.fit(X_train_df,y_train)
print(grid.best_estimator_)
print(grid.best_params_)
print(grid.best_score_)

#Classifer performance evalution

print(classification_report(y_test, prediction['MultinomialNB'], target_names = ["Ham", "Spam"]))
conf_mat = confusion_matrix(y_test, prediction['MultinomialNB'])
conf_mat_normalized = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
sns.heatmap(conf_mat_normalized)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()



